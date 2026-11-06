use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    hash::Hash,
    ops::{Add, Mul, Sub},
    rc::Rc,
    vec,
};

use crate::number::Number;

#[derive(Debug)]
pub struct TensorNode<T: Number> {
    data: RefCell<Vec<T>>,
    dim: Vec<u16>,
    size: usize,

    op: Option<TensorOperation<T>>,
    gradients: RefCell<Vec<T>>,
}

impl<T: Number> TensorNode<T> {
    pub fn new(data: Vec<T>, dim: Vec<u16>, op: Option<TensorOperation<T>>) -> Self {
        let size = dim.iter().product::<u16>() as usize;
        Self {
            data: RefCell::new(data),
            dim,
            size,
            op,
            gradients: RefCell::new(vec![T::zero(); size]),
        }
    }
}

#[derive(Debug)]
pub enum TensorOperationType {
    ADD,
    SUB,
    MUL,
}

#[derive(Debug)]
pub struct TensorOperation<T: Number> {
    pub kind: TensorOperationType,
    pub parents: Vec<Tensor<T>>,
}

#[derive(Debug)]
pub struct Tensor<T: Number> {
    node: Rc<TensorNode<T>>,
}

impl<T: Number> Tensor<T> {
    pub fn new(data: Vec<T>, dim: Vec<u16>) -> Self {
        let node = TensorNode::new(data, dim, None);
        Self {
            node: Rc::new(node),
        }
    }

    pub fn get_op(&self) -> &Option<TensorOperation<T>> {
        &self.node.op
    }

    pub fn get_data(&self) -> Ref<'_, Vec<T>> {
        self.node.data.borrow()
    }

    pub fn get_gradients(&self) -> Ref<'_, Vec<T>> {
        self.node.gradients.borrow()
    }

    pub fn set_gradients(&self, gradients: Vec<T>) {
        *self.node.gradients.borrow_mut() = gradients;
    }

    pub fn add_gradients(&self, gradients: &Vec<T>, scale: T) {
        for (a, b) in (*self.node.gradients.borrow_mut())
            .iter_mut()
            .zip(gradients)
        {
            *a = *a + (*b * scale);
        }
    }

    pub fn add_data(&self, data: &Vec<T>, scale: T) {
        for (a, b) in (*self.node.data.borrow_mut()).iter_mut().zip(data) {
            *a = *a + (*b * scale);
        }
    }

    fn topo_sort(&self, reset_gradients: bool) -> Vec<Tensor<T>> {
        let mut visited: HashSet<Tensor<T>> = HashSet::new();
        let mut tensors: Vec<Tensor<T>> = vec![];

        fn topo_sort_inner<T: Number>(
            vis: &mut HashSet<Tensor<T>>,
            tens: &mut Vec<Tensor<T>>,
            t: &Tensor<T>,
            reset: bool,
        ) {
            if !vis.insert(t.clone()) {
                return;
            }

            if reset {
                t.set_gradients(vec![T::zero(); t.node.size]);
            }

            let op = t.get_op();
            if let Some(o) = op {
                for each_t in o.parents.iter() {
                    topo_sort_inner(vis, tens, each_t, reset);
                }
            }
            tens.push(t.clone());
        }
        topo_sort_inner(&mut visited, &mut tensors, self, reset_gradients);
        tensors
    }

    pub fn backward(&self, seed: T) {
        let sorted = self.topo_sort(true);
        self.set_gradients(vec![seed; self.node.size]);

        for each in sorted.iter().rev() {
            if let Some(op) = each.get_op() {
                let grad = each.get_gradients();
                let p1 = &op.parents[0];
                let p2 = &op.parents[1];

                match op.kind {
                    TensorOperationType::ADD => {
                        p1.add_gradients(&grad, T::one());
                        p2.add_gradients(&grad, T::one());
                    }
                    TensorOperationType::SUB => {
                        p1.add_gradients(&grad, T::one());
                        p2.add_gradients(&grad, -T::one());
                    }
                    TensorOperationType::MUL => {
                        let a = p1.get_data()[0];
                        let b = p2.get_data()[0];

                        p1.add_gradients(&grad, b);
                        p2.add_gradients(&grad, a);
                    }
                }
            }
        }
    }

    pub fn step(&self, learning_rate: T) {
        let sorted = self.topo_sort(false);
        let mut i = 0;
        for each in sorted.iter().rev() {
            if (i == 5) {
                return;
            }
            i += 1;
            each.add_data(&each.get_gradients(), -T::one() * learning_rate);
        }
    }
}

impl<T: Number> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }
}
impl<T: Number> Eq for Tensor<T> {}
impl<T: Number> Hash for Tensor<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.node).hash(state);
    }
}

impl<T: Number> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

impl<T: Number> Add for &Tensor<T>
where
    for<'a> &'a T: Add<&'a T, Output = T>,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.node.dim, rhs.node.dim);

        Tensor {
            node: Rc::new(TensorNode::new(
                self.node
                    .data
                    .borrow()
                    .iter()
                    .zip(rhs.node.data.borrow().iter())
                    .map(|(a, b)| a + b)
                    .collect(),
                self.node.dim.clone(),
                Some(TensorOperation {
                    kind: TensorOperationType::ADD,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }
}

impl<T: Number> Sub for &Tensor<T>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.node.dim, rhs.node.dim);

        Tensor {
            node: Rc::new(TensorNode::new(
                self.node
                    .data
                    .borrow()
                    .iter()
                    .zip(rhs.node.data.borrow().iter())
                    .map(|(a, b)| a - b)
                    .collect(),
                self.node.dim.clone(),
                Some(TensorOperation {
                    kind: TensorOperationType::SUB,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }
}

impl<T: Number> Mul for &Tensor<T>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.node.dim, rhs.node.dim);

        Tensor {
            node: Rc::new(TensorNode::new(
                self.node
                    .data
                    .borrow()
                    .iter()
                    .zip(rhs.node.data.borrow().iter())
                    .map(|(a, b)| a * b)
                    .collect(),
                self.node.dim.clone(),
                Some(TensorOperation {
                    kind: TensorOperationType::MUL,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }
}
