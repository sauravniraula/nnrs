use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    hash::Hash,
    ops::{Add, Mul, Range, Sub},
    rc::Rc,
    vec,
};

use crate::number::Number;

#[derive(Debug)]
pub struct TensorNode<T: Number> {
    data: RefCell<Vec<T>>,
    dim: Vec<usize>,
    size: usize,

    op: Option<TensorOperation<T>>,
    gradients: RefCell<Vec<T>>,
}

impl<T: Number> TensorNode<T> {
    pub fn new(data: Vec<T>, dim: Vec<usize>, op: Option<TensorOperation<T>>) -> Self {
        let size = dim.iter().product();
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
    MATMUL,
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
    pub fn new(data: Vec<T>, dim: Vec<usize>) -> Self {
        let node = TensorNode::new(data, dim, None);
        Self {
            node: Rc::new(node),
        }
    }

    pub fn new_filled(with: impl Fn() -> T, dim: Vec<usize>) -> Self {
        let size = dim.iter().product();
        Self::new((0..size).map(|_| with()).collect(), dim)
    }

    pub fn zeros(dim: Vec<usize>) -> Self {
        Self::new_filled(|| T::zero(), dim)
    }

    pub fn ones(dim: Vec<usize>) -> Self {
        Self::new_filled(|| T::one(), dim)
    }

    pub fn random(range: Range<T>, dim: Vec<usize>) -> Self {
        Self::new_filled(|| rand::random_range(range.clone()), dim)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.node.dim.len(), 2, "matmul operands must be 2-D");
        assert_eq!(rhs.node.dim.len(), 2, "matmul operands must be 2-D");
        let rows = self.node.dim[0];
        let inner = self.node.dim[1];
        let cols = rhs.node.dim[1];
        assert_eq!(inner, rhs.node.dim[0], "matmul inner dimensions must match");

        let a = self.get_data();
        let b = rhs.get_data();
        let mut data = vec![T::zero(); rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                for k in 0..inner {
                    data[row * cols + col] += a[row * inner + k] * b[k * cols + col];
                }
            }
        }
        Self {
            node: Rc::new(TensorNode::new(
                data,
                vec![rows, cols],
                Some(TensorOperation {
                    kind: TensorOperationType::MATMUL,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }

    pub fn get_op(&self) -> &Option<TensorOperation<T>> {
        &self.node.op
    }

    pub fn get_data(&self) -> Ref<'_, Vec<T>> {
        self.node.data.borrow()
    }

    pub fn get_dim(&self) -> &Vec<usize> {
        &self.node.dim
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
                        let da = grad
                            .iter()
                            .zip(p2.get_data().iter())
                            .map(|(g, b)| *g * *b)
                            .collect();
                        let db = grad
                            .iter()
                            .zip(p1.get_data().iter())
                            .map(|(g, a)| *g * *a)
                            .collect();
                        p1.add_gradients(&da, T::one());
                        p2.add_gradients(&db, T::one());
                    }
                    TensorOperationType::MATMUL => {
                        let rows = p1.node.dim[0];
                        let inner = p1.node.dim[1];
                        let cols = p2.node.dim[1];
                        let a = p1.get_data();
                        let b = p2.get_data();
                        let mut da = vec![T::zero(); p1.node.size];
                        let mut db = vec![T::zero(); p2.node.size];
                        for row in 0..rows {
                            for col in 0..cols {
                                for k in 0..inner {
                                    da[row * inner + k] +=
                                        grad[row * cols + col] * b[k * cols + col];
                                    db[k * cols + col] +=
                                        grad[row * cols + col] * a[row * inner + k];
                                }
                            }
                        }
                        p1.add_gradients(&da, T::one());
                        p2.add_gradients(&db, T::one());
                    }
                }
            }
        }
    }

    pub fn step(&self, learning_rate: T) {
        self.add_data(&self.get_gradients(), -T::one() * learning_rate);
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
