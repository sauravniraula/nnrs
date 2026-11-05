use std::{
    default,
    ops::{Add, Sub},
    rc::Rc,
};

#[derive(Debug)]
pub struct TensorNode<T> {
    data: Vec<T>,
    dim: Vec<u16>,

    op: Option<TensorOperation<T>>,
    gradient: f32,
}

#[derive(Debug)]
pub enum TensorOperationType {
    ADD,
    SUB,
}

#[derive(Debug)]
pub struct TensorOperation<T> {
    pub kind: TensorOperationType,
    pub parents: Vec<Tensor<T>>,
}

#[derive(Debug)]
pub struct Tensor<T> {
    node: Rc<TensorNode<T>>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, dim: Vec<u16>) -> Self {
        let node = TensorNode {
            data,
            dim,
            op: None,
            gradient: 0.0,
        };
        Self {
            node: Rc::new(node),
        }
    }

    pub fn get_op(&self) -> &Option<TensorOperation<T>> {
        &self.node.op
    }

    pub fn get_data(&self) -> &Vec<T> {
        &self.node.data
    }
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
        }
    }
}

impl<T> Add for &Tensor<T>
where
    for<'a> &'a T: Add<&'a T, Output = T>,
{
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.node.dim, rhs.node.dim);

        Tensor {
            node: Rc::new(TensorNode {
                data: self
                    .node
                    .data
                    .iter()
                    .zip(rhs.node.data.iter())
                    .map(|(a, b)| a + b)
                    .collect(),
                dim: self.node.dim.clone(),
                op: Some(TensorOperation {
                    kind: TensorOperationType::ADD,
                    parents: vec![self.clone(), rhs.clone()],
                }),
                gradient: 0.0,
            }),
        }
    }
}

impl<T> Sub for &Tensor<T>
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.node.dim, rhs.node.dim);

        Tensor {
            node: Rc::new(TensorNode {
                data: self
                    .node
                    .data
                    .iter()
                    .zip(rhs.node.data.iter())
                    .map(|(a, b)| a - b)
                    .collect(),
                dim: self.node.dim.clone(),
                op: Some(TensorOperation {
                    kind: TensorOperationType::SUB,
                    parents: vec![self.clone(), rhs.clone()],
                }),
                gradient: 0.0
            }),
        }
    }
}
