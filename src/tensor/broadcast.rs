use std::{
    ops::{Add, Mul, Sub},
    rc::Rc,
};

use super::{Tensor, TensorNode, TensorOperation, TensorOperationType};
use crate::number::Number;

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $kind:ident) => {
        impl<T: Number> $trait for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.elementwise(rhs, TensorOperationType::$kind)
            }
        }

        impl<T: Number> $trait<&Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: &Tensor<T>) -> Self::Output {
                self.elementwise(rhs, TensorOperationType::$kind)
            }
        }

        impl<T: Number> $trait<Tensor<T>> for &Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Tensor<T>) -> Self::Output {
                self.elementwise(&rhs, TensorOperationType::$kind)
            }
        }

        impl<T: Number> $trait for Tensor<T> {
            type Output = Tensor<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.elementwise(&rhs, TensorOperationType::$kind)
            }
        }
    };
}

impl_binary_op!(Add, add, ADD);
impl_binary_op!(Sub, sub, SUB);
impl_binary_op!(Mul, mul, MUL);

struct BroadcastLayout {
    dim: Vec<usize>,
    lhs_strides: Vec<usize>,
    rhs_strides: Vec<usize>,
}

impl BroadcastLayout {
    fn new(lhs: &[usize], rhs: &[usize]) -> Self {
        let rank = lhs.len().max(rhs.len());
        let mut dim = vec![1; rank];
        for (axis, output_dim) in dim.iter_mut().enumerate() {
            let lhs_dim = axis.checked_sub(rank - lhs.len()).map_or(1, |i| lhs[i]);
            let rhs_dim = axis.checked_sub(rank - rhs.len()).map_or(1, |i| rhs[i]);
            assert!(
                lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1,
                "tensor shapes must be broadcast-compatible: {lhs:?} and {rhs:?}"
            );
            *output_dim = if lhs_dim == 1 { rhs_dim } else { lhs_dim };
        }
        Self {
            dim,
            lhs_strides: Self::strides(lhs, rank),
            rhs_strides: Self::strides(rhs, rank),
        }
    }

    fn strides(dim: &[usize], rank: usize) -> Vec<usize> {
        let mut strides = vec![0; rank];
        let mut stride = 1;
        for axis in (0..dim.len()).rev() {
            if dim[axis] != 1 {
                strides[rank - dim.len() + axis] = stride;
            }
            stride *= dim[axis];
        }
        strides
    }

    fn for_each_index(&self, mut visit: impl FnMut(usize, usize, usize)) {
        for out in 0..self.dim.iter().product() {
            let (mut lhs, mut rhs) = (0, 0);
            let mut remaining = out;
            for axis in (0..self.dim.len()).rev() {
                let coordinate = remaining % self.dim[axis];
                remaining /= self.dim[axis];
                lhs += coordinate * self.lhs_strides[axis];
                rhs += coordinate * self.rhs_strides[axis];
            }
            visit(lhs, rhs, out);
        }
    }
}

impl<T: Number> Tensor<T> {
    pub(super) fn elementwise(&self, rhs: &Self, kind: TensorOperationType) -> Self {
        let layout = BroadcastLayout::new(&self.node.dim, &rhs.node.dim);
        let a = self.get_data();
        let b = rhs.get_data();
        let mut data = vec![T::zero(); layout.dim.iter().product()];
        layout.for_each_index(|i, j, out| {
            data[out] = match kind {
                TensorOperationType::ADD => a[i] + b[j],
                TensorOperationType::SUB => a[i] - b[j],
                TensorOperationType::MUL => a[i] * b[j],
                TensorOperationType::MATMUL => unreachable!("matmul is not elementwise"),
            };
        });
        Self {
            node: Rc::new(TensorNode::new(
                data,
                layout.dim,
                Some(TensorOperation {
                    kind,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }

    pub(super) fn elementwise_backward(&self, rhs: &Self, grad: &[T], kind: &TensorOperationType) {
        let layout = BroadcastLayout::new(&self.node.dim, &rhs.node.dim);
        let a = self.get_data();
        let b = rhs.get_data();
        let mut da = vec![T::zero(); self.node.size];
        let mut db = vec![T::zero(); rhs.node.size];
        layout.for_each_index(|i, j, out| {
            let (lhs_grad, rhs_grad) = match kind {
                TensorOperationType::ADD => (grad[out], grad[out]),
                TensorOperationType::SUB => (grad[out], -grad[out]),
                TensorOperationType::MUL => (grad[out] * b[j], grad[out] * a[i]),
                TensorOperationType::MATMUL => unreachable!("matmul is not elementwise"),
            };
            da[i] += lhs_grad;
            db[j] += rhs_grad;
        });
        self.add_gradients(&da, T::one());
        rhs.add_gradients(&db, T::one());
    }
}
