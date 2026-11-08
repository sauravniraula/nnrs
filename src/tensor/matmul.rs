use std::rc::Rc;

use super::{Tensor, TensorNode, TensorOperation, TensorOperationType};
use crate::number::Number;

struct MatmulLayout {
    output_dim: Vec<usize>,
    batch_dim: Vec<usize>,
    lhs_batch_strides: Vec<usize>,
    rhs_batch_strides: Vec<usize>,
    rows: usize,
    inner: usize,
    cols: usize,
}

impl MatmulLayout {
    fn new(lhs: &[usize], rhs: &[usize]) -> Self {
        assert!(
            !lhs.is_empty() && !rhs.is_empty(),
            "matmul operands must have at least one dimension"
        );
        let lhs_vector = lhs.len() == 1;
        let rhs_vector = rhs.len() == 1;
        let promoted_lhs = [1, lhs[0]];
        let promoted_rhs = [rhs[0], 1];
        let lhs = if lhs_vector { &promoted_lhs } else { lhs };
        let rhs = if rhs_vector { &promoted_rhs } else { rhs };
        let rows = lhs[lhs.len() - 2];
        let inner = lhs[lhs.len() - 1];
        let cols = rhs[rhs.len() - 1];
        assert_eq!(
            inner,
            rhs[rhs.len() - 2],
            "matmul inner dimensions must match"
        );

        let batch_rank = (lhs.len() - 2).max(rhs.len() - 2);
        let mut batch_dim = vec![1; batch_rank];
        for (axis, dim) in batch_dim.iter_mut().enumerate() {
            let lhs_dim = axis
                .checked_sub(batch_rank - (lhs.len() - 2))
                .map_or(1, |index| lhs[index]);
            let rhs_dim = axis
                .checked_sub(batch_rank - (rhs.len() - 2))
                .map_or(1, |index| rhs[index]);
            assert!(
                lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1,
                "matmul batch dimensions must be broadcast-compatible"
            );
            *dim = if lhs_dim == 1 { rhs_dim } else { lhs_dim };
        }

        let mut output_dim = batch_dim.clone();
        if !lhs_vector {
            output_dim.push(rows);
        }
        if !rhs_vector {
            output_dim.push(cols);
        }
        Self {
            output_dim,
            batch_dim,
            lhs_batch_strides: Self::batch_strides(lhs, batch_rank),
            rhs_batch_strides: Self::batch_strides(rhs, batch_rank),
            rows,
            inner,
            cols,
        }
    }

    fn batch_strides(dim: &[usize], batch_rank: usize) -> Vec<usize> {
        let mut strides = vec![0; batch_rank];
        let input_batch_rank = dim.len() - 2;
        let mut stride = dim[input_batch_rank] * dim[input_batch_rank + 1];
        for axis in (0..input_batch_rank).rev() {
            if dim[axis] != 1 {
                strides[batch_rank - input_batch_rank + axis] = stride;
            }
            stride *= dim[axis];
        }
        strides
    }

    fn for_each_product(&self, mut visit: impl FnMut(usize, usize, usize)) {
        for batch in 0..self.batch_dim.iter().product() {
            let (mut lhs_offset, mut rhs_offset) = (0, 0);
            let mut remaining = batch;
            for axis in (0..self.batch_dim.len()).rev() {
                let coordinate = remaining % self.batch_dim[axis];
                remaining /= self.batch_dim[axis];
                lhs_offset += coordinate * self.lhs_batch_strides[axis];
                rhs_offset += coordinate * self.rhs_batch_strides[axis];
            }
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let output = (batch * self.rows + row) * self.cols + col;
                    for k in 0..self.inner {
                        visit(
                            lhs_offset + row * self.inner + k,
                            rhs_offset + k * self.cols + col,
                            output,
                        );
                    }
                }
            }
        }
    }
}

impl<T: Number> Tensor<T> {
    pub fn matmul(&self, rhs: &Self) -> Self {
        let layout = MatmulLayout::new(&self.node.dim, &rhs.node.dim);
        let a = self.get_data();
        let b = rhs.get_data();
        let mut data = vec![T::zero(); layout.output_dim.iter().product()];
        layout.for_each_product(|i, j, out| data[out] += a[i] * b[j]);
        Self {
            node: Rc::new(TensorNode::new(
                data,
                layout.output_dim,
                Some(TensorOperation {
                    kind: TensorOperationType::MATMUL,
                    parents: vec![self.clone(), rhs.clone()],
                }),
            )),
        }
    }

    pub(super) fn matmul_backward(&self, rhs: &Self, grad: &[T]) {
        let layout = MatmulLayout::new(&self.node.dim, &rhs.node.dim);
        let a = self.get_data();
        let b = rhs.get_data();
        let mut da = vec![T::zero(); self.node.size];
        let mut db = vec![T::zero(); rhs.node.size];
        layout.for_each_product(|i, j, out| {
            da[i] += grad[out] * b[j];
            db[j] += grad[out] * a[i];
        });
        self.add_gradients(&da, T::one());
        rhs.add_gradients(&db, T::one());
    }
}
