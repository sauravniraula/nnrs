use crate::{number::Number, tensor::Tensor};

pub struct DenseLayer<T: Number> {
    pub n_inputs: usize,
    pub n_outputs: usize,

    weights: Tensor<T>,
    bias: Tensor<T>,
}

impl<T: Number> DenseLayer<T> {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        let weights = Tensor::random(-T::one()..T::one(), vec![n_inputs, n_outputs]);
        let bias = Tensor::random(-T::one()..T::one(), vec![1, n_outputs]);
        Self {
            n_inputs,
            n_outputs,
            weights,
            bias,
        }
    }

    pub fn forward(&self, inputs: &Tensor<T>) -> Tensor<T>
    where
        for<'a> &'a T: std::ops::Add<&'a T, Output = T>,
    {
        assert_eq!(inputs.get_dim().len(), 2, "dense inputs must be 2-D");
        assert_eq!(inputs.get_dim()[1], self.n_inputs);
        let output = inputs.matmul(&self.weights);
        let bias = Tensor::ones(vec![inputs.get_dim()[0], 1]).matmul(&self.bias);
        &output + &bias
    }

    pub fn step(&self, learning_rate: T) {
        self.weights.step(learning_rate);
        self.bias.step(learning_rate);
    }
}
