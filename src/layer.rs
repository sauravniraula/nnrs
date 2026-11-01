use std::fmt::Display;

use crate::neuron::Neuron;

pub struct Layer {
    n_inputs: usize,
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let neurons: Vec<Neuron> = (0..n_neurons).map(|_| Neuron::new(n_inputs)).collect();

        Self { n_inputs, neurons }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), self.n_inputs);

        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(&inputs))
            .collect::<Vec<f32>>()
    }

    pub fn backward(&mut self, grads: &Vec<f32>) {}
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
