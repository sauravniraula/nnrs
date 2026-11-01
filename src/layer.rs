use std::fmt::Display;

use crate::neuron::Neuron;

pub struct Layer {
    n_inputs: usize,
    neurons: Vec<Neuron>,

    last_inputs: Option<Vec<f32>>,
}

impl Layer {
    pub fn new(n_inputs: usize, n_neurons: usize, activation: bool) -> Self {
        let neurons: Vec<Neuron> = (0..n_neurons).map(|_| Neuron::new(n_inputs, activation)).collect();

        Self {
            n_inputs,
            neurons,
            last_inputs: None,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        assert_eq!(inputs.len(), self.n_inputs);

        self.last_inputs = Some(inputs.clone());
        self.neurons
            .iter_mut()
            .map(|neuron| neuron.forward(&inputs))
            .collect::<Vec<f32>>()
    }

    pub fn backward(&mut self, grads: &Vec<f32>) -> Vec<f32> {
        if let Some(last_inputs) = &self.last_inputs {
            assert_eq!(grads.len(), self.neurons.len());

            let mut back_gradients = vec![0_f32; self.n_inputs];
            for (i, neuron) in self.neurons.iter_mut().enumerate() {
                let neuron_back_gradients = neuron.backward(last_inputs, grads[i]);
                for (i, back_grad) in neuron_back_gradients.iter().enumerate() {
                    back_gradients[i] += back_grad;
                }
            }
            return back_gradients;
        }
        panic!("run forward before running backward");
    }
}

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}
