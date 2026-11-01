use std::fmt::Display;

pub struct Neuron {
    n_inputs: usize,

    weights: Vec<f32>,
    bias: f32,

    _last_inputs: Option<Vec<f32>>
}

impl Neuron {
    pub fn new(n_inputs: usize) -> Self {
        let weights = (0..n_inputs)
            .map(|_| rand::random_range(-1.0..1.0))
            .collect();
        let bias = rand::random_range(-1.0..1.0);

        Self {
            n_inputs,

            weights,
            bias,
            _last_inputs: None
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> f32 {
        assert_eq!(inputs.len(), self.n_inputs);

        let mut sum = 0_f32;
        for i in 0..self.n_inputs {
            sum += self.weights[i] * inputs[i];
        }

        self._last_inputs = Some(inputs.clone());
        sum + self.bias
    }

    pub fn backward(&mut self, gradient: f32) {
        if let Some(inputs) = &self._last_inputs {
            for i in 0..self.n_inputs {
                self.weights[i] -= inputs[i] * gradient;
            }
            self.bias -= gradient;
        } else {
            println!("Neuron.forward() not called yet")
        }
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!();
    }
}
