use std::fmt::Display;

pub struct Neuron {
    n_inputs: usize,
    activation: bool,

    weights: Vec<f32>,
    bias: f32,

    last_output: Option<f32>,
}

impl Neuron {
    pub fn new(n_inputs: usize, activation: bool) -> Self {
        let weights = (0..n_inputs)
            .map(|_| rand::random_range(-1.0..1.0))
            .collect();
        let bias = rand::random_range(-1.0..1.0);

        Self {
            n_inputs,
            activation,

            weights,
            bias,

            last_output: None,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> f32 {
        assert_eq!(inputs.len(), self.n_inputs);

        let mut sum = 0_f32;
        for i in 0..self.n_inputs {
            sum += self.weights[i] * inputs[i];
        }

        let mut x = sum + self.bias;
        if self.activation {
            x = x.tanh()
        }
        self.last_output = Some(x);

        x
    }

    pub fn backward(&mut self, last_inputs: &Vec<f32>, grad: f32) -> Vec<f32> {
        if let Some(last_output) = self.last_output {
            let mut back_gradients = Vec::<f32>::with_capacity(self.n_inputs);
            let val = if self.activation {
                1.0 - last_output.powi(2)
            } else {
                1.0
            };
            for i in 0..self.n_inputs {
                back_gradients.push(self.weights[i] * val * grad);
                self.weights[i] -= last_inputs[i] * val * grad * 0.00005;
            }
            self.bias -= val * grad * 0.00005;
            return back_gradients;
        }
        panic!("run forward before running backward")
    }
}

impl Display for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!();
    }
}
