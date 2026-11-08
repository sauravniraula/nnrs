use nnrs::{dense_layer::DenseLayer, tensor::Tensor};
use num_traits::pow;

fn main() {
    let goal = 10.0_f32;

    let layer1 = DenseLayer::<f32>::new(2, 10);
    let layer2 = DenseLayer::<f32>::new(10, 1);
    let x = Tensor::new(
        vec![1.0_f32, 2.0_f32, 1.9_f32, 8.4_f32, 3.5_f32, 7.0_f32],
        vec![3, 2],
    );

    let max_iteration = 100;
    for i in 0..max_iteration {
        let out1 = layer1.forward(&x);
        let out2 = layer2.forward(&out1);

        let out = out2.get_data()[0];
        let seed = out - goal;
        let loss = pow(seed, 2);
        println!(
            "{}/{} Prediction: {}, Loss: {}",
            i, max_iteration, out, loss
        );

        out2.backward(seed);
        layer2.step(0.001);
        layer1.step(0.001);
    }
}
