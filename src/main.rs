use nnrs::layer::Layer;

fn main() {
    let inputs = vec![1_f32, 2_f32];
    let goal = 5_f32;

    let mut l1 = Layer::new(2, 3);
    let mut l2 = Layer::new(3, 1);

    for i in 0..10 {
        let l1_output = l1.forward(&inputs);
        let l2_output = l2.forward(&l1_output);

        let loss = goal - l2_output[0];

        println!("Output: {:?}", l2_output);
        println!("Loss: {:?}", loss);

        l2.backward(grads);
    }
}
