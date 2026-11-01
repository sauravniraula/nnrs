use nnrs::layer::Layer;

fn main() {
    let inputs = vec![1_f32, 2_f32];
    let goal = 5_f32;

    let mut l1 = Layer::new(2, 10000, true);
    let mut l2 = Layer::new(10000, 1, false);

    for i in 0..10 {
        let l1_output = l1.forward(&inputs);
        let l2_output = l2.forward(&l1_output);

        let loss = goal - l2_output[0];

        println!("-------- Iteration {} ---------", i+1);
        println!("Output: {:?}", l2_output);
        println!("Loss: {:?}", loss);

        let l2_grads = l2.backward(&vec![-loss]);
        l1.backward(&l2_grads);
    }
}
