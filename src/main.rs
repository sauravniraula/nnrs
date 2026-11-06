use nnrs::{draw::draw_computation_graph, tensor::Tensor};

fn main() {
    let t1 = Tensor::new([2.0].into(), [1, 1].into());
    let t2 = Tensor::new([-5.0].into(), [1, 1].into());
    let t3 = Tensor::new([3.0].into(), [1, 1].into());

    let t4 = Tensor::new([9.0].into(), [1, 1].into());
    let t5 = Tensor::new([4.0].into(), [1, 1].into());
    let t6 = Tensor::new([7.0].into(), [1, 1].into());

    let t7 = &t1 * &t2;
    let t8 = &t7 + &t3;

    let t9 = &t4 * &t5;
    let t10 = &t9 + &t6;

    let t11 = Tensor::new([2.0].into(), [1, 1].into());
    let t12 = Tensor::new([-3.0].into(), [1, 1].into());
    let t13 = Tensor::new([1.0].into(), [1, 1].into());

    let t14 = &t8 * &t11;
    let t15 = &t10 * &t12;

    let t16 = &t14 + &t15;
    let t17 = &t16 + &t13;

    t17.backward(10.0);

    draw_computation_graph(&t17).unwrap();
}
