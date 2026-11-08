use nnrs::tensor::Tensor;

#[test]
fn backward_uses_each_operand_element() {
    let a_data = vec![1.0, -2.0, 0.0, 4.0];
    let b_data = vec![-3.0, 5.0, 7.0, 0.0];
    let a = Tensor::new(a_data.clone(), vec![2, 2]);
    let b = Tensor::new(b_data.clone(), vec![2, 2]);
    let output = &a * &b;

    for seed in [1.0, -0.5] {
        output.backward(seed);
        assert_eq!(
            *a.get_gradients(),
            b_data.iter().map(|b| seed * b).collect::<Vec<_>>()
        );
        assert_eq!(
            *b.get_gradients(),
            a_data.iter().map(|a| seed * a).collect::<Vec<_>>()
        );
    }
}

#[test]
fn backward_handles_nonuniform_upstream_gradients() {
    let a_data = vec![1.0, -2.0, 0.0, 4.0];
    let b_data = vec![-3.0, 5.0, 7.0, 0.0];
    let weights_data = vec![2.0, -1.0, 0.5, 3.0];
    let a = Tensor::new(a_data.clone(), vec![2, 1, 2]);
    let b = Tensor::new(b_data.clone(), vec![2, 1, 2]);
    let weights = Tensor::new(weights_data.clone(), vec![2, 1, 2]);
    let output = &a * &b;
    (&output * &weights).backward(0.5);

    for (input, other) in [(&a, &b_data), (&b, &a_data)] {
        assert_eq!(
            *input.get_gradients(),
            other
                .iter()
                .zip(&weights_data)
                .map(|(value, weight)| 0.5 * value * weight)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(
        *weights.get_gradients(),
        a_data
            .iter()
            .zip(&b_data)
            .map(|(a, b)| 0.5 * a * b)
            .collect::<Vec<_>>()
    );
}

#[test]
fn shared_operands_and_branches_accumulate_gradients() {
    let data = vec![1.0, -2.0, 0.0, 4.0];
    let a = Tensor::new(data.clone(), vec![2, 2]);
    let square = &a * &a;
    let output = &square + &square;
    output.backward(0.5);
    assert_eq!(
        *a.get_gradients(),
        data.iter().map(|a| 2.0 * a).collect::<Vec<_>>()
    );
}

#[test]
fn backward_handles_scalars_and_empty_tensors() {
    for dim in [vec![], vec![0], vec![2, 0, 3]] {
        let a = Tensor::<f32>::new_filled(|| -2.0, dim.clone());
        let b = Tensor::<f32>::new_filled(|| 3.0, dim);
        (&a * &b).backward(0.5);
        assert_eq!(*a.get_gradients(), vec![1.5; a.get_data().len()]);
        assert_eq!(*b.get_gradients(), vec![-1.0; b.get_data().len()]);
    }
}
