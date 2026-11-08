use nnrs::tensor::{Tensor, TensorOperationType};

#[test]
fn matrix_matmul_propagates_gradients() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let output = a.matmul(&b);

    assert_eq!(output.get_dim(), &[2, 2]);
    assert_eq!(*output.get_data(), vec![58.0, 64.0, 139.0, 154.0]);
    assert!(matches!(
        output.get_op().as_ref().unwrap().kind,
        TensorOperationType::MATMUL
    ));

    output.backward(1.0);
    assert_eq!(*a.get_gradients(), vec![15.0, 19.0, 23.0, 15.0, 19.0, 23.0]);
    assert_eq!(*b.get_gradients(), vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn broadcasts_multiple_batch_axes_in_backward() {
    let a = Tensor::<f64>::ones(vec![2, 1, 2, 3]);
    let b = Tensor::<f64>::ones(vec![1, 4, 3, 2]);
    let output = a.matmul(&b);

    assert_eq!(output.get_dim(), &[2, 4, 2, 2]);
    assert_eq!(*output.get_data(), vec![3.0; 32]);
    output.backward(0.5);
    assert_eq!(*a.get_gradients(), vec![4.0; 12]);
    assert_eq!(*b.get_gradients(), vec![2.0; 24]);
}

#[test]
fn promotes_vectors_and_removes_only_promoted_axes() {
    for (a_dim, b_dim, output_dim, da, db) in [
        (vec![3], vec![3], vec![], 1.0, 1.0),
        (vec![3], vec![2, 3, 2], vec![2, 2], 4.0, 1.0),
        (vec![2, 2, 3], vec![3], vec![2, 2], 1.0, 4.0),
        (vec![1, 3], vec![3], vec![1], 1.0, 1.0),
    ] {
        let a = Tensor::<f64>::ones(a_dim);
        let b = Tensor::<f64>::ones(b_dim);
        let output = a.matmul(&b);
        assert_eq!(output.get_dim(), output_dim);
        assert!(output.get_data().iter().all(|&value| value == 3.0));
        output.backward(1.0);
        assert!(a.get_gradients().iter().all(|&value| value == da));
        assert!(b.get_gradients().iter().all(|&value| value == db));
    }
}

fn weighted_loss(a: &Tensor<f64>, b: &Tensor<f64>, weights: &Tensor<f64>) -> f64 {
    a.matmul(b)
        .get_data()
        .iter()
        .zip(weights.get_data().iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

#[test]
fn backward_matches_finite_differences_with_nonuniform_upstream_gradients() {
    for (a_dim, b_dim) in [
        (vec![2, 3], vec![3, 2]),
        (vec![2, 2, 3], vec![2, 3, 2]),
        (vec![2, 1, 2, 3], vec![1, 3, 3, 2]),
        (vec![2, 3], vec![2, 3, 2]),
        (vec![2, 2, 3], vec![3, 2]),
        (vec![3], vec![3]),
        (vec![3], vec![2, 3, 2]),
        (vec![2, 2, 3], vec![3]),
    ] {
        let a = Tensor::new(
            (0..a_dim.iter().product())
                .map(|i| (i as f64 - 2.5) / 7.0)
                .collect(),
            a_dim,
        );
        let b = Tensor::new(
            (0..b_dim.iter().product())
                .map(|i| (i as f64 + 1.0) / 5.0)
                .collect(),
            b_dim,
        );
        let output = a.matmul(&b);
        let weights = Tensor::new(
            (0..output.get_data().len())
                .map(|i| (i as f64 % 7.0 - 3.0) / 4.0)
                .collect(),
            output.get_dim().to_vec(),
        );
        (&output * &weights).backward(1.0);

        let epsilon = 1e-6;
        for input in [&a, &b] {
            for (index, &analytic) in input.get_gradients().iter().enumerate() {
                let mut delta = vec![0.0; input.get_data().len()];
                delta[index] = epsilon;
                input.add_data(&delta, 1.0);
                let plus = weighted_loss(&a, &b, &weights);
                input.add_data(&delta, -2.0);
                let minus = weighted_loss(&a, &b, &weights);
                input.add_data(&delta, 1.0);
                let numeric = (plus - minus) / (2.0 * epsilon);
                assert!(
                    (analytic - numeric).abs() < 1e-6,
                    "shapes {:?} @ {:?}, input {:?}, index {index}: analytic {analytic}, numeric {numeric}",
                    a.get_dim(),
                    b.get_dim(),
                    input.get_dim()
                );
            }
        }
    }
}

#[test]
fn broadcast_batches_use_the_correct_input_offsets() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 1, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![1, 3, 2, 1]);
    let output = a.matmul(&b);
    assert_eq!(output.get_dim(), &[2, 3, 1, 1]);
    assert_eq!(*output.get_data(), vec![17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
}

#[test]
fn shared_operands_accumulate_gradients_and_backward_resets_them() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let output = a.matmul(&a);
    let branched = &output + &output;
    let expected = [7.0, 11.0, 9.0, 13.0];
    for seed in [1.0, -0.5] {
        branched.backward(seed);
        assert_eq!(
            *a.get_gradients(),
            expected
                .iter()
                .map(|value| value * 2.0 * seed)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn chained_matmul_propagates_to_all_leaves() {
    let a = Tensor::<f32>::ones(vec![2, 3]);
    let b = Tensor::<f32>::ones(vec![3, 4]);
    let c = Tensor::<f32>::ones(vec![4, 1]);
    a.matmul(&b).matmul(&c).backward(1.0);
    assert_eq!(*a.get_gradients(), vec![4.0; 6]);
    assert_eq!(*b.get_gradients(), vec![2.0; 12]);
    assert_eq!(*c.get_gradients(), vec![6.0; 4]);
}

#[test]
fn zero_sized_axes_produce_valid_outputs_and_gradients() {
    for (a_dim, b_dim, output_dim) in [
        (vec![2, 0], vec![0, 3], vec![2, 3]),
        (vec![0, 2], vec![2, 3], vec![0, 3]),
        (vec![2, 3], vec![3, 0], vec![2, 0]),
        (vec![0, 2, 3], vec![1, 3, 2], vec![0, 2, 2]),
        (vec![1, 2, 3], vec![0, 3, 2], vec![0, 2, 2]),
        (vec![0], vec![0], vec![]),
    ] {
        let a = Tensor::<f64>::ones(a_dim);
        let b = Tensor::<f64>::ones(b_dim);
        let output = a.matmul(&b);
        assert_eq!(output.get_dim(), output_dim);
        assert_eq!(output.get_data().len(), output_dim.iter().product());
        assert!(output.get_data().iter().all(|&value| value == 0.0));
        output.backward(1.0);
        assert!(a.get_gradients().iter().all(|&value| value == 0.0));
        assert!(b.get_gradients().iter().all(|&value| value == 0.0));
    }
}

#[test]
#[should_panic(expected = "matmul inner dimensions must match")]
fn rejects_mismatched_inner_dimensions() {
    Tensor::<f64>::ones(vec![2, 3]).matmul(&Tensor::ones(vec![4, 2]));
}

#[test]
#[should_panic(expected = "matmul batch dimensions must be broadcast-compatible")]
fn rejects_incompatible_batch_dimensions() {
    Tensor::<f64>::ones(vec![2, 2, 3]).matmul(&Tensor::ones(vec![3, 3, 2]));
}

#[test]
#[should_panic(expected = "matmul operands must have at least one dimension")]
fn rejects_scalar_left_operand() {
    Tensor::<f64>::ones(vec![]).matmul(&Tensor::ones(vec![1, 1]));
}

#[test]
#[should_panic(expected = "matmul operands must have at least one dimension")]
fn rejects_scalar_right_operand() {
    Tensor::<f64>::ones(vec![1, 1]).matmul(&Tensor::ones(vec![]));
}
