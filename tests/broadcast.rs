use nnrs::tensor::Tensor;

#[test]
fn addition_broadcasts_bias_and_reduces_its_gradients() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let output = &a + &bias;
    assert_eq!(output.get_dim(), &[2, 3]);
    assert_eq!(*output.get_data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    output.backward(0.5);
    assert_eq!(*a.get_gradients(), vec![0.5; 6]);
    assert_eq!(*bias.get_gradients(), vec![1.0; 3]);
}

#[test]
fn matmul_plus_vector_bias_supports_owned_results() {
    let x = Tensor::<f64>::ones(vec![2, 3]);
    let weights = Tensor::<f64>::ones(vec![3, 4]);
    let bias_data = vec![1.0, -2.0, 3.0, -4.0];
    let bias = Tensor::new(bias_data.clone(), vec![4]);
    let output = x.matmul(&weights) + &bias;
    assert_eq!(output.get_dim(), &[2, 4]);
    assert_eq!(
        *output.get_data(),
        bias_data
            .iter()
            .map(|bias| 3.0 + bias)
            .cycle()
            .take(8)
            .collect::<Vec<_>>()
    );
    output.backward(1.0);
    assert_eq!(*x.get_gradients(), vec![4.0; 6]);
    assert_eq!(*weights.get_gradients(), vec![2.0; 12]);
    assert_eq!(*bias.get_gradients(), vec![2.0; 4]);
}

#[test]
fn all_operators_support_owned_and_borrowed_operands() {
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0], vec![2]);
    macro_rules! check_forms {
        ($op:tt) => {{
            let expected = &a $op &b;
            expected.backward(0.5);
            let da = a.get_gradients().clone();
            let db = b.get_gradients().clone();
            for output in [a.clone() $op &b, &a $op b.clone(), a.clone() $op b.clone()] {
                assert_eq!(*output.get_data(), *expected.get_data());
                assert_eq!(output.get_dim(), expected.get_dim());
                output.backward(0.5);
                assert_eq!(*a.get_gradients(), da);
                assert_eq!(*b.get_gradients(), db);
            }
        }};
    }
    check_forms!(+);
    check_forms!(-);
    check_forms!(*);
}

fn apply(a: &Tensor<f64>, b: &Tensor<f64>, op: char) -> Tensor<f64> {
    match op {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        _ => unreachable!(),
    }
}

#[test]
fn multiple_broadcast_axes_use_correct_input_elements() {
    let a_data = [1.0, -2.0, 3.0, -4.0];
    let b_data = [5.0, -6.0, 7.0];
    let a = Tensor::new(a_data.to_vec(), vec![2, 1, 2]);
    let b = Tensor::new(b_data.to_vec(), vec![1, 3, 1]);
    for op in ['+', '-', '*'] {
        let output = apply(&a, &b, op);
        let mut expected = vec![];
        for row in a_data.chunks(2) {
            for b in b_data {
                for a in row {
                    expected.push(match op {
                        '+' => a + b,
                        '-' => a - b,
                        '*' => a * b,
                        _ => unreachable!(),
                    });
                }
            }
        }
        assert_eq!(output.get_dim(), &[2, 3, 2]);
        assert_eq!(*output.get_data(), expected);
    }
}

fn weighted_loss(a: &Tensor<f64>, b: &Tensor<f64>, weights: &Tensor<f64>, op: char) -> f64 {
    apply(a, b, op)
        .get_data()
        .iter()
        .zip(weights.get_data().iter())
        .map(|(value, weight)| value * weight)
        .sum()
}

#[test]
fn broadcast_gradients_match_finite_differences() {
    for op in ['+', '-', '*'] {
        for (a_dim, b_dim) in [
            (vec![2, 3], vec![3]),
            (vec![3], vec![2, 3]),
            (vec![2, 1, 3], vec![1, 4, 1]),
            (vec![2, 3], vec![]),
            (vec![], vec![2, 3]),
            (vec![], vec![]),
            (vec![1], vec![2, 1, 3]),
            (vec![2, 3], vec![2, 3]),
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
            let output = apply(&a, &b, op);
            let weights = Tensor::new(
                (0..output.get_data().len())
                    .map(|i| (i as f64 % 7.0 - 3.0) / 4.0)
                    .collect(),
                output.get_dim().to_vec(),
            );
            let seed = -0.5;
            (&output * &weights).backward(seed);
            let epsilon = 1e-6;
            for input in [&a, &b] {
                assert_eq!(input.get_gradients().len(), input.get_data().len());
                for (index, &analytic) in input.get_gradients().iter().enumerate() {
                    let mut delta = vec![0.0; input.get_data().len()];
                    delta[index] = epsilon;
                    input.add_data(&delta, 1.0);
                    let plus = weighted_loss(&a, &b, &weights, op);
                    input.add_data(&delta, -2.0);
                    let minus = weighted_loss(&a, &b, &weights, op);
                    input.add_data(&delta, 1.0);
                    let numeric = seed * (plus - minus) / (2.0 * epsilon);
                    assert!(
                        (analytic - numeric).abs() < 1e-6,
                        "{:?} {op} {:?}, input {:?}, index {index}: analytic {analytic}, numeric {numeric}",
                        a.get_dim(),
                        b.get_dim(),
                        input.get_dim()
                    );
                }
            }
        }
    }
}

#[test]
fn empty_broadcast_outputs_leave_zero_input_gradients() {
    for (a_dim, b_dim, output_dim) in [
        (vec![0, 3], vec![3], vec![0, 3]),
        (vec![1, 3], vec![0, 3], vec![0, 3]),
        (vec![2, 0, 3], vec![1, 3], vec![2, 0, 3]),
        (vec![], vec![0], vec![0]),
        (vec![0], vec![], vec![0]),
    ] {
        let a = Tensor::<f64>::ones(a_dim);
        let b = Tensor::<f64>::ones(b_dim);
        for op in ['+', '-', '*'] {
            let output = apply(&a, &b, op);
            assert_eq!(output.get_dim(), &output_dim);
            assert!(output.get_data().is_empty());
            output.backward(1.0);
            assert_eq!(*a.get_gradients(), vec![0.0; a.get_data().len()]);
            assert_eq!(*b.get_gradients(), vec![0.0; b.get_data().len()]);
        }
    }
}

#[test]
fn all_operators_reject_incompatible_shapes() {
    for op in ['+', '-', '*'] {
        for (a_dim, b_dim) in [
            (vec![2, 3], vec![2]),
            (vec![2, 2, 3], vec![4, 3]),
            (vec![0], vec![2]),
        ] {
            let result = std::panic::catch_unwind(|| {
                let a = Tensor::<f64>::ones(a_dim);
                let b = Tensor::<f64>::ones(b_dim);
                apply(&a, &b, op)
            });
            let error = result.unwrap_err();
            let message = error.downcast_ref::<String>().unwrap();
            assert!(message.contains("tensor shapes must be broadcast-compatible"));
        }
    }
}

#[test]
fn scalar_broadcasting_supports_f32() {
    let a = Tensor::new(vec![2.0_f32], vec![]);
    let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let output = &a * &b;
    assert_eq!(output.get_dim(), &[3]);
    assert_eq!(*output.get_data(), vec![2.0, 4.0, 6.0]);
    output.backward(1.0);
    assert_eq!(*a.get_gradients(), vec![6.0]);
    assert_eq!(*b.get_gradients(), vec![2.0; 3]);
}
