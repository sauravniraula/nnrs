# nnrs

A small, educational neural-network and automatic-differentiation library written in Rust. It implements tensor arithmetic, reverse-mode backpropagation, trainable dense layers, and computation-graph export without a machine-learning framework.

The repository includes both a library and a runnable training demonstration. It is an experimental project, not a production training framework.

## Quick start

### Requirements

- Rust and Cargo with support for the **2024 edition**. Install a current stable toolchain using [rustup](https://rustup.rs/).
- Graphviz, only if you want to render computation graphs. Its `dot` executable must be on your `PATH`.
- Make, only for the convenience commands in the Makefile.

There are no datasets, API keys, Python packages, or GPU dependencies to configure. Cargo downloads the Rust dependencies on the first build.

```sh
git clone https://github.com/sauravniraula/nnrs.git
cd nnrs
mkdir -p outputs
cargo run --locked
```

The program prints the iteration, prediction, and squared error. Weights and biases are randomly initialized, so numeric results differ between runs.

**Create `outputs/` first:** the graph exporter writes `outputs/graph.dot` relative to the working directory but does not create the directory. The demo currently ignores graph-export errors, so training output alone does not prove the graph was saved.

### Render the computation graph

Install Graphviz if needed:

```sh
# macOS with Homebrew
brew install graphviz

# Ubuntu / Debian
sudo apt-get install graphviz
```

From the repository root:

```sh
mkdir -p outputs
make preview
```

`make preview` runs the demo and converts its DOT graph into `outputs/graph.svg`. Open the SVG in a browser or image viewer. Without Make, the equivalent commands are:

```sh
cargo run --locked
dot -Tsvg outputs/graph.dot -o outputs/graph.svg
```

`make run` is a shortcut for `cargo run`. Both `outputs/` and Cargo's `target/` directory are ignored by Git.

## What the demo does

The committed example in [`src/main.rs`](src/main.rs) builds two dense layers:

```text
Input [batch, 2]
    │
    ▼
DenseLayer(2, 10)
    │
    ▼
DenseLayer(10, 1)
    │
    ▼
Output [batch, 1]
```

Its input is a single row, `[1.0, 2.0]`, represented as a tensor of shape `[1, 2]`. The output has shape `[1, 1]`, and the target is `10.0`.

For each iteration, it:

1. Runs the input through both layers.
2. Reads the **first** output value and prints `(prediction - target)²`.
3. Calls `out2.backward(prediction - target)` to propagate gradients.
4. Updates both layers' parameters with gradient descent.
5. Exports the output's computation graph to `outputs/graph.dot`, replacing the previous file.

The committed defaults are **10 iterations** and a **learning rate of `0.01`**. Change `max_iteration`, the input values, `goal`, and the arguments to `step` in `src/main.rs` to experiment. Each run starts with fresh random parameters; nothing is saved for later inference.

### Important details about this example

- The printed loss is computed as an ordinary Rust scalar, **outside the tensor graph**. Backpropagation starts at the output tensor, not at a loss tensor.
- `backward(seed)` assigns the same seed to **every output element**. The demo has only one output element. If you expand it to a multi-row batch without changing the loss and seeding logic, all predictions will receive the error derived from the first prediction; that would not be a correctly reduced batch squared-error objective.
- For a single prediction, `prediction - target` is the derivative of **half** the squared error. The derivative of the printed full squared error would have an additional factor of two.
- Neither dense layer applies an activation function. Stacking these layers still represents an affine transformation, not a nonlinear model.
- Graph export occurs **after** parameter updates. Parameter nodes show updated values, while intermediate nodes retain the values computed during the forward pass. Treat the graph as a structural/debugging aid, not a consistent pre-update numerical snapshot.

## How the library works

### Tensors and shared graph nodes

[`Tensor<T>`](src/tensor.rs) wraps an `Rc<TensorNode<T>>`. Each node stores:

- A flat, row-major `Vec<T>` of values.
- A shape (`Vec<usize>`) and its element count.
- A gradient vector.
- An optional operation recording how the tensor was produced and which tensors are its parents.

Values and gradients use `RefCell` for interior mutability. Cloning a tensor shares its underlying node; it does **not** copy the data. Tensor equality and hashing use node identity, allowing graph traversal to recognize shared parents.

`Tensor::new(data, shape)` creates a leaf node. `zeros`, `ones`, `random`, and `new_filled` provide convenience constructors. `get_data`, `get_dim`, `get_gradients`, and `get_op` expose the corresponding state. The generic [`Number`](src/number.rs) trait combines the floating-point, random-sampling, formatting, and arithmetic bounds needed by the implementation; `f32` and `f64` are natural choices.

### Supported operations

Operations create new tensors and retain shared references to their parents:

| Operation | API | Shape contract |
| --- | --- | --- |
| Element-wise addition | `&a + &b` | Identical shapes |
| Element-wise subtraction | `&a - &b` | Identical shapes |
| Element-wise multiplication | `&a * &b` | Identical shapes |
| Matrix multiplication | `a.matmul(&b)` | `[m, k] × [k, n] → [m, n]`, both operands 2-D |

There is **no general broadcasting**, scalar/tensor arithmetic overload, vector matmul, or batched/N-dimensional matmul. Matrix multiplication uses straightforward nested loops on the CPU.

### Reverse-mode automatic differentiation

`output.backward(seed)`:

1. Traverses all nodes reachable from the output in topological order, visiting shared nodes only once.
2. Resets those nodes' gradients to zero.
3. Seeds every output element with `seed`.
4. Walks the order backward, accumulating each operation's contributions into its parents.

For incoming gradient `G`, the implemented rules are:

| Forward operation | Gradient for `A` | Gradient for `B` |
| --- | --- | --- |
| `A + B` | `G` | `G` |
| `A - B` | `G` | `-G` |
| `A * B` (element-wise) | `G * B` | `G * A` |
| `A.matmul(B)` | `G.matmul(Bᵀ)` | `Aᵀ.matmul(G)` |

The matmul backward pass computes these products directly with loops; a public transpose operation is not required. Shared parents accumulate contributions from every path. A subsequent `backward` call resets reachable gradients rather than accumulating across calls.

`tensor.step(learning_rate)` updates that tensor's values in place:

```text
value ← value - learning_rate × gradient
```

It does not update the entire graph or clear gradients. Compute gradients before stepping, and perform a new forward pass after modifying parameters so intermediate values are fresh.

### Dense layers

[`DenseLayer<T>`](src/dense_layer.rs) owns weights of shape `[n_inputs, n_outputs]` and a bias of shape `[1, n_outputs]`, both initialized randomly in `[-1, 1)`.

For an input `X` of shape `[batch, n_inputs]`, `forward` computes:

```text
Y = X.matmul(weights) + ones([batch, 1]).matmul(bias)
```

The second matrix product repeats the bias across the batch explicitly, so ordinary tensor addition still receives two identically shaped tensors. The output shape is `[batch, n_outputs]`. `layer.step(rate)` updates only that layer's weights and bias.

### Computation-graph export

[`draw_computation_graph(&tensor)`](src/draw.rs) walks the tensor's ancestors and builds a directed graph using `petgraph`. Tensor nodes display their values and gradients; separate operation nodes are labeled `ADD`, `SUB`, `MUL`, or `MATMUL`. Shared tensor nodes appear once, with edges connecting parents through operations to results.

The function returns an I/O `Result` and writes Graphviz DOT to the fixed relative path `outputs/graph.dot`. Graphviz itself is not needed to export DOT, only to render it.

## Source map

| File | Responsibility |
| --- | --- |
| [`src/lib.rs`](src/lib.rs) | Public library modules |
| [`src/number.rs`](src/number.rs) | Numeric trait bounds |
| [`src/tensor.rs`](src/tensor.rs) | Tensor storage, arithmetic, graph traversal, backpropagation, updates |
| [`src/dense_layer.rs`](src/dense_layer.rs) | Dense-layer initialization, forward pass, parameter updates |
| [`src/draw.rs`](src/draw.rs) | DOT computation-graph export |
| [`src/main.rs`](src/main.rs) | Runnable training demonstration |
| [`Makefile`](Makefile) | Run and graph-preview shortcuts |
| [`Cargo.toml`](Cargo.toml) | Package metadata and dependencies |

Dependencies are `num-traits` for numeric traits and scalar exponentiation, `rand` for parameter initialization, and `petgraph` for graph construction and DOT formatting. Tensor math and differentiation are implemented in this repository.

## Development

```sh
cargo build --locked
cargo test --locked
cargo doc --locked --no-deps --open
```

`cargo test` currently discovers **no automated tests**; a successful result verifies compilation and the test harness, not numerical correctness. Use the runnable demo and graph preview for a basic smoke check.

## Current limitations

- Educational CPU-only implementation, with no accelerated kernels or parallel execution.
- `Rc`/`RefCell` graph storage is single-threaded, not a thread-safe tensor backend.
- No activation functions, tensor reductions, built-in loss functions, advanced optimizers, dataset loader, or model serialization.
- Tensor construction does not validate that the data length matches the shape's element count. Callers must supply consistent data and shapes; gradient setters likewise require correctly sized vectors.
- Unsupported operation shapes panic through assertions rather than returning recoverable errors.
- Demo metrics and graph export have the caveats described above; neither establishes model quality or numerical-gradient correctness.
