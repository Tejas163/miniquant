# Step 8: Quantized MatMul

## What You'll Learn

- INT8 matrix multiplication
- Dequantize → matmul → requantize pattern
- Accuracy vs speed tradeoffs

## Context

The core operation in quantized neural networks. We use dequantize → matmul → requantize to leverage integer arithmetic while maintaining accuracy.

## Task

Implement `qmatmul` in `src/ops/qmatmul.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Dequantize both inputs to float first.

</details>

<details>
<summary>Hint 2</summary>

Use `ndarray::Array2::dot()` for matrix multiplication.

</details>

## Formula

```
output_f32 = dequantize(input) @ dequantize(weights)
output_q = round(output_f32 / output_scale)
```

## Template

```rust
pub fn qmatmul(
    input_q: &Array2<i8>,
    input_scale: f32,
    weight_q: &Array2<i8>, 
    weight_scale: f32,
    output_scale: f32,
) -> Array2<i8> {
    unimplemented!()
}
```

## Run Tests

```bash
cargo test ops::qmatmul
```
