# Step 2: Scale Calculation

## What You'll Learn

- Compute optimal scale factors for quantization
- Per-tensor vs per-channel scaling
- INT8 range limits

## Context

The scale factor determines how float values map to the INT8 range. For symmetric quantization: `scale = 127.0 / max(|x|)` - this ensures the maximum absolute value maps to 127.

## Task

Implement scale calculation functions in `src/core/scale.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Use `fold` to find the maximum absolute value in an array.

</details>

<details>
<summary>Hint 2</summary>

For per-channel, use `ndarray::Axis` to iterate along a specific dimension.

</details>

## Formula

```rust
// Per-tensor
scale = 127.0 / max(abs(x))

// Per-channel  
scale[i] = 127.0 / max(abs(x[:, i]))
```

## Template

```rust
pub fn calculate_per_tensor_scale(x: &ArrayD<f32>) -> f32 {
    // Your code here
    unimplemented!()
}

pub fn calculate_per_channel_scale(x: &ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
    // Your code here
    unimplemented!()
}
```

## Run Tests

```bash
cargo test core::scale
```
