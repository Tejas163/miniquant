# Step 3: Symmetric vs Asymmetric Quantization

## What You'll Learn

- Symmetric quantization (zero-point = 0)
- Asymmetric quantization (learned zero-point)
- Compare accuracy between modes

## Context

- **Symmetric**: Range is [-max, max], zero-point = 0. Simpler, faster.
- **Asymmetric**: Range is [min, max], zero-point != 0. More accurate for biased distributions.

## Task

Implement both quantization modes in `src/core/modes.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Symmetric is simpler - just divide by scale.

</details>

<details>
<summary>Hint 2</summary>

Asymmetric needs offset: `q = round((x / scale) + zero_point)`

</details>

## Formula

```rust
// Symmetric
q = round(x / scale)

// Asymmetric  
q = round(x / scale + zero_point)
x_dequant = (q - zero_point) * scale
```

## Template

```rust
pub fn quantize_symmetric(x: &ArrayD<f32>, scale: f32) -> ArrayD<i8> {
    unimplemented!()
}

pub fn quantize_asymmetric(x: &ArrayD<f32>, scale: f32, zero_point: i8) -> ArrayD<i8> {
    unimplemented!()
}

pub fn calculate_asymmetric_params(x: &ArrayD<f32>) -> (f32, i8) {
    unimplemented!()
}
```

## Run Tests

```bash
cargo test core::modes
```
