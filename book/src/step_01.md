# Step 1: Basic Quantization

## What You'll Learn

- FP32 → INT8 conversion
- Scale-based quantization
- Clamping and rounding

## Context

Quantization is the process of mapping values from a large set (FP32) to a smaller set (INT8). This reduces model size and enables faster inference using integer arithmetic.

The basic formula is: `q = round(x / scale)`

where scale determines the mapping between the two ranges.

## Task

Implement `quantize()` and `dequantize()` functions in `src/core/quantize.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Use `ndarray::ArrayD` for multi-dimensional arrays.

</details>

<details>
<summary>Hint 2</summary>

Round before casting to integer: `let q = (v / scale).round();`

</details>

<details>
<summary>Hint 3</summary>

Clamp to avoid overflow: `q.clamp(-128.0, 127.0) as i8`

</details>

## Formula

```
q[i] = clamp(round(x[i] / scale), -128, 127)
x_dequant[i] = q[i] * scale
```

## Template

```rust
pub fn quantize(x: &ArrayD<f32>, scale: f32) -> ArrayD<i8> {
    // Your code here
    unimplemented!()
}

pub fn dequantize(x: &ArrayD<i8>, scale: f32) -> ArrayD<f32> {
    // Your code here
    unimplemented!()
}
```

## Run Tests

```bash
cargo test core::quantize
```

## Solution

<details>
<summary>Click to reveal</summary>

```rust
pub fn quantize(x: &ArrayD<f32>, scale: f32) -> ArrayD<i8> {
    x.mapv(|v| {
        let q = (v / scale).round();
        q.clamp(-128.0, 127.0) as i8
    })
}

pub fn dequantize(x: &ArrayD<i8>, scale: f32) -> ArrayD<f32> {
    x.mapv(|v| v as f32 * scale)
}
```

</details>
