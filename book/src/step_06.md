# Step 6: Dynamic Quantization

## What You'll Learn

- Quantize activations on-the-fly
- Dynamic vs static quantization
- Per-batch quantization

## Context

Dynamic quantization quantizes weights statically but computes activation scales dynamically for each batch.

## Task

Implement `DynamicQuantizer` in `src/calibration/dynamic.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Pre-quantize weights once during creation.

</details>

<details>
<summary>Hint 2</summary>

Compute input scale dynamically for each forward pass.

</details>

## Template

```rust
pub struct DynamicQuantizer {
    quantized_weights: Array2<i8>,
    weight_scale: f32,
    zero_point: i8,
}

impl DynamicQuantizer {
    pub fn new(weights: &Array2<f32>, scale: f32, zero_point: i8) -> Self {
        unimplemented!()
    }
    
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        unimplemented!()
    }
}
```

## Run Tests

```bash
cargo test calibration::dynamic
```
