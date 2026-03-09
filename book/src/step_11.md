# Step 11: Model Converter

## What You'll Learn

- Convert full models to quantized versions
- Walk through model layers
- Complete quantization pipeline

## Context

The converter walks through model layers, applies calibration, quantizes weights, and replaces layers with quantized versions.

## Task

Implement `quantize_model` in `src/pipeline/converter.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Iterate through layers, quantize weights, create quantized layer wrappers.

</details>

<details>
<summary>Hint 2</summary>

Use the calibration data to determine activation scales.

## Template

```rust
pub fn quantize_model(
    model: &SimpleModel,
    calibration_data: &[ArrayD<f32>],
    config: &QuantizationConfig,
) -> QuantizedModel {
    // Your code here
    unimplemented!()
}
```

## Run Tests

```bash
cargo test pipeline::converter
```

## Complete!

Congratulations! You've built a complete quantization library from scratch. The final test verifies that your quantized model produces results close to the original.
