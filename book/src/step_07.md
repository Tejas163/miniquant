# Step 7: QuantizedArray

## What You'll Learn

- Design a tensor struct with metadata
- Separate storage from quantization parameters
- Serialization

## Context

`QuantizedArray` stores INT8 data along with scale and zero_point for accurate dequantization.

## Task

Implement `QuantizedArray` in `src/tensor/quantized_array.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Store data as `Vec<i8>` with separate `shape`, `scale`, and `zero_point`.

</details>

<details>
<summary>Hint 2</summary>

Use `serde` derive for serialization support.

## Template

```rust
#[derive(Serialize, Deserialize)]
pub struct QuantizedArray {
    data: Vec<i8>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: i8,
}

impl QuantizedArray {
    pub fn new<D: IntoDimension>(data: Array<f32, D>, scale: f32, zero_point: i8) -> Self {
        unimplemented!()
    }
    
    pub fn dequantize(&self) -> ArrayD<f32> {
        unimplemented!()
    }
}
```

## Run Tests

```bash
cargo test tensor::quantized_array
```
