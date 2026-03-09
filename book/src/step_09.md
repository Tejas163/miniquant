# Step 9: Operation Fusion

## What You'll Learn

- Fold quantization parameters into weights
- Pre-compute fused scales
- Reduce memory bandwidth

## Context

Fusion combines operations to reduce memory traffic. We can fuse dequantization into weight storage.

## Task

Implement `FusedWeight` and `fused_matmul` in `src/ops/fusion.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Store weight with a single fused scale instead of separate scale+zero_point.

</details>

## Template

```rust
pub struct FusedWeight {
    pub data: Array2<i8>,
    pub fused_scale: f32,
}

impl FusedWeight {
    pub fn from_float(weight: &Array2<f32>, target_scale: f32) -> Self {
        unimplemented!()
    }
    
    pub fn to_float(&self) -> Array2<f32> {
        unimplemented!()
    }
}

pub fn fused_matmul(
    input_q: &Array2<i8>,
    input_scale: f32,
    weight: &FusedWeight,
) -> Array2<f32> {
    unimplemented!()
}
```

## Run Tests

```bash
cargo test ops::fusion
```
