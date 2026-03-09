# Step 10: Observer Pattern

## What You'll Learn

- Hook into forward passes
- Collect calibration data
- Don't modify model outputs

## Context

Observers run inference on sample data to collect statistics without modifying outputs. Essential for post-training quantization.

## Task

Implement `TensorObserver` in `src/pipeline/observer.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Track running min/max like MinMaxCalibrator but don't change forward pass.

</details>

## Template

```rust
pub trait Observer {
    fn observe(&mut self, data: &ArrayD<f32>);
}

pub struct TensorObserver {
    min_val: f32,
    max_val: f32,
}

impl TensorObserver {
    pub fn new() -> Self { unimplemented!() }
    
    pub fn compute_scale(&self) -> f32 {
        unimplemented!()
    }
}

impl Observer for TensorObserver {
    fn observe(&mut self, data: &ArrayD<f32>) {
        // Your code here
    }
}
```

## Run Tests

```bash
cargo test pipeline::observer
```
