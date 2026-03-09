# Step 4: Min-Max Calibration

## What You'll Learn

- Determine quantization ranges from sample data
- Track running min/max values
- Compute scale from calibration

## Context

Calibration collects statistics from representative sample data to determine the optimal quantization range. Min-max is the simplest approach: use the observed min and max values.

## Task

Implement `MinMaxCalibrator` in `src/calibration/minmax.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Store running min and max as instance variables.

</details>

<details>
<summary>Hint 2</summary>

Update with `self.min_val = self.min_val.min(batch_min)`.

</details>

## Template

```rust
pub struct MinMaxCalibrator {
    // Add fields
}

impl MinMaxCalibrator {
    pub fn new() -> Self { unimplemented!() }
    
    pub fn update(&mut self, batch: &ArrayD<f32>) {
        // Your code here
    }
    
    pub fn compute(&self) -> (f32, i8) {
        // Your code here  
        unimplemented!()
    }
}
```

## Run Tests

```bash
cargo test calibration::minmax
```
