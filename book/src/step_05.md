# Step 5: Percentile Calibration

## What You'll Learn

- Handle outliers in calibration data
- Percentile-based range selection
- Why min-max can be suboptimal

## Context

Min-max calibration is sensitive to outliers. Percentile calibration ignores extreme values by specifying what percentile to use.

## Task

Implement `PercentileCalibrator` in `src/calibration/percentile.rs`.

## Hints

<details>
<summary>Hint 1</summary>

Collect all values, sort them, then pick values at the specified percentile.

</details>

<details>
<summary>Hint 2</summary>

Use `sorted[kth]` to get percentile values efficiently.

</details>

## Template

```rust
pub struct PercentileCalibrator {
    percentile: f32,
    all_values: Vec<f32>,
}

impl PercentileCalibrator {
    pub fn new(percentile: f32) -> Self { unimplemented!() }
    
    pub fn update(&mut self, batch: &ArrayD<f32>) {
        // Your code here
    }
    
    pub fn compute(&self) -> (f32, i8) {
        unimplemented!()
    }
}
```

## Run Tests

```bash
cargo test calibration::percentile
```
