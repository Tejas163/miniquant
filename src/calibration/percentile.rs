//! Step 5: Percentile Calibration
//!
//! # Learning Objectives
//! - Handle outliers in calibration data
//! - Implement percentile-based range selection
//! - Understand why min-max can be suboptimal
//!
//! # Context
//! Min-max calibration is sensitive to outliers - a single extreme value
//! can waste precision for the rest of the data. Percentile calibration
//! ignores the most extreme values by specifying what percentile to use.

use ndarray::{Array, ArrayD, Axis};

/// Percentile Calibrator that uses percentile-based range selection.
///
/// # Usage
/// 1. Create with desired percentile (e.g., 99.9)
/// 2. Feed sample batches via `update()`
/// 3. Call `compute()` to get quantization parameters
pub struct PercentileCalibrator {
    percentile: f32,
    all_values: Vec<f32>,
}

impl PercentileCalibrator {
    /// Create a new PercentileCalibrator.
    ///
    /// # Arguments
    /// * `percentile` - Percentile to use (e.g., 99.9 for 99.9th percentile)
    pub fn new(percentile: f32) -> Self {
        Self {
            percentile: percentile.clamp(0.0, 100.0),
            all_values: Vec::new(),
        }
    }

    /// Update with a new batch of data.
    pub fn update(&mut self, batch: &ArrayD<f32>) {
        let values: Vec<f32> = batch.iter().cloned().collect();
        self.all_values.extend(values);
    }

    /// Compute scale and zero-point from percentile.
    ///
    /// Uses the specified percentile as both min and max bounds,
    /// effectively clipping outliers.
    pub fn compute(&self) -> (f32, i8) {
        if self.all_values.is_empty() {
            return (1.0, 0);
        }

        let mut sorted = self.all_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let low_idx = ((self.percentile / 100.0) * n as f32) as usize;
        let high_idx = (((100.0 - self.percentile) / 100.0) * n as f32) as usize;

        let min_val = sorted[low_idx.min(n - 1)];
        let max_val = sorted[high_idx.min(n - 1)];

        let scale = (max_val - min_val) / 255.0;
        let zero_point = if scale == 0.0 {
            0
        } else {
            (-min_val / scale).round() as i8
        };

        (scale.max(1e-8), zero_point.clamp(-128, 127))
    }

    /// Compute symmetric scale using percentile.
    pub fn compute_symmetric(&self) -> f32 {
        if self.all_values.is_empty() {
            return 1.0;
        }

        let mut abs_values: Vec<f32> = self.all_values.iter().map(|v| v.abs()).collect();
        abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = abs_values.len();
        let idx = ((self.percentile / 100.0) * n as f32) as usize;
        let max_abs = abs_values[idx.min(n - 1)];

        if max_abs == 0.0 {
            1.0
        } else {
            127.0 / max_abs
        }
    }

    /// Get the number of samples processed.
    pub fn num_samples(&self) -> usize {
        self.all_values.len()
    }

    /// Reset the calibrator.
    pub fn reset(&mut self) {
        self.all_values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_percentile_basic() {
        let mut cal = PercentileCalibrator::new(99.0);
        let batch = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        assert!(scale > 0.0);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_percentile_with_outliers() {
        let mut cal = PercentileCalibrator::new(99.0);
        // Most values in [0, 10], but have outliers at 1000
        let batch = array![[0.0, 1.0, 2.0, 3.0, 1000.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        // Scale should be based on lower percentile, not include outlier
        // This prevents wasting precision on normal values
        assert!(scale > 0.03); // Would be ~4 if using min-max
    }

    #[test]
    fn test_percentile_symmetric() {
        let mut cal = PercentileCalibrator::new(99.9);
        // Symmetric around 0 with some outliers
        let batch = array![[-100.0, -1.0, 0.0, 1.0, 100.0]];
        cal.update(&batch);

        let scale = cal.compute_symmetric();

        // Should use 99.9th percentile of abs values
        // abs values: [0, 1, 1, 100, 100] -> sorted [0,1,1,100,100]
        // 99.9% of 5 = 4.995 -> index 4 -> 100
        // scale = 127/100 = 1.27
        assert!((scale - 1.27).abs() < 0.1);
    }

    #[test]
    fn test_percentile_multiple_batches() {
        let mut cal = PercentileCalibrator::new(95.0);

        cal.update(&array![[1.0, 2.0]]);
        cal.update(&array![[3.0, 4.0]]);
        cal.update(&array![[5.0, 6.0]]);

        assert_eq!(cal.num_samples(), 6);
    }

    #[test]
    fn test_percentile_100() {
        let mut cal = PercentileCalibrator::new(100.0);
        let batch = array![[1.0, 10.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        // 100% should be same as min-max
        let mut cal_mm = crate::calibration::MinMaxCalibrator::new();
        cal_mm.update(&batch);
        let (scale_mm, zp_mm) = cal_mm.compute();

        assert!((scale - scale_mm).abs() < 0.001);
        assert_eq!(zp, zp_mm);
    }

    #[test]
    fn test_percentile_0() {
        let mut cal = PercentileCalibrator::new(0.0);
        let batch = array![[1.0, 10.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        // 0% percentile would give degenerate scale
        // Should handle gracefully
        assert!(scale > 0.0);
    }

    #[test]
    fn test_percentile_reset() {
        let mut cal = PercentileCalibrator::new(99.0);

        cal.update(&array![[1.0, 2.0]]);
        assert_eq!(cal.num_samples(), 2);

        cal.reset();
        assert_eq!(cal.num_samples(), 0);
    }
}
