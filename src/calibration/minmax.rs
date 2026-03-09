//! Step 4: Min-Max Calibration
//!
//! # Learning Objectives
//! - Implement min-max calibration for quantization
//! - Track running min/max values across multiple batches
//! - Compute scale from calibration data
//!
//! # Context
//! Calibration collects statistics from representative sample data
//! to determine the optimal quantization range. Min-max is the simplest
//! approach: use the observed min and max values.

use ndarray::{Array1, ArrayD};

/// Min-Max Calibrator that tracks running min/max values.
///
/// # Usage
/// 1. Create a new calibrator
/// 2. Feed sample batches via `update()`
/// 3. Call `compute()` to get quantization parameters
pub struct MinMaxCalibrator {
    min_val: f32,
    max_val: f32,
    num_samples: usize,
}

impl MinMaxCalibrator {
    /// Create a new MinMaxCalibrator.
    pub fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            num_samples: 0,
        }
    }

    /// Update running min/max with a new batch of data.
    ///
    /// # Arguments
    /// * `batch` - A batch of activation data to calibrate on
    pub fn update(&mut self, batch: &ArrayD<f32>) {
        let batch_min = batch.iter().cloned().fold(f32::INFINITY, f32::min);
        let batch_max = batch.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.min_val = self.min_val.min(batch_min);
        self.max_val = self.max_val.max(batch_max);
        self.num_samples += 1;
    }

    /// Compute scale and zero-point from collected statistics.
    ///
    /// Returns (scale, zero_point) for asymmetric quantization.
    pub fn compute(&self) -> (f32, i8) {
        let scale = (self.max_val - self.min_val) / 255.0;
        let zero_point = if scale == 0.0 {
            0
        } else {
            (-self.min_val / scale).round() as i8
        };

        (scale.max(1e-8), zero_point.clamp(-128, 127))
    }

    /// Compute scale for symmetric quantization.
    pub fn compute_symmetric(&self) -> f32 {
        let max_abs = self.min_val.abs().max(self.max_val.abs());
        if max_abs == 0.0 {
            1.0
        } else {
            127.0 / max_abs
        }
    }

    /// Get the observed min value.
    pub fn min(&self) -> f32 {
        self.min_val
    }

    /// Get the observed max value.
    pub fn max(&self) -> f32 {
        self.max_val
    }

    /// Get the number of samples processed.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Reset the calibrator.
    pub fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.num_samples = 0;
    }
}

impl Default for MinMaxCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_minmax_single_batch() {
        let mut cal = MinMaxCalibrator::new();
        let batch = array![[1.0, 2.0], [3.0, 4.0]];

        cal.update(&batch);

        assert_eq!(cal.min(), 1.0);
        assert_eq!(cal.max(), 4.0);
    }

    #[test]
    fn test_minmax_multiple_batches() {
        let mut cal = MinMaxCalibrator::new();

        cal.update(&array![[1.0, 5.0]]);
        cal.update(&array![[3.0, 7.0]]);
        cal.update(&array![[2.0, 6.0]]);

        assert_eq!(cal.min(), 1.0);
        assert_eq!(cal.max(), 7.0);
        assert_eq!(cal.num_samples(), 3);
    }

    #[test]
    fn test_minmax_compute_asymmetric() {
        let mut cal = MinMaxCalibrator::new();
        let batch = array![[0.0, 10.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        // scale = (10 - 0) / 255 ≈ 0.0392
        // zero_point = -0 / scale = 0
        assert!((scale - 0.039215686).abs() < 0.001);
        assert_eq!(zp, 0);
    }

    #[test]
    fn test_minmax_compute_symmetric() {
        let mut cal = MinMaxCalibrator::new();
        let batch = array![[-10.0, 10.0]];
        cal.update(&batch);

        let scale = cal.compute_symmetric();

        // scale = 127 / 10 = 12.7
        assert!((scale - 12.7).abs() < 0.1);
    }

    #[test]
    fn test_minmax_biased_distribution() {
        let mut cal = MinMaxCalibrator::new();
        // Data skewed positive: [1, 2, 3, 4, 5]
        let batch = array![[1.0, 2.0, 3.0, 4.0, 5.0]];
        cal.update(&batch);

        let (scale, zp) = cal.compute();

        // scale = (5-1)/255 ≈ 0.0157
        // zero_point = -1/0.0157 ≈ -64
        assert!(scale > 0.0);
        assert!(zp < 0); // Negative zero-point to handle bias
    }

    #[test]
    fn test_minmax_reset() {
        let mut cal = MinMaxCalibrator::new();

        cal.update(&array![[1.0, 5.0]]);
        assert_eq!(cal.min(), 1.0);

        cal.reset();
        assert_eq!(cal.min(), f32::INFINITY);
        assert_eq!(cal.num_samples(), 0);
    }

    #[test]
    fn test_minmax_empty_computation() {
        let cal = MinMaxCalibrator::new();
        let (scale, zp) = cal.compute();

        // With no data, should return safe defaults
        assert!(scale > 0.0);
        assert_eq!(zp, 0);
    }
}
