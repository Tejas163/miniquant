//! Step 2: Scale Calculation
//!
//! # Learning Objectives
//! - Compute optimal scale factors for quantization
//! - Understand per-tensor vs per-channel scaling
//! - Learn about INT8 range limits
//!
//! # Context
//! The scale factor determines how float values map to the INT8 range.
//! For symmetric quantization: scale = 127.0 / max(|x|)
//! This ensures the maximum absolute value maps to 127 (or -128).

use ndarray::{Array, Array2, ArrayD, Axis};

/// Calculate per-tensor scale (single scale for entire array).
///
/// # Formula
/// scale = 127.0 / max(abs(x))
///
/// This ensures all values fit within [-128, 127] when quantized.
pub fn calculate_per_tensor_scale(x: &ArrayD<f32>) -> f32 {
    let max_val = x.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
    if max_val == 0.0 {
        1.0
    } else {
        127.0 / max_val
    }
}

/// Calculate per-channel scale (one scale per row/feature dimension).
///
/// # Arguments
/// * `x` - Input array (typically 2D: [batch, features])
/// * `axis` - Axis along which to compute per-channel scales
///
/// # Formula
/// scale[i] = 127.0 / max(abs(x[:, i]))
pub fn calculate_per_channel_scale(x: &ArrayD<f32>, axis: Axis) -> ArrayD<f32> {
    let max_vals = x.map_axis(axis, |row| {
        row.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b))
    });

    max_vals.mapv(|max_val| if max_val == 0.0 { 1.0 } else { 127.0 / max_val })
}

/// Calculate scale using a custom max value (e.g., for INT4 use 7.0).
pub fn calculate_scale(x: &ArrayD<f32>, max_value: f32) -> f32 {
    let max_val = x.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
    if max_val == 0.0 {
        1.0
    } else {
        max_value / max_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_per_tensor_scale_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = calculate_per_tensor_scale(&x);
        // max = 4, so scale = 127/4 = 31.75
        assert!((scale - 31.75).abs() < 0.01);
    }

    #[test]
    fn test_per_tensor_scale_all_positive() {
        let x = array![[1.0, 10.0], [100.0, 1000.0]];
        let scale = calculate_per_tensor_scale(&x);
        // max = 1000, so scale = 127/1000 = 0.127
        assert!((scale - 0.127).abs() < 0.001);
    }

    #[test]
    fn test_per_tensor_scale_all_negative() {
        let x = array![[-1.0, -10.0], [-100.0, -1000.0]];
        let scale = calculate_per_tensor_scale(&x);
        // max = 1000, so scale = 127/1000 = 0.127
        assert!((scale - 0.127).abs() < 0.001);
    }

    #[test]
    fn test_per_tensor_scale_zero() {
        let x = array![[0.0, 0.0], [0.0, 0.0]];
        let scale = calculate_per_tensor_scale(&x);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_per_channel_scale_2d() {
        let x = array![[1.0, 2.0], [10.0, 20.0]];
        let scales = calculate_per_channel_scale(&x, Axis(1));
        // Column 0: max = 10, scale = 12.7
        // Column 1: max = 20, scale = 6.35
        assert!((scales[[0]] - 12.7).abs() < 0.1);
        assert!((scales[[1]] - 6.35).abs() < 0.1);
    }

    #[test]
    fn test_custom_max_value() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = calculate_scale(&x, 7.0); // For INT4
                                              // max = 4, so scale = 7/4 = 1.75
        assert!((scale - 1.75).abs() < 0.01);
    }

    #[test]
    fn test_per_tensor_roundtrip() {
        use crate::core::quantize::{dequantize, quantize};

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = calculate_per_tensor_scale(&x);

        let q = quantize(&x, scale);
        let reconstructed = dequantize(&q, scale);

        // Check reconstruction error is minimal
        let max_error = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(max_error < 0.1);
    }
}
