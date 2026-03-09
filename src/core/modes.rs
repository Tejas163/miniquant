//! Step 3: Symmetric vs Asymmetric Quantization
//!
//! # Learning Objectives
//! - Understand symmetric quantization (zero-point = 0)
//! - Understand asymmetric quantization (learned zero-point)
//! - Compare accuracy between the two modes
//!
//! # Context
//! - **Symmetric**: Range is [-max, max], zero-point = 0
//!   - Simpler, faster
//!   - Works well when data is centered around 0
//!   
//! - **Asymmetric**: Range is [min, max], zero-point != 0
//!   - More accurate for biased distributions
//!   - Requires storing zero-point offset

use ndarray::{Array1, Array2, ArrayD};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationMode {
    Symmetric,
    Asymmetric,
}

pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i8,
    pub mode: QuantizationMode,
}

/// Quantize using symmetric mode (zero-point = 0).
///
/// Range: [-128, 127] mapped to [-scale * 128, scale * 127]
pub fn quantize_symmetric(x: &ArrayD<f32>, scale: f32) -> ArrayD<i8> {
    x.mapv(|v| {
        let q = (v / scale).round();
        q.clamp(-128.0, 127.0) as i8
    })
}

/// Dequantize symmetric quantized array.
pub fn dequantize_symmetric(x: &ArrayD<i8>, scale: f32) -> ArrayD<f32> {
    x.mapv(|v| v as f32 * scale)
}

/// Quantize using asymmetric mode (with zero-point offset).
///
/// Formula: q = round((x / scale) + zero_point)
///
/// # Arguments
/// * `x` - Input float array
/// * `scale` - Scale factor
/// * `zero_point` - Offset to shift the range (typically computed from min value)
pub fn quantize_asymmetric(x: &ArrayD<f32>, scale: f32, zero_point: i8) -> ArrayD<i8> {
    x.mapv(|v| {
        let q = (v / scale + zero_point as f32).round();
        q.clamp(-128.0, 127.0) as i8
    })
}

/// Dequantize asymmetric quantized array.
pub fn dequantize_asymmetric(x: &ArrayD<i8>, scale: f32, zero_point: i8) -> ArrayD<f32> {
    x.mapv(|v| (v as f32 - zero_point as f32) * scale)
}

/// Calculate quantization parameters for asymmetric mode.
///
/// # Formula
/// scale = (max - min) / 255
/// zero_point = -round(min / scale)
pub fn calculate_asymmetric_params(x: &ArrayD<f32>) -> (f32, i8) {
    let min_val = x.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let scale = (max_val - min_val) / 255.0;
    let zero_point = if scale == 0.0 {
        0
    } else {
        (-min_val / scale).round() as i8
    };

    (scale.max(1e-8), zero_point.clamp(-128, 127))
}

/// Calculate quantization parameters for symmetric mode.
pub fn calculate_symmetric_params(x: &ArrayD<f32>) -> f32 {
    let max_abs = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        1.0
    } else {
        127.0 / max_abs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_symmetric_quantization() {
        let x = array![[-10.0, 0.0, 10.0]];
        let scale = calculate_symmetric_params(&x);
        let q = quantize_symmetric(&x, scale);

        // Should quantize symmetrically around 0
        assert_eq!(q[[0]], -127i8);
        assert_eq!(q[[1]], 0i8);
        assert_eq!(q[[2]], 127i8);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let x = array![[0.0, 5.0, 10.0]];
        let (scale, zero_point) = calculate_asymmetric_params(&x);
        let q = quantize_asymmetric(&x, scale, zero_point);

        // Should quantize the full range [0, 10]
        assert_eq!(q[[0]], 0i8);
        assert_eq!(q[[1]], 127i8);
        assert_eq!(q[[2]], -128i8); // Clamped
    }

    #[test]
    fn test_symmetric_dequantize() {
        let q = array![[-127i8, 0, 127]];
        let scale = 10.0 / 127.0;
        let x = dequantize_symmetric(&q, scale);

        assert!((x[[0]] - (-10.0)).abs() < 0.1);
        assert!((x[[1]] - 0.0).abs() < 0.1);
        assert!((x[[2]] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_asymmetric_dequantize() {
        let q = array![[0i8, 127]];
        let scale = 10.0 / 127.0;
        let zero_point = 0i8;
        let x = dequantize_asymmetric(&q, scale, zero_point);

        assert!((x[[0]] - 0.0).abs() < 0.1);
        assert!((x[[1]] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_asymmetric_with_nonzero_zp() {
        let x = array![[2.5, 5.0, 7.5]];
        let (scale, zero_point) = calculate_asymmetric_params(&x);

        // min=2.5, max=7.5, range=5.0
        // scale = 5.0/255 ≈ 0.0196
        // zero_point = -2.5/0.0196 ≈ -128
        assert!(scale > 0.0);

        let q = quantize_asymmetric(&x, scale, zero_point);
        let reconstructed = dequantize_asymmetric(&q, scale, zero_point);

        // Check reconstruction
        let max_error = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, |a, b| a.max(b));

        assert!(max_error < 0.2);
    }

    #[test]
    fn test_compare_modes() {
        let x = array![[1.0, 2.0, 100.0]];

        // Symmetric
        let scale_s = calculate_symmetric_params(&x);
        let q_s = quantize_symmetric(&x, scale_s);
        let r_s = dequantize_symmetric(&q_s, scale_s);

        // Asymmetric
        let (scale_a, zp_a) = calculate_asymmetric_params(&x);
        let q_a = quantize_asymmetric(&x, scale_a, zp_a);
        let r_a = dequantize_asymmetric(&q_a, scale_a, zp_a);

        let error_s = x
            .iter()
            .zip(r_s.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let error_a = x
            .iter()
            .zip(r_a.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        // Asymmetric should have lower or equal error for biased data
        assert!(error_a <= error_s);
    }
}
