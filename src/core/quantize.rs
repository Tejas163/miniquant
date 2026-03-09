//! Step 1: Basic Quantization
//!
//! # Learning Objectives
//! - Understand FP32 → INT8 conversion
//! - Implement scale-based quantization
//! - Learn about clamping and rounding
//!
//! # Context
//! Quantization is the process of mapping values from a large set (FP32)
//! to a smaller set (INT8). This reduces model size and enables faster
//! inference using integer arithmetic.
//!
//! The basic formula is: q = round(x / scale)
//!
//! where scale determines the mapping between the two ranges.

use ndarray::{Array, Array2, ArrayD};

/// Quantize a float array to INT8 using the given scale.
///
/// # Arguments
/// * `x` - Input f32 array
/// * `scale` - Scale factor for quantization
///
/// # Returns
/// Quantized i8 array with same shape as input
///
/// # Formula
/// q[i] = clamp(round(x[i] / scale), -128, 127)
///
/// # Example
/// ```
/// use ndarray::array;
/// use miniquant::core::quantize;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0]];
/// let scale = 1.0;
/// let result = quantize(&x, scale);
/// // result = [[1, 2], [3, 4]]
/// ```
pub fn quantize(x: &ArrayD<f32>, scale: f32) -> ArrayD<i8> {
    x.mapv(|v| {
        let q = (v / scale).round();
        q.clamp(-128.0, 127.0) as i8
    })
}

/// Dequantize an INT8 array back to FP32 using the given scale.
///
/// # Arguments
/// * `x` - Input i8 array
/// * `scale` - Scale factor used during quantization
///
/// # Returns
/// Dequantized f32 array with same shape as input
///
/// # Formula
/// x[i] = q[i] * scale
///
/// # Example
/// ```
/// use ndarray::array;
/// use miniquant::core::dequantize;
///
/// let q = array![[1i8, 2], [3, 4]];
/// let scale = 1.0;
/// let result = dequantize(&q, scale);
/// // result = [[1.0, 2.0], [3.0, 4.0]]
/// ```
pub fn dequantize(x: &ArrayD<i8>, scale: f32) -> ArrayD<f32> {
    x.mapv(|v| v as f32 * scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantize_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = 1.0;
        let result = quantize(&x, scale);
        let expected = array![[1i8, 2], [3, 4]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_quantize_with_scale() {
        let x = array![[2.0, 4.0], [6.0, 8.0]];
        let scale = 2.0;
        let result = quantize(&x, scale);
        let expected = array![[1i8, 2], [3, 4]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_quantize_clamp() {
        let x = array![[300.0, -300.0]];
        let scale = 1.0;
        let result = quantize(&x, scale);
        let expected = array![[127i8, -128]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dequantize_basic() {
        let q = array![[1i8, 2], [3, 4]];
        let scale = 1.0;
        let result = dequantize(&q, scale);
        let expected = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dequantize_with_scale() {
        let q = array![[1i8, 2], [3, 4]];
        let scale = 2.0;
        let result = dequantize(&q, scale);
        let expected = array![[2.0, 4.0], [6.0, 8.0]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let x = array![[1.5, 2.7], [-1.2, 3.8]];
        let scale = 0.1;

        let q = quantize(&x, scale);
        let reconstructed = dequantize(&q, scale);

        // Check that values are close (accounting for quantization error)
        let diff = (&reconstructed - &x).mapv(|v| v.abs());
        assert!(diff.iter().all(|&v| v < 0.1));
    }

    #[test]
    fn test_quantize_1d() {
        let x = array![1.0, 2.0, 3.0];
        let scale = 1.0;
        let result = quantize(&x, scale);
        let expected = array![1i8, 2, 3];
        assert_eq!(result, expected);
    }
}
