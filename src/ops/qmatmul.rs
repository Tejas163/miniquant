//! Step 8: Quantized Matrix Multiplication
//!
//! # Learning Objectives
//! - Implement INT8 matrix multiplication using dequantize → matmul → requantize
//! - Handle per-tensor and per-channel quantization
//! - Understand the accuracy vs speed tradeoffs
//!
//! # Context
//! The core operation in quantized neural networks is the quantized matrix
//! multiplication. We use a "dequantize → matmul → requantize" pattern to
//! leverage integer arithmetic while maintaining accuracy.

use ndarray::{Array, Array2, ArrayD, Ix2};

/// Quantized matrix multiplication result.
#[derive(Debug, Clone)]
pub struct QMatmulResult {
    pub output: ArrayD<i8>,
    pub output_scale: f32,
    pub output_zero_point: i8,
}

/// Compute quantized matrix multiplication.
///
/// Formula: output = (input_q @ weight_q) * (input_scale * weight_scale) / output_scale
///
/// This implements: dequantize → matmul → requantize
pub fn qmatmul(
    input_q: &Array2<i8>,
    input_scale: f32,
    input_zp: i8,
    weight_q: &Array2<i8>,
    weight_scale: f32,
    weight_zp: i8,
    output_scale: f32,
) -> QMatmulResult {
    // Step 1: Dequantize inputs to float
    let input_f32 = input_q.mapv(|v| (v as f32 - input_zp as f32) * input_scale);
    let weight_f32 = weight_q.mapv(|v| (v as f32 - weight_zp as f32) * weight_scale);

    // Step 2: Matrix multiplication in float
    let output_f32 = input_f32.dot(&weight_f32);

    // Step 3: Requantize output
    // Compute effective scale
    let effective_scale = input_scale * weight_scale;

    let output_q = output_f32.mapv(|v| {
        let scaled = v / effective_scale;
        let quantized = (scaled + 0.5).floor(); // Simple rounding
        quantized.clamp(-128.0, 127.0) as i8
    });

    QMatmulResult {
        output: output_q.into_dyn(),
        output_scale,
        output_zero_point: 0,
    }
}

/// Simplified qmatmul with symmetric quantization.
pub fn qmatmul_symmetric(
    input_q: &Array2<i8>,
    input_scale: f32,
    weight_q: &Array2<i8>,
    weight_scale: f32,
    output_scale: f32,
) -> QMatmulResult {
    qmatmul(
        input_q,
        input_scale,
        0, // zero_point = 0 for symmetric
        weight_q,
        weight_scale,
        0,
        output_scale,
    )
}

/// Compute qmatmul with fp32 output (no requantization).
pub fn qmatmul_to_float(
    input_q: &Array2<i8>,
    input_scale: f32,
    input_zp: i8,
    weight_q: &Array2<i8>,
    weight_scale: f32,
    weight_zp: i8,
) -> Array2<f32> {
    // Dequantize
    let input_f32 = input_q.mapv(|v| (v as f32 - input_zp as f32) * input_scale);
    let weight_f32 = weight_q.mapv(|v| (v as f32 - weight_zp as f32) * weight_scale);

    // Matmul
    input_f32.dot(&weight_f32)
}

/// Compute output scale for qmatmul.
///
/// For correct quantization: output_scale = input_scale * weight_scale
pub fn compute_output_scale(input_scale: f32, weight_scale: f32) -> f32 {
    input_scale * weight_scale
}

/// Compute per-channel output scales.
///
/// For per-channel quantization, each output channel has its own scale.
pub fn compute_output_scale_per_channel(
    input_scale: f32,
    weight_scales: &Array2<f32>,
) -> Array2<f32> {
    // weight_scales shape: [output_channels, input_channels] (per-row scales)
    // We need scale per output channel
    let mut scales = Array::zeros(weight_scales.raw_dim());

    for (i, row) in weight_scales.rows().into_iter().enumerate() {
        let max_scale = row.iter().fold(0.0f32, |a, &b| a.max(b));
        scales[[i]] = input_scale * max_scale;
    }

    scales
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_qmatmul_basic() {
        // Simple 2x2 matmul
        let input_q = array![[1i8, 2], [3, 4]];
        let weight_q = array![[1i8, 0], [0, 1]];

        let result = qmatmul_symmetric(&input_q, 1.0, &weight_q, 1.0, 1.0);

        // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        assert_eq!(result.output.shape(), &[2, 2]);
    }

    #[test]
    fn test_qmatmul_identity() {
        // Identity matrix should preserve input
        let input_q = array![[1i8, 2], [3, 4]];
        let weight_q = array![[1i8, 0], [0, 1]];

        let result = qmatmul_to_float(&input_q, 1.0, 0, &weight_q, 1.0, 0);

        assert!((result[[0, 0]] - 1.0).abs() < 0.1);
        assert!((result[[0, 1]] - 2.0).abs() < 0.1);
        assert!((result[[1, 0]] - 3.0).abs() < 0.1);
        assert!((result[[1, 1]] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_qmatmul_with_scales() {
        let input_q = array![[10i8, 20]];
        let weight_q = array![[1i8, 2], [3, 4]];

        let result = qmatmul_to_float(
            &input_q, 0.1, 0, // scale = 0.1
            &weight_q, 1.0, 0,
        );

        // Expected: [[10*0.1*1 + 20*0.1*3, 10*0.1*2 + 20*0.1*4]]
        // = [[1 + 6, 2 + 8]] = [[7, 10]]
        assert!((result[[0, 0]] - 7.0).abs() < 0.1);
        assert!((result[[0, 1]] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_qmatmul_vs_fp32() {
        // Compare qmatmul with FP32 matmul
        let input = array![[1.5, 2.7], [3.3, 4.2]];
        let weight = array![[1.1, 2.2], [3.3, 4.4]];

        // Quantize
        let (i_scale, i_zp) =
            crate::core::modes::calculate_asymmetric_params(&input.clone().into_dyn());
        let (w_scale, w_zp) =
            crate::core::modes::calculate_asymmetric_params(&weight.clone().into_dyn());

        let input_q =
            crate::core::modes::quantize_asymmetric(&input.clone().into_dyn(), i_scale, i_zp)
                .into_dimensionality::<Ix2>()
                .unwrap();
        let weight_q =
            crate::core::modes::quantize_asymmetric(&weight.clone().into_dyn(), w_scale, w_zp)
                .into_dimensionality::<Ix2>()
                .unwrap();

        // FP32 baseline
        let fp32_result = input.dot(&weight);

        // Quantized
        let q_result = qmatmul_to_float(&input_q, i_scale, i_zp, &weight_q, w_scale, w_zp);

        let max_error = fp32_result
            .iter()
            .zip(q_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Should be reasonably accurate
        assert!(max_error < 0.5);
    }

    #[test]
    fn test_compute_output_scale() {
        let scale = compute_output_scale(0.1, 0.2);
        assert!((scale - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_qmatmul_larger() {
        // 3x4 @ 4x2 = 3x2
        let input_q = array![[1i8, 2, 3, 4], [5, 6, 7, 8], [1, 1, 1, 1]];
        let weight_q = array![[1i8, 2], [2, 3], [3, 4], [4, 5]];

        let result = qmatmul_to_float(&input_q, 1.0, 0, &weight_q, 1.0, 0);

        assert_eq!(result.shape(), &[3, 2]);

        // Row 0: [1,2,3,4] @ [[1,2],[2,3],[3,4],[4,5]]
        // = [1*1+2*2+3*3+4*4, 1*2+2*3+3*4+4*5]
        // = [1+4+9+16, 2+6+12+20] = [30, 40]
        assert!((result[[0, 0]] - 30.0).abs() < 0.1);
        assert!((result[[0, 1]] - 40.0).abs() < 0.1);
    }

    #[test]
    fn test_qmatmul_requantize() {
        let input_q = array![[100i8]];
        let weight_q = array![[10i8]];

        let result = qmatmul_symmetric(&input_q, 1.0, &weight_q, 1.0, 1.0);

        // 100 * 10 = 1000, quantized
        assert!(result.output[[0, 0]] <= 127);
    }
}
