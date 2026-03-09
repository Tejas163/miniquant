//! Step 9: Operation Fusion
//!
//! # Learning Objectives
//! - Fold quantization parameters into weights for faster inference
//! - Pre-compute fused scales
//! - Reduce memory bandwidth with fusion
//!
//! # Context
//! Fusion combines multiple operations into one to reduce memory traffic
//! and computation. For quantization, we can fuse the dequantization
//! into the weight storage, eliminating a separate dequantization step.

use ndarray::{Array2, ArrayD, Ix2};

/// Fused weight with pre-computed scale.
///
/// Instead of storing (weight_q, scale, zero_point) separately,
/// we store (fused_weight, fused_scale) where:
/// fused_weight = round(weight_q / weight_scale)
/// fused_scale = weight_scale
pub struct FusedWeight {
    pub data: Array2<i8>,
    pub fused_scale: f32,
}

impl FusedWeight {
    /// Create fused weight from float weight.
    pub fn from_float(weight: &Array2<f32>, target_scale: f32) -> Self {
        let scale = crate::core::modes::calculate_symmetric_params(&weight.clone().into_dyn());

        // Quantize then "dequantize" to get integer representation with scale baked in
        let q = crate::core::modes::quantize_symmetric(&weight.clone().into_dyn(), scale);
        let q_2d = q.into_dimensionality::<Ix2>().unwrap();

        // The fused scale is what we multiply by to get back float
        let fused_scale = scale;

        Self {
            data: q_2d,
            fused_scale,
        }
    }

    /// Create from pre-quantized weight.
    pub fn from_quantized(weight: Array2<i8>, scale: f32) -> Self {
        Self {
            data: weight,
            fused_scale: scale,
        }
    }

    /// Get the float weight (dequantized).
    pub fn to_float(&self) -> Array2<f32> {
        self.data.mapv(|v| v as f32 * self.fused_scale)
    }

    /// Get the shape.
    pub fn shape(&self) -> [usize; 2] {
        [self.data.nrows(), self.data.ncols()]
    }
}

/// Fast matrix multiplication using fused weights.
///
/// This avoids separate dequantization by using the pre-fused scale.
pub fn fused_matmul(
    input_q: &Array2<i8>,
    input_scale: f32,
    input_zp: i8,
    weight: &FusedWeight,
) -> Array2<f32> {
    // Dequantize input
    let input_f32 = input_q.mapv(|v| (v as f32 - input_zp as f32) * input_scale);

    // Matmul with fused weight
    input_f32.dot(&weight.to_float())
}

/// Compute effective output scale for fusion.
///
/// When fusing, output_scale = input_scale * weight_fused_scale
pub fn compute_fused_output_scale(input_scale: f32, weight: &FusedWeight) -> f32 {
    input_scale * weight.fused_scale
}

/// Pre-compute fused parameters for a weight matrix.
///
/// This optimizes the quantization by reducing the number of scales
/// that need to be applied during computation.
pub struct FusedQuantizedLinear {
    pub weight: FusedWeight,
    pub bias: Option<Array2<f32>>,
    pub input_scale: f32,
    pub input_zero_point: i8,
}

impl FusedQuantizedLinear {
    /// Create from float linear layer.
    pub fn from_linear(weight: &Array2<f32>, bias: Option<&Array2<f32>>, input_scale: f32) -> Self {
        let fused_weight = FusedWeight::from_float(weight, 1.0);
        let bias_clone = bias.map(|b| b.clone());

        Self {
            weight: fused_weight,
            bias: bias_clone,
            input_scale,
            input_zero_point: 0,
        }
    }

    /// Forward pass with fused computation.
    pub fn forward(&self, input: &Array2<i8>) -> Array2<f32> {
        let output = fused_matmul(input, self.input_scale, self.input_zero_point, &self.weight);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let mut result = output;
            for (i, row) in result.rows_mut().into_iter().enumerate() {
                *row += &bias.row(i);
            }
            result
        } else {
            output
        }
    }

    /// Compute output quantization scale.
    pub fn output_scale(&self) -> f32 {
        compute_fused_output_scale(self.input_scale, &self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fused_weight_creation() {
        let weight = array![[1.0, 2.0], [3.0, 4.0]];
        let fused = FusedWeight::from_float(&weight, 1.0);

        assert_eq!(fused.data.shape(), &[2, 2]);
        assert!(fused.fused_scale > 0.0);
    }

    #[test]
    fn test_fused_weight_roundtrip() {
        let weight = array![[1.0, 2.0], [3.0, 4.0]];
        let fused = FusedWeight::from_float(&weight, 1.0);

        let reconstructed = fused.to_float();

        let max_error = weight
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_error < 0.1);
    }

    #[test]
    fn test_fused_matmul() {
        let input_q = array![[1i8, 2], [3, 4]];
        let weight = array![[1.0, 0.0], [0.0, 1.0]];
        let fused = FusedWeight::from_float(&weight, 1.0);

        let result = fused_matmul(&input_q, 1.0, 0, &fused);

        // Identity matrix
        assert!((result[[0, 0]] - 1.0).abs() < 0.1);
        assert!((result[[0, 1]] - 2.0).abs() < 0.1);
        assert!((result[[1, 0]] - 3.0).abs() < 0.1);
        assert!((result[[1, 1]] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_fused_matmul_with_scale() {
        let input_q = array![[10i8, 20]];
        let weight = array![[1.0, 2.0], [3.0, 4.0]];
        let fused = FusedWeight::from_float(&weight, 1.0);

        let result = fused_matmul(&input_q, 0.1, 0, &fused);

        // input_f32 = [[1, 2]]
        // result = [[1*1+2*3, 1*2+2*4]] = [[7, 10]]
        assert!((result[[0, 0]] - 7.0).abs() < 0.1);
        assert!((result[[0, 1]] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_fused_quantized_linear() {
        let weight = array![[1.0, 2.0], [3.0, 4.0]];
        let bias = array![[0.1], [0.2]];

        let layer = FusedQuantizedLinear::from_linear(&weight, Some(&bias), 1.0);

        let input_q = array![[1i8, 2]];
        let output = layer.forward(&input_q);

        // [1,2] @ [[1,2],[3,4]] = [7, 10]
        // + bias = [7.1, 10.2]
        assert!((output[[0, 0]] - 7.1).abs() < 0.1);
        assert!((output[[0, 1]] - 10.2).abs() < 0.1);
    }

    #[test]
    fn test_fused_quantized_linear_no_bias() {
        let weight = array![[1.0, 2.0], [3.0, 4.0]];

        let layer = FusedQuantizedLinear::from_linear(&weight, None, 1.0);

        let input_q = array![[1i8, 2]];
        let output = layer.forward(&input_q);

        assert!((output[[0, 0]] - 7.0).abs() < 0.1);
        assert!((output[[0, 1]] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_fused_output_scale() {
        let weight = array![[1.0, 2.0]];
        let fused = FusedWeight::from_float(&weight, 1.0);

        let output_scale = compute_fused_output_scale(0.5, &fused);

        assert!(output_scale > 0.0);
    }

    #[test]
    fn test_fused_vs_standard() {
        let input = array![[1.5, 2.5], [3.5, 4.5]];
        let weight = array![[1.0, 2.0], [3.0, 4.0]];

        // FP32 baseline
        let fp32_result = input.dot(&weight);

        // Standard quantized
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

        let q_result =
            crate::ops::qmatmul_to_float(&input_q, i_scale, i_zp, &weight_q, w_scale, w_zp);

        // Fused
        let fused_weight = FusedWeight::from_float(&weight, 1.0);
        let fused_result = fused_matmul(&input_q, i_scale, i_zp, &fused_weight);

        // Compare errors
        let standard_error = fp32_result
            .iter()
            .zip(q_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let fused_error = fp32_result
            .iter()
            .zip(fused_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Both should be reasonably accurate
        assert!(standard_error < 0.5);
        assert!(fused_error < 0.5);
    }
}
