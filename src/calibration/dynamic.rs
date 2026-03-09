//! Step 6: Dynamic Quantization
//!
//! # Learning Objectives
//! - Quantize activations on-the-fly during inference
//! - Understand dynamic vs static quantization
//! - Implement per-batch quantization
//!
//! # Context
//! Dynamic quantization quantizes weights statically but computes
//! activation scales dynamically for each batch. This is simpler to
//! implement and works well for models with dynamic activation ranges.

use ndarray::{Array1, Array2, ArrayD};

/// Dynamic Quantizer that quantizes activations per-batch.
///
/// # Usage
/// 1. Pre-quantize weights once using static quantization
/// 2. For each batch, compute scale dynamically
/// 3. Quantize activations, compute, dequantize
pub struct DynamicQuantizer {
    quantized_weights: Array2<i8>,
    weight_scale: f32,
    zero_point: i8,
}

impl DynamicQuantizer {
    /// Create a new DynamicQuantizer with pre-quantized weights.
    pub fn new(weights: &Array2<f32>, scale: f32, zero_point: i8) -> Self {
        let quantized =
            crate::core::modes::quantize_asymmetric(&weights.clone().into_dyn(), scale, zero_point)
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

        Self {
            quantized_weights: quantized,
            weight_scale: scale,
            zero_point,
        }
    }

    /// Quantize input activations dynamically for the current batch.
    pub fn quantize_input(&self, input: &ArrayD<f32>) -> (ArrayD<i8>, f32, i8) {
        // Compute scale dynamically for this batch
        let (scale, zp) = crate::core::modes::calculate_asymmetric_params(input);

        let quantized = crate::core::modes::quantize_asymmetric(input, scale, zp);

        (quantized, scale, zp)
    }

    /// Compute output using dynamic quantization.
    ///
    /// Formula: output = dequantize(quantize(input) @ weights)
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let (input_q, input_scale, input_zp) = self.quantize_input(input);

        // Convert to 2D if needed
        let input_2d = if input_q.ndim() == 1 {
            input_q.into_dimensionality::<ndarray::Ix2>().unwrap()
        } else if input_q.ndim() == 2 {
            input_q.into_dimensionality::<ndarray::Ix2>().unwrap()
        } else {
            // Flatten to 2D: [batch, features]
            let shape = input_q.shape();
            let batch_size = shape[0];
            let features: usize = shape[1..].iter().product();
            input_q.into_shape((batch_size, features)).unwrap()
        };

        // Matrix multiplication in INT8 (approximation)
        // For full INT8 matmul, we'd need proper integer ops
        // Here we approximate with dequantized computation
        let weight_f32 = crate::core::modes::dequantize_asymmetric(
            &self.quantized_weights.clone().into_dyn(),
            self.weight_scale,
            self.zero_point,
        )
        .into_dimensionality::<ndarray::Ix2>()
        .unwrap();

        let input_f32 =
            crate::core::modes::dequantize_asymmetric(&input_2d.into_dyn(), input_scale, input_zp);

        // Matmul
        let output = input_f32.dot(&weight_f32);

        output.into_dyn()
    }

    /// Get the quantized weights.
    pub fn weights(&self) -> &Array2<i8> {
        &self.quantized_weights
    }

    /// Get the weight scale.
    pub fn weight_scale(&self) -> f32 {
        self.weight_scale
    }
}

/// Convenience function for dynamic quantization of a linear layer.
pub fn dynamic_quantize_linear(weights: &Array2<f32>, input: &ArrayD<f32>) -> ArrayD<f32> {
    // Compute weight scale once (static)
    let (w_scale, w_zp) =
        crate::core::modes::calculate_asymmetric_params(&weights.clone().into_dyn());

    // Create quantizer
    let quantizer = DynamicQuantizer::new(weights, w_scale, w_zp);

    // Forward with dynamic input quantization
    quantizer.forward(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dynamic_quantizer_creation() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = 1.0;
        let zp = 0i8;

        let quantizer = DynamicQuantizer::new(&weights, scale, zp);

        assert_eq!(quantizer.weights().shape(), &[2, 2]);
    }

    #[test]
    fn test_dynamic_quantize_input() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let quantizer = DynamicQuantizer::new(&weights, 1.0, 0);

        let input = array![[1.0, 2.0]];
        let (q, scale, zp) = quantizer.quantize_input(&input);

        assert_eq!(q.shape(), input.shape());
        assert!(scale > 0.0);
    }

    #[test]
    fn test_dynamic_forward() {
        let weights = array![[1.0, 0.0], [0.0, 1.0]];
        let quantizer = DynamicQuantizer::new(&weights, 1.0, 0);

        let input = array![[1.0, 2.0]];
        let output = quantizer.forward(&input);

        // Identity matrix should give same output
        assert!((output[[0, 0]] - 1.0).abs() < 0.1);
        assert!((output[[0, 1]] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_dynamic_forward_1d_input() {
        let weights = array![[1.0, 2.0, 3.0]];
        let quantizer = DynamicQuantizer::new(&weights, 1.0, 0);

        // 1D input [1, 2, 3] -> treated as batch of 1
        let input = array![1.0, 2.0, 3.0];
        let output = quantizer.forward(&input);

        // [1,2,3] @ [[1,2,3]] = [14]
        assert!((output[[0]] - 14.0).abs() < 0.1);
    }

    #[test]
    fn test_dynamic_quantize_linear() {
        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let input = array![[1.0, 2.0]];

        let output = dynamic_quantize_linear(&weights, &input);

        // Expected: [1,2] @ [[1,2],[3,4]] = [7, 10]
        assert_eq!(output.shape(), &[1, 2]);
        assert!((output[[0, 0]] - 7.0).abs() < 0.5);
        assert!((output[[0, 1]] - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_dynamic_vs_static_accuracy() {
        use crate::core::modes::{
            calculate_asymmetric_params, dequantize_asymmetric, quantize_asymmetric,
        };

        let weights = array![[1.5, 2.3], [3.7, 4.1]];
        let input = array![[1.2, 2.8]];

        // Full precision baseline
        let fp_output = input.dot(&weights);

        // Dynamic quantization
        let dy_output = dynamic_quantize_linear(&weights, &input);

        let error = fp_output
            .iter()
            .zip(dy_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Dynamic should be reasonably accurate
        assert!(error < 1.0);
    }

    #[test]
    fn test_dynamic_batch_processing() {
        let weights = array![[1.0, 0.0], [0.0, 1.0]];
        let quantizer = DynamicQuantizer::new(&weights, 1.0, 0);

        // Process multiple batches with different scales
        let batch1 = array![[1.0, 2.0]];
        let batch2 = array![[10.0, 20.0]];

        let out1 = quantizer.forward(&batch1);
        let out2 = quantizer.forward(&batch2);

        // Each batch gets its own input scale
        assert_eq!(out1.shape(), out2.shape());
    }
}
