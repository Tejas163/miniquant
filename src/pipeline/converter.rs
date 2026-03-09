//! Step 11: Model Converter
//!
//! # Learning Objectives
//! - Convert full neural network models to quantized versions
//! - Walk through model layers and apply quantization
//! - Implement complete quantization pipeline
//!
//! # Context
//! The converter walks through model layers, applies calibration,
//! quantizes weights, and replaces layers with quantized versions.

use ndarray::{Array2, ArrayD, Ix2};
use std::collections::HashMap;

use crate::ops::fusion::FusedQuantizedLinear;
use crate::tensor::QuantizedArray;

/// A simple neural network model.
#[derive(Clone)]
pub struct SimpleModel {
    layers: Vec<ModelLayer>,
}

impl SimpleModel {
    /// Create a new model from layers.
    pub fn new(layers: Vec<ModelLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass.
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer weights.
    pub fn get_weights(&self) -> Vec<(Array2<f32>, Option<Array2<f32>>)> {
        self.layers
            .iter()
            .filter_map(|l| match l {
                ModelLayer::Linear(w, b) => Some((w.clone(), b.clone())),
                _ => None,
            })
            .collect()
    }
}

/// A layer in the model.
pub enum ModelLayer {
    Linear(Array2<f32>, Option<Array2<f32>>),
    ReLU,
}

impl ModelLayer {
    /// Forward pass through a layer.
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        match self {
            ModelLayer::Linear(weight, bias) => {
                let input_2d = Self::ensure_2d(input);
                let mut output = input_2d.dot(weight);

                if let Some(b) = bias {
                    for (i, row) in output.rows_mut().into_iter().enumerate() {
                        *row += &b.row(i);
                    }
                }

                output.into_dyn()
            }
            ModelLayer::ReLU => input.mapv(|v| v.max(0.0)),
        }
    }

    fn ensure_2d(x: &ArrayD<f32>) -> Array2<f32> {
        if x.ndim() == 2 {
            x.clone().into_dimensionality::<Ix2>().unwrap()
        } else if x.ndim() == 1 {
            x.clone().into_dimensionality::<Ix2>().unwrap()
        } else {
            let shape = x.shape();
            let batch = shape[0];
            let features: usize = shape[1..].iter().product();
            x.clone().into_shape((batch, features)).unwrap()
        }
    }

    /// Get weight data if this is a linear layer.
    pub fn weight(&self) -> Option<&Array2<f32>> {
        match self {
            ModelLayer::Linear(w, _) => Some(w),
            _ => None,
        }
    }

    /// Get bias data if this is a linear layer.
    pub fn bias(&self) -> Option<&Array2<f32>> {
        match self {
            ModelLayer::Linear(_, b) => b.as_ref(),
            _ => None,
        }
    }
}

/// Configuration for quantization.
#[derive(Clone)]
pub struct QuantizationConfig {
    pub mode: QuantizationMethod,
    pub percentiles: Option<f32>,
}

#[derive(Clone, Copy)]
pub enum QuantizationMethod {
    Symmetric,
    Asymmetric,
    Dynamic,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            mode: QuantizationMethod::Asymmetric,
            percentiles: None,
        }
    }
}

/// Result of model quantization.
#[derive(Clone)]
pub struct QuantizedModel {
    pub layers: Vec<QuantizedModelLayer>,
}

impl QuantizedModel {
    /// Forward pass through quantized model.
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// A layer in the quantized model.
pub enum QuantizedModelLayer {
    QuantizedLinear(QuantizedLinearLayer),
    ReLU,
}

impl QuantizedModelLayer {
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        match self {
            QuantizedModelLayer::QuantizedLinear(l) => l.forward(input),
            QuantizedModelLayer::ReLU => input.mapv(|v| v.max(0.0)),
        }
    }
}

/// A quantized linear layer.
pub struct QuantizedLinearLayer {
    pub weight: QuantizedArray,
    pub bias: Option<Array2<f32>>,
    pub input_scale: f32,
}

impl QuantizedLinearLayer {
    pub fn new(weight: QuantizedArray, bias: Option<Array2<f32>>, input_scale: f32) -> Self {
        Self {
            weight,
            bias,
            input_scale,
        }
    }

    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
        // Quantize input
        let input_2d = if input.ndim() == 1 {
            input.clone().into_dimensionality::<Ix2>().unwrap()
        } else if input.ndim() == 2 {
            input.clone().into_dimensionality::<Ix2>().unwrap()
        } else {
            let shape = input.shape();
            let batch = shape[0];
            let features: usize = shape[1..].iter().product();
            input.clone().into_shape((batch, features)).unwrap()
        };

        // Compute input scale dynamically
        let (i_scale, i_zp) =
            crate::core::modes::calculate_asymmetric_params(&input.clone().into_dyn());

        // Quantize input
        let input_q =
            crate::core::modes::quantize_asymmetric(&input.clone().into_dyn(), i_scale, i_zp)
                .into_dimensionality::<Ix2>()
                .unwrap();

        // Dequantize weight
        let weight_f32 = self
            .weight
            .dequantize()
            .into_dimensionality::<Ix2>()
            .unwrap();

        // Matmul
        let mut output = input_q.dot(&weight_f32);

        // Scale correction
        let w_scale = self.weight.scale();
        let effective_scale = i_scale * w_scale;
        output.mapv_inplace(|v| v * effective_scale);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for (i, row) in output.rows_mut().into_iter().enumerate() {
                *row += &bias.row(i);
            }
        }

        output.into_dyn()
    }
}

/// Convert a model to quantized version.
pub fn quantize_model(
    model: &SimpleModel,
    calibration_data: &[ArrayD<f32>],
    config: &QuantizationConfig,
) -> QuantizedModel {
    let mut quantized_layers = Vec::new();

    for (i, layer) in model.layers.iter().enumerate() {
        match layer {
            ModelLayer::Linear(weight, bias) => {
                // First, determine input scale from calibration
                let input_scale = if let Some(data) = calibration_data.first() {
                    // Run forward pass to get activation scales
                    let mut x = data.clone();
                    for (j, l) in model.layers.iter().enumerate() {
                        if j == i {
                            // Get scale from input
                            let (s, _) = crate::core::modes::calculate_asymmetric_params(&x);
                            break;
                        }
                        x = l.forward(&x);
                    }
                    s
                } else {
                    1.0
                };

                // Quantize weights
                let (scale, zp) = match config.mode {
                    QuantizationMethod::Symmetric => {
                        let s = crate::core::modes::calculate_symmetric_params(
                            &weight.clone().into_dyn(),
                        );
                        (s, 0i8)
                    }
                    QuantizationMethod::Asymmetric | QuantizationMethod::Dynamic => {
                        crate::core::modes::calculate_asymmetric_params(&weight.clone().into_dyn())
                    }
                };

                let q_weight = QuantizedArray::new(weight.clone(), scale, zp);
                let bias_clone = bias.clone();

                quantized_layers.push(QuantizedModelLayer::QuantizedLinear(
                    QuantizedLinearLayer::new(q_weight, bias_clone, input_scale),
                ));
            }
            ModelLayer::ReLU => {
                quantized_layers.push(QuantizedModelLayer::ReLU);
            }
        }
    }

    QuantizedModel {
        layers: quantized_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simple_model() {
        let layers = vec![
            ModelLayer::Linear(array![[1.0, 0.0], [0.0, 1.0]], Some(array![[0.0], [0.0]])),
            ModelLayer::ReLU,
        ];

        let model = SimpleModel::new(layers);
        let input = array![[-1.0, 2.0], [3.0, -4.0]];
        let output = model.forward(&input);

        // After ReLU: max(0, x)
        assert!((output[[0, 0]] - 0.0).abs() < 0.1);
        assert!((output[[0, 1]] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantized_linear_layer() {
        let weight = array![[1.0, 2.0], [3.0, 4.0]];
        let (scale, zp) =
            crate::core::modes::calculate_asymmetric_params(&weight.clone().into_dyn());

        let q_weight = QuantizedArray::new(weight.clone(), scale, zp);

        let layer = QuantizedLinearLayer::new(q_weight, None, 1.0);

        let input = array![[1.0, 1.0]];
        let output = layer.forward(&input);

        // [1,1] @ [[1,2],[3,4]] = [4, 6]
        // With scale correction, should be close
        assert!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_quantize_model() {
        let layers = vec![ModelLayer::Linear(array![[1.0, 0.0], [0.0, 1.0]], None)];

        let model = SimpleModel::new(layers);

        let calibration_data = vec![array![[1.0, 2.0]]];

        let config = QuantizationConfig::default();
        let q_model = quantize_model(&model, &calibration_data, &config);

        assert_eq!(q_model.num_layers(), 1);
    }

    #[test]
    fn test_model_vs_quantized_accuracy() {
        let layers = vec![ModelLayer::Linear(
            array![[1.5, 2.3], [3.7, 4.1]],
            Some(array![[0.1], [0.2]]),
        )];

        let model = SimpleModel::new(layers.clone());

        let input = array![[1.2, 2.8]];
        let fp32_output = model.forward(&input);

        let calibration_data = vec![input.clone()];
        let config = QuantizationConfig::default();
        let q_model = quantize_model(&model, &calibration_data, &config);

        let q_output = q_model.forward(&input);

        let max_error = fp32_output
            .iter()
            .zip(q_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Should be reasonably accurate
        assert!(max_error < 1.0);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();

        match config.mode {
            QuantizationMethod::Asymmetric => {}
            _ => panic!("Default should be asymmetric"),
        }
    }

    #[test]
    fn test_multilayer_quantization() {
        let layers = vec![
            ModelLayer::Linear(array![[1.0, 2.0], [3.0, 4.0]], None),
            ModelLayer::ReLU,
            ModelLayer::Linear(array![[1.0, 0.0], [0.0, 1.0]], None),
        ];

        let model = SimpleModel::new(layers);

        let calibration_data = vec![array![[1.0, 1.0]]];
        let config = QuantizationConfig::default();
        let q_model = quantize_model(&model, &calibration_data, &config);

        assert_eq!(q_model.num_layers(), 3);
    }
}
