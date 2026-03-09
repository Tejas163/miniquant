//! Step 10: Observer Pattern
//!
//! # Learning Objectives
//! - Hook into forward passes to observe activation ranges
//! - Collect calibration data without modifying model outputs
//! - Implement observer pattern for quantization
//!
//! # Context
//! Observers run inference on sample data to collect statistics
//! without modifying the model's outputs. This is essential for
//! post-training quantization (PTQ).

use ndarray::{Array2, ArrayD, Ix2};

/// Trait for objects that can observe activation data.
pub trait Observer {
    /// Observe a batch of activation data.
    fn observe(&mut self, data: &ArrayD<f32>);

    /// Get the observed data for calibration.
    fn finalize(&self);
}

/// Simple tensor observer that collects min/max statistics.
pub struct TensorObserver {
    name: String,
    min_val: f32,
    max_val: f32,
    num_samples: usize,
}

impl TensorObserver {
    /// Create a new TensorObserver.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            num_samples: 0,
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

    /// Compute scale from observed statistics.
    pub fn compute_scale(&self) -> f32 {
        let max_abs = self.min_val.abs().max(self.max_val.abs());
        if max_abs == 0.0 {
            1.0
        } else {
            127.0 / max_abs
        }
    }

    /// Compute asymmetric params.
    pub fn compute_params(&self) -> (f32, i8) {
        let scale = (self.max_val - self.min_val) / 255.0;
        let zero_point = if scale == 0.0 {
            0
        } else {
            (-self.min_val / scale).round() as i8
        };
        (scale.max(1e-8), zero_point.clamp(-128, 127))
    }
}

impl Observer for TensorObserver {
    fn observe(&mut self, data: &ArrayD<f32>) {
        let batch_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let batch_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        self.min_val = self.min_val.min(batch_min);
        self.max_val = self.max_val.max(batch_max);
        self.num_samples += 1;
    }

    fn finalize(&self) {
        // No cleanup needed for simple min-max
    }
}

/// A simple neural network layer with observer support.
pub struct Layer {
    pub weight: Array2<f32>,
    pub bias: Option<Array2<f32>>,
    pub observer: Option<TensorObserver>,
}

impl Layer {
    /// Create a new layer.
    pub fn new(weight: Array2<f32>, bias: Option<Array2<f32>>) -> Self {
        Self {
            weight,
            bias,
            observer: None,
        }
    }

    /// Enable observation for this layer.
    pub fn observe(&mut self, name: &str) {
        self.observer = Some(TensorObserver::new(name));
    }

    /// Forward pass with optional observation.
    pub fn forward(&self, input: &ArrayD<f32>) -> ArrayD<f32> {
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

        // Observe input if observer is set
        if let Some(ref mut obs) = self.observer {
            obs.observe(&input.clone());
        }

        // Compute
        let mut output = input_2d.dot(&self.weight);

        if let Some(ref bias) = self.bias {
            for (i, row) in output.rows_mut().into_iter().enumerate() {
                *row += &bias.row(i);
            }
        }

        // Observe output if observer is set
        if let Some(ref mut obs) = self.observer {
            obs.observe(&output.clone().into_dyn());
        }

        output.into_dyn()
    }

    /// Get the observer's computed scale.
    pub fn get_observed_scale(&self) -> Option<f32> {
        self.observer.as_ref().map(|o| o.compute_scale())
    }
}

/// Model observer that manages multiple layer observers.
pub struct ModelObserver {
    observers: std::collections::HashMap<String, TensorObserver>,
}

impl ModelObserver {
    /// Create a new ModelObserver.
    pub fn new() -> Self {
        Self {
            observers: std::collections::HashMap::new(),
        }
    }

    /// Register an observer for a layer.
    pub fn register(&mut self, name: &str) {
        self.observers
            .insert(name.to_string(), TensorObserver::new(name));
    }

    /// Observe data for a specific layer.
    pub fn observe(&mut self, name: &str, data: &ArrayD<f32>) {
        if let Some(obs) = self.observers.get_mut(name) {
            obs.observe(data);
        }
    }

    /// Get scale for a specific layer.
    pub fn get_scale(&self, name: &str) -> Option<f32> {
        self.observers.get(name).map(|o| o.compute_scale())
    }

    /// Get all computed scales.
    pub fn get_all_scales(&self) -> std::collections::HashMap<String, f32> {
        self.observers
            .iter()
            .map(|(k, v)| (k.clone(), v.compute_scale()))
            .collect()
    }
}

impl Default for ModelObserver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tensor_observer() {
        let mut obs = TensorObserver::new("test");

        obs.observe(&array![[1.0, 2.0]]);
        obs.observe(&array![[3.0, 4.0]]);

        assert_eq!(obs.min(), 1.0);
        assert_eq!(obs.max(), 4.0);
        assert_eq!(obs.num_samples(), 2);
    }

    #[test]
    fn test_tensor_observer_compute_scale() {
        let mut obs = TensorObserver::new("test");
        obs.observe(&array![[-10.0, 10.0]]);

        let scale = obs.compute_scale();
        // scale = 127 / 10 = 12.7
        assert!((scale - 12.7).abs() < 0.1);
    }

    #[test]
    fn test_layer_with_observer() {
        let mut layer = Layer::new(array![[1.0, 0.0], [0.0, 1.0]], None);
        layer.observe("layer1");

        let input = array![[1.0, 2.0]];
        let output = layer.forward(&input);

        // Should observe input
        let scale = layer.get_observed_scale();
        assert!(scale.is_some());
    }

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(array![[1.0, 2.0], [3.0, 4.0]], Some(array![[0.1], [0.2]]));

        let input = array![[1.0, 1.0]];
        let output = layer.forward(&input);

        // [1,1] @ [[1,2],[3,4]] = [4, 6]
        // + bias = [4.1, 6.2]
        assert!((output[[0, 0]] - 4.1).abs() < 0.1);
        assert!((output[[0, 1]] - 6.2).abs() < 0.1);
    }

    #[test]
    fn test_model_observer() {
        let mut model_obs = ModelObserver::new();

        model_obs.register("layer1");
        model_obs.register("layer2");

        model_obs.observe("layer1", &array![[1.0, 2.0]]);
        model_obs.observe("layer2", &array![[3.0, 4.0]]);

        let scales = model_obs.get_all_scales();

        assert_eq!(scales.len(), 2);
        assert!(scales.contains_key("layer1"));
        assert!(scales.contains_key("layer2"));
    }

    #[test]
    fn test_observer_roundtrip() {
        // Test that observer doesn't affect forward pass
        let layer_with = Layer::new(array![[1.0, 0.0], [0.0, 1.0]], None);
        layer_with.observe("test");

        let layer_without = Layer::new(array![[1.0, 0.0], [0.0, 1.0]], None);

        let input = array![[5.0, 10.0]];
        let out_with = layer_with.forward(&input);
        let out_without = layer_without.forward(&input);

        assert_eq!(out_with, out_without);
    }
}
