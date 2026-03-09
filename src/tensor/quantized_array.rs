//! Step 7: QuantizedArray
//!
//! # Learning Objectives
//! - Design a tensor struct that stores quantized data with metadata
//! - Separate raw integer data from quantization parameters
//! - Implement serialization for quantized tensors
//!
//! # Context
//! A QuantizedArray stores INT8 data along with scale and zero_point
//! for accurate dequantization. This is the core data structure
//! used throughout the quantization pipeline.

use ndarray::{Array, Array2, ArrayD, Dim, IntoDimension, Ix1, Ix2};
use serde::{Deserialize, Serialize};

/// A quantized array with scale and zero-point metadata.
///
/// # Storage
/// - `data`: INT8 array storing quantized values
/// - `scale`: Float scale factor for dequantization
/// - `zero_point`: INT8 offset for asymmetric quantization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedArray {
    data: Vec<i8>,
    shape: Vec<usize>,
    scale: f32,
    zero_point: i8,
}

impl QuantizedArray {
    /// Create a new QuantizedArray from float data.
    ///
    /// # Arguments
    /// * `data` - Float array to quantize
    /// * `scale` - Scale factor
    /// * `zero_point` - Zero-point offset
    pub fn new<D: IntoDimension<Dim = Dim<ndarray::ConstDyn>>>(
        data: Array<f32, D>,
        scale: f32,
        zero_point: i8,
    ) -> Self {
        let shape = data.shape().to_vec();
        let flat_data = data
            .into_iter()
            .map(|&v| {
                let q = (v / scale + zero_point as f32).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect();

        Self {
            data: flat_data,
            shape,
            scale,
            zero_point,
        }
    }

    /// Create from pre-quantized data.
    pub fn from_quantized<D: IntoDimension<Dim = Dim<ndarray::ConstDyn>>>(
        data: Array<i8, D>,
        scale: f32,
        zero_point: i8,
    ) -> Self {
        let shape = data.shape().to_vec();
        let flat_data: Vec<i8> = data.into_iter().collect();

        Self {
            data: flat_data,
            shape,
            scale,
            zero_point,
        }
    }

    /// Get the scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the zero-point.
    pub fn zero_point(&self) -> i8 {
        self.zero_point
    }

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dequantize back to float array.
    pub fn dequantize(&self) -> ArrayD<f32> {
        let mut arr = Array::from_shape_vec(&self.shape[..], self.data.clone()).unwrap();

        // Apply dequantization in place
        arr.mapv_inplace(|v| (v as f32 - self.zero_point as f32) * self.scale);

        arr.into_dyn()
    }

    /// Get raw quantized data as a flat slice.
    pub fn as_slice(&self) -> &[i8] {
        &self.data
    }

    /// Quantize float data and wrap in QuantizedArray.
    pub fn quantize(data: &ArrayD<f32>, scale: f32, zero_point: i8) -> Self {
        let shape = data.shape().to_vec();
        let flat_data = data
            .iter()
            .map(|&v| {
                let q = (v / scale + zero_point as f32).round();
                q.clamp(-128.0, 127.0) as i8
            })
            .collect();

        Self {
            data: flat_data,
            shape,
            scale,
            zero_point,
        }
    }
}

impl From<QuantizedArray> for ArrayD<f32> {
    fn from(q: QuantizedArray) -> Self {
        q.dequantize()
    }
}

impl From<QuantizedArray> for Array2<f32> {
    fn from(q: QuantizedArray) -> Self {
        let shape = q.shape();
        if shape.len() != 2 {
            panic!("Cannot convert to Array2 from shape {:?}", shape);
        }
        q.dequantize().into_dimensionality::<Ix2>().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_quantized_array_creation() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let q = QuantizedArray::new(x, 1.0, 0);

        assert_eq!(q.shape(), &[2, 2]);
        assert_eq!(q.scale(), 1.0);
        assert_eq!(q.zero_point(), 0);
    }

    #[test]
    fn test_quantized_array_roundtrip() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let scale = crate::core::modes::calculate_symmetric_params(&x.clone().into_dyn());

        let q = QuantizedArray::new(x.clone(), scale, 0);
        let reconstructed = q.dequantize();

        let max_error = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_error < 0.1);
    }

    #[test]
    fn test_from_quantized() {
        let q_data = array![[1i8, 2], [3, 4]];
        let q = QuantizedArray::from_quantized(q_data, 1.0, 0);

        let reconstructed = q.dequantize();
        let expected = array![[1.0, 2.0], [3.0, 4.0]];

        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn test_quantized_with_zero_point() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        // Asymmetric quantization
        let (scale, zp) = crate::core::modes::calculate_asymmetric_params(&x.clone().into_dyn());

        let q = QuantizedArray::new(x.clone(), scale, zp);
        let reconstructed = q.dequantize();

        let max_error = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_error < 0.2);
    }

    #[test]
    fn test_quantized_array_metadata() {
        let x = array![[1.0, 2.0]];
        let q = QuantizedArray::new(x, 0.5, 10);

        assert_eq!(q.ndim(), 2);
        assert_eq!(q.len(), 2);
        assert!(!q.is_empty());
    }

    #[test]
    fn test_quantized_array_serialize() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let q = QuantizedArray::new(x, 1.0, 0);

        let serialized = serde_json::to_string(&q).unwrap();
        let deserialized: QuantizedArray = serde_json::from_str(&serialized).unwrap();

        assert_eq!(q.shape(), deserialized.shape());
        assert_eq!(q.scale(), deserialized.scale());
        assert_eq!(q.zero_point(), deserialized.zero_point());
    }

    #[test]
    fn test_into_array2() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let q = QuantizedArray::new(x.clone(), 1.0, 0);

        let arr2: Array2<f32> = q.into();
        assert_eq!(arr2, x);
    }
}
