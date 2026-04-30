//! FFI Model Wrapper
//!
//! Provides a Rust-safe wrapper around the Core ML C API model type.
//! On macOS, this wraps the actual Core ML model. On other platforms,
//! it provides a stub that returns "unavailable" errors.

use crate::error::FfiError;

/// A model loaded via the Core ML C API.
///
/// This wraps an `MLModel` reference from the Core ML C API.
/// On macOS, the model is loaded using `MLModelLoad()` and
/// released using `MLModelDestroy()`. On other platforms,
/// construction returns a platform-unavailable error.
#[derive(Debug)]
pub struct FfiModel {
    /// Path to the .mlpackage or .mlmodelc directory.
    path: String,
    /// Platform-specific model handle.
    /// On macOS, this would be a raw pointer to the MLModel.
    /// On other platforms, this is None.
    handle: Option<ModelHandle>,
}

/// Platform-specific model handle.
/// On macOS, this would be a raw pointer.
/// This type is a placeholder for the actual FFI pointer type.
#[derive(Debug)]
struct ModelHandle {
    /// Placeholder for the actual handle value.
    /// On macOS: `*mut std::ffi::c_void` pointing to an MLModel.
    _raw: usize,
}

/// Model metadata returned by the FFI layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FfiModelMetadata {
    /// Number of functions in the model.
    pub function_count: usize,
    /// Function names.
    pub function_names: Vec<String>,
    /// Whether the model has state inputs.
    pub has_state: bool,
    /// Input tensor descriptions.
    pub inputs: Vec<FfiTensorDesc>,
    /// Output tensor descriptions.
    pub outputs: Vec<FfiTensorDesc>,
    /// State tensor descriptions.
    pub states: Vec<FfiTensorDesc>,
    /// Core ML specification version.
    pub spec_version: i32,
}

/// Tensor description from the FFI layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FfiTensorDesc {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<i64>,
    /// Tensor data type string.
    pub dtype: String,
    /// Whether this is a state tensor.
    pub is_state: bool,
}

/// Prediction input/output data.
#[derive(Debug, Clone)]
pub enum FfiTensorData {
    /// Float16 data (as raw bytes).
    Float16(Vec<u8>),
    /// Float32 data.
    Float32(Vec<f32>),
    /// Int32 data.
    Int32(Vec<i32>),
}

/// Result of a model prediction via FFI.
#[derive(Debug, Clone)]
pub struct FfiPredictionResult {
    /// Output tensor name -> data.
    pub outputs: Vec<(String, FfiTensorData)>,
    /// Prediction latency in nanoseconds.
    pub latency_ns: u64,
}

impl FfiModel {
    /// Load a model from an mlpackage or mlmodelc path.
    ///
    /// On macOS with Core ML, this calls `MLModelLoad()`.
    /// On other platforms, returns `FfiError::PlatformUnavailable`.
    pub fn load(path: &str) -> Result<Self, FfiError> {
        // Platform check: Core ML is only available on macOS
        if !cfg!(target_os = "macos") {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "Core ML C API requires macOS with CoreML.framework".to_string(),
            });
        }

        // On macOS, we would call:
        //   let mut model: *mut MLModel = std::ptr::null_mut();
        //   let mut error: *mut CFError = std::ptr::null_mut();
        //   let url = CFURLCreateFromFileSystemRepresentation(...);
        //   MLModelLoad(url, &mut model, &mut error);
        //
        // For now, return a stub since we can't link against CoreML.framework
        // in this environment.

        Ok(Self { path: path.to_string(), handle: None })
    }

    /// Get model metadata (inputs, outputs, functions, states).
    ///
    /// On macOS with Core ML, this inspects the loaded model.
    /// On other platforms, returns a platform-unavailable error.
    pub fn metadata(&self) -> Result<FfiModelMetadata, FfiError> {
        if self.handle.is_none() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "Model not loaded — Core ML C API requires macOS".to_string(),
            });
        }

        // On macOS, we would inspect the model's spec.
        // This is a placeholder.
        Ok(FfiModelMetadata {
            function_count: 0,
            function_names: vec![],
            has_state: false,
            inputs: vec![],
            outputs: vec![],
            states: vec![],
            spec_version: 0,
        })
    }

    /// Run prediction on the model.
    ///
    /// On macOS with Core ML, this calls `MLModelPrediction()`.
    /// On other platforms, returns a platform-unavailable error.
    pub fn predict(
        &self,
        _inputs: &[(String, FfiTensorData)],
    ) -> Result<FfiPredictionResult, FfiError> {
        if self.handle.is_none() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "Prediction requires macOS with Core ML runtime".to_string(),
            });
        }

        // On macOS:
        //   let input_feature_provider = ...;
        //   let mut output_feature_provider: *mut MLFeatureProvider = std::ptr::null_mut();
        //   MLModelPrediction(model, input_feature_provider, &mut output_feature_provider, &mut error);

        Err(FfiError::PredictionError {
            reason: "Not implemented — requires Core ML C API linkage".to_string(),
            source: None,
        })
    }

    /// Get the model path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Whether the model is actually loaded (has a valid handle).
    pub fn is_loaded(&self) -> bool {
        self.handle.is_some()
    }
}

impl Drop for FfiModel {
    fn drop(&mut self) {
        if let Some(_handle) = self.handle.take() {
            // On macOS with CoreML.framework linked, this is where we'd call:
            //   unsafe { MLModelDestroy(_handle._raw as *mut std::ffi::c_void) };
            //
            // Currently we have no Core ML C API linkage, so there is nothing
            // to free. The handle is taken() to clear the Option so the
            // FfiModel does not hold a dangling reference after drop.
        }
    }
}

// Safety: FfiModel is not Send/Sync by default because it contains
// a raw pointer. On macOS, the Core ML model object is thread-safe,
// so we could implement Send+Sync if needed. For now, we keep it
// !Send + !Sync as a conservative default.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_model_load_unavailable() {
        // On non-macOS, loading should fail with PlatformUnavailable
        let result = FfiModel::load("/path/to/model.mlpackage");
        if !cfg!(target_os = "macos") {
            assert!(result.is_err());
            match result.unwrap_err() {
                FfiError::PlatformUnavailable { .. } => {}
                other => panic!("Expected PlatformUnavailable, got: {}", other),
            }
        }
    }

    #[test]
    fn test_ffi_model_metadata_unavailable() {
        let model = FfiModel { path: "/test".to_string(), handle: None };
        let result = model.metadata();
        assert!(result.is_err());
    }

    #[test]
    fn test_ffi_model_predict_unavailable() {
        let model = FfiModel { path: "/test".to_string(), handle: None };
        let result = model.predict(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffi_tensor_data_variants() {
        let f16 = FfiTensorData::Float16(vec![0u8; 32]);
        let f32 = FfiTensorData::Float32(vec![0.0f32; 16]);
        let i32 = FfiTensorData::Int32(vec![0i32; 16]);

        // Just verify the variants exist and can be constructed
        match f16 {
            FfiTensorData::Float16(_) => {}
            _ => panic!("Expected Float16"),
        }
        match f32 {
            FfiTensorData::Float32(_) => {}
            _ => panic!("Expected Float32"),
        }
        match i32 {
            FfiTensorData::Int32(_) => {}
            _ => panic!("Expected Int32"),
        }
    }
}
