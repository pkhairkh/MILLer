//! FFI Error Types

use std::fmt;

/// Error type for Core ML FFI operations.
#[derive(Debug, Clone)]
pub enum FfiError {
    /// The Core ML framework is not available on this platform.
    PlatformUnavailable { platform: String, reason: String },

    /// The Core ML C API returned an error.
    ApiError { function: String, code: i32, message: String },

    /// The model could not be loaded.
    ModelLoadError { path: String, reason: String },

    /// The model could not be compiled.
    ModelCompileError { path: String, reason: String },

    /// Prediction failed.
    PredictionError { reason: String },

    /// The requested feature requires a newer OS version.
    InsufficientOsVersion { required: String, actual: String },

    /// Serialization or deserialization error.
    SerializationError { reason: String },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::PlatformUnavailable { platform, reason } => {
                write!(f, "Core ML unavailable on {}: {}", platform, reason)
            }
            FfiError::ApiError { function, code, message } => {
                write!(f, "Core ML API error in {} (code {}): {}", function, code, message)
            }
            FfiError::ModelLoadError { path, reason } => {
                write!(f, "Failed to load model at {}: {}", path, reason)
            }
            FfiError::ModelCompileError { path, reason } => {
                write!(f, "Failed to compile model at {}: {}", path, reason)
            }
            FfiError::PredictionError { reason } => {
                write!(f, "Prediction failed: {}", reason)
            }
            FfiError::InsufficientOsVersion { required, actual } => {
                write!(f, "Requires {} but running {}", required, actual)
            }
            FfiError::SerializationError { reason } => {
                write!(f, "Serialization error: {}", reason)
            }
        }
    }
}

impl std::error::Error for FfiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_error_display() {
        let err = FfiError::PlatformUnavailable {
            platform: "Linux".to_string(),
            reason: "Core ML requires macOS".to_string(),
        };
        assert!(err.to_string().contains("Core ML unavailable on Linux"));
    }
}
