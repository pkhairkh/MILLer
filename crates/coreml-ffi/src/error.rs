//! FFI Error Types
//!
//! Error type for Core ML FFI operations with full `std::error::Error`
//! source chain support. Uses `thiserror` for consistent, boilerplate-free
//! error definitions across the crate.

/// Error type for Core ML FFI operations.
///
/// All variants that can wrap an underlying error carry an optional
/// `source` field, enabling full `std::error::Error` source chain
/// traversal (e.g., for structured logging or error reporting).
#[derive(Debug, thiserror::Error)]
pub enum FfiError {
    /// The Core ML framework is not available on this platform.
    #[error("Core ML unavailable on {platform}: {reason}")]
    PlatformUnavailable { platform: String, reason: String },

    /// The Core ML C API returned an error.
    #[error("Core ML API error in {function} (code {code}): {message}")]
    ApiError {
        function: String,
        code: i32,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// The model could not be loaded.
    #[error("Failed to load model at {path}: {reason}")]
    ModelLoadError {
        path: String,
        reason: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// The model could not be compiled.
    #[error("Failed to compile model at {path}: {reason}")]
    ModelCompileError {
        path: String,
        reason: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Prediction failed.
    #[error("Prediction failed: {reason}")]
    PredictionError {
        reason: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// The requested feature requires a newer OS version.
    #[error("Requires {required} but running {actual}")]
    InsufficientOsVersion { required: String, actual: String },

    /// Serialization or deserialization error.
    #[error("Serialization error: {reason}")]
    SerializationError {
        reason: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

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

    #[test]
    fn test_ffi_error_source_chain() {
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = FfiError::ModelLoadError {
            path: "/tmp/model.mlmodel".to_string(),
            reason: "I/O error".to_string(),
            source: Some(Box::new(io_err)),
        };
        assert!(err.source().is_some());
        assert!(err.source().unwrap().to_string().contains("file not found"));
    }

    #[test]
    fn test_ffi_error_no_source() {
        use std::error::Error;
        let err = FfiError::InsufficientOsVersion {
            required: "macOS 15.0".to_string(),
            actual: "macOS 14.0".to_string(),
        };
        assert!(err.source().is_none());
    }
}
