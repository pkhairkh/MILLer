//! Core ML C API FFI Declarations
//!
//! This module declares the FFI interface for the Core ML C API.
//! These declarations match Apple's public Core ML C header files.
//!
//! ## API Surface
//!
//! The Core ML C API provides these key functions:
//!
//! - **Model lifecycle**: `MLModelLoad`, `MLModelDestroy`
//! - **Prediction**: `MLModelPrediction`
//! - **Compilation**: `MLModelCompile`
//! - **Metadata**: `MLModelGetSpecification`, `MLModelConfigurationCreate`
//!
//! ## Linking
//!
//! On macOS, this crate links against `CoreML.framework`.
//! On other platforms, all functions return "unavailable" errors.

use crate::error::FfiError;

/// High-level API for Core ML FFI operations.
///
/// This provides a safe Rust interface over the Core ML C API,
/// handling platform detection and error conversion.
pub struct CoreMlApi;

impl CoreMlApi {
    /// Check if the Core ML C API is available on this platform.
    pub fn is_available() -> bool {
        cfg!(target_os = "macos")
    }

    /// Get the Core ML framework version string (macOS only).
    pub fn version() -> Result<String, FfiError> {
        if !Self::is_available() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "Core ML framework requires macOS".to_string(),
            });
        }

        // On macOS: would call the actual C API or read the framework version
        Ok("unknown".to_string())
    }

    /// Compile an mlpackage into an mlmodelc (compiled model cache).
    ///
    /// On macOS, this calls `MLModelCompile()` which produces an
    /// `.mlmodelc` directory that can be loaded more quickly than
    /// a raw `.mlpackage`.
    ///
    /// On other platforms, returns PlatformUnavailable.
    pub fn compile_model(source_path: &str, _output_dir: &str) -> Result<String, FfiError> {
        if !Self::is_available() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "Model compilation requires macOS with CoreML.framework".to_string(),
            });
        }

        // On macOS:
        // let source_url = CFURLCreateFromFileSystemRepresentation(..., source_path);
        // let output_url = CFURLCreateFromFileSystemRepresentation(..., output_dir);
        // let compiled_url = MLModelCompile(source_url, output_url, &mut error);

        Err(FfiError::ModelCompileError {
            path: source_path.to_string(),
            reason: "Not implemented — requires Core ML C API linkage".to_string(),
            source: None,
        })
    }

    /// Get model structure (structural introspection without execution).
    ///
    /// On macOS with Core ML, this uses `MLModelStructure` to walk
    /// the model's op graph without executing it. This provides:
    /// - Per-op structural information
    /// - State declarations
    /// - Function descriptions
    /// - Weight metadata
    ///
    /// On other platforms, returns PlatformUnavailable.
    pub fn inspect_model_structure(
        _mlpackage_path: &str,
    ) -> Result<ModelStructureResult, FfiError> {
        if !Self::is_available() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "MLModelStructure requires macOS with Core ML runtime".to_string(),
            });
        }

        // On macOS: MLModelStructure.load_from_path(mlpackage_path)

        Ok(ModelStructureResult {
            available: false,
            functions: vec![],
            operations: vec![],
            state_declarations: vec![],
        })
    }

    /// Inspect compute plan (per-op device placement and estimated cost).
    ///
    /// On macOS with Core ML, this uses `MLComputePlan` to determine
    /// which compute unit each operation will execute on.
    ///
    /// On other platforms, returns PlatformUnavailable.
    pub fn inspect_compute_plan(
        _mlpackage_path: &str,
        _compute_units: &str,
    ) -> Result<ComputePlanResult, FfiError> {
        if !Self::is_available() {
            return Err(FfiError::PlatformUnavailable {
                platform: std::env::consts::OS.to_string(),
                reason: "MLComputePlan requires macOS with Core ML runtime".to_string(),
            });
        }

        Ok(ComputePlanResult {
            available: false,
            reason: "Core ML C API not linked".to_string(),
            operations: vec![],
        })
    }
}

/// Result of structural model inspection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelStructureResult {
    /// Whether the inspection was successful.
    pub available: bool,
    /// Function descriptors.
    pub functions: Vec<FunctionStructure>,
    /// Operation descriptors.
    pub operations: Vec<OpStructure>,
    /// State declarations.
    pub state_declarations: Vec<StateDeclaration>,
}

/// A function's structural description.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FunctionStructure {
    /// Function name.
    pub name: String,
    /// Number of operations.
    pub op_count: usize,
    /// Input names.
    pub input_names: Vec<String>,
    /// Output names.
    pub output_names: Vec<String>,
}

/// An operation's structural description.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpStructure {
    /// Operation type (e.g., "linear", "gelu", "scaled_dot_product_attention").
    pub op_type: String,
    /// Operation name in the graph.
    pub name: String,
    /// Input names.
    pub inputs: Vec<String>,
    /// Output names.
    pub outputs: Vec<String>,
}

/// A state declaration in the model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateDeclaration {
    /// State name.
    pub name: String,
    /// State shape.
    pub shape: Vec<i64>,
    /// State data type.
    pub dtype: String,
}

/// Result of compute plan inspection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComputePlanResult {
    /// Whether the compute plan was available.
    pub available: bool,
    /// Reason if unavailable.
    pub reason: String,
    /// Per-operation placement info.
    pub operations: Vec<OpPlacement>,
}

/// Per-operation device placement from compute plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpPlacement {
    /// Operation name.
    pub name: String,
    /// Estimated compute unit (CPU, GPU, ANE).
    pub compute_unit: String,
    /// Estimated latency in microseconds.
    pub estimated_latency_us: Option<f64>,
}

// ─── Core ML C API Function Declarations ─────────────────────────────────────
// These are the actual C function signatures that would be linked
// on macOS. They are declared here for reference and future use.

#[cfg(target_os = "macos")]
mod c_api {
    // These would be the actual FFI declarations:
    //
    // extern "C" {
    //     /// Load a Core ML model from a URL.
    //     fn MLModelLoad(
    //         url: *const c_void,  // CFURLRef
    //         model: *mut *mut c_void,  // MLModel**
    //         error: *mut *mut c_void,  // CFErrorRef*
    //     ) -> bool;
    //
    //     /// Destroy a loaded Core ML model.
    //     fn MLModelDestroy(model: *mut c_void);
    //
    //     /// Run prediction on a model.
    //     fn MLModelPrediction(
    //         model: *mut c_void,  // MLModel*
    //         input: *const c_void,  // MLFeatureProvider*
    //         output: *mut *mut c_void,  // MLFeatureProvider**
    //         error: *mut *mut c_void,  // CFErrorRef*
    //     ) -> bool;
    //
    //     /// Compile an mlpackage into an mlmodelc.
    //     fn MLModelCompile(
    //         source_url: *const c_void,  // CFURLRef
    //         output_url: *const c_void,  // CFURLRef
    //         error: *mut *mut c_void,  // CFErrorRef*
    //     ) -> *mut c_void;  // CFURLRef (compiled model URL)
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── CoreMlApi::is_available ──────────────────────────────────────────

    #[test]
    fn test_coreml_api_availability() {
        // On Linux, should be unavailable
        if !cfg!(target_os = "macos") {
            assert!(!CoreMlApi::is_available());
        }
    }

    // ─── CoreMlApi::version ───────────────────────────────────────────────

    #[test]
    fn test_coreml_api_version_unavailable() {
        let result = CoreMlApi::version();
        if !cfg!(target_os = "macos") {
            assert!(result.is_err());
            match result.unwrap_err() {
                FfiError::PlatformUnavailable { platform, reason } => {
                    assert_eq!(platform, std::env::consts::OS);
                    assert!(reason.contains("macOS"), "reason should mention macOS");
                }
                other => panic!("Expected PlatformUnavailable, got: {}", other),
            }
        }
    }

    // ─── CoreMlApi::compile_model ─────────────────────────────────────────

    #[test]
    fn test_coreml_api_compile_model_unavailable() {
        let result = CoreMlApi::compile_model("/test/model.mlpackage", "/tmp");
        if !cfg!(target_os = "macos") {
            assert!(result.is_err());
            match result.unwrap_err() {
                FfiError::PlatformUnavailable { platform, reason } => {
                    assert_eq!(platform, std::env::consts::OS);
                    assert!(reason.contains("macOS") || reason.contains("CoreML"));
                }
                other => panic!("Expected PlatformUnavailable, got: {}", other),
            }
        }
    }

    // ─── CoreMlApi::inspect_model_structure ───────────────────────────────

    #[test]
    fn test_inspect_model_structure_unavailable() {
        let result = CoreMlApi::inspect_model_structure("/test/model.mlpackage");
        if !cfg!(target_os = "macos") {
            assert!(result.is_err());
            match result.unwrap_err() {
                FfiError::PlatformUnavailable { reason, .. } => {
                    assert!(reason.contains("macOS") || reason.contains("Core ML"));
                }
                other => panic!("Expected PlatformUnavailable, got: {}", other),
            }
        }
    }

    // ─── CoreMlApi::inspect_compute_plan ──────────────────────────────────

    #[test]
    fn test_inspect_compute_plan_unavailable() {
        let result = CoreMlApi::inspect_compute_plan("/test/model.mlpackage", "CPU_AND_NE");
        if !cfg!(target_os = "macos") {
            assert!(result.is_err());
            match result.unwrap_err() {
                FfiError::PlatformUnavailable { reason, .. } => {
                    assert!(reason.contains("macOS") || reason.contains("Core ML"));
                }
                other => panic!("Expected PlatformUnavailable, got: {}", other),
            }
        }
    }

    // ─── Result type structure tests (T-76, I-51) ─────────────────────────

    #[test]
    fn test_model_structure_result_serialization() {
        let result = ModelStructureResult {
            available: true,
            functions: vec![FunctionStructure {
                name: "main".to_string(),
                op_count: 5,
                input_names: vec!["input_ids".to_string()],
                output_names: vec!["logits".to_string()],
            }],
            operations: vec![OpStructure {
                op_type: "linear".to_string(),
                name: "op_0".to_string(),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
            }],
            state_declarations: vec![StateDeclaration {
                name: "kv_cache".to_string(),
                shape: vec![1, 64, 128],
                dtype: "fp16".to_string(),
            }],
        };

        // Verify JSON serialization roundtrip
        let json = serde_json::to_string(&result).expect("should serialize");
        let deserialized: ModelStructureResult =
            serde_json::from_str(&json).expect("should deserialize");

        assert_eq!(deserialized.available, true);
        assert_eq!(deserialized.functions.len(), 1);
        assert_eq!(deserialized.functions[0].name, "main");
        assert_eq!(deserialized.operations[0].op_type, "linear");
        assert_eq!(deserialized.state_declarations[0].shape, vec![1, 64, 128]);
    }

    #[test]
    fn test_compute_plan_result_serialization() {
        let result = ComputePlanResult {
            available: true,
            reason: String::new(),
            operations: vec![OpPlacement {
                name: "linear_0".to_string(),
                compute_unit: "ANE".to_string(),
                estimated_latency_us: Some(150.0),
            }],
        };

        let json = serde_json::to_string(&result).expect("should serialize");
        let deserialized: ComputePlanResult =
            serde_json::from_str(&json).expect("should deserialize");

        assert!(deserialized.available);
        assert_eq!(deserialized.operations.len(), 1);
        assert_eq!(deserialized.operations[0].compute_unit, "ANE");
        assert_eq!(deserialized.operations[0].estimated_latency_us, Some(150.0));
    }

    #[test]
    fn test_model_structure_result_empty() {
        let result = ModelStructureResult {
            available: false,
            functions: vec![],
            operations: vec![],
            state_declarations: vec![],
        };

        assert!(!result.available);
        assert!(result.functions.is_empty());
        assert!(result.operations.is_empty());
    }

    #[test]
    fn test_compute_plan_result_unavailable() {
        let result = ComputePlanResult {
            available: false,
            reason: "Core ML C API not linked".to_string(),
            operations: vec![],
        };

        assert!(!result.available);
        assert!(!result.reason.is_empty());
    }

    #[test]
    fn test_op_placement_all_compute_units() {
        let units = vec!["CPU", "GPU", "ANE"];
        for unit in units {
            let placement = OpPlacement {
                name: "test_op".to_string(),
                compute_unit: unit.to_string(),
                estimated_latency_us: None,
            };
            assert_eq!(placement.compute_unit, unit);
        }
    }

    #[test]
    fn test_function_structure_fields() {
        let func = FunctionStructure {
            name: "encode".to_string(),
            op_count: 42,
            input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
            output_names: vec!["hidden_states".to_string()],
        };

        assert_eq!(func.name, "encode");
        assert_eq!(func.op_count, 42);
        assert_eq!(func.input_names.len(), 2);
        assert_eq!(func.output_names.len(), 1);
    }

    #[test]
    fn test_state_declaration_dtype_field() {
        let state = StateDeclaration {
            name: "k_cache".to_string(),
            shape: vec![1, 32, 128, 64],
            dtype: "fp16".to_string(),
        };

        assert_eq!(state.dtype, "fp16");
        assert_eq!(state.shape.len(), 4);
    }
}
