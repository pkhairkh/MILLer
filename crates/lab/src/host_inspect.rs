//! Host-Side Inspector
//!
//! Performs honest host-side inspection of mlpackage artifacts.
//! Checks what can be determined without executing the model on
//! a device runtime. NEVER infers ANE behavior or compute unit
//! placement from host-only evidence.
//!
//! The inspector works by:
//! 1. Checking that the mlpackage directory exists and has the expected structure
//! 2. Reading the mlpackage Manifest.json for metadata
//! 3. (Via Python bridge) Attempting to load the model with coremltools
//! 4. (Via Python bridge) Checking compute plan availability
//! 5. (Via Python bridge) Performing structural verification via MLModelStructure (Sprint 34)
//! 6. Building an InspectionStepResult that honestly reports what was found
//!
//! Sprint 34 adds step 5: structural verification. When MLModelStructure is
//! available (macOS with Core ML runtime), the inspector walks the emitted
//! model structure and reports op inventory, function signatures, and state
//! declarations. When unavailable, it falls back to file-based heuristics
//! and explicitly labels the result as "fallback_file_check" rather than
//! "mlmodel_structure".

use crate::harness::{InspectionStepResult, TensorSpecRecord};
use std::path::Path;

/// Host-side inspector for mlpackage artifacts.
pub struct HostInspector {
    /// Path to the Python bridge script.
    bridge_script: String,
    /// Path to the Python interpreter.
    python_path: String,
}

impl HostInspector {
    /// Create a new host-side inspector.
    pub fn new(python_path: &str, bridge_script: &str) -> Self {
        Self { bridge_script: bridge_script.to_string(), python_path: python_path.to_string() }
    }

    /// Perform host-side inspection of an mlpackage.
    ///
    /// This first does Rust-side file checks, then optionally calls
    /// the Python bridge for model loading, compute plan checks, and
    /// structural verification via MLModelStructure.
    pub fn inspect(&self, mlpackage_path: &str) -> InspectionStepResult {
        let mut warnings = Vec::new();
        let pkg_path = Path::new(mlpackage_path);

        // Step 1: Check package presence
        let package_present = pkg_path.exists() && pkg_path.is_dir();
        if !package_present {
            return InspectionStepResult {
                package_present: false,
                manifest_readable: false,
                model_loadable: false,
                model_load_failure_reason: Some("mlpackage directory does not exist".to_string()),
                function_count: None,
                input_specs: vec![],
                output_specs: vec![],
                warnings: vec!["mlpackage directory not found".to_string()],
                structure_inspection_available: None,
                structure_inspection_failure_reason: Some(
                    "mlpackage directory does not exist".to_string(),
                ),
                structure_op_names: vec![],
                structure_op_count: None,
                structure_function_count: None,
                structure_state_declarations: vec![],
                op_fidelity_score: None,
                missing_ops: vec![],
                extra_ops: vec![],
                inspection_method: "none".to_string(),
            };
        }

        // Step 2: Read mlpackage Manifest.json
        let manifest_path = pkg_path.join("Manifest.json");
        let manifest_readable = if manifest_path.exists() {
            match std::fs::read_to_string(&manifest_path) {
                Ok(content) => {
                    // Try to parse as JSON to verify it's valid
                    match serde_json::from_str::<serde_json::Value>(&content) {
                        Ok(_) => true,
                        Err(e) => {
                            warnings.push(format!("Manifest.json is not valid JSON: {}", e));
                            false
                        }
                    }
                }
                Err(e) => {
                    warnings.push(format!("Failed to read Manifest.json: {}", e));
                    false
                }
            }
        } else {
            warnings.push("Manifest.json not found in mlpackage".to_string());
            false
        };

        // Step 3: Check for model.mlmodel (core mlpackage structure)
        let model_path = pkg_path.join("Data").join("com.apple.CoreML").join("model.mlmodel");
        if !model_path.exists() {
            warnings.push("model.mlmodel not found in expected path".to_string());
        }

        // Step 4: Check for weights directory
        let weights_dir = pkg_path.join("Data").join("com.apple.CoreML").join("weights");
        if weights_dir.exists() {
            // Count weight files
            if let Ok(entries) = std::fs::read_dir(&weights_dir) {
                let weight_count = entries.flatten().count();
                if weight_count == 0 {
                    warnings.push("weights directory exists but is empty".to_string());
                }
            }
        }

        // Step 5: Attempt Python bridge host_inspect for model loading
        let (model_loadable, model_load_failure_reason, function_count, input_specs, output_specs) =
            self.python_inspect(mlpackage_path, &mut warnings);

        // Step 6: Attempt Python bridge model_structure for structural verification (Sprint 34)
        let struct_result = self.python_model_structure(mlpackage_path, &mut warnings);

        InspectionStepResult {
            package_present,
            manifest_readable,
            model_loadable,
            model_load_failure_reason,
            function_count,
            input_specs,
            output_specs,
            warnings,
            // Sprint 34: Structural verification fields
            structure_inspection_available: struct_result.available,
            structure_inspection_failure_reason: struct_result.failure_reason,
            structure_op_names: struct_result.op_names,
            structure_op_count: struct_result.op_count,
            structure_function_count: struct_result.function_count,
            structure_state_declarations: struct_result.state_declarations,
            op_fidelity_score: struct_result.op_fidelity_score,
            missing_ops: struct_result.missing_ops,
            extra_ops: struct_result.extra_ops,
            inspection_method: struct_result.method,
        }
    }

    /// Perform Python-side inspection via the bridge.
    ///
    /// This calls the `host_inspect` command on the Python bridge,
    /// which attempts model loading and compute plan checks.
    fn python_inspect(
        &self,
        mlpackage_path: &str,
        warnings: &mut Vec<String>,
    ) -> (bool, Option<String>, Option<usize>, Vec<TensorSpecRecord>, Vec<TensorSpecRecord>) {
        use ane_bridge::subprocess::PythonBridge;

        let bridge = PythonBridge::new(&self.python_path, &self.bridge_script);

        let payload = serde_json::json!({
            "command": "host_inspect",
            "bridge_version": 1,
            "mlpackage_path": mlpackage_path,
            "compute_units": "CPU_AND_NE",
        });

        match bridge.execute_raw_payload(&payload) {
            Ok(result) => {
                if result.status != "success" {
                    warnings.push(format!(
                        "Python host_inspect failed: {}",
                        result.error_message.as_deref().unwrap_or("unknown error")
                    ));
                    return (false, result.error_message, None, vec![], vec![]);
                }

                // Parse the inspection result from metadata
                let meta = &result.metadata;

                let model_loadable =
                    meta.get("model_loadable").and_then(|v| v.as_bool()).unwrap_or(false);

                let model_load_failure_reason = meta
                    .get("model_load_failure_reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let function_count =
                    meta.get("function_count").and_then(|v| v.as_u64()).map(|n| n as usize);

                let input_specs = meta
                    .get("input_specs")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                Some(TensorSpecRecord {
                                    name: v.get("name")?.as_str()?.to_string(),
                                    shape: v
                                        .get("shape")?
                                        .as_array()?
                                        .iter()
                                        .filter_map(|s| s.as_u64().map(|n| n as usize))
                                        .collect(),
                                    dtype: v.get("dtype")?.as_str()?.to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let output_specs = meta
                    .get("output_specs")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                Some(TensorSpecRecord {
                                    name: v.get("name")?.as_str()?.to_string(),
                                    shape: v
                                        .get("shape")?
                                        .as_array()?
                                        .iter()
                                        .filter_map(|s| s.as_u64().map(|n| n as usize))
                                        .collect(),
                                    dtype: v.get("dtype")?.as_str()?.to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                // Collect warnings from Python side
                if let Some(py_warnings) = meta.get("warnings").and_then(|v| v.as_array()) {
                    for w in py_warnings {
                        if let Some(s) = w.as_str() {
                            warnings.push(s.to_string());
                        }
                    }
                }

                (
                    model_loadable,
                    model_load_failure_reason,
                    function_count,
                    input_specs,
                    output_specs,
                )
            }
            Err(e) => {
                warnings.push(format!("Python bridge invocation failed: {}", e));
                (false, Some(format!("Bridge error: {}", e)), None, vec![], vec![])
            }
        }
    }

    /// Perform Python-side structural verification via the model_structure bridge command.
    ///
    /// This calls the `model_structure` command on the Python bridge, which
    /// uses MLModelStructure.load_from_path() when available (macOS) or
    /// reports unavailability on non-Apple platforms.
    ///
    /// Returns a `StructureInspectionResult` with structural verification data.
    fn python_model_structure(
        &self,
        mlpackage_path: &str,
        warnings: &mut Vec<String>,
    ) -> StructureInspectionResult {
        use ane_bridge::subprocess::PythonBridge;

        let bridge = PythonBridge::new(&self.python_path, &self.bridge_script);

        let payload = serde_json::json!({
            "command": "model_structure",
            "bridge_version": 1,
            "mlpackage_path": mlpackage_path,
            "include_fallback": true,
        });

        match bridge.execute_raw_payload(&payload) {
            Ok(result) => {
                if result.status != "success" {
                    warnings.push(format!(
                        "Python model_structure failed: {}",
                        result.error_message.as_deref().unwrap_or("unknown error")
                    ));
                    return StructureInspectionResult {
                        available: Some(false),
                        failure_reason: result.error_message,
                        op_names: vec![],
                        op_count: None,
                        function_count: None,
                        state_declarations: vec![],
                        op_fidelity_score: None,
                        missing_ops: vec![],
                        extra_ops: vec![],
                        method: "none".to_string(),
                    };
                }

                let meta = &result.metadata;

                let available = meta.get("available").and_then(|v| v.as_bool());

                let failure_reason = if available == Some(false) {
                    meta.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string())
                } else {
                    None
                };

                let method = meta
                    .get("inspection_method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Extract op names from operations list
                let op_names: Vec<String> = meta
                    .get("operations")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|op| {
                                op.get("op_type").and_then(|t| t.as_str()).map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let op_count =
                    meta.get("total_operation_count").and_then(|v| v.as_u64()).map(|n| n as usize);

                // Extract function count from functions list
                let function_count =
                    meta.get("functions").and_then(|v| v.as_array()).map(|arr| arr.len());

                // Extract state declarations
                let state_declarations: Vec<TensorSpecRecord> = meta
                    .get("state_declarations")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| {
                                Some(TensorSpecRecord {
                                    name: v.get("name")?.as_str()?.to_string(),
                                    shape: v
                                        .get("shape")
                                        .and_then(|s| s.as_array())
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(|v| v.as_u64().map(|n| n as usize))
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                    dtype: v
                                        .get("dtype")
                                        .and_then(|d| d.as_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                // Extract MIR comparison results
                let mir_comparison = meta.get("mir_comparison");
                let op_fidelity_score = mir_comparison
                    .and_then(|mc| mc.get("op_fidelity_score"))
                    .and_then(|v| v.as_f64());

                let missing_ops: Vec<String> = mir_comparison
                    .and_then(|mc| mc.get("missing_from_structure"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|op| {
                                op.get("op_type").and_then(|t| t.as_str()).map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let extra_ops: Vec<String> = meta
                    .get("mir_comparison")
                    .and_then(|mc| mc.get("extra_in_structure"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|op| {
                                op.get("op_type").and_then(|t| t.as_str()).map(|s| s.to_string())
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                // Add informational warning about structural inspection status
                match available {
                    Some(true) => {
                        // Structural inspection succeeded — this is the gold standard
                    }
                    Some(false) => {
                        warnings.push(
                            "MLModelStructure inspection unavailable — structural verification "
                                .to_string()
                                + "uses weaker fallback methods. See inspection_method field.",
                        );
                    }
                    None => {
                        warnings.push(
                            "MLModelStructure inspection result missing from bridge response"
                                .to_string(),
                        );
                    }
                }

                StructureInspectionResult {
                    available,
                    failure_reason,
                    op_names,
                    op_count,
                    function_count,
                    state_declarations,
                    op_fidelity_score,
                    missing_ops,
                    extra_ops,
                    method,
                }
            }
            Err(e) => {
                warnings.push(format!("Python bridge model_structure invocation failed: {}", e));
                StructureInspectionResult {
                    available: Some(false),
                    failure_reason: Some(format!("Bridge error: {}", e)),
                    op_names: vec![],
                    op_count: None,
                    function_count: None,
                    state_declarations: vec![],
                    op_fidelity_score: None,
                    missing_ops: vec![],
                    extra_ops: vec![],
                    method: "none".to_string(),
                }
            }
        }
    }
}

/// Internal helper struct for structure inspection results before
/// they are folded into InspectionStepResult.
struct StructureInspectionResult {
    available: Option<bool>,
    failure_reason: Option<String>,
    op_names: Vec<String>,
    op_count: Option<usize>,
    function_count: Option<usize>,
    state_declarations: Vec<TensorSpecRecord>,
    op_fidelity_score: Option<f64>,
    missing_ops: Vec<String>,
    extra_ops: Vec<String>,
    method: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a HostInspector with a dummy Python path and bridge script.
    /// On Linux, the Python bridge calls will always fail, which is expected.
    fn make_inspector() -> HostInspector {
        HostInspector::new("python3", "bridge_script.py")
    }

    #[test]
    fn test_host_inspector_new() {
        let inspector = HostInspector::new("/usr/bin/python3", "/path/to/bridge.py");
        // We can't directly access private fields, but we can verify the inspector
        // was constructed successfully by using it.
        let result = inspector.inspect("/nonexistent/path");
        // The inspector should have been constructed and is usable
        assert!(!result.package_present, "Nonexistent path should report package_present=false");
    }

    #[test]
    fn test_inspect_nonexistent_path() {
        let inspector = make_inspector();
        let result = inspector.inspect("/nonexistent/mlpackage_that_does_not_exist");

        assert!(!result.package_present, "package_present should be false for nonexistent path");
        assert!(
            !result.manifest_readable,
            "manifest_readable should be false for nonexistent path"
        );
        assert!(!result.model_loadable, "model_loadable should be false for nonexistent path");
        assert_eq!(
            result.inspection_method, "none",
            "inspection_method should be 'none' for nonexistent path"
        );
        assert!(!result.warnings.is_empty(), "warnings should be non-empty for nonexistent path");
    }

    #[test]
    fn test_inspect_empty_directory() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a directory that exists but has no Manifest.json
        let mlpackage_dir = tmp.path().join("TestModel.mlpackage");
        std::fs::create_dir_all(&mlpackage_dir).unwrap();

        let inspector = make_inspector();
        let result = inspector.inspect(mlpackage_dir.to_str().unwrap());

        assert!(result.package_present, "package_present should be true for existing directory");
        assert!(
            !result.manifest_readable,
            "manifest_readable should be false when Manifest.json is missing"
        );
        // Should have a warning about missing Manifest.json
        assert!(
            result.warnings.iter().any(|w| w.contains("Manifest.json")),
            "Should warn about missing Manifest.json, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_inspect_with_valid_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let mlpackage_dir = tmp.path().join("TestModel.mlpackage");
        std::fs::create_dir_all(&mlpackage_dir).unwrap();

        // Write a valid Manifest.json
        let manifest = serde_json::json!({
            "model_name": "TestModel",
            "model_version": "1.0"
        });
        std::fs::write(
            mlpackage_dir.join("Manifest.json"),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        // Create the Data/com.apple.CoreML/ directory and a model.mlmodel file
        let coreml_dir = mlpackage_dir.join("Data").join("com.apple.CoreML");
        std::fs::create_dir_all(&coreml_dir).unwrap();
        std::fs::write(coreml_dir.join("model.mlmodel"), "dummy model data").unwrap();

        let inspector = make_inspector();
        let result = inspector.inspect(mlpackage_dir.to_str().unwrap());

        assert!(result.package_present, "package_present should be true");
        assert!(
            result.manifest_readable,
            "manifest_readable should be true with valid Manifest.json"
        );
        // On Linux, model_loadable should be false (Python bridge fails)
        assert!(
            !result.model_loadable,
            "model_loadable should be false on Linux (no Core ML runtime)"
        );
    }

    #[test]
    fn test_inspect_with_invalid_manifest_json() {
        let tmp = tempfile::tempdir().unwrap();
        let mlpackage_dir = tmp.path().join("BadManifest.mlpackage");
        std::fs::create_dir_all(&mlpackage_dir).unwrap();

        // Write an invalid Manifest.json
        std::fs::write(mlpackage_dir.join("Manifest.json"), "this is not valid json {{{{").unwrap();

        let inspector = make_inspector();
        let result = inspector.inspect(mlpackage_dir.to_str().unwrap());

        assert!(result.package_present, "package_present should be true");
        assert!(!result.manifest_readable, "manifest_readable should be false with invalid JSON");
        assert!(
            result.warnings.iter().any(|w| w.contains("not valid JSON")),
            "Should warn about invalid JSON in Manifest.json, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_inspect_with_empty_weights_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mlpackage_dir = tmp.path().join("EmptyWeights.mlpackage");
        std::fs::create_dir_all(&mlpackage_dir).unwrap();

        // Write a valid Manifest.json
        std::fs::write(
            mlpackage_dir.join("Manifest.json"),
            serde_json::json!({"model_name": "Test"}).to_string(),
        )
        .unwrap();

        // Create Data/com.apple.CoreML/weights/ directory (empty)
        let weights_dir = mlpackage_dir.join("Data").join("com.apple.CoreML").join("weights");
        std::fs::create_dir_all(&weights_dir).unwrap();

        let inspector = make_inspector();
        let result = inspector.inspect(mlpackage_dir.to_str().unwrap());

        assert!(
            result.warnings.iter().any(|w| w.contains("empty")),
            "Should warn about empty weights directory, got: {:?}",
            result.warnings
        );
    }

    #[test]
    fn test_structure_inspection_result_default_fields() {
        // On Linux, the Python bridge will always fail, so structure inspection
        // fields should reflect that unavailability.
        let tmp = tempfile::tempdir().unwrap();
        let mlpackage_dir = tmp.path().join("StructTest.mlpackage");
        std::fs::create_dir_all(&mlpackage_dir).unwrap();

        // Write a valid Manifest.json so we get past the early checks
        std::fs::write(
            mlpackage_dir.join("Manifest.json"),
            serde_json::json!({"model_name": "Test"}).to_string(),
        )
        .unwrap();

        let inspector = make_inspector();
        let result = inspector.inspect(mlpackage_dir.to_str().unwrap());

        // On Linux, the Python bridge fails, so structure inspection should be unavailable
        assert!(
            result.structure_inspection_available == Some(false)
                || result.structure_inspection_available.is_none(),
            "structure_inspection_available should be Some(false) or None on Linux, got: {:?}",
            result.structure_inspection_available
        );
        assert!(
            result.structure_op_names.is_empty(),
            "structure_op_names should be empty when bridge fails"
        );
        assert!(
            result.structure_op_count.is_none(),
            "structure_op_count should be None when bridge fails"
        );
        assert!(
            result.structure_function_count.is_none(),
            "structure_function_count should be None when bridge fails"
        );
        assert!(
            result.structure_state_declarations.is_empty(),
            "structure_state_declarations should be empty when bridge fails"
        );
        assert!(
            result.op_fidelity_score.is_none(),
            "op_fidelity_score should be None when bridge fails"
        );
        assert!(result.missing_ops.is_empty(), "missing_ops should be empty when bridge fails");
        assert!(result.extra_ops.is_empty(), "extra_ops should be empty when bridge fails");
    }
}
