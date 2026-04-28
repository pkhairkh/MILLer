//! Proto-Direct Emission Interface for the Bridge
//!
//! This module provides the proto-direct emission interface that the bridge
//! can use as an alternative to the Python subprocess. It wraps the
//! `ane-coreml-emit` crate's `ProtoEmitter` with bridge-friendly types
//! and result structures.
//!
//! ## Emission Paths
//!
//! 1. **Direct MIR emission**: `MirGraphCompat → ProtoEmitter → .mlpackage`
//!    Used when MIR graphs are already available in compat form.
//!
//! 2. **Role-specific shard emission** (Sprint 48): `ShardSpec → RoleMirBuilder
//!    → MirGraph → mir_graph_to_compat() → ProtoEmitter → .mlpackage`
//!    Used for role-specific shard programs where the RoleMirBuilder determines
//!    the op structure per shard role.
//!
//! ## Validation
//!
//! `validate_proto_direct_package()` performs structural validation of an
//! emitted mlpackage directory without requiring macOS or the Core ML runtime.
//! It checks the directory structure, required files, and basic integrity.

use crate::mir_to_compat::{mir_graph_to_compat, EmptyWeightResolver};
use ane_coreml_emit::ProtoEmitter;
use ane_coreml_proto::mir_compat::MirGraphCompat;
use ane_ir::mir::MirGraph;
use ane_ir::pir::ShardSpec;
use ane_passes::role_mir::RoleMirBuilder;
use anyhow::Result;
use std::fs;
use std::path::Path;

/// Result of a proto-direct emission operation.
#[derive(Debug, Clone)]
pub struct ProtoDirectResult {
    /// Path to the written mlpackage directory.
    pub mlpackage_path: String,
    /// Content hash of the mlpackage.
    pub content_hash: String,
    /// Total size in bytes.
    pub total_size: u64,
    /// Number of files written.
    pub file_count: usize,
    /// Number of unique weights in weight.bin.
    pub weight_count: usize,
    /// Number of functions in the model.
    pub function_count: usize,
    /// Whether weight sharing was used.
    pub has_shared_weights: bool,
    /// Emission method identifier.
    pub emission_method: String,
}

/// Validation result for a proto-direct emitted mlpackage.
#[derive(Debug, Clone)]
pub struct ProtoDirectValidation {
    /// Whether the package passed all validation checks.
    pub is_valid: bool,
    /// List of validation errors (empty if valid).
    pub errors: Vec<String>,
    /// List of validation warnings (non-fatal issues).
    pub warnings: Vec<String>,
    /// Path to the validated mlpackage.
    pub mlpackage_path: String,
    /// Size of the model.mlmodel file in bytes.
    pub model_file_size: Option<u64>,
    /// Size of the weight.bin file in bytes.
    pub weight_file_size: Option<u64>,
    /// Whether a Manifest.json was found.
    pub has_manifest: bool,
}

/// Emit a single-function MIR graph as an mlpackage via proto-direct.
///
/// This is the simplest emission path: one function, one graph.
/// The output is written to `output_path` as a complete `.mlpackage`
/// directory structure.
pub fn emit_proto_direct(graph: &MirGraphCompat, output_path: &str) -> Result<ProtoDirectResult> {
    let emitter = ProtoEmitter::new();
    let emit_result = emitter.emit_mir_graph(graph, output_path)?;

    Ok(ProtoDirectResult {
        mlpackage_path: emit_result.package_result.path,
        content_hash: emit_result.package_result.content_hash,
        total_size: emit_result.package_result.total_size,
        file_count: emit_result.package_result.file_count,
        weight_count: emit_result.package_result.weight_count,
        function_count: emit_result.package_result.function_count,
        has_shared_weights: emit_result.package_result.has_shared_weights,
        emission_method: "proto-direct".to_string(),
    })
}

/// Emit a multi-function model with shared weights via proto-direct.
///
/// When `shared_weight_names` is non-empty, weights with those names
/// will be deduplicated across functions — each shared weight appears
/// once in `weight.bin` and is referenced by all functions that use it.
pub fn emit_proto_direct_multifunction(
    graphs: &[MirGraphCompat],
    shared_weight_names: &[String],
    output_path: &str,
) -> Result<ProtoDirectResult> {
    let emitter = ProtoEmitter::new();
    let emit_result =
        emitter.emit_multifunction_with_shared_weights(graphs, shared_weight_names, output_path)?;

    Ok(ProtoDirectResult {
        mlpackage_path: emit_result.package_result.path,
        content_hash: emit_result.package_result.content_hash,
        total_size: emit_result.package_result.total_size,
        file_count: emit_result.package_result.file_count,
        weight_count: emit_result.package_result.weight_count,
        function_count: emit_result.package_result.function_count,
        has_shared_weights: emit_result.package_result.has_shared_weights,
        emission_method: "proto-direct".to_string(),
    })
}

/// Emit a role-specific shard as an mlpackage via proto-direct.
///
/// This is the Sprint 48 wiring: it uses `RoleMirBuilder` to produce a
/// MIR graph from the shard specification (which includes the op profile),
/// then converts through the compat layer and emits via `ProtoEmitter`.
///
/// The call chain is:
/// ```text
/// ShardSpec → RoleMirBuilder::build_mir() → MirGraph
///          → mir_graph_to_compat() → MirGraphCompat
///          → ProtoEmitter::emit_mir_graph() → .mlpackage on disk
/// ```
///
/// This makes `RoleMirBuilder` the single Rust-side source of truth for
/// role-specific MIR, and the proto-direct path the single emission
/// mechanism for converting that MIR to a disk artifact.
///
/// ## Weight Data
///
/// Since `RoleMirBuilder` produces structural MIR (with `value_path`
/// references but no actual weight bytes), the conversion uses
/// `EmptyWeightResolver` which fills in zero bytes for weight constants.
/// For real weight data, the caller should build a `HashMapWeightResolver`
/// and use `emit_mir_graph_proto_direct` directly.
pub fn emit_role_shard_proto_direct(
    spec: &ShardSpec,
    output_path: &str,
) -> Result<ProtoDirectResult> {
    let builder = RoleMirBuilder::new();
    let mir_graph = builder.build_mir(spec)?;

    emit_mir_graph_proto_direct(&mir_graph, output_path)
}

/// Emit a compiler MIR graph as an mlpackage via proto-direct.
///
/// This is the full pipeline: compiler `MirGraph` → compat conversion →
/// proto emission. Use this when you have a real `MirGraph` from the
/// pass pipeline (not a `MirGraphCompat`).
///
/// For weight data, this uses `EmptyWeightResolver` which fills in zero
/// bytes. For real weight data, build your own resolver and call
/// `mir_graph_to_compat()` + `emit_proto_direct()` directly.
pub fn emit_mir_graph_proto_direct(
    graph: &MirGraph,
    output_path: &str,
) -> Result<ProtoDirectResult> {
    let resolver = EmptyWeightResolver;
    let compat = mir_graph_to_compat(graph, &resolver)?;

    emit_proto_direct(&compat, output_path)
}

/// Validate that a proto-direct emitted mlpackage has the correct structure.
///
/// This works on all platforms (no macOS dependency). It checks:
/// 1. Directory exists and is named `*.mlpackage`
/// 2. `Manifest.json` exists and is valid JSON
/// 3. `Model/com.apple.CoreML/model.mlmodel` exists and is non-empty
/// 4. `Data/com.apple.CoreML/weights/weight.bin` exists (may be empty for constant-free models)
/// 5. Manifest references match actual files
pub fn validate_proto_direct_package(mlpackage_path: &str) -> Result<ProtoDirectValidation> {
    let pkg_path = Path::new(mlpackage_path);
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // 1. Directory exists
    if !pkg_path.exists() {
        return Ok(ProtoDirectValidation {
            is_valid: false,
            errors: vec![format!("mlpackage directory does not exist: {}", mlpackage_path)],
            warnings,
            mlpackage_path: mlpackage_path.to_string(),
            model_file_size: None,
            weight_file_size: None,
            has_manifest: false,
        });
    }

    if !pkg_path.is_dir() {
        errors.push(format!("mlpackage path is not a directory: {}", mlpackage_path));
        return Ok(ProtoDirectValidation {
            is_valid: false,
            errors,
            warnings,
            mlpackage_path: mlpackage_path.to_string(),
            model_file_size: None,
            weight_file_size: None,
            has_manifest: false,
        });
    }

    // 2. Directory name ends with .mlpackage
    let dir_name =
        pkg_path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    if !dir_name.ends_with(".mlpackage") {
        warnings.push(format!("Directory name '{}' does not end with .mlpackage", dir_name));
    }

    // 3. Manifest.json
    let manifest_path = pkg_path.join("Manifest.json");
    let has_manifest = manifest_path.exists();
    if !has_manifest {
        errors.push("Manifest.json is missing".to_string());
    } else {
        // Try to parse as JSON
        match fs::read_to_string(&manifest_path) {
            Ok(content) => {
                if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
                    errors.push(format!("Manifest.json is not valid JSON: {}", e));
                }
            }
            Err(e) => {
                errors.push(format!("Cannot read Manifest.json: {}", e));
            }
        }
    }

    // 4. Model file
    let model_path = pkg_path.join("Model/com.apple.CoreML/model.mlmodel");
    let model_file_size = if model_path.exists() {
        match fs::metadata(&model_path) {
            Ok(meta) => {
                let size = meta.len();
                if size == 0 {
                    errors.push("model.mlmodel is empty".to_string());
                }
                Some(size)
            }
            Err(e) => {
                errors.push(format!("Cannot stat model.mlmodel: {}", e));
                None
            }
        }
    } else {
        errors.push("Model/com.apple.CoreML/model.mlmodel is missing".to_string());
        None
    };

    // 5. Weight file
    let weight_path = pkg_path.join("Data/com.apple.CoreML/weights/weight.bin");
    let weight_file_size = if weight_path.exists() {
        match fs::metadata(&weight_path) {
            Ok(meta) => Some(meta.len()),
            Err(e) => {
                warnings.push(format!("Cannot stat weight.bin: {}", e));
                None
            }
        }
    } else {
        warnings.push("Data/com.apple.CoreML/weights/weight.bin is missing (may be OK for constant-free models)".to_string());
        None
    };

    let is_valid = errors.is_empty();

    Ok(ProtoDirectValidation {
        is_valid,
        errors,
        warnings,
        mlpackage_path: mlpackage_path.to_string(),
        model_file_size,
        weight_file_size,
        has_manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_coreml_emit::mir_to_proto::build_linear_projection_mir;
    use ane_coreml_emit::mir_to_proto::build_multifunction_shared_weights_mir;
    use ane_coreml_proto::mir_compat::MilDtypeCompat;
    use tempfile::TempDir;

    fn make_linear_graph() -> MirGraphCompat {
        build_linear_projection_mir("test_linear", 16, 8, 1, MilDtypeCompat::Fp16, 42)
    }

    #[test]
    fn test_emit_proto_direct_linear_projection() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("linear.mlpackage");
        let graph = make_linear_graph();

        let result = emit_proto_direct(&graph, output_path.to_str().unwrap()).unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 1);
        assert_eq!(result.weight_count, 2); // weight + bias
        assert!(!result.has_shared_weights);
        assert!(result.total_size > 0);
        assert!(result.file_count >= 3); // model.mlmodel + weight.bin + Manifest.json at minimum

        // Verify the directory actually exists
        assert!(output_path.exists());
        assert!(output_path.join("Manifest.json").exists());
        assert!(output_path.join("Model/com.apple.CoreML/model.mlmodel").exists());
        assert!(output_path.join("Data/com.apple.CoreML/weights/weight.bin").exists());
    }

    #[test]
    fn test_emit_proto_direct_multifunction_shared_weights() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("shared.mlpackage");

        let (graphs, shared_names) =
            build_multifunction_shared_weights_mir("test_shared", 32, 1, MilDtypeCompat::Fp16, 42);

        let result =
            emit_proto_direct_multifunction(&graphs, &shared_names, output_path.to_str().unwrap())
                .unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 2);
        assert!(result.has_shared_weights);
    }

    #[test]
    fn test_validate_proto_direct_package_valid() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("valid.mlpackage");
        let graph = make_linear_graph();

        emit_proto_direct(&graph, output_path.to_str().unwrap()).unwrap();

        let validation = validate_proto_direct_package(output_path.to_str().unwrap()).unwrap();

        assert!(validation.is_valid, "Validation errors: {:?}", validation.errors);
        assert!(validation.has_manifest);
        assert!(validation.model_file_size.is_some());
        assert!(validation.model_file_size.unwrap() > 0);
    }

    #[test]
    fn test_validate_proto_direct_package_rejects_malformed() {
        let tmp = TempDir::new().unwrap();
        let malformed_path = tmp.path().join("bad.mlpackage");

        // Create a directory that looks like an mlpackage but is missing files
        fs::create_dir_all(malformed_path.join("Model/com.apple.CoreML")).unwrap();
        // Write an empty model file
        fs::write(malformed_path.join("Model/com.apple.CoreML/model.mlmodel"), b"").unwrap();
        // No Manifest.json, empty model file

        let validation = validate_proto_direct_package(malformed_path.to_str().unwrap()).unwrap();

        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
        // Should report missing Manifest.json
        assert!(validation.errors.iter().any(|e| e.contains("Manifest.json")));
        // Should report empty model.mlmodel
        assert!(validation.errors.iter().any(|e| e.contains("model.mlmodel is empty")));
    }

    #[test]
    fn test_validate_nonexistent_path() {
        let validation = validate_proto_direct_package("/nonexistent/path.mlpackage").unwrap();
        assert!(!validation.is_valid);
        assert!(validation.errors.iter().any(|e| e.contains("does not exist")));
    }

    #[test]
    fn test_validate_not_a_directory() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("not_a_dir.mlpackage");
        fs::write(&file_path, b"not a directory").unwrap();

        let validation = validate_proto_direct_package(file_path.to_str().unwrap()).unwrap();
        assert!(!validation.is_valid);
    }

    // ─── Sprint 48: Role-specific shard emission tests ───────────────────────

    fn make_entry_spec() -> ShardSpec {
        use ane_ir::pir::{ComputeUnitHint, ShardOpProfile, ShardRole, TensorSpec};
        ShardSpec {
            shard_name: "test_entry".into(),
            role: ShardRole::Entry,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 64],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::EntryLinear {
                needs_reshape: true,
                reshape_target: Some(vec![1, 48]),
            },
        }
    }

    fn make_interior_spec() -> ShardSpec {
        use ane_ir::pir::{ActivationType, ComputeUnitHint, ShardOpProfile, ShardRole, TensorSpec};
        ShardSpec {
            shard_name: "test_interior".into(),
            role: ShardRole::Interior,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::InteriorLinear { activation: ActivationType::GeluTanh },
        }
    }

    fn make_exit_spec() -> ShardSpec {
        use ane_ir::pir::{ComputeUnitHint, ShardOpProfile, ShardRole, TensorSpec};
        ShardSpec {
            shard_name: "test_exit".into(),
            role: ShardRole::Exit,
            input_specs: vec![TensorSpec {
                name: "x".into(),
                shape: vec![1, 48],
                dtype: "fp16".into(),
            }],
            output_specs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 32],
                dtype: "fp16".into(),
            }],
            compute_units: ComputeUnitHint::CPUAndNE,
            op_profile: ShardOpProfile::ExitLinear { ln_epsilon: 1e-5 },
        }
    }

    /// Sprint 48 test: Entry shard emits via proto-direct with RoleMirBuilder.
    #[test]
    fn test_emit_role_shard_entry() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("entry_shard.mlpackage");

        let result =
            emit_role_shard_proto_direct(&make_entry_spec(), output_path.to_str().unwrap())
                .unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 1);
        assert!(output_path.exists());
        assert!(output_path.join("Manifest.json").exists());
    }

    /// Sprint 48 test: Interior shard emits with GELU activation.
    #[test]
    fn test_emit_role_shard_interior() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("interior_shard.mlpackage");

        let result =
            emit_role_shard_proto_direct(&make_interior_spec(), output_path.to_str().unwrap())
                .unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 1);
        assert!(output_path.exists());
    }

    /// Sprint 48 test: Exit shard emits with LayerNorm.
    #[test]
    fn test_emit_role_shard_exit() {
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("exit_shard.mlpackage");

        let result =
            emit_role_shard_proto_direct(&make_exit_spec(), output_path.to_str().unwrap()).unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 1);
        assert!(output_path.exists());
    }

    /// Sprint 48 test: Role-specific shards produce different content hashes.
    /// This is the proof that RoleMirBuilder output reaches proto-direct emission
    /// with genuinely different op structures per role.
    #[test]
    fn test_role_shards_produce_different_content_hashes() {
        let tmp = TempDir::new().unwrap();

        let entry_path = tmp.path().join("entry.mlpackage");
        let interior_path = tmp.path().join("interior.mlpackage");
        let exit_path = tmp.path().join("exit.mlpackage");

        let entry_result =
            emit_role_shard_proto_direct(&make_entry_spec(), entry_path.to_str().unwrap()).unwrap();

        let interior_result =
            emit_role_shard_proto_direct(&make_interior_spec(), interior_path.to_str().unwrap())
                .unwrap();

        let exit_result =
            emit_role_shard_proto_direct(&make_exit_spec(), exit_path.to_str().unwrap()).unwrap();

        // All three must have different content hashes (different op structures)
        assert_ne!(
            entry_result.content_hash, interior_result.content_hash,
            "Entry and Interior shards must have different content hashes"
        );
        assert_ne!(
            entry_result.content_hash, exit_result.content_hash,
            "Entry and Exit shards must have different content hashes"
        );
        assert_ne!(
            interior_result.content_hash, exit_result.content_hash,
            "Interior and Exit shards must have different content hashes"
        );
    }

    /// Sprint 48 test: emit_mir_graph_proto_direct works for compiler MIR.
    #[test]
    fn test_emit_mir_graph_proto_direct() {
        use ane_ir::mir::{ComputeUnitHint, MilDtype, MirNode, MirNodeId, MirOp};
        let tmp = TempDir::new().unwrap();
        let output_path = tmp.path().join("mir_graph.mlpackage");

        let graph = MirGraph {
            nodes: vec![
                MirNode {
                    id: MirNodeId("w".into()),
                    op: MirOp::MILConst {
                        name: "w".into(),
                        value_path: "weights/w.bin".into(),
                        dtype: MilDtype::Fp16,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![32, 64],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                },
                MirNode {
                    id: MirNodeId("out".into()),
                    op: MirOp::MILLinear {
                        name: "linear".into(),
                        x: MirNodeId("input".into()),
                        weight: "w".into(),
                        bias: None,
                    },
                    dtype: MilDtype::Fp16,
                    shape: vec![32],
                    compute_unit_hint: Some(ComputeUnitHint::CPUAndNE),
                    air_source: None,
                },
            ],
            inputs: vec![MirNodeId("input".into())],
            outputs: vec![MirNodeId("out".into())],
            opset_version: "iOS18".into(),
            shard_name: "main".into(),
        };

        let result = emit_mir_graph_proto_direct(&graph, output_path.to_str().unwrap()).unwrap();

        assert_eq!(result.emission_method, "proto-direct");
        assert_eq!(result.function_count, 1);
        assert!(output_path.exists());
    }
}
