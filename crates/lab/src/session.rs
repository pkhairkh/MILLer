//! Lab session orchestration.
//!
//! This module defines the session types and helper functions for running
//! lab and lab-loop subcommands. The logic was extracted from the CLI crate
//! to establish a clean boundary between command-line parsing and lab
//! orchestration.
//!
//! Key types:
//! - `LabSession` — configuration for a single lab run
//! - `LabLoopSession` — configuration for a lab-loop run (with knowledge ingestion)
//! - `LabResult` — structured output of a lab session
//! - `StoreKnowledgeQuery` — adapter from `KnowledgeStore` to `PassKnowledgeQuery`

use std::collections::HashMap;
use std::fmt::Write;
use std::path::PathBuf;

use crate::baseline::BaselineComputer;
use crate::drift::DriftDetector;
use crate::harness::{
    CompileStepResult, EnvironmentSummary, GeneratorProvenance, InspectionStepResult,
    LabRunBuilder, VerificationScope,
};
use crate::run_dir::{generate_run_id, layout, LabRunWriter};
use ane_bridge::subprocess::PythonBridge;
use ane_ir::linear_slice::{
    lower_linear_projection_to_mir, sir_from_linear_projection, FamilyPayload,
};
use ane_ir::task_spec::load_synthetic_task;
use sha2::Digest;

// ---------------------------------------------------------------------------
// Session configuration structs
// ---------------------------------------------------------------------------

/// Configuration for a single lab run.
///
/// A lab run compiles a task spec, performs host-side inspection (optional),
/// computes a baseline, drift report, and knowledge update, and writes all
/// artifacts to the output directory.
pub struct LabSession {
    /// Path to the task specification TOML file.
    pub input: String,
    /// Output directory for compiled packages and artifacts.
    pub output: String,
    /// Path to the Python bridge script.
    pub bridge_script: String,
    /// Path to the Python interpreter.
    pub python_path: String,
    /// Whether to perform host-side inspection.
    pub do_inspect: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Provenance info if this run used a generated task (format: "family,seed,version").
    pub generated_from: Option<String>,
}

/// Configuration for a lab-loop run.
///
/// A lab-loop run is the same as a lab run, but additionally ingests
/// observations into the knowledge store, closing the host-side evidence
/// loop.
pub struct LabLoopSession {
    /// Path to the task specification TOML file.
    pub input: String,
    /// Output directory for compiled packages and artifacts.
    pub output: String,
    /// Path to the Python bridge script.
    pub bridge_script: String,
    /// Path to the Python interpreter.
    pub python_path: String,
    /// Path to the knowledge store directory.
    pub knowledge_dir: String,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Provenance info if this run used a generated task (format: "family,seed,version").
    pub generated_from: Option<String>,
}

/// Structured output of a lab session.
pub struct LabResult {
    /// Whether the lab session completed successfully.
    pub success: bool,
    /// Name of the task that was run.
    pub task_name: String,
    /// Path to the output directory.
    pub output_dir: String,
    /// Path to the manifest file, if written.
    pub manifest_path: Option<String>,
    /// Error message, if the session failed.
    pub error_message: Option<String>,
}

// ---------------------------------------------------------------------------
// Helper functions (moved from CLI)
// ---------------------------------------------------------------------------

/// Compute a deterministic SHA-256 hash for a task spec.
///
/// Uses the canonical identity string from `TaskOp` as the single source
/// of truth for all op-specific fields, eliminating per-variant match arms.
pub fn compute_task_hash(spec: &ane_ir::task_spec::SyntheticTaskSpec) -> String {
    let mut hash_input = String::new();
    // SAFETY: write! to String is infallible (std::fmt::Write for String never errors).
    write!(hash_input, "family={}", spec.family).expect("write to String cannot fail");
    write!(hash_input, ";name={}", spec.name).expect("write to String cannot fail");
    write!(hash_input, ";{}", spec.op.identity_string()).expect("write to String cannot fail");

    let digest = sha2::Sha256::digest(hash_input.as_bytes());
    let hex: String = digest.iter().fold(String::new(), |mut output, b| {
        write!(output, "{:02x}", b).expect("write to String cannot fail");
        output
    });
    format!("sha256:{}", hex)
}

/// Build an artifact manifest from the task spec and bridge result.
pub fn build_artifact_manifest(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
    compiler_version: &str,
) -> serde_json::Value {
    use ane_artifacts::manifest::{ArtifactManifest, FunctionDescriptor, PackageEntry, TensorSpec};

    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, batch_size, dtype) = spec.op.primary_dims();

    let functions: Vec<FunctionDescriptor> = if !bridge_result.function_descriptors.is_empty() {
        bridge_result
            .function_descriptors
            .iter()
            .map(|fd| {
                let inputs: Vec<TensorSpec> = fd
                    .inputs
                    .iter()
                    .map(|inp| TensorSpec {
                        name: inp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: inp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: inp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                let outputs: Vec<TensorSpec> = fd
                    .outputs
                    .iter()
                    .map(|outp| TensorSpec {
                        name: outp
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        shape: outp
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect()
                            })
                            .unwrap_or_default(),
                        dtype: outp
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("fp16")
                            .to_string(),
                    })
                    .collect();
                FunctionDescriptor {
                    name: fd.name.clone(),
                    inputs,
                    outputs,
                    stateful: fd.stateful,
                    emission_status: "emitted".to_string(),
                    mir_ops: vec![],
                }
            })
            .collect()
    } else {
        vec![FunctionDescriptor {
            name: "main".to_string(),
            inputs: vec![TensorSpec {
                name: "x".to_string(),
                shape: vec![batch_size, input_dim],
                dtype: dtype.clone(),
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![batch_size, output_dim],
                dtype: dtype.clone(),
            }],
            stateful: false,
            emission_status: if bridge_result.status == "success" {
                "emitted".to_string()
            } else {
                "seam_only".to_string()
            },
            mir_ops: vec![],
        }]
    };

    let packages: Vec<PackageEntry> = if bridge_result.status == "success" {
        vec![PackageEntry {
            name: spec.name.clone(),
            role: "synthetic_microkernel".to_string(),
            path: bridge_result.output_path.clone(),
            content_hash: bridge_result.content_hash.clone(),
            size_bytes: 0,
            functions,
        }]
    } else {
        vec![]
    };

    let manifest = ArtifactManifest {
        version: "0.3.0".to_string(),
        model_id: spec.name.clone(),
        task_hash: task_hash.to_string(),
        created_at: timestamp,
        packages,
        state_declarations: vec![],
        handoffs: vec![],
        compiler_version: compiler_version.to_string(),
        implementation_status: "host_compiled".to_string(),
        verification_scope: "host_compile_only".to_string(),
        environment_limitations: vec![
            "no_apple_hardware".to_string(),
            "ane_placement_not_verified".to_string(),
            "no_on_device_predict".to_string(),
        ],
    };

    serde_json::to_value(&manifest)
        .unwrap_or_else(|_| serde_json::json!({"error": "manifest serialization failed"}))
}

/// Build a backend-knowledge update from the compilation result.
///
/// Includes the deterministic task hash for identity tracking.
pub fn build_knowledge_update(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
) -> serde_json::Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();

    serde_json::json!({
        "version": 2,
        "timestamp": timestamp,
        "source": "vertical_slice_compile",
        "task_hash": task_hash,
        "task_name": spec.name,
        "task_family": spec.family,
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.matmul",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": format!("LinearProjection {}x{}", input_dim, output_dim),
            },
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.add",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": "bias addition after matmul",
            },
        ],
        "compilation_result": {
            "status": bridge_result.status,
            "mlpackage_produced": bridge_result.output_path.is_some(),
            "content_hash": bridge_result.content_hash,
        },
        "residuals": [
            "Device-specific ANE placement not verified (requires Apple hardware)",
            "Numerical drift not measured (requires Apple hardware for predict())",
            "Fallback suspicion not assessed (requires compute plan on Apple hardware)",
        ],
    })
}

/// Build a backend-knowledge update that includes drift evidence.
///
/// This extends the standard knowledge update with:
/// - baseline provenance (FP32 reference was computed, linked by task_hash)
/// - drift observation (if available, with scope/confidence fields)
/// - honest residual about what could not be measured
pub fn build_knowledge_update_with_drift(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    bridge_result: &ane_bridge::subprocess::BridgeResult,
    task_hash: &str,
    baseline: &crate::baseline::BaselineResult,
    drift: &crate::drift::DriftReport,
) -> serde_json::Value {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let (input_dim, output_dim, _batch_size, _dtype) = spec.op.primary_dims();

    let drift_observation = if drift.is_computed() {
        serde_json::json!({
            "knowledge_type": "PrecisionHazard",
            "op_pattern": "linear_projection_fp16_vs_fp32",
            "max_absolute_error": drift.max_absolute_error,
            "mean_absolute_error": drift.mean_absolute_error,
            "rmse": drift.rmse,
            "cosine_distance": drift.cosine_distance,
            "relative_error_p99": drift.relative_error_p99,
            "has_drift": drift.has_drift,
            "confidence": 0.3,
            "evidence_source": "SyntheticRun",
            "evidence_count": 1,
            "scope": {
                "device_classes": ["unknown"],
                "os_versions": ["unknown"],
                "opset_versions": ["iOS18"],
            },
            "context": format!("FP16 vs FP32 drift for LinearProjection {}x{}", input_dim, output_dim),
        })
    } else {
        serde_json::json!({
            "knowledge_type": "PrecisionHazard",
            "op_pattern": "linear_projection_fp16_vs_fp32",
            "computation_status": "unavailable",
            "reason": match &drift.computation_status {
                crate::drift::DriftComputationStatus::Unavailable { reason } => reason.clone(),
                _ => "unknown".to_string(),
            },
            "confidence": 0.0,
            "evidence_source": "None",
            "evidence_count": 0,
            "scope": {
                "device_classes": [],
                "os_versions": [],
                "opset_versions": ["iOS18"],
            },
            "note": "Drift could not be computed — requires predict() output from Apple hardware",
        })
    };

    serde_json::json!({
        "version": 3,
        "timestamp": timestamp,
        "source": "lab_run_with_drift",
        "task_hash": task_hash,
        "task_name": spec.name,
        "task_family": spec.family,
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.matmul",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": format!("LinearProjection {}x{}", input_dim, output_dim),
            },
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.add",
                "ane_legal": bridge_result.status == "success",
                "confidence": if bridge_result.status == "success" { 0.3 } else { 0.7 },
                "evidence_source": "SyntheticRun",
                "evidence_count": 1,
                "scope": {
                    "device_classes": ["unknown"],
                    "os_versions": ["unknown"],
                    "opset_versions": ["iOS18"],
                },
                "context": "bias addition after matmul",
            },
            drift_observation,
        ],
        "baseline_provenance": {
            "baseline_schema_version": baseline.baseline_schema_version,
            "task_id": baseline.task_id,
            "task_hash": baseline.task_hash,
            "seed": baseline.seed,
            "precision": baseline.precision,
            "output_element_count": baseline.output_tensor.len(),
            "compute_time_ms": baseline.compute_time_ms,
        },
        "drift_evidence": {
            "drift_report_schema_version": drift.drift_report_schema_version,
            "computation_status": match &drift.computation_status {
                crate::drift::DriftComputationStatus::Computed => "computed",
                crate::drift::DriftComputationStatus::Unavailable { .. } => "unavailable",
                crate::drift::DriftComputationStatus::LengthMismatch { .. } => "length_mismatch",
                crate::drift::DriftComputationStatus::EmptyInput => "empty_input",
            },
            "has_drift": drift.has_drift,
            "max_absolute_error": drift.max_absolute_error,
            "mean_absolute_error": drift.mean_absolute_error,
            "rmse": drift.rmse,
            "scope_note": drift.scope_note,
        },
        "compilation_result": {
            "status": bridge_result.status,
            "mlpackage_produced": bridge_result.output_path.is_some(),
            "content_hash": bridge_result.content_hash,
        },
        "residuals": [
            "Device-specific ANE placement not verified (requires Apple hardware)",
            "Numerical drift not fully measured — baseline computed but actual model output requires Apple hardware for predict()",
            "Fallback suspicion not assessed (requires compute plan on Apple hardware)",
        ],
    })
}

// ---------------------------------------------------------------------------
// StoreKnowledgeQuery (moved from CLI)
// ---------------------------------------------------------------------------

/// Adapter from `KnowledgeStore` to `PassKnowledgeQuery` trait.
///
/// This struct wraps a reference to a `KnowledgeStore` and implements the
/// `PassKnowledgeQuery` trait used by the pass pipeline to query legality,
/// risk, precision hazard, and compute plan placement knowledge.
pub struct StoreKnowledgeQuery<'a> {
    store: &'a ane_knowledge::store::KnowledgeStore,
}

impl<'a> StoreKnowledgeQuery<'a> {
    /// Create a new `StoreKnowledgeQuery` wrapping the given store.
    pub fn new(store: &'a ane_knowledge::store::KnowledgeStore) -> Self {
        Self { store }
    }
}

impl<'a> ane_passes::knowledge_query::PassKnowledgeQuery for StoreKnowledgeQuery<'a> {
    fn query_legality(
        &self,
        op_pattern: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::LegalityInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let query =
            KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule).with_min_confidence(0.1);

        let results = self.store.query(&query).ok()?;

        for unit in results {
            if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                if pattern.split('|').any(|p| p.trim() == op_pattern) {
                    let ane_legal =
                        unit.payload.get("ane_legal").and_then(|v| v.as_bool()).unwrap_or(false);
                    return Some(ane_passes::knowledge_query::LegalityInfo {
                        ane_legal,
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                    });
                }
            }
        }
        None
    }

    fn query_risk(
        &self,
        op_pattern: &str,
        scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::RiskInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let survival_query = KnowledgeQuery::new()
            .with_type(KnowledgeType::SurvivalMatrixEntry)
            .with_min_confidence(0.1);

        if let Ok(survival_results) = self.store.query(&survival_query) {
            for unit in survival_results {
                if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                    if pattern.split('|').any(|p| p.trim() == op_pattern) {
                        let fallback_risk = unit
                            .payload
                            .get("fallback_risk")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.1) as f32;
                        let drift_risk =
                            unit.payload.get("drift_risk").and_then(|v| v.as_f64()).unwrap_or(0.05)
                                as f32;
                        return Some(ane_passes::knowledge_query::RiskInfo {
                            fallback_risk,
                            drift_risk,
                            confidence: unit.confidence,
                            evidence_count: unit.evidence_count,
                            source_id: Some(unit.id.clone()),
                        });
                    }
                }
            }
        }

        let legality = self.query_legality(op_pattern, scope)?;
        let fallback_risk =
            if legality.ane_legal { 1.0 - legality.confidence } else { legality.confidence };
        let drift_risk = fallback_risk * 0.5;

        Some(ane_passes::knowledge_query::RiskInfo {
            fallback_risk: fallback_risk.min(1.0),
            drift_risk: drift_risk.min(1.0),
            confidence: legality.confidence,
            evidence_count: legality.evidence_count,
            source_id: legality.source_id,
        })
    }

    fn query_precision_hazard(
        &self,
        op_pattern: &str,
        current_dtype: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::PrecisionHazardInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let hazard_query = KnowledgeQuery::new()
            .with_type(KnowledgeType::PrecisionHazard)
            .with_min_confidence(0.1);

        let results = self.store.query(&hazard_query).ok()?;

        for unit in results {
            let op_match = unit
                .payload
                .get("op")
                .and_then(|v| v.as_str())
                .map(|op| op == op_pattern)
                .unwrap_or(false);

            let pattern_match = unit
                .payload
                .get("op_pattern")
                .and_then(|v| v.as_str())
                .map(|p| p.split('|').any(|s| s.trim() == op_pattern))
                .unwrap_or(false);

            if op_match || pattern_match {
                let quality_impact =
                    unit.payload.get("quality_impact").and_then(|v| v.as_str()).unwrap_or("none");

                let applies = match quality_impact {
                    "high" | "medium" => current_dtype == "fp16",
                    _ => false,
                };

                if applies {
                    return Some(ane_passes::knowledge_query::PrecisionHazardInfo {
                        op_pattern: op_pattern.to_string(),
                        hazardous_dtype: "fp16".to_string(),
                        recommended_dtype: "fp32".to_string(),
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                        description: unit
                            .payload
                            .get("note")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                    });
                }
            }
        }
        None
    }

    fn query_compute_plan_placement(
        &self,
        op_pattern: &str,
        _scope: Option<&ane_ir::kir::KnowledgeScope>,
    ) -> Option<ane_passes::knowledge_query::ComputePlanPlacementInfo> {
        use ane_ir::kir::KnowledgeType;
        use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};

        let query = KnowledgeQuery::new()
            .with_type(KnowledgeType::SurvivalMatrixEntry)
            .with_min_confidence(0.1);

        let results = self.store.query(&query).ok()?;

        for unit in results {
            if let Some(pattern) = unit.payload.get("op_pattern").and_then(|v| v.as_str()) {
                if pattern.split('|').any(|p| p.trim() == op_pattern) {
                    let ane_placed =
                        unit.payload.get("ane_placed").and_then(|v| v.as_bool()).unwrap_or(false);
                    let preferred_device = unit
                        .payload
                        .get("preferred_device")
                        .and_then(|v| v.as_str())
                        .unwrap_or("CPU")
                        .to_string();
                    return Some(ane_passes::knowledge_query::ComputePlanPlacementInfo {
                        op_pattern: op_pattern.to_string(),
                        ane_placed,
                        preferred_device,
                        confidence: unit.confidence,
                        evidence_count: unit.evidence_count,
                        source_id: Some(unit.id.clone()),
                    });
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Ingest knowledge observations (moved from CLI)
// ---------------------------------------------------------------------------

/// Ingest knowledge observations from a knowledge update JSON into the knowledge store.
///
/// This is the key function that closes the host-side evidence loop: it converts
/// observations from the JSON format produced by `build_knowledge_update_with_drift`
/// into proper `KnowledgeUnit` structs and ingests them via `UpdatePipeline`.
///
/// Returns the number of observations successfully ingested.
pub fn ingest_knowledge_observations(
    store: &mut ane_knowledge::store::KnowledgeStore,
    knowledge_update: &serde_json::Value,
    task_hash: &str,
) -> Result<usize, String> {
    use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
    use ane_knowledge::update::UpdatePipeline;

    let observations = knowledge_update
        .get("observations")
        .and_then(|v| v.as_array())
        .ok_or("No observations found in knowledge update")?;

    let mut pipeline = UpdatePipeline::new(store);
    let mut ingested = 0;

    for obs in observations {
        let knowledge_type_str =
            obs.get("knowledge_type").and_then(|v| v.as_str()).unwrap_or("LegalityRule");

        let knowledge_type = match knowledge_type_str {
            "LegalityRule" => KnowledgeType::LegalityRule,
            "PrecisionHazard" => KnowledgeType::PrecisionHazard,
            "SurvivalMatrixEntry" => KnowledgeType::SurvivalMatrixEntry,
            "FallbackSignature" => KnowledgeType::FallbackSignature,
            "MotifCatalog" => KnowledgeType::MotifCatalog,
            "ShardTemplateKnowledge" => KnowledgeType::ShardTemplateKnowledge,
            "DeviceFingerprint" => KnowledgeType::DeviceFingerprint,
            "StateTopologyOutcome" => KnowledgeType::StateTopologyOutcome,
            "SyntheticTransferAnnotation" => KnowledgeType::SyntheticTransferAnnotation,
            _ => KnowledgeType::LegalityRule,
        };

        let confidence = obs.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

        let evidence_source_str =
            obs.get("evidence_source").and_then(|v| v.as_str()).unwrap_or("SyntheticRun");

        let evidence_source = match evidence_source_str {
            "SyntheticRun" => EvidenceSource::SyntheticRun,
            "RealModelRun" => EvidenceSource::RealModelRun,
            "CompileFailure" => EvidenceSource::CompileFailure,
            "LoadFailure" => EvidenceSource::LoadFailure,
            "RuntimeAnomaly" => EvidenceSource::RuntimeAnomaly,
            "ManualEntry" => EvidenceSource::ManualEntry,
            "CrossValidated" => EvidenceSource::CrossValidated,
            _ => EvidenceSource::SyntheticRun,
        };

        let evidence_count =
            obs.get("evidence_count").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let obs_id = format!("obs_{}_{}", task_hash.replace(":", "_"), ingested);

        let scope_json = obs.get("scope").cloned().unwrap_or(serde_json::json!({
            "device_classes": ["unknown"],
            "os_versions": ["unknown"],
            "opset_versions": ["iOS18"],
        }));

        let scope = KnowledgeScope {
            device_classes: scope_json
                .get("device_classes")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            os_versions: scope_json
                .get("os_versions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            opset_versions: scope_json
                .get("opset_versions")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
        };

        let mut payload = HashMap::new();
        if let Some(obj) = obs.as_object() {
            for (key, value) in obj {
                if !matches!(
                    key.as_str(),
                    "knowledge_type"
                        | "confidence"
                        | "evidence_source"
                        | "evidence_count"
                        | "scope"
                ) {
                    payload.insert(key.clone(), value.clone());
                }
            }
        }

        let unit = KnowledgeUnit {
            id: obs_id,
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type,
            confidence,
            evidence_source,
            evidence_count,
            scope,
            conflict_priority: 0,
            payload,
        };

        match pipeline.ingest(unit) {
            Ok(()) => ingested += 1,
            Err(e) => eprintln!("  Warning: failed to ingest observation: {}", e),
        }
    }

    Ok(ingested)
}

// ---------------------------------------------------------------------------
// Baseline computation helper
// ---------------------------------------------------------------------------

/// Compute the FP32 baseline for a given task spec, dispatching on the op type.
fn compute_baseline(
    spec: &ane_ir::task_spec::SyntheticTaskSpec,
    seed: u64,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
) -> Result<crate::baseline::BaselineResult, String> {
    let baseline_computer = BaselineComputer::new(seed);
    match &spec.op {
        ane_ir::task_spec::TaskOp::MlpBlock {
            input_dim,
            hidden_dim,
            output_dim,
            activation,
            batch_size,
            ..
        } => baseline_computer
            .compute_mlp_block(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                activation,
                *batch_size,
            )
            .map_err(|e| format!("MLP baseline computation failed: {}", e)),
        ane_ir::task_spec::TaskOp::DecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Decode-step baseline computation failed: {}", e)),
        ane_ir::task_spec::TaskOp::Attention {
            embed_dim,
            num_heads,
            head_dim,
            seq_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_attention(&spec.name, *embed_dim, *num_heads, *head_dim, *seq_len, *batch_size)
            .map_err(|e| format!("Attention baseline computation failed: {}", e)),
        ane_ir::task_spec::TaskOp::LutProjection {
            vocab_size,
            embed_dim,
            num_groups,
            lut_bitwidth,
            batch_size,
            ..
        } => baseline_computer
            .compute_lut_projection(
                &spec.name,
                *vocab_size,
                *embed_dim,
                *num_groups,
                *lut_bitwidth,
                *batch_size,
            )
            .map_err(|e| format!("LUT baseline computation failed: {}", e)),
        ane_ir::task_spec::TaskOp::ShardedLinearPipeline {
            input_dim,
            hidden_dim,
            output_dim,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_linear_pipeline(
                &spec.name,
                *input_dim,
                *hidden_dim,
                *output_dim,
                *batch_size,
            )
            .map_err(|e| format!("Sharded linear pipeline baseline computation failed: {}", e)),
        ane_ir::task_spec::TaskOp::ShardedDecodeStep {
            embed_dim,
            num_heads,
            head_dim,
            kv_len,
            batch_size,
            ..
        } => baseline_computer
            .compute_sharded_decode_step(
                &spec.name,
                *embed_dim,
                *num_heads,
                *head_dim,
                *kv_len,
                *batch_size,
            )
            .map_err(|e| format!("Sharded decode-step baseline computation failed: {}", e)),
        _ => baseline_computer
            .compute_linear_projection(&spec.name, input_dim, output_dim, batch_size)
            .map_err(|e| format!("Baseline computation failed: {}", e)),
    }
}

// ---------------------------------------------------------------------------
// LabSession::run
// ---------------------------------------------------------------------------

impl LabSession {
    /// Run the lab session.
    ///
    /// Steps:
    /// 1. Load task spec
    /// 2. Compile via bridge
    /// 3. Write artifacts in lab run directory layout
    /// 4. Perform host-side inspection (if not skipped)
    /// 5. Compute FP32 baseline reference output
    /// 6. Compute drift between baseline and actual output (if available)
    /// 7. Write knowledge update with drift evidence
    /// 8. Build and write the LabRun record
    pub fn run(&self) -> Result<LabResult, String> {
        println!("=== MILLer — Lab Run ===\n");

        // Step 1: Load task spec
        println!("[1/8] Loading task spec: {}", self.input);
        let spec = load_synthetic_task(&self.input)?;
        let task_hash = compute_task_hash(&spec);
        println!("  Task: {} (family: {})", spec.name, spec.family);
        println!("  Task hash: {}", task_hash);

        if spec.op.is_sharded() {
            return Err(format!("Use 'compile-sharded' command for {} tasks", spec.op.family_id()));
        }
        let (input_dim, output_dim, batch_size, _dtype) = spec.op.primary_dims();

        // Step 2: Build IR and compile
        println!("[2/8] Compiling...");
        let sir = sir_from_linear_projection(&spec)?;
        println!("  SIR: {} nodes", sir.nodes.len());

        let shard_name = format!("{}_shard_0", spec.name);
        let mir = lower_linear_projection_to_mir(&spec, &shard_name)?;

        let output_path = PathBuf::from(&self.output);
        let mlpackage_output = output_path.join(layout::MLPACKAGE_DIR);
        let payload = FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?;
        let payload_json = serde_json::to_value(&payload)
            .map_err(|e| format!("Payload serialization failed: {}", e))?;

        let bridge = PythonBridge::new(&self.python_path, &self.bridge_script);
        let result = bridge
            .execute_raw_payload(&payload_json)
            .map_err(|e| format!("Bridge execution failed: {}", e))?;

        let compile_step = CompileStepResult {
            success: result.status == "success",
            error: result.error_message.clone(),
            output_path: result.output_path.clone(),
            content_hash: result.content_hash.clone(),
            file_count: if result.package_files.is_empty() {
                None
            } else {
                Some(result.package_files.len())
            },
            coremltools_version: result.coremltools_version.clone(),
        };

        if compile_step.success {
            println!("  Compilation: SUCCESS");
            if let Some(ref hash) = compile_step.content_hash {
                println!("  Content hash: {}", hash);
            }
        } else {
            println!("  Compilation: FAILED");
            if let Some(ref err) = compile_step.error {
                println!("  Error: {}", err);
            }
        }

        // Step 3: Create lab run directory and write artifacts
        println!("[3/8] Writing lab run artifacts...");
        let run_id = generate_run_id(&task_hash);
        let writer = LabRunWriter::new(&output_path);
        let run_dir = writer
            .create_run_directory(&run_id)
            .map_err(|e| format!("Failed to create run directory: {}", e))?;
        println!("  Run directory: {}", run_dir.display());

        // Write manifest
        let compiler_version = env!("CARGO_PKG_VERSION");
        let manifest = build_artifact_manifest(&spec, &result, &task_hash, compiler_version);
        writer
            .write_manifest(&run_dir, &manifest)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;

        // Write MIR
        let mir_json =
            serde_json::to_value(&mir).map_err(|e| format!("MIR serialization failed: {}", e))?;
        writer.write_mir(&run_dir, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;

        // Step 4: Host-side inspection
        let inspect_step = if self.do_inspect && compile_step.success {
            println!("[4/8] Performing host-side inspection...");
            let inspector =
                crate::host_inspect::HostInspector::new(&self.python_path, &self.bridge_script);
            let mlpackage_path = result.output_path.as_deref().unwrap_or("");
            let inspect_result = inspector.inspect(mlpackage_path);

            println!("  Package present: {}", inspect_result.package_present);
            println!("  Manifest readable: {}", inspect_result.manifest_readable);
            println!("  Model loadable: {}", inspect_result.model_loadable);
            if !inspect_result.model_loadable {
                if let Some(ref reason) = inspect_result.model_load_failure_reason {
                    println!("  Load failure: {}", reason);
                }
            }
            if !inspect_result.warnings.is_empty() {
                println!("  Warnings:");
                for w in &inspect_result.warnings {
                    println!("    - {}", w);
                }
            }

            let inspect_json = serde_json::to_value(&inspect_result)
                .map_err(|e| format!("Inspection serialization failed: {}", e))?;
            writer
                .write_inspection(&run_dir, &inspect_json)
                .map_err(|e| format!("Failed to write inspection: {}", e))?;

            inspect_result
        } else {
            if !self.do_inspect {
                println!("[4/8] Host-side inspection: SKIPPED");
            } else {
                println!("[4/8] Host-side inspection: SKIPPED (compilation failed)");
            }
            InspectionStepResult {
                package_present: false,
                manifest_readable: false,
                model_loadable: false,
                model_load_failure_reason: Some("Inspection not performed".to_string()),
                function_count: None,
                input_specs: vec![],
                output_specs: vec![],
                warnings: vec!["Host-side inspection was not performed".to_string()],
                structure_inspection_available: None,
                structure_inspection_failure_reason: Some("Inspection not performed".to_string()),
                structure_op_names: vec![],
                structure_op_count: None,
                structure_function_count: None,
                structure_state_declarations: vec![],
                op_fidelity_score: None,
                missing_ops: vec![],
                extra_ops: vec![],
                inspection_method: "none".to_string(),
            }
        };

        // Step 5: Compute FP32 baseline reference
        println!("[5/8] Computing FP32 baseline reference...");
        let mut baseline_result =
            compute_baseline(&spec, self.seed, input_dim, output_dim, batch_size)?;
        baseline_result.task_hash = Some(task_hash.clone());
        println!(
            "  Baseline: {} output elements, computed in {:.3}ms",
            baseline_result.output_tensor.len(),
            baseline_result.compute_time_ms
        );

        let baseline_json = serde_json::to_value(&baseline_result)
            .map_err(|e| format!("Baseline serialization failed: {}", e))?;
        writer
            .write_baseline(&run_dir, &baseline_json)
            .map_err(|e| format!("Failed to write baseline: {}", e))?;
        println!("  Baseline: {}", run_dir.join(layout::BASELINE_JSON).display());

        // Step 6: Compute drift
        println!("[6/8] Computing drift metrics...");
        let drift_report = if compile_step.success {
            let unavailable_report = DriftDetector::unavailable(
                "predict() requires Apple hardware with Core ML runtime",
            );
            println!("  Drift: UNAVAILABLE (no on-device predict output)");
            unavailable_report
        } else {
            let unavailable_report =
                DriftDetector::unavailable("compilation failed — no model output to compare");
            println!("  Drift: UNAVAILABLE (compilation failed)");
            unavailable_report
        };

        let drift_json = serde_json::to_value(&drift_report)
            .map_err(|e| format!("Drift serialization failed: {}", e))?;
        writer
            .write_drift(&run_dir, &drift_json)
            .map_err(|e| format!("Failed to write drift report: {}", e))?;
        println!("  Drift report: {}", run_dir.join(layout::DRIFT_JSON).display());

        // Step 7: Write knowledge update with drift evidence
        println!("[7/8] Writing knowledge update...");
        let knowledge_update = build_knowledge_update_with_drift(
            &spec,
            &result,
            &task_hash,
            &baseline_result,
            &drift_report,
        );
        writer
            .write_knowledge_update(&run_dir, &spec.name, &knowledge_update)
            .map_err(|e| format!("Failed to write knowledge update: {}", e))?;
        println!("  Knowledge: {}", run_dir.join(layout::KNOWLEDGE_DIR).display());

        // Step 8: Build and write LabRun record
        println!("[8/8] Writing lab run record...");
        let env = EnvironmentSummary::detect(1);
        let verification_scope = VerificationScope::HostOnlyInspection;

        let mut builder =
            LabRunBuilder::new(run_id, task_hash, spec.name.clone(), verification_scope, env)
                .compile_result(compile_step)
                .inspect_result(inspect_step)
                .artifact_directory(run_dir.to_string_lossy().to_string())
                .adaptation_readiness("artifacts_only".to_string())
                .warning(
                    "No device-backed profiling performed — requires Apple hardware".to_string(),
                )
                .warning(
                    "Drift metrics unavailable — requires Apple hardware for predict() output"
                        .to_string(),
                );

        if let Some(ref gen_info) = self.generated_from {
            let parts: Vec<&str> = gen_info.splitn(3, ',').collect();
            if parts.len() == 3 {
                if let Ok(gen_seed) = parts[1].parse::<u64>() {
                    builder = builder.generator_provenance(GeneratorProvenance {
                        generator_version: parts[2].to_string(),
                        family: parts[0].to_string(),
                        seed: gen_seed,
                        task_name: spec.name.clone(),
                    });
                    println!(
                        "  Generator provenance: family={}, seed={}, version={}",
                        parts[0], gen_seed, parts[2]
                    );
                }
            } else {
                eprintln!(
                    "  Warning: --generated-from format should be 'family,seed,version', got: {}",
                    gen_info
                );
            }
        }

        let lab_run = builder.build();

        writer
            .write_run_record(&run_dir, &lab_run)
            .map_err(|e| format!("Failed to write run record: {}", e))?;
        println!("  Run record: {}", run_dir.join(layout::RUN_JSON).display());

        println!("\n=== Lab run summary ===");
        println!("  Run ID: {}", lab_run.run_id);
        println!("  Verification scope: {:?}", lab_run.verification_scope);
        println!(
            "  Compilation: {}",
            if lab_run.compile_result.success { "SUCCESS" } else { "FAILED" }
        );
        println!(
            "  Baseline: {} FP32 reference values computed",
            baseline_result.output_tensor.len()
        );
        println!(
            "  Drift: {}",
            if drift_report.is_computed() { "computed" } else { "unavailable" }
        );
        println!("  Artifacts: {}", run_dir.display());

        println!("\n=== Lab run complete ===");

        Ok(LabResult {
            success: true,
            task_name: spec.name,
            output_dir: run_dir.to_string_lossy().to_string(),
            manifest_path: Some(run_dir.join("manifest.json").to_string_lossy().to_string()),
            error_message: None,
        })
    }
}

// ---------------------------------------------------------------------------
// LabLoopSession::run
// ---------------------------------------------------------------------------

impl LabLoopSession {
    /// Run the lab-loop session.
    ///
    /// This closes the loop from task → compile → baseline → drift →
    /// knowledge store persistence. The key difference from `LabSession`
    /// is that after computing the knowledge update JSON, this command
    /// actually ingests the observations into the KnowledgeStore using
    /// UpdatePipeline, making them queryable by the pass pipeline in
    /// subsequent compiles.
    ///
    /// Steps:
    /// 1. Load task spec from input TOML
    /// 2. Compute task hash
    /// 3. Build SIR and MIR
    /// 4. Build bridge payload and invoke Python bridge
    /// 5. Compute baseline
    /// 6. Compute drift (unavailable on non-Apple hardware, but the path must exist)
    /// 7. Open knowledge store and ingest observations
    /// 8. Write all run artifacts
    /// 9. Determine and record adaptation_readiness metadata
    pub fn run(&self) -> Result<LabResult, String> {
        println!("=== MILLer — Lab-Loop (Host-Side Evidence Loop) ===\n");

        // Step 1: Load task spec
        println!("[1/9] Loading task spec: {}", self.input);
        let spec = load_synthetic_task(&self.input)?;
        let task_hash = compute_task_hash(&spec);
        println!("  Task: {} (family: {})", spec.name, spec.family);
        println!("  Task hash: {}", task_hash);

        if spec.op.is_sharded() {
            return Err(format!("Use 'compile-sharded' command for {} tasks", spec.op.family_id()));
        }
        let (input_dim, output_dim, batch_size, _dtype) = spec.op.primary_dims();

        // Step 2: Build IR and compile
        println!("[2/9] Compiling...");
        let sir = sir_from_linear_projection(&spec)?;
        println!("  SIR: {} nodes", sir.nodes.len());

        let shard_name = format!("{}_shard_0", spec.name);
        let mir = lower_linear_projection_to_mir(&spec, &shard_name)?;

        let output_path = PathBuf::from(&self.output);
        let mlpackage_output = output_path.join(layout::MLPACKAGE_DIR);
        let payload = FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?;
        let payload_json = serde_json::to_value(&payload)
            .map_err(|e| format!("Payload serialization failed: {}", e))?;

        let bridge = PythonBridge::new(&self.python_path, &self.bridge_script);
        let result = bridge
            .execute_raw_payload(&payload_json)
            .map_err(|e| format!("Bridge execution failed: {}", e))?;

        let compile_step = CompileStepResult {
            success: result.status == "success",
            error: result.error_message.clone(),
            output_path: result.output_path.clone(),
            content_hash: result.content_hash.clone(),
            file_count: if result.package_files.is_empty() {
                None
            } else {
                Some(result.package_files.len())
            },
            coremltools_version: result.coremltools_version.clone(),
        };

        if compile_step.success {
            println!("  Compilation: SUCCESS");
            if let Some(ref hash) = compile_step.content_hash {
                println!("  Content hash: {}", hash);
            }
        } else {
            println!("  Compilation: FAILED");
            if let Some(ref err) = compile_step.error {
                println!("  Error: {}", err);
            }
        }

        // Step 3: Create lab run directory and write initial artifacts
        println!("[3/9] Writing lab-loop run artifacts...");
        let run_id = generate_run_id(&task_hash);
        let writer = LabRunWriter::new(&output_path);
        let run_dir = writer
            .create_run_directory(&run_id)
            .map_err(|e| format!("Failed to create run directory: {}", e))?;
        println!("  Run directory: {}", run_dir.display());

        // Write manifest (will be updated later with adaptation_readiness)
        let compiler_version = env!("CARGO_PKG_VERSION");
        let mut manifest = build_artifact_manifest(&spec, &result, &task_hash, compiler_version);

        // Write MIR
        let mir_json =
            serde_json::to_value(&mir).map_err(|e| format!("MIR serialization failed: {}", e))?;
        writer.write_mir(&run_dir, &mir_json).map_err(|e| format!("Failed to write MIR: {}", e))?;

        // Step 4: Host-side inspection
        println!("[4/9] Performing host-side inspection...");
        let inspect_step = if compile_step.success {
            let inspector =
                crate::host_inspect::HostInspector::new(&self.python_path, &self.bridge_script);
            let mlpackage_path = result.output_path.as_deref().unwrap_or("");
            let inspect_result = inspector.inspect(mlpackage_path);

            println!("  Package present: {}", inspect_result.package_present);
            println!("  Manifest readable: {}", inspect_result.manifest_readable);
            println!("  Model loadable: {}", inspect_result.model_loadable);
            if !inspect_result.model_loadable {
                if let Some(ref reason) = inspect_result.model_load_failure_reason {
                    println!("  Load failure: {}", reason);
                }
            }

            let inspect_json = serde_json::to_value(&inspect_result)
                .map_err(|e| format!("Inspection serialization failed: {}", e))?;
            writer
                .write_inspection(&run_dir, &inspect_json)
                .map_err(|e| format!("Failed to write inspection: {}", e))?;

            inspect_result
        } else {
            println!("  Host-side inspection: SKIPPED (compilation failed)");
            InspectionStepResult {
                package_present: false,
                manifest_readable: false,
                model_loadable: false,
                model_load_failure_reason: Some("Inspection not performed".to_string()),
                function_count: None,
                input_specs: vec![],
                output_specs: vec![],
                warnings: vec!["Host-side inspection was not performed".to_string()],
                structure_inspection_available: None,
                structure_inspection_failure_reason: Some("Inspection not performed".to_string()),
                structure_op_names: vec![],
                structure_op_count: None,
                structure_function_count: None,
                structure_state_declarations: vec![],
                op_fidelity_score: None,
                missing_ops: vec![],
                extra_ops: vec![],
                inspection_method: "none".to_string(),
            }
        };

        // Step 5: Compute FP32 baseline reference
        println!("[5/9] Computing FP32 baseline reference...");
        let mut baseline_result =
            compute_baseline(&spec, self.seed, input_dim, output_dim, batch_size)?;
        baseline_result.task_hash = Some(task_hash.clone());
        println!(
            "  Baseline: {} output elements, computed in {:.3}ms",
            baseline_result.output_tensor.len(),
            baseline_result.compute_time_ms
        );

        let baseline_json = serde_json::to_value(&baseline_result)
            .map_err(|e| format!("Baseline serialization failed: {}", e))?;
        writer
            .write_baseline(&run_dir, &baseline_json)
            .map_err(|e| format!("Failed to write baseline: {}", e))?;

        // Step 6: Compute drift
        println!("[6/9] Computing drift metrics...");
        let drift_report = if compile_step.success {
            let unavailable_report = DriftDetector::unavailable(
                "predict() requires Apple hardware with Core ML runtime",
            );
            println!("  Drift: UNAVAILABLE (no on-device predict output)");
            unavailable_report
        } else {
            let unavailable_report =
                DriftDetector::unavailable("compilation failed — no model output to compare");
            println!("  Drift: UNAVAILABLE (compilation failed)");
            unavailable_report
        };

        let drift_json = serde_json::to_value(&drift_report)
            .map_err(|e| format!("Drift serialization failed: {}", e))?;
        writer
            .write_drift(&run_dir, &drift_json)
            .map_err(|e| format!("Failed to write drift report: {}", e))?;

        // Step 7: Build knowledge update and ingest observations into the store
        println!("[7/9] Ingesting observations into knowledge store...");
        let knowledge_update = build_knowledge_update_with_drift(
            &spec,
            &result,
            &task_hash,
            &baseline_result,
            &drift_report,
        );

        writer
            .write_knowledge_update(&run_dir, &spec.name, &knowledge_update)
            .map_err(|e| format!("Failed to write knowledge update: {}", e))?;

        // Open the knowledge store and ingest observations
        let knowledge_store_path = PathBuf::from(&self.knowledge_dir);
        let mut store = if knowledge_store_path.join("store_index.json").exists() {
            ane_knowledge::store::KnowledgeStore::open(&self.knowledge_dir).map_err(|e| {
                format!("Failed to open knowledge store at {}: {}", self.knowledge_dir, e)
            })?
        } else {
            std::fs::create_dir_all(&knowledge_store_path)
                .map_err(|e| format!("Failed to create knowledge store directory: {}", e))?;
            ane_knowledge::store::KnowledgeStore::open(&self.knowledge_dir).map_err(|e| {
                format!("Failed to create knowledge store at {}: {}", self.knowledge_dir, e)
            })?
        };

        let ingested_count =
            ingest_knowledge_observations(&mut store, &knowledge_update, &task_hash)?;
        println!(
            "  Ingested {} observations into knowledge store at {}",
            ingested_count, self.knowledge_dir
        );

        // Step 8: Determine adaptation_readiness
        println!("[8/9] Determining adaptation readiness...");
        let readiness_level = if ingested_count > 0 {
            let empty_observations: Vec<serde_json::Value> = vec![];
            let observations = knowledge_update
                .get("observations")
                .and_then(|v| v.as_array())
                .unwrap_or(&empty_observations);
            let has_compiler_consumable = observations.iter().any(|obs| {
                let conf = obs.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ev_count = obs.get("evidence_count").and_then(|v| v.as_u64()).unwrap_or(0);
                conf > 0.0 && ev_count >= 1
            });
            if has_compiler_consumable {
                "artifacts_observation_compiler_consumable"
            } else {
                "artifacts_and_observation"
            }
        } else {
            "artifacts_only"
        };
        println!("  Adaptation readiness: {}", readiness_level);

        // Step 9: Build and write LabRun record with adaptation_readiness
        println!("[9/9] Writing lab-loop run record...");
        let env = EnvironmentSummary::detect(1);
        let verification_scope = VerificationScope::HostOnlyInspection;

        let mut builder =
            LabRunBuilder::new(run_id, task_hash, spec.name.clone(), verification_scope, env)
                .compile_result(compile_step)
                .inspect_result(inspect_step)
                .artifact_directory(run_dir.to_string_lossy().to_string())
                .adaptation_readiness(readiness_level.to_string())
                .warning(
                    "No device-backed profiling performed — requires Apple hardware".to_string(),
                )
                .warning(
                    "Drift metrics unavailable — requires Apple hardware for predict() output"
                        .to_string(),
                );

        if let Some(ref gen_info) = self.generated_from {
            let parts: Vec<&str> = gen_info.splitn(3, ',').collect();
            if parts.len() == 3 {
                if let Ok(gen_seed) = parts[1].parse::<u64>() {
                    builder = builder.generator_provenance(GeneratorProvenance {
                        generator_version: parts[2].to_string(),
                        family: parts[0].to_string(),
                        seed: gen_seed,
                        task_name: spec.name.clone(),
                    });
                    println!(
                        "  Generator provenance: family={}, seed={}, version={}",
                        parts[0], gen_seed, parts[2]
                    );
                }
            } else {
                eprintln!(
                    "  Warning: --generated-from format should be 'family,seed,version', got: {}",
                    gen_info
                );
            }
        }

        let lab_run = builder.build();

        writer
            .write_run_record(&run_dir, &lab_run)
            .map_err(|e| format!("Failed to write run record: {}", e))?;

        // Add adaptation_readiness to manifest
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert("adaptation_readiness".to_string(), serde_json::json!(readiness_level));
            obj.insert("knowledge_store_path".to_string(), serde_json::json!(self.knowledge_dir));
            obj.insert("observations_ingested".to_string(), serde_json::json!(ingested_count));
        }
        writer
            .write_manifest(&run_dir, &manifest)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;

        println!("  Run record: {}", run_dir.join(layout::RUN_JSON).display());

        println!("\n=== Lab-loop run summary ===");
        println!("  Run ID: {}", lab_run.run_id);
        println!("  Verification scope: {:?}", lab_run.verification_scope);
        println!(
            "  Compilation: {}",
            if lab_run.compile_result.success { "SUCCESS" } else { "FAILED" }
        );
        println!(
            "  Baseline: {} FP32 reference values computed",
            baseline_result.output_tensor.len()
        );
        println!(
            "  Drift: {}",
            if drift_report.is_computed() { "computed" } else { "unavailable" }
        );
        println!("  Observations ingested: {}", ingested_count);
        println!("  Adaptation readiness: {}", readiness_level);
        println!("  Knowledge store: {}", self.knowledge_dir);
        println!("  Artifacts: {}", run_dir.display());

        println!("\n=== Lab-loop run complete ===");

        Ok(LabResult {
            success: true,
            task_name: spec.name,
            output_dir: run_dir.to_string_lossy().to_string(),
            manifest_path: Some(run_dir.join("manifest.json").to_string_lossy().to_string()),
            error_message: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_bridge::subprocess::{BridgeResult, EmissionPath};
    use ane_ir::task_spec::{MeasurementConfig, SyntheticTaskSpec, TaskOp};

    fn make_test_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "test_task".to_string(),
            family: "LinearProjection".to_string(),
            description: Some("test description".to_string()),
            op: TaskOp::LinearProjection {
                input_dim: 64,
                output_dim: 32,
                batch_size: 1,
                has_bias: true,
                dtype: "fp16".to_string(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".to_string()],
            },
        }
    }

    fn make_other_spec() -> SyntheticTaskSpec {
        SyntheticTaskSpec {
            name: "other_task".to_string(),
            family: "LinearProjection".to_string(),
            description: None,
            op: TaskOp::LinearProjection {
                input_dim: 128,
                output_dim: 64,
                batch_size: 2,
                has_bias: false,
                dtype: "fp32".to_string(),
            },
            measurement: MeasurementConfig {
                warmup_iterations: 5,
                measured_iterations: 20,
                metrics: vec!["Latency".to_string()],
            },
        }
    }

    fn make_success_bridge_result() -> BridgeResult {
        BridgeResult {
            status: "success".to_string(),
            error_message: None,
            output_path: Some("/tmp/test.mlpackage".to_string()),
            coremltools_version: Some("9.0".to_string()),
            content_hash: Some("sha256:abc123".to_string()),
            package_files: vec![],
            compute_plan: None,
            function_descriptors: vec![],
            metadata: serde_json::Value::Null,
            stderr: String::new(),
            emission_path: EmissionPath::PythonBridge,
        }
    }

    fn make_failure_bridge_result() -> BridgeResult {
        BridgeResult {
            status: "error".to_string(),
            error_message: Some("compilation failed".to_string()),
            output_path: None,
            coremltools_version: None,
            content_hash: None,
            package_files: vec![],
            compute_plan: None,
            function_descriptors: vec![],
            metadata: serde_json::Value::Null,
            stderr: String::new(),
            emission_path: EmissionPath::PythonBridge,
        }
    }

    #[test]
    fn test_compute_task_hash_deterministic() {
        let spec = make_test_spec();
        let hash1 = compute_task_hash(&spec);
        let hash2 = compute_task_hash(&spec);
        assert_eq!(hash1, hash2, "Same spec must produce the same hash");
    }

    #[test]
    fn test_compute_task_hash_different_specs() {
        let spec1 = make_test_spec();
        let spec2 = make_other_spec();
        let hash1 = compute_task_hash(&spec1);
        let hash2 = compute_task_hash(&spec2);
        assert_ne!(hash1, hash2, "Different specs must produce different hashes");
    }

    #[test]
    fn test_compute_task_hash_format() {
        let spec = make_test_spec();
        let hash = compute_task_hash(&spec);
        assert!(
            hash.starts_with("sha256:"),
            "Hash must start with 'sha256:' prefix, got: {}",
            hash
        );
        let hex_part = &hash[7..];
        assert_eq!(hex_part.len(), 64, "SHA-256 hex digest must be 64 chars");
        assert!(hex_part.chars().all(|c| c.is_ascii_hexdigit()), "Hex portion must be valid hex");
    }

    #[test]
    fn test_compute_task_hash_uses_identity_string() {
        let mut spec_a = make_test_spec();
        let mut spec_b = make_test_spec();
        spec_a.op = TaskOp::LinearProjection {
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            has_bias: true,
            dtype: "fp16".to_string(),
        };
        spec_b.op = TaskOp::LinearProjection {
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            has_bias: false,
            dtype: "fp16".to_string(),
        };
        assert_ne!(
            compute_task_hash(&spec_a),
            compute_task_hash(&spec_b),
            "Hash must incorporate the op identity string (has_bias differs)"
        );
    }

    #[test]
    fn test_build_artifact_manifest_success() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let task_hash = "sha256:abcdef1234567890";
        let manifest = build_artifact_manifest(&spec, &bridge_result, task_hash, "0.1.0");
        assert_eq!(manifest["version"], "0.3.0");
        assert_eq!(manifest["model_id"], "test_task");
        assert_eq!(manifest["task_hash"], task_hash);
        assert_eq!(manifest["compiler_version"], "0.1.0");
        assert_eq!(manifest["implementation_status"], "host_compiled");
        assert_eq!(manifest["verification_scope"], "host_compile_only");
        let packages = manifest["packages"].as_array().unwrap();
        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0]["name"], "test_task");
        assert_eq!(packages[0]["role"], "synthetic_microkernel");
        assert_eq!(packages[0]["path"], "/tmp/test.mlpackage");
        assert_eq!(packages[0]["content_hash"], "sha256:abc123");
    }

    #[test]
    fn test_build_artifact_manifest_failure() {
        let spec = make_test_spec();
        let bridge_result = make_failure_bridge_result();
        let manifest = build_artifact_manifest(&spec, &bridge_result, "sha256:abc", "0.1.0");
        let packages = manifest["packages"].as_array().unwrap();
        assert!(packages.is_empty(), "Failure should produce no packages");
        assert_eq!(manifest["model_id"], "test_task");
    }

    #[test]
    fn test_build_artifact_manifest_has_environment_limitations() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let manifest = build_artifact_manifest(&spec, &bridge_result, "sha256:test", "0.1.0");
        let limitations = manifest["environment_limitations"].as_array().unwrap();
        assert_eq!(limitations.len(), 3);
        let strs: Vec<&str> = limitations.iter().filter_map(|v| v.as_str()).collect();
        assert!(strs.contains(&"no_apple_hardware"));
        assert!(strs.contains(&"ane_placement_not_verified"));
        assert!(strs.contains(&"no_on_device_predict"));
    }

    #[test]
    fn test_build_knowledge_update_success() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let task_hash = "sha256:testhash";
        let update = build_knowledge_update(&spec, &bridge_result, task_hash);
        assert_eq!(update["version"], 2);
        assert_eq!(update["source"], "vertical_slice_compile");
        assert_eq!(update["task_hash"], task_hash);
        assert_eq!(update["task_name"], "test_task");
        assert_eq!(update["task_family"], "LinearProjection");
        let observations = update["observations"].as_array().unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0]["knowledge_type"], "LegalityRule");
        assert_eq!(observations[0]["op_pattern"], "mb.matmul");
        assert_eq!(observations[0]["ane_legal"], true);
        assert_eq!(observations[0]["confidence"], 0.3);
        assert_eq!(observations[0]["evidence_source"], "SyntheticRun");
        assert_eq!(observations[1]["op_pattern"], "mb.add");
        assert_eq!(observations[1]["ane_legal"], true);
        assert_eq!(observations[1]["confidence"], 0.3);
    }

    #[test]
    fn test_build_knowledge_update_failure() {
        let spec = make_test_spec();
        let bridge_result = make_failure_bridge_result();
        let update = build_knowledge_update(&spec, &bridge_result, "sha256:testhash");
        let observations = update["observations"].as_array().unwrap();
        assert_eq!(observations[0]["ane_legal"], false);
        assert_eq!(observations[0]["confidence"], 0.7);
        assert_eq!(observations[1]["ane_legal"], false);
        assert_eq!(observations[1]["confidence"], 0.7);
    }

    #[test]
    fn test_build_knowledge_update_has_residuals() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let update = build_knowledge_update(&spec, &bridge_result, "sha256:testhash");
        let residuals = update["residuals"].as_array().unwrap();
        assert_eq!(residuals.len(), 3);
        let strs: Vec<&str> = residuals.iter().filter_map(|v| v.as_str()).collect();
        assert!(strs[0].contains("ANE placement not verified"));
        assert!(strs[1].contains("Numerical drift not measured"));
        assert!(strs[2].contains("Fallback suspicion not assessed"));
    }

    #[test]
    fn test_build_knowledge_update_with_drift_computed() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let task_hash = "sha256:drifttest";
        let baseline = crate::baseline::BaselineResult {
            baseline_schema_version: "1.0.0".to_string(),
            task_id: "test_task".to_string(),
            task_hash: Some(task_hash.to_string()),
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            seed: 42,
            precision: "fp32".to_string(),
            output_tensor: vec![0.0; 32],
            output_shape: vec![1, 32],
            compute_time_ms: 1.0,
        };
        let drift =
            crate::drift::DriftDetector::new().detect(&[1.0f32, 2.0, 3.0], &[1.01, 2.01, 3.01]);
        assert!(drift.is_computed());
        let update =
            build_knowledge_update_with_drift(&spec, &bridge_result, task_hash, &baseline, &drift);
        assert_eq!(update["version"], 3);
        assert_eq!(update["source"], "lab_run_with_drift");
        let observations = update["observations"].as_array().unwrap();
        assert_eq!(observations.len(), 3);
        let drift_obs = &observations[2];
        assert_eq!(drift_obs["knowledge_type"], "PrecisionHazard");
        assert_eq!(drift_obs["op_pattern"], "linear_projection_fp16_vs_fp32");
        assert!(drift_obs.get("max_absolute_error").is_some());
        assert!(drift_obs.get("mean_absolute_error").is_some());
        assert!(drift_obs.get("rmse").is_some());
        assert!(drift_obs.get("cosine_distance").is_some());
        assert!(drift_obs.get("relative_error_p99").is_some());
        assert_eq!(drift_obs["confidence"], 0.3);
        assert_eq!(drift_obs["evidence_source"], "SyntheticRun");
        assert_eq!(drift_obs["evidence_count"], 1);
    }

    #[test]
    fn test_build_knowledge_update_with_drift_unavailable() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let task_hash = "sha256:driftunavail";
        let baseline = crate::baseline::BaselineResult {
            baseline_schema_version: "1.0.0".to_string(),
            task_id: "test_task".to_string(),
            task_hash: Some(task_hash.to_string()),
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            seed: 42,
            precision: "fp32".to_string(),
            output_tensor: vec![0.0; 32],
            output_shape: vec![1, 32],
            compute_time_ms: 1.0,
        };
        let drift = crate::drift::DriftDetector::unavailable("no Apple hardware");
        let update =
            build_knowledge_update_with_drift(&spec, &bridge_result, task_hash, &baseline, &drift);
        let observations = update["observations"].as_array().unwrap();
        let drift_obs = &observations[2];
        assert_eq!(drift_obs["knowledge_type"], "PrecisionHazard");
        assert_eq!(drift_obs["computation_status"], "unavailable");
        assert_eq!(drift_obs["reason"], "no Apple hardware");
        assert_eq!(drift_obs["confidence"], 0.0);
    }

    #[test]
    fn test_build_knowledge_update_with_drift_version() {
        let spec = make_test_spec();
        let bridge_result = make_success_bridge_result();
        let task_hash = "sha256:versiontest";
        let baseline = crate::baseline::BaselineResult {
            baseline_schema_version: "1.0.0".to_string(),
            task_id: "test_task".to_string(),
            task_hash: None,
            input_dim: 64,
            output_dim: 32,
            batch_size: 1,
            seed: 42,
            precision: "fp32".to_string(),
            output_tensor: vec![0.0; 32],
            output_shape: vec![1, 32],
            compute_time_ms: 1.0,
        };
        let drift = crate::drift::DriftDetector::unavailable("test");
        let update =
            build_knowledge_update_with_drift(&spec, &bridge_result, task_hash, &baseline, &drift);
        assert_eq!(update["version"], 3, "Drift variant must use version 3");
    }

    #[test]
    fn test_ingest_knowledge_observations_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("store").to_string_lossy().to_string();
        let mut store = ane_knowledge::store::KnowledgeStore::open(&store_path).unwrap();
        let knowledge_update = serde_json::json!({
            "observations": [
                {
                    "knowledge_type": "LegalityRule",
                    "op_pattern": "mb.matmul",
                    "ane_legal": true,
                    "confidence": 0.3,
                    "evidence_source": "SyntheticRun",
                    "evidence_count": 1,
                    "scope": {
                        "device_classes": ["unknown"],
                        "os_versions": ["unknown"],
                        "opset_versions": ["iOS18"],
                    },
                },
                {
                    "knowledge_type": "LegalityRule",
                    "op_pattern": "mb.add",
                    "ane_legal": true,
                    "confidence": 0.3,
                    "evidence_source": "SyntheticRun",
                    "evidence_count": 1,
                    "scope": {
                        "device_classes": ["unknown"],
                        "os_versions": ["unknown"],
                        "opset_versions": ["iOS18"],
                    },
                },
            ],
        });
        let count =
            ingest_knowledge_observations(&mut store, &knowledge_update, "sha256:abc").unwrap();
        assert_eq!(count, 2, "Should ingest 2 valid observations");
    }

    #[test]
    fn test_ingest_knowledge_observations_empty_observations() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("store").to_string_lossy().to_string();
        let mut store = ane_knowledge::store::KnowledgeStore::open(&store_path).unwrap();
        let knowledge_update = serde_json::json!({"observations": []});
        let count =
            ingest_knowledge_observations(&mut store, &knowledge_update, "sha256:abc").unwrap();
        assert_eq!(count, 0, "Empty observations array should ingest 0");
    }

    #[test]
    fn test_ingest_knowledge_observations_missing_observations() {
        let tmp = tempfile::tempdir().unwrap();
        let store_path = tmp.path().join("store").to_string_lossy().to_string();
        let mut store = ane_knowledge::store::KnowledgeStore::open(&store_path).unwrap();
        let knowledge_update = serde_json::json!({"version": 2, "source": "test"});
        let result = ingest_knowledge_observations(&mut store, &knowledge_update, "sha256:abc");
        assert!(result.is_err(), "Missing observations field should return Err");
        assert!(
            result.unwrap_err().contains("No observations found"),
            "Error message should mention missing observations"
        );
    }
}
