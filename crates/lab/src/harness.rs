//! Lab Run Harness
//!
//! Orchestrates lab runs: compilation, host-side inspection, and (when available)
//! device-backed profiling. Every run produces a structured LabRun record that
//! honestly distinguishes host-only evidence from device-backed evidence.
//!
//! The run schema is the single source of truth for what happened during a lab
//! execution. No run record ever claims device-backed verification when only
//! host-side inspection was performed.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Schema version for the LabRun format.
/// Increment this when the serialized structure changes incompatibly.
pub const LAB_RUN_SCHEMA_VERSION: &str = "1.0.0";

/// Verification scope of a lab run.
///
/// This enum is the structural mechanism that prevents host-only evidence
/// from being misread as device-backed evidence. Every LabRun carries one
/// of these, and downstream consumers must check it before interpreting
/// timing or placement results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerificationScope {
    /// Only host-side operations were performed: compilation, file inspection,
    /// metadata extraction. No model was executed on any runtime.
    /// Timing fields in the run MUST be None.
    HostOnlyInspection,
    /// The compiled model was loaded and executed on a host runtime (CPU/GPU),
    /// but not on Apple hardware with ANE. Timing is real but reflects host
    /// execution, not ANE execution.
    HostRuntimeExecution,
    /// The model was executed on Apple hardware with Core ML runtime.
    /// Timing reflects real device execution. Compute unit assignment may
    /// still be uncertain unless a compute plan was obtained.
    DeviceBackedExecution,
}

/// Summary of the environment where the lab run was performed.
///
/// This is NOT a device metadata record — it describes the host environment
/// where the compiler and inspector ran. Device metadata (for device-backed
/// runs) is captured separately in the profiling result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSummary {
    /// Host OS (e.g., "Linux x86_64", "macOS arm64").
    pub host_os: String,
    /// Whether Core ML runtime is available on this host.
    pub coreml_runtime_available: bool,
    /// Whether coremltools Python package is available.
    pub coremltools_available: bool,
    /// Rust compiler version used.
    pub compiler_version: String,
    /// Bridge version in use.
    pub bridge_version: u32,
}

impl EnvironmentSummary {
    /// Build an environment summary for the current host.
    ///
    /// This performs honest detection: it checks what is actually available
    /// rather than assuming capabilities.
    pub fn detect(bridge_version: u32) -> Self {
        let host_os = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);

        // coremltools availability can only be confirmed by the Python side;
        // on the Rust side we conservatively report unknown.
        let coremltools_available = false; // Rust cannot confirm this; Python bridge reports it

        // Core ML runtime is only available on Apple platforms.
        let coreml_runtime_available = cfg!(target_vendor = "apple");

        Self {
            host_os,
            coreml_runtime_available,
            coremltools_available,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            bridge_version,
        }
    }
}

/// Result of the compilation step within a lab run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileStepResult {
    /// Whether compilation succeeded.
    pub success: bool,
    /// Error message if compilation failed.
    pub error: Option<String>,
    /// Path to the output mlpackage (if successful).
    pub output_path: Option<String>,
    /// Content hash of the mlpackage directory (sha256:<hex>).
    pub content_hash: Option<String>,
    /// Number of files in the mlpackage.
    pub file_count: Option<usize>,
    /// coremltools version used for emission.
    pub coremltools_version: Option<String>,
}

/// Result of the host-side inspection step within a lab run.
///
/// Host-side inspection checks what can honestly be determined without
/// executing the model on a device runtime. It NEVER infers ANE behavior
/// or compute unit placement.
///
/// Sprint 34 adds structural verification fields: if MLModelStructure
/// inspection is available, the result includes op inventory, function
/// signatures, and state declarations from the actual emitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionStepResult {
    /// Whether the mlpackage directory exists and is well-formed.
    pub package_present: bool,
    /// Whether the mlpackage Manifest.json could be read.
    pub manifest_readable: bool,
    /// Whether the model could be loaded via coremltools (requires Core ML runtime).
    pub model_loadable: bool,
    /// Reason model load failed (if not loadable).
    pub model_load_failure_reason: Option<String>,
    /// Number of functions in the package (from metadata).
    pub function_count: Option<usize>,
    /// Input specifications extracted from package metadata.
    pub input_specs: Vec<TensorSpecRecord>,
    /// Output specifications extracted from package metadata.
    pub output_specs: Vec<TensorSpecRecord>,
    /// Warnings from inspection (e.g., "compute plan not available on this platform").
    pub warnings: Vec<String>,

    // --- Sprint 34: Structural Verification Fields ---
    /// Whether MLModelStructure inspection was available and succeeded.
    /// None means structural inspection was not attempted.
    /// Some(true) means structural inspection succeeded.
    /// Some(false) means structural inspection was attempted but unavailable
    /// (e.g., non-Apple platform or missing coremltools).
    pub structure_inspection_available: Option<bool>,

    /// Reason structural inspection was unavailable (if applicable).
    pub structure_inspection_failure_reason: Option<String>,

    /// Op names found in the emitted model structure via MLModelStructure.
    /// Empty if structural inspection was not available.
    pub structure_op_names: Vec<String>,

    /// Total number of operations found in the emitted model.
    pub structure_op_count: Option<usize>,

    /// Number of functions found in the emitted model structure.
    pub structure_function_count: Option<usize>,

    /// State declarations found in the emitted model structure.
    /// Each entry is a dict-like record with name, shape, dtype.
    pub structure_state_declarations: Vec<TensorSpecRecord>,

    /// Op fidelity score (0-1) from MIR-vs-structure comparison.
    /// None if comparison was not performed.
    pub op_fidelity_score: Option<f64>,

    /// MIR ops that were NOT found in the emitted structure.
    /// Each entry is the op type name (e.g., "MILLinear").
    pub missing_ops: Vec<String>,

    /// Ops in the emitted structure that were NOT expected by the MIR.
    /// Each entry is the op type name (e.g., "const").
    pub extra_ops: Vec<String>,

    /// The inspection method used: "mlmodel_structure", "fallback_file_check",
    /// or "none" (if no structural inspection was performed).
    pub inspection_method: String,
}

/// A tensor specification as recorded from inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpecRecord {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// A complete lab run record.
///
/// This is the primary output artifact of a lab run. It captures everything
/// that happened during the run in a structured, serializable format.
///
/// The `verification_scope` field is authoritative: consumers MUST check it
/// before interpreting any timing, placement, or correctness claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabRun {
    /// Schema version of this run record.
    pub schema_version: String,
    /// Unique run identifier. Format: "run_<timestamp>_<task_hash_prefix>".
    pub run_id: String,
    /// ISO-8601 timestamp of run start.
    pub started_at: String,
    /// ISO-8601 timestamp of run completion.
    pub completed_at: Option<String>,
    /// Verification scope — what class of evidence this run provides.
    pub verification_scope: VerificationScope,
    /// Environment where the run was performed.
    pub environment: EnvironmentSummary,
    /// Task identity hash (sha256:<hex>), matching the manifest task_hash.
    pub task_id: String,
    /// Task name (e.g., "linear_proj_slice").
    pub task_name: String,
    /// Payload hash — the hash of the bridge request payload.
    /// Links this run to a specific compilation input.
    pub payload_hash: Option<String>,
    /// Result of the compilation step.
    pub compile_result: CompileStepResult,
    /// Result of the host-side inspection step.
    pub inspect_result: InspectionStepResult,
    /// Timing results (only present for device-backed or host-runtime runs).
    pub timing: Option<TimingResult>,
    /// Fallback suspicion assessment (only present for device-backed runs).
    pub fallback_suspicion: Option<FallbackSuspicionResult>,
    /// Warnings accumulated during the run.
    pub warnings: Vec<String>,
    /// Directory where run artifacts were written.
    pub artifact_directory: Option<String>,
    /// Generator provenance: if this run used a generated task rather than
    /// a hand-authored TOML, this records the generator metadata.
    pub generator_provenance: Option<GeneratorProvenance>,

    /// Adaptation readiness: how far the evidence loop closed.
    /// - "artifacts_only": only artifacts produced, no knowledge persistence
    /// - "artifacts_and_observation": artifacts + stored observation
    /// - "artifacts_observation_compiler_consumable": artifacts + stored observation that the pass pipeline can query
    pub adaptation_readiness: Option<String>,
}

/// Provenance metadata for tasks generated by the task generator.
///
/// Records that a task was generated (not hand-authored), which family
/// and seed were used, and the generator version. This allows downstream
/// consumers to distinguish generated tasks from manually authored ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorProvenance {
    /// The generator version that produced this task.
    pub generator_version: String,
    /// The task family (e.g., "LinearProjection").
    pub family: String,
    /// The seed used for deterministic generation.
    pub seed: u64,
    /// The task name within the generated set.
    pub task_name: String,
}

/// Timing results from repeated execution.
///
/// Only meaningful when verification_scope is HostRuntimeExecution or
/// DeviceBackedExecution. For HostOnlyInspection runs, this MUST be None.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingResult {
    /// Number of warmup iterations (not included in statistics).
    pub warmup_iterations: usize,
    /// Number of measured iterations.
    pub measured_iterations: usize,
    /// Median latency in milliseconds.
    pub p50_ms: f64,
    /// 90th percentile latency in milliseconds.
    pub p90_ms: f64,
    /// 99th percentile latency in milliseconds.
    pub p99_ms: f64,
    /// Minimum latency in milliseconds.
    pub min_ms: f64,
    /// Maximum latency in milliseconds.
    pub max_ms: f64,
    /// Mean latency in milliseconds.
    pub mean_ms: f64,
    /// Standard deviation in milliseconds.
    pub std_dev_ms: f64,
    /// Compute units used for execution (e.g., "CPU_AND_NE", "CPU_ONLY").
    pub compute_units: String,
    /// Scope note: what this timing actually measures.
    /// E.g., "Host CPU execution on Linux x86_64 — NOT ANE timing"
    /// or "Device execution on Apple M2 with CPU_AND_NE hint".
    pub scope_note: String,
}

/// Fallback suspicion result.
///
/// This is a deliberately weak and honest assessment. It does NOT make
/// hard claims about compute unit placement. It can only express levels
/// of suspicion based on available evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackSuspicionResult {
    /// Overall suspicion level.
    pub suspicion_level: FallbackSuspicionLevel,
    /// Human-readable explanation of the suspicion assessment.
    pub explanation: String,
    /// Evidence items that contributed to this assessment.
    pub evidence: Vec<SuspicionEvidence>,
}

/// Level of fallback suspicion.
///
/// This is explicitly NOT a binary "fell back / didn't fall back" result.
/// It represents the honest range of conclusions that can be drawn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FallbackSuspicionLevel {
    /// Fallback assessment is not available.
    /// This is the default for host-only runs and runs without compute plan access.
    Unavailable,
    /// There is some weak evidence suggesting fallback, but it is not conclusive.
    /// E.g., latency is higher than expected but the benchmark is imprecise.
    LowConfidenceSuspicion,
    /// No evidence of fallback was found, but absence of evidence is not
    /// evidence of absence. The model may still have fallen back.
    NoConclusion,
}

/// A single piece of evidence contributing to fallback suspicion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspicionEvidence {
    /// What kind of evidence this is.
    pub kind: String,
    /// Description of the evidence.
    pub description: String,
    /// How strongly this evidence points toward fallback (0.0 = none, 1.0 = certain).
    /// Values above 0.5 suggest fallback; values below 0.5 suggest no fallback.
    /// This is always a weak signal — no single evidence item is conclusive.
    pub strength: f64,
}

/// Builder for constructing a LabRun incrementally.
pub struct LabRunBuilder {
    run: LabRun,
}

impl LabRunBuilder {
    /// Start building a new lab run with the given identity and scope.
    pub fn new(
        run_id: String,
        task_id: String,
        task_name: String,
        verification_scope: VerificationScope,
        environment: EnvironmentSummary,
    ) -> Self {
        let started_at = chrono::Utc::now().to_rfc3339();
        Self {
            run: LabRun {
                schema_version: LAB_RUN_SCHEMA_VERSION.to_string(),
                run_id,
                started_at,
                completed_at: None,
                verification_scope,
                environment,
                task_id,
                task_name,
                payload_hash: None,
                compile_result: CompileStepResult {
                    success: false,
                    error: None,
                    output_path: None,
                    content_hash: None,
                    file_count: None,
                    coremltools_version: None,
                },
                inspect_result: InspectionStepResult {
                    package_present: false,
                    manifest_readable: false,
                    model_loadable: false,
                    model_load_failure_reason: None,
                    function_count: None,
                    input_specs: vec![],
                    output_specs: vec![],
                    warnings: vec![],
                    structure_inspection_available: None,
                    structure_inspection_failure_reason: None,
                    structure_op_names: vec![],
                    structure_op_count: None,
                    structure_function_count: None,
                    structure_state_declarations: vec![],
                    op_fidelity_score: None,
                    missing_ops: vec![],
                    extra_ops: vec![],
                    inspection_method: "none".to_string(),
                },
                timing: None,
                fallback_suspicion: None,
                warnings: vec![],
                artifact_directory: None,
                generator_provenance: None,
                adaptation_readiness: None,
            },
        }
    }

    /// Set the payload hash.
    pub fn payload_hash(mut self, hash: String) -> Self {
        self.run.payload_hash = Some(hash);
        self
    }

    /// Set the compilation result.
    pub fn compile_result(mut self, result: CompileStepResult) -> Self {
        self.run.compile_result = result;
        self
    }

    /// Set the inspection result.
    pub fn inspect_result(mut self, result: InspectionStepResult) -> Self {
        self.run.inspect_result = result;
        self
    }

    /// Set the timing result (only for runs with execution).
    pub fn timing(mut self, timing: TimingResult) -> Self {
        self.run.timing = Some(timing);
        self
    }

    /// Set the fallback suspicion result.
    pub fn fallback_suspicion(mut self, suspicion: FallbackSuspicionResult) -> Self {
        self.run.fallback_suspicion = Some(suspicion);
        self
    }

    /// Add a warning.
    pub fn warning(mut self, warning: String) -> Self {
        self.run.warnings.push(warning);
        self
    }

    /// Set the artifact directory.
    pub fn artifact_directory(mut self, dir: String) -> Self {
        self.run.artifact_directory = Some(dir);
        self
    }

    /// Set the generator provenance.
    pub fn generator_provenance(mut self, provenance: GeneratorProvenance) -> Self {
        self.run.generator_provenance = Some(provenance);
        self
    }

    /// Set the adaptation readiness level.
    pub fn adaptation_readiness(mut self, readiness: String) -> Self {
        self.run.adaptation_readiness = Some(readiness);
        self
    }

    /// Finalize the lab run, recording completion time.
    pub fn build(self) -> LabRun {
        let mut run = self.run;
        run.completed_at = Some(chrono::Utc::now().to_rfc3339());
        run
    }
}

impl LabRun {
    /// Serialize this run to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Write this run to a JSON file.
    pub fn write_to_file(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env() -> EnvironmentSummary {
        EnvironmentSummary::detect(1)
    }

    fn make_builder() -> LabRunBuilder {
        LabRunBuilder::new(
            "run_123".to_string(),
            "sha256:abc".to_string(),
            "test_task".to_string(),
            VerificationScope::HostOnlyInspection,
            make_env(),
        )
    }

    #[test]
    fn test_lab_run_builder_minimal() {
        let run = make_builder().build();
        assert_eq!(run.run_id, "run_123");
        assert_eq!(run.task_id, "sha256:abc");
        assert_eq!(run.task_name, "test_task");
        assert_eq!(run.verification_scope, VerificationScope::HostOnlyInspection);
        assert!(run.payload_hash.is_none());
        assert!(run.timing.is_none());
        assert!(run.fallback_suspicion.is_none());
        assert!(run.artifact_directory.is_none());
        assert!(run.generator_provenance.is_none());
        assert!(run.adaptation_readiness.is_none());
        assert!(run.warnings.is_empty());
    }

    #[test]
    fn test_lab_run_builder_all_fields() {
        let run = make_builder()
            .payload_hash("hash123".to_string())
            .compile_result(CompileStepResult {
                success: true,
                error: None,
                output_path: Some("/tmp/out".to_string()),
                content_hash: Some("sha256:xxx".to_string()),
                file_count: Some(5),
                coremltools_version: Some("9.0".to_string()),
            })
            .inspect_result(InspectionStepResult {
                package_present: true,
                manifest_readable: true,
                model_loadable: false,
                model_load_failure_reason: Some("no runtime".to_string()),
                function_count: Some(1),
                input_specs: vec![],
                output_specs: vec![],
                warnings: vec![],
                structure_inspection_available: None,
                structure_inspection_failure_reason: None,
                structure_op_names: vec![],
                structure_op_count: None,
                structure_function_count: None,
                structure_state_declarations: vec![],
                op_fidelity_score: None,
                missing_ops: vec![],
                extra_ops: vec![],
                inspection_method: "none".to_string(),
            })
            .timing(TimingResult {
                warmup_iterations: 3,
                measured_iterations: 10,
                p50_ms: 1.0,
                p90_ms: 1.5,
                p99_ms: 2.0,
                min_ms: 0.8,
                max_ms: 2.5,
                mean_ms: 1.1,
                std_dev_ms: 0.3,
                compute_units: "CPU_AND_NE".to_string(),
                scope_note: "test".to_string(),
            })
            .fallback_suspicion(FallbackSuspicionResult {
                suspicion_level: FallbackSuspicionLevel::NoConclusion,
                explanation: "no evidence".to_string(),
                evidence: vec![],
            })
            .warning("watch out".to_string())
            .warning("another warning".to_string())
            .artifact_directory("/tmp/artifacts".to_string())
            .generator_provenance(GeneratorProvenance {
                generator_version: "1.0".to_string(),
                family: "LinearProjection".to_string(),
                seed: 42,
                task_name: "gen_task".to_string(),
            })
            .adaptation_readiness("artifacts_and_observation".to_string())
            .build();
        assert_eq!(run.payload_hash, Some("hash123".to_string()));
        assert!(run.compile_result.success);
        assert!(run.timing.is_some());
        assert!(run.fallback_suspicion.is_some());
        assert_eq!(run.warnings.len(), 2);
        assert_eq!(run.artifact_directory, Some("/tmp/artifacts".to_string()));
        assert!(run.generator_provenance.is_some());
        assert_eq!(run.adaptation_readiness, Some("artifacts_and_observation".to_string()));
    }

    #[test]
    fn test_lab_run_builder_payload_hash() {
        let run = make_builder().payload_hash("phash".to_string()).build();
        assert_eq!(run.payload_hash, Some("phash".to_string()));
    }

    #[test]
    fn test_lab_run_builder_compile_result() {
        let cr = CompileStepResult {
            success: true,
            error: None,
            output_path: None,
            content_hash: None,
            file_count: None,
            coremltools_version: None,
        };
        let run = make_builder().compile_result(cr.clone()).build();
        assert!(run.compile_result.success);
    }

    #[test]
    fn test_lab_run_builder_inspect_result() {
        let ir = InspectionStepResult {
            package_present: true,
            manifest_readable: false,
            model_loadable: false,
            model_load_failure_reason: None,
            function_count: None,
            input_specs: vec![],
            output_specs: vec![],
            warnings: vec!["w1".to_string()],
            structure_inspection_available: None,
            structure_inspection_failure_reason: None,
            structure_op_names: vec![],
            structure_op_count: None,
            structure_function_count: None,
            structure_state_declarations: vec![],
            op_fidelity_score: None,
            missing_ops: vec![],
            extra_ops: vec![],
            inspection_method: "none".to_string(),
        };
        let run = make_builder().inspect_result(ir).build();
        assert!(run.inspect_result.package_present);
    }

    #[test]
    fn test_lab_run_builder_timing() {
        let t = TimingResult {
            warmup_iterations: 1,
            measured_iterations: 5,
            p50_ms: 1.0,
            p90_ms: 1.0,
            p99_ms: 1.0,
            min_ms: 1.0,
            max_ms: 1.0,
            mean_ms: 1.0,
            std_dev_ms: 0.0,
            compute_units: "CPU_ONLY".to_string(),
            scope_note: "test".to_string(),
        };
        let run = make_builder().timing(t).build();
        assert!(run.timing.is_some());
        assert_eq!(run.timing.as_ref().unwrap().p50_ms, 1.0);
    }

    #[test]
    fn test_lab_run_builder_fallback_suspicion() {
        let fs = FallbackSuspicionResult {
            suspicion_level: FallbackSuspicionLevel::LowConfidenceSuspicion,
            explanation: "slow".to_string(),
            evidence: vec![],
        };
        let run = make_builder().fallback_suspicion(fs).build();
        assert!(run.fallback_suspicion.is_some());
        assert_eq!(
            run.fallback_suspicion.as_ref().unwrap().suspicion_level,
            FallbackSuspicionLevel::LowConfidenceSuspicion
        );
    }

    #[test]
    fn test_lab_run_builder_warnings() {
        let run = make_builder()
            .warning("w1".to_string())
            .warning("w2".to_string())
            .warning("w3".to_string())
            .build();
        assert_eq!(run.warnings, vec!["w1", "w2", "w3"]);
    }

    #[test]
    fn test_lab_run_builder_generator_provenance() {
        let gp = GeneratorProvenance {
            generator_version: "2.0".to_string(),
            family: "MlpBlock".to_string(),
            seed: 99,
            task_name: "gen_mlp".to_string(),
        };
        let run = make_builder().generator_provenance(gp).build();
        let prov = run.generator_provenance.unwrap();
        assert_eq!(prov.generator_version, "2.0");
        assert_eq!(prov.family, "MlpBlock");
        assert_eq!(prov.seed, 99);
        assert_eq!(prov.task_name, "gen_mlp");
    }

    #[test]
    fn test_lab_run_builder_adaptation_readiness() {
        let run = make_builder().adaptation_readiness("artifacts_only".to_string()).build();
        assert_eq!(run.adaptation_readiness, Some("artifacts_only".to_string()));
    }

    #[test]
    fn test_lab_run_completed_at_set() {
        let run = make_builder().build();
        assert!(run.completed_at.is_some(), "build() must set completed_at");
        assert!(!run.completed_at.unwrap().is_empty());
    }

    #[test]
    fn test_lab_run_to_json() {
        let run = make_builder().build();
        let json = run.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["run_id"], "run_123");
        assert_eq!(parsed["task_name"], "test_task");
    }

    #[test]
    fn test_lab_run_json_roundtrip() {
        let run = make_builder()
            .payload_hash("ph".to_string())
            .warning("test warning".to_string())
            .build();
        let json = run.to_json().unwrap();
        let back: LabRun = serde_json::from_str(&json).unwrap();
        assert_eq!(back.run_id, run.run_id);
        assert_eq!(back.task_id, run.task_id);
        assert_eq!(back.task_name, run.task_name);
        assert_eq!(back.verification_scope, run.verification_scope);
        assert_eq!(back.payload_hash, run.payload_hash);
        assert_eq!(back.warnings, run.warnings);
    }

    #[test]
    fn test_lab_run_write_to_file() {
        let run = make_builder().build();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("run.json");
        run.write_to_file(&path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["run_id"], "run_123");
    }

    #[test]
    fn test_verification_scope_serialization() {
        for scope in [
            VerificationScope::HostOnlyInspection,
            VerificationScope::HostRuntimeExecution,
            VerificationScope::DeviceBackedExecution,
        ] {
            let json = serde_json::to_string(&scope).unwrap();
            let back: VerificationScope = serde_json::from_str(&json).unwrap();
            assert_eq!(scope, back);
        }
    }

    #[test]
    fn test_environment_summary_detect() {
        let env = EnvironmentSummary::detect(1);
        assert!(!env.host_os.is_empty());
        // On non-Apple CI, coreml_runtime_available should be false
        assert!(!env.coreml_runtime_available || cfg!(target_vendor = "apple"));
    }

    #[test]
    fn test_environment_summary_bridge_version() {
        let env = EnvironmentSummary::detect(42);
        assert_eq!(env.bridge_version, 42);
    }

    #[test]
    fn test_compile_step_result_serialization() {
        let r = CompileStepResult {
            success: true,
            error: Some("err".to_string()),
            output_path: Some("/tmp/out".to_string()),
            content_hash: Some("sha256:ab".to_string()),
            file_count: Some(3),
            coremltools_version: Some("8.0".to_string()),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: CompileStepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.success, r.success);
        assert_eq!(back.error, r.error);
        assert_eq!(back.file_count, r.file_count);
    }

    #[test]
    fn test_inspection_step_result_serialization() {
        let r = InspectionStepResult {
            package_present: true,
            manifest_readable: true,
            model_loadable: false,
            model_load_failure_reason: None,
            function_count: Some(2),
            input_specs: vec![],
            output_specs: vec![],
            warnings: vec!["w".to_string()],
            structure_inspection_available: Some(true),
            structure_inspection_failure_reason: None,
            structure_op_names: vec!["linear".to_string()],
            structure_op_count: Some(1),
            structure_function_count: Some(1),
            structure_state_declarations: vec![],
            op_fidelity_score: Some(0.95),
            missing_ops: vec![],
            extra_ops: vec![],
            inspection_method: "mlmodel_structure".to_string(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: InspectionStepResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.package_present, r.package_present);
        assert_eq!(back.structure_op_count, r.structure_op_count);
    }

    #[test]
    fn test_timing_result_serialization() {
        let r = TimingResult {
            warmup_iterations: 5,
            measured_iterations: 20,
            p50_ms: 1.0,
            p90_ms: 1.5,
            p99_ms: 2.0,
            min_ms: 0.8,
            max_ms: 2.5,
            mean_ms: 1.1,
            std_dev_ms: 0.3,
            compute_units: "CPU_AND_NE".to_string(),
            scope_note: "test timing".to_string(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: TimingResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.p50_ms, r.p50_ms);
        assert_eq!(back.compute_units, r.compute_units);
    }

    #[test]
    fn test_fallback_suspicion_result_serialization() {
        let r = FallbackSuspicionResult {
            suspicion_level: FallbackSuspicionLevel::LowConfidenceSuspicion,
            explanation: "slow".to_string(),
            evidence: vec![SuspicionEvidence {
                kind: "latency_anomaly".to_string(),
                description: "observed 5x expected".to_string(),
                strength: 0.4,
            }],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: FallbackSuspicionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.suspicion_level, r.suspicion_level);
        assert_eq!(back.evidence.len(), 1);
        assert_eq!(back.evidence[0].kind, "latency_anomaly");
    }

    #[test]
    fn test_generator_provenance_serialization() {
        let gp = GeneratorProvenance {
            generator_version: "1.0".to_string(),
            family: "Attention".to_string(),
            seed: 7,
            task_name: "attn_7".to_string(),
        };
        let json = serde_json::to_string(&gp).unwrap();
        let back: GeneratorProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(back.generator_version, gp.generator_version);
        assert_eq!(back.seed, gp.seed);
    }

    #[test]
    fn test_lab_run_schema_version() {
        assert_eq!(LAB_RUN_SCHEMA_VERSION, "1.0.0");
    }

    #[test]
    fn test_lab_run_builder_chained() {
        let run = make_builder()
            .payload_hash("a".to_string())
            .warning("b".to_string())
            .adaptation_readiness("c".to_string())
            .build();
        assert_eq!(run.payload_hash, Some("a".to_string()));
        assert_eq!(run.warnings, vec!["b"]);
        assert_eq!(run.adaptation_readiness, Some("c".to_string()));
    }
}
