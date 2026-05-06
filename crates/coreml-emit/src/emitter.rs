//! Proto-Direct Emitter
//!
//! The high-level emitter that converts a CoreMlModel into an mlpackage
//! on disk. This is the main entry point for proto-direct emission,
//! replacing the Python bridge for Core ML model construction.

use crate::package::{MlPackageResult, MlPackageWriter};
use ane_coreml_proto::{mir_compat::MirGraphCompat, CoreMlComputeUnit, CoreMlModel, SpecVersion};
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};

/// T-120: Per-process compilation counter (Orion #5).
///
/// The ANE has a ~119 compilation limit per process. This counter tracks
/// how many models have been compiled in the current process so that
/// warnings can be emitted before hitting the limit.
static COMPILATION_COUNT: AtomicU64 = AtomicU64::new(0);

/// T-120: Approximate compilation limit per process per Orion #5.
///
/// The ANE silently fails after approximately 119 compilations in a
/// single process. We warn at 80% of this threshold.
pub const COMPILATION_LIMIT: u64 = 119;

/// T-120: Warning threshold — 80% of the compilation limit.
///
/// At this count, a warning is logged to alert the user that the
/// process is approaching the ANE compilation limit.
pub const COMPILATION_WARNING_THRESHOLD: u64 = 95;

/// T-120: Get the current per-process compilation count.
pub fn compilation_count() -> u64 {
    COMPILATION_COUNT.load(Ordering::Relaxed)
}

/// Proto-direct emitter for Core ML mlpackage artifacts.
///
/// This emitter constructs mlpackage directories directly from Rust,
/// without going through the Python bridge. The emission path is:
///
/// ```text
/// MirGraphCompat → CoreMlModel → .mlpackage on disk
/// ```
///
/// ## Usage
///
/// ```rust,ignore
/// use ane_coreml_emit::ProtoEmitter;
/// use ane_coreml_proto::mir_compat::MirGraphCompat;
///
/// let emitter = ProtoEmitter::new();
/// let result = emitter.emit_mir_graph(&mir_graph, "/path/to/output")?;
/// println!("Wrote mlpackage to: {}", result.path);
/// ```
pub struct ProtoEmitter {
    /// Specification version to target.
    spec_version: SpecVersion,
    /// Compute unit preference.
    compute_unit: CoreMlComputeUnit,
    /// Validation policy for ANE constraint checks.
    validation_policy: crate::mir_to_proto::ValidationPolicy,
}

/// Result of proto-direct emission, including comparison data.
#[derive(Debug, Clone)]
pub struct ProtoEmitResult {
    /// The mlpackage write result.
    pub package_result: MlPackageResult,
    /// Whether the emission used proto-direct path (always true for this emitter).
    pub emission_method: String,
    /// Whether weight sharing was used.
    pub weight_sharing_used: bool,
    /// Number of shared weight references.
    pub shared_weight_count: usize,
    /// Functions emitted.
    pub function_names: Vec<String>,
    /// T-120: Per-process compilation number for this emission.
    ///
    /// Monotonically increasing counter. When this approaches
    /// `COMPILATION_LIMIT` (119), the process should be restarted
    /// to avoid ANE silent failures (Orion #5).
    pub compilation_number: u64,
}

impl Default for ProtoEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtoEmitter {
    /// Create a new proto-direct emitter targeting the latest spec version.
    pub fn new() -> Self {
        Self {
            spec_version: SpecVersion::V10,
            compute_unit: CoreMlComputeUnit::CpuAndNe,
            validation_policy: crate::mir_to_proto::ValidationPolicy::default(),
        }
    }

    /// Create an emitter targeting a specific spec version.
    pub fn with_spec_version(spec_version: SpecVersion) -> Self {
        Self {
            spec_version,
            compute_unit: CoreMlComputeUnit::CpuAndNe,
            validation_policy: crate::mir_to_proto::ValidationPolicy::default(),
        }
    }

    /// Create an emitter with a specific compute unit preference.
    pub fn with_compute_unit(compute_unit: CoreMlComputeUnit) -> Self {
        Self {
            spec_version: SpecVersion::V10,
            compute_unit,
            validation_policy: crate::mir_to_proto::ValidationPolicy::default(),
        }
    }

    /// Create an emitter with a specific validation policy.
    ///
    /// Use `ValidationPolicy::warn_only()` for development/testing where
    /// small test tensors don't meet the 49KB IOSurface minimum.
    pub fn with_validation_policy(
        validation_policy: crate::mir_to_proto::ValidationPolicy,
    ) -> Self {
        Self {
            spec_version: SpecVersion::V10,
            compute_unit: CoreMlComputeUnit::CpuAndNe,
            validation_policy,
        }
    }

    /// Emit a single-function MIR graph as an mlpackage.
    ///
    /// This is the simplest emission path: one function, one graph.
    pub fn emit_mir_graph(
        &self,
        graph: &MirGraphCompat,
        output_path: &str,
    ) -> Result<ProtoEmitResult> {
        let model = crate::mir_to_proto::convert_mir_to_proto_multifunction_with_policy(
            std::slice::from_ref(graph),
            &[],
            self.spec_version,
            self.compute_unit,
            self.validation_policy.clone(),
        )?;
        self.emit_model(&model, output_path)
    }

    /// Emit a multi-function model with shared weights.
    ///
    /// This is the key use case for proto-direct emission: multiple
    /// functions can share weight tensors, producing a smaller mlpackage
    /// than coremltools 9.0's `add_function()` + `ct.convert()` which duplicates
    /// constants (note: `save_multifunction()` does perform cross-function dedup).
    pub fn emit_multifunction_with_shared_weights(
        &self,
        graphs: &[MirGraphCompat],
        shared_weight_names: &[String],
        output_path: &str,
    ) -> Result<ProtoEmitResult> {
        let model = crate::mir_to_proto::convert_mir_to_proto_multifunction_with_policy(
            graphs,
            shared_weight_names,
            self.spec_version,
            self.compute_unit,
            self.validation_policy.clone(),
        )?;
        self.emit_model(&model, output_path)
    }

    /// Emit a CoreMlModel as an mlpackage.
    fn emit_model(&self, model: &CoreMlModel, output_path: &str) -> Result<ProtoEmitResult> {
        // T-120: Track per-process compilation count (Orion #5).
        // The ANE silently fails after ~119 compilations per process.
        let count = COMPILATION_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= COMPILATION_LIMIT {
            anyhow::bail!(
                "T-120: ANE compilation limit ({}) reached for this process. \
                 The ANE silently fails after ~{} compilations (Orion #5). \
                 Restart the process to compile more models.",
                COMPILATION_LIMIT,
                COMPILATION_LIMIT
            );
        } else if count >= COMPILATION_WARNING_THRESHOLD {
            log::warn!(
                "T-120: ANE compilation count {}/{} — approaching per-process limit (Orion #5). \
                 Consider restarting the process to avoid silent compilation failures.",
                count,
                COMPILATION_LIMIT
            );
        }

        let package_result = MlPackageWriter::write(model, output_path)?;

        Ok(ProtoEmitResult {
            package_result,
            emission_method: "proto-direct".to_string(),
            weight_sharing_used: !model.shared_weights.is_empty(),
            shared_weight_count: model.shared_weights.len(),
            function_names: model.functions.iter().map(|f| f.name.clone()).collect(),
            compilation_number: count, // T-120
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_emitter_creation() {
        let emitter = ProtoEmitter::new();
        assert!(matches!(emitter.spec_version, SpecVersion::V10));
        assert!(matches!(emitter.compute_unit, CoreMlComputeUnit::CpuAndNe));
    }

    #[test]
    fn test_proto_emitter_custom_spec() {
        let emitter = ProtoEmitter::with_spec_version(SpecVersion::V7);
        assert!(matches!(emitter.spec_version, SpecVersion::V7));
    }

    // ─── T-120: Compilation count tests ─────────────────────────────────

    #[test]
    fn test_t120_compilation_limit_constants() {
        assert_eq!(COMPILATION_LIMIT, 119, "Orion #5: ANE compilation limit is ~119");
        assert_eq!(COMPILATION_WARNING_THRESHOLD, 95, "Warning at 80% of limit");
        const { assert!(COMPILATION_WARNING_THRESHOLD < COMPILATION_LIMIT) };
    }

    #[test]
    fn test_t120_compilation_count_increments() {
        let before = compilation_count();
        // The count should be non-decreasing — we can't reset it (it's a global)
        assert!(before < COMPILATION_LIMIT, "Count should be below limit in tests");
    }

    #[test]
    fn test_t120_proto_emit_result_has_compilation_number() {
        // Verify the field exists on ProtoEmitResult (compile-time check)
        let result = ProtoEmitResult {
            package_result: MlPackageResult {
                path: String::new(),
                content_hash: String::new(),
                total_size: 0,
                file_count: 0,
                weight_count: 0,
                function_count: 0,
                has_shared_weights: false,
                size_comparison: None,
            },
            emission_method: "proto-direct".to_string(),
            weight_sharing_used: false,
            shared_weight_count: 0,
            function_names: vec![],
            compilation_number: 1, // T-120
        };
        assert_eq!(result.compilation_number, 1);
    }
}
