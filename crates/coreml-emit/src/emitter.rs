//! Proto-Direct Emitter
//!
//! The high-level emitter that converts a CoreMlModel into an mlpackage
//! on disk. This is the main entry point for proto-direct emission,
//! replacing the Python bridge for Core ML model construction.

use crate::package::{MlPackageResult, MlPackageWriter};
use ane_coreml_proto::{mir_compat::MirGraphCompat, CoreMlComputeUnit, CoreMlModel, SpecVersion};
use anyhow::Result;

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
}

impl Default for ProtoEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtoEmitter {
    /// Create a new proto-direct emitter targeting the latest spec version.
    pub fn new() -> Self {
        Self { spec_version: SpecVersion::V8, compute_unit: CoreMlComputeUnit::CpuAndNe }
    }

    /// Create an emitter targeting a specific spec version.
    pub fn with_spec_version(spec_version: SpecVersion) -> Self {
        Self { spec_version, compute_unit: CoreMlComputeUnit::CpuAndNe }
    }

    /// Create an emitter with a specific compute unit preference.
    pub fn with_compute_unit(compute_unit: CoreMlComputeUnit) -> Self {
        Self { spec_version: SpecVersion::V8, compute_unit }
    }

    /// Emit a single-function MIR graph as an mlpackage.
    ///
    /// This is the simplest emission path: one function, one graph.
    pub fn emit_mir_graph(
        &self,
        graph: &MirGraphCompat,
        output_path: &str,
    ) -> Result<ProtoEmitResult> {
        let model =
            crate::mir_to_proto::convert_mir_to_proto(graph, self.spec_version, self.compute_unit)?;
        self.emit_model(&model, output_path)
    }

    /// Emit a multi-function model with shared weights.
    ///
    /// This is the key use case for proto-direct emission: multiple
    /// functions can share weight tensors, producing a smaller mlpackage
    /// than coremltools 9.0's `add_function()` which duplicates constants.
    pub fn emit_multifunction_with_shared_weights(
        &self,
        graphs: &[MirGraphCompat],
        shared_weight_names: &[String],
        output_path: &str,
    ) -> Result<ProtoEmitResult> {
        let model = crate::mir_to_proto::convert_mir_to_proto_multifunction(
            graphs,
            shared_weight_names,
            self.spec_version,
            self.compute_unit,
        )?;
        self.emit_model(&model, output_path)
    }

    /// Emit a CoreMlModel as an mlpackage.
    fn emit_model(&self, model: &CoreMlModel, output_path: &str) -> Result<ProtoEmitResult> {
        let package_result = MlPackageWriter::write(model, output_path)?;

        Ok(ProtoEmitResult {
            package_result,
            emission_method: "proto-direct".to_string(),
            weight_sharing_used: !model.shared_weights.is_empty(),
            shared_weight_count: model.shared_weights.len(),
            function_names: model.functions.iter().map(|f| f.name.clone()).collect(),
        })
    }

    /// Compare proto-direct emission with Python bridge emission.
    ///
    /// This emits the same model both ways and compares:
    /// - weight.bin size (proto-direct should be smaller with shared weights)
    /// - Structural equivalence (both should produce valid mlpackages)
    /// - Function count and op counts
    ///
    /// Returns a comparison report.
    pub fn compare_with_python_bridge(
        &self,
        graph: &MirGraphCompat,
        proto_output_path: &str,
        _python_output_path: &str,
    ) -> Result<ComparisonReport> {
        // Emit via proto-direct
        let proto_result = self.emit_mir_graph(graph, proto_output_path)?;

        // Python bridge comparison would go here when the bridge is available.
        // For now, we report the proto-direct result only.

        Ok(ComparisonReport {
            proto_direct_result: proto_result,
            python_bridge_result: None,
            weight_bin_comparison: None,
            structural_equivalence: None,
        })
    }
}

/// Report comparing proto-direct and Python bridge emission.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Proto-direct emission result.
    pub proto_direct_result: ProtoEmitResult,
    /// Python bridge emission result (if available).
    pub python_bridge_result: Option<ProtoEmitResult>,
    /// Weight.bin size comparison.
    pub weight_bin_comparison: Option<WeightBinComparison>,
    /// Whether both paths produce structurally equivalent mlpackages.
    pub structural_equivalence: Option<bool>,
}

/// Comparison of weight.bin sizes between emission paths.
#[derive(Debug, Clone)]
pub struct WeightBinComparison {
    /// Size of proto-direct weight.bin.
    pub proto_direct_size: u64,
    /// Size of Python bridge weight.bin.
    pub python_bridge_size: u64,
    /// Whether proto-direct is smaller.
    pub proto_direct_is_smaller: bool,
    /// Bytes saved.
    pub bytes_saved: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_emitter_creation() {
        let emitter = ProtoEmitter::new();
        assert!(matches!(emitter.spec_version, SpecVersion::V8));
        assert!(matches!(emitter.compute_unit, CoreMlComputeUnit::CpuAndNe));
    }

    #[test]
    fn test_proto_emitter_custom_spec() {
        let emitter = ProtoEmitter::with_spec_version(SpecVersion::V7);
        assert!(matches!(emitter.spec_version, SpecVersion::V7));
    }
}
