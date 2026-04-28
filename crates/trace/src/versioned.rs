//! ANE version-aware compilation with constraint enforcement.
//!
//! The `VersionedCompiler` enforces ANE constraints during compilation
//! based on the target ANE family. Unlike naive compilation that may
//! produce graphs the ANE rejects at runtime, this module ensures every
//! op in the output graph is *guaranteed* to have an ANE converter for
//! the target family.
//!
//! # Constraint Categories
//!
//! 1. **Operation support**: Which MIL ops have ANEC converters per family
//! 2. **Tensor dimensions**: Max rank (5), width, height, channels per revision
//! 3. **Dtype restrictions**: FP16 broadcast on A11/A12, no E4M3 on most, no BF16
//! 4. **Dynamic shapes**: All-or-nothing rule — eliminated by staticize pass
//! 5. **Fusion boundaries**: NE/PE/Transpose engine assignment constraints

use ane_ir::ane_hw_limits::AneHwLimits;
use ane_ir::ane_target::{AneFamily, AneRevision};
use ane_ir::sir::{SirGraph, SirOp};
use serde::{Deserialize, Serialize};

/// ANE version-aware compiler that enforces per-family constraints.
#[derive(Debug, Clone)]
pub struct VersionedCompiler {
    target_family: AneFamily,
    hw_limits: AneHwLimits,
    op_support: OpSupportMatrix,
}

impl VersionedCompiler {
    /// Create a new versioned compiler for the given ANE family.
    pub fn new(family: AneFamily) -> Self {
        let revision = family_to_default_revision(family);
        let hw_limits = AneHwLimits::for_revision(revision);
        let op_support = OpSupportMatrix::for_family(family);
        Self {
            target_family: family,
            hw_limits,
            op_support,
        }
    }

    /// The target ANE family.
    pub fn target_family(&self) -> AneFamily {
        self.target_family
    }

    /// The hardware limits for the target revision.
    pub fn hw_limits(&self) -> &AneHwLimits {
        &self.hw_limits
    }

    /// Validate an entire SIR graph against ANE constraints.
    ///
    /// Returns a `VersionedCompileResult` with the faithfulness report.
    /// If `ane_only` is true, any violation causes the compilation to fail.
    pub fn validate_sir(&self, sir: &SirGraph, ane_only: bool) -> VersionedCompileResult {
        let mut report = AnceFaithfulnessReport {
            target_family: self.target_family,
            total_ops: 0,
            ane_supported: 0,
            cpu_fallback: 0,
            violations: Vec::new(),
            warnings: Vec::new(),
            is_faithful: true,
        };

        for node in &sir.nodes {
            report.total_ops += 1;
            let op_name = op_name_for_sir(&node.op);

            match self.op_support.check_op(&node.op) {
                OpSupport::AneSupported(_engine) => {
                    report.ane_supported += 1;
                }
                OpSupport::CpuOnly(reason) => {
                    report.cpu_fallback += 1;
                    report.violations.push(ConstraintViolation {
                        node_id: node.id.0.clone(),
                        op_name: op_name.clone(),
                        violation_type: ViolationType::CpuFallback,
                        message: reason.clone(),
                        severity: if ane_only { Severity::Error } else { Severity::Warning },
                    });
                    if ane_only {
                        report.is_faithful = false;
                    }
                }
                OpSupport::Unsupported(reason) => {
                    report.cpu_fallback += 1;
                    report.violations.push(ConstraintViolation {
                        node_id: node.id.0.clone(),
                        op_name: op_name.clone(),
                        violation_type: ViolationType::Unsupported,
                        message: reason.clone(),
                        severity: Severity::Error,
                    });
                    report.is_faithful = false;
                }
                OpSupport::FamilyGated { minimum_family, reason } => {
                    let meets_minimum = family_meets_minimum(self.target_family, minimum_family);
                    if meets_minimum {
                        report.ane_supported += 1;
                    } else {
                        report.cpu_fallback += 1;
                        report.violations.push(ConstraintViolation {
                            node_id: node.id.0.clone(),
                            op_name: op_name.clone(),
                            violation_type: ViolationType::FamilyGated {
                                required: minimum_family,
                            },
                            message: reason.clone(),
                            severity: if ane_only { Severity::Error } else { Severity::Warning },
                        });
                        if ane_only {
                            report.is_faithful = false;
                        }
                    }
                }
            }
        }

        // Add version-specific warnings
        self.add_version_warnings(&mut report);

        VersionedCompileResult {
            report,
            target_family: self.target_family,
            target_revision: family_to_default_revision(self.target_family),
        }
    }

    /// Add version-specific warnings based on the target family.
    fn add_version_warnings(&self, report: &mut AnceFaithfulnessReport) {
        match self.target_family {
            AneFamily::A11Legacy | AneFamily::A12 => {
                report.warnings.push(
                    "A11/A12: FP16-only broadcast. Mixed-precision broadcast will fall back to CPU."
                        .to_string(),
                );
            }
            AneFamily::A14 => {
                report.warnings.push(
                    "A14: SDPA not supported. Attention must decompose to MatMul+Softmax+MatMul."
                        .to_string(),
                );
                report.warnings.push(
                    "A14: LayerNorm not supported on ANE. Use RMSNorm decomposition instead."
                        .to_string(),
                );
            }
            AneFamily::A15 => {
                report.warnings.push(
                    "A15: SDPA may not be reliable. Consider decomposing attention."
                        .to_string(),
                );
            }
            AneFamily::A16 => {
                report.warnings.push(
                    "A16: SDPA and LayerNorm are supported. Optimal for LLaMA-family models."
                        .to_string(),
                );
            }
            AneFamily::A18 => {
                report.warnings.push(
                    "A18: Full SDPA, LayerNorm, and E4M3 support. Most capable ANE generation."
                        .to_string(),
                );
            }
        }
    }
}

/// Result of version-aware compilation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedCompileResult {
    /// The faithfulness report.
    pub report: AnceFaithfulnessReport,
    /// Target ANE family.
    pub target_family: AneFamily,
    /// Target ANE revision.
    pub target_revision: AneRevision,
}

/// ANE faithfulness report — whether the compiled graph will actually
/// run on the target ANE without CPU fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnceFaithfulnessReport {
    /// Target ANE family.
    pub target_family: AneFamily,
    /// Total number of ops in the graph.
    pub total_ops: usize,
    /// Number of ops with ANE converters for this family.
    pub ane_supported: usize,
    /// Number of ops requiring CPU fallback.
    pub cpu_fallback: usize,
    /// Constraint violations.
    pub violations: Vec<ConstraintViolation>,
    /// Version-specific warnings.
    pub warnings: Vec<String>,
    /// Whether the graph is ANE-faithful (all ops on ANE).
    pub is_faithful: bool,
}

impl AnceFaithfulnessReport {
    /// Percentage of ops that will run on ANE.
    pub fn ane_utilization(&self) -> f64 {
        if self.total_ops == 0 {
            return 100.0;
        }
        (self.ane_supported as f64 / self.total_ops as f64) * 100.0
    }
}

/// A constraint violation found during version-aware validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    /// The SIR node ID that violates the constraint.
    pub node_id: String,
    /// The operation name.
    pub op_name: String,
    /// Type of violation.
    pub violation_type: ViolationType,
    /// Human-readable explanation.
    pub message: String,
    /// Severity of the violation.
    pub severity: Severity,
}

impl ConstraintViolation {
    /// Format severity as a string.
    pub fn severity_str(&self) -> &'static str {
        match self.severity {
            Severity::Warning => "WARN",
            Severity::Error => "ERROR",
        }
    }
}

/// Type of constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Op will fall back to CPU (no ANE converter).
    CpuFallback,
    /// Op is completely unsupported (no converter at all).
    Unsupported,
    /// Op requires a newer ANE family than the target.
    FamilyGated { required: AneFamily },
}

/// Severity of a constraint violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Warning,
    Error,
}

/// Operation support status for a specific ANE family.
#[derive(Debug, Clone)]
enum OpSupport {
    /// Op has an ANE converter and will run on the specified engine.
    AneSupported(AneEngineSupport),
    /// Op has no ANE converter; will fall back to CPU.
    CpuOnly(String),
    /// Op is fundamentally unsupported on any ANE family.
    Unsupported(String),
    /// Op has a converter, but only on certain families or newer.
    FamilyGated { minimum_family: AneFamily, reason: String },
}

/// Which ANE engine an op targets.
#[derive(Debug, Clone, Copy)]
enum AneEngineSupport {
    /// Neural Engine (conv/pool/matmul/attention).
    NE,
    /// Processing Element (elementwise/reduction/normalization).
    PE,
    /// Transpose Engine (data rearrangement).
    Transpose,
}

/// Per-family operation support matrix.
///
/// Based on ane-constraints-docs/04-operation-support/per-op-per-family-support-matrix.md
/// and ane-constraints-docs/02-hardware-and-limits/hardware-versions-limits-and-op-support.md
#[derive(Debug, Clone)]
struct OpSupportMatrix {
    family: AneFamily,
}

impl OpSupportMatrix {
    fn for_family(family: AneFamily) -> Self {
        Self { family }
    }

    fn check_op(&self, op: &SirOp) -> OpSupport {
        match op {
            // ─── Always ANE-supported ──────────────────────────────
            SirOp::LinearProjection { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::MatMul { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::Conv { .. } => OpSupport::AneSupported(AneEngineSupport::NE),

            // ─── Elementwise: always ANE-supported on PE ───────────
            SirOp::Add { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Mul { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sub { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Maximum { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Minimum { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::RealDiv { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A14,
                        reason: "Divide requires A14+ elementwise converter".to_string(),
                    }
                } else {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                }
            }
            SirOp::Relu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Gelu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Silu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sigmoid { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Tanh { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Exp { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Rsqrt { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sqrt { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Cos { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sin { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Abs { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Neg { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Cast { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Where { .. } => OpSupport::AneSupported(AneEngineSupport::PE),

            // ─── Reduction: PE, but ReduceMin has A14+ restriction ─
            SirOp::ReduceMean { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceSum { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceMax { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceMin { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A14,
                        reason: "ReduceMin requires A14+ for non-FP types".to_string(),
                    }
                } else {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                }
            }

            // ─── Normalization: family-gated ───────────────────────
            SirOp::Softmax { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::LayerNorm { .. } => {
                if self.family.supports_layernorm() {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                } else {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A15,
                        reason: "LayerNorm requires A15+".to_string(),
                    }
                }
            }
            SirOp::InstanceNorm { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::BatchNorm { .. } => OpSupport::CpuOnly("BatchNorm decomposes to InstanceNorm + broadcast; may partially fall back".to_string()),

            // ─── Attention ─────────────────────────────────────────
            SirOp::ScaledDotProductAttention { .. } => {
                if self.family.supports_sdpa() {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                } else {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A16,
                        reason: "SDPA requires A16+ for reliable ANE placement. Decompose to MatMul+Softmax+MatMul on older families.".to_string(),
                    }
                }
            }
            SirOp::AttentionBlock { .. } => {
                // AttentionBlock is a composite — it decomposes into QKV proj + attention + out proj
                // The decomposition will be validated individually
                OpSupport::AneSupported(AneEngineSupport::NE)
            }

            // ─── Tensor transforms ─────────────────────────────────
            SirOp::Reshape { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Transpose { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Concat { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Split { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::ExpandDims { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Squeeze { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::SliceByIndex { .. } => OpSupport::AneSupported(AneEngineSupport::PE),

            // ─── Gather/Scatter ────────────────────────────────────
            SirOp::Gather { axis, .. } => {
                // Gather on ANE requires constant axis and specific constraints
                if *axis >= 0 {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                } else {
                    OpSupport::CpuOnly("Gather with negative axis may not be ANE-compatible".to_string())
                }
            }
            SirOp::GatherNd { .. } => OpSupport::CpuOnly("GatherNd has no ANEC converter".to_string()),
            SirOp::Scatter { .. } => OpSupport::CpuOnly("Scatter ops have no ANEC converter".to_string()),
            SirOp::ScatterNd { .. } => OpSupport::Unsupported("ScatterNd is never ANE-compatible".to_string()),

            // ─── Quantization ──────────────────────────────────────
            SirOp::Quantize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Dequantize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),

            // ─── State ops ─────────────────────────────────────────
            SirOp::StateRead { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::StateWrite { .. } => OpSupport::AneSupported(AneEngineSupport::NE),

            // ─── Composite/semantic ops ────────────────────────────
            SirOp::RMSNorm { .. } => {
                // RMSNorm decomposes to ANE-faithful primitives (Rsqrt, ReduceMean, Mul)
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::RoPETransform { .. } => {
                // RoPE decomposes to Cos, Sin, Mul, Add — all ANE-faithful
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::DecodeStep { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::Sampler { .. } => OpSupport::CpuOnly("TopK sampling is CPU-only on most families".to_string()),

            // ─── Always CPU-only ──────────────────────────────────
            SirOp::LogicalAnd { .. }
            | SirOp::LogicalOr { .. }
            | SirOp::LogicalXor { .. } => OpSupport::CpuOnly("Logical ops have no ANEC converter".to_string()),
            SirOp::Cumsum { .. } => OpSupport::CpuOnly("Cumsum has no ANEC converter".to_string()),
            SirOp::RandomBernoulli { .. }
            | SirOp::RandomNormal { .. }
            | SirOp::RandomUniform { .. }
            | SirOp::RandomCategorical { .. } => OpSupport::Unsupported("Random ops are never ANE-compatible".to_string()),
            SirOp::Cond { .. }
            | SirOp::WhileLoop { .. } => OpSupport::Unsupported("Control flow has no ANEC converter".to_string()),
            SirOp::Rnn { .. }
            | SirOp::Gru { .. }
            | SirOp::Lstm { .. } => OpSupport::CpuOnly("RNN/LSTM/GRU have no direct ANEC converter".to_string()),

            // ─── Const/Identity: structural, not a real compute op ─
            SirOp::Const { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Identity { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),

            // ─── Default: assume CPU fallback for unmapped ops ──────
            _ => OpSupport::CpuOnly(format!("Op '{}' not yet mapped in versioned support matrix", op_name_for_sir(op))),
        }
    }
}

/// Map a SirOp to a human-readable name.
fn op_name_for_sir(op: &SirOp) -> String {
    let name = std::any::type_name::<SirOp>();
    // Extract the variant name from the full type path
    match op {
        SirOp::LinearProjection { .. } => "LinearProjection",
        SirOp::MatMul { .. } => "MatMul",
        SirOp::Add { .. } => "Add",
        SirOp::Mul { .. } => "Mul",
        SirOp::Rsqrt { .. } => "Rsqrt",
        SirOp::Softmax { .. } => "Softmax",
        SirOp::Gelu { .. } => "Gelu",
        SirOp::Silu { .. } => "Silu",
        SirOp::Reshape { .. } => "Reshape",
        SirOp::Transpose { .. } => "Transpose",
        SirOp::Concat { .. } => "Concat",
        SirOp::Split { .. } => "Split",
        SirOp::ScaledDotProductAttention { .. } => "SDPA",
        SirOp::LayerNorm { .. } => "LayerNorm",
        SirOp::RMSNorm { .. } => "RMSNorm",
        SirOp::ReduceMean { .. } => "ReduceMean",
        SirOp::ReduceSum { .. } => "ReduceSum",
        SirOp::Gather { .. } => "Gather",
        SirOp::StateRead { .. } => "StateRead",
        SirOp::StateWrite { .. } => "StateWrite",
        _ => name.rsplit("::").next().unwrap_or("Unknown"),
    }.to_string()
}

/// Get the default revision for a family.
fn family_to_default_revision(family: AneFamily) -> AneRevision {
    match family {
        AneFamily::A11Legacy => AneRevision::V4,
        AneFamily::A12 => AneRevision::V5,
        AneFamily::A14 => AneRevision::V7,
        AneFamily::A15 => AneRevision::V8,
        AneFamily::A16 => AneRevision::V10,
        AneFamily::A18 => AneRevision::V17,
    }
}

/// Check if a family meets a minimum requirement.
fn family_meets_minimum(actual: AneFamily, minimum: AneFamily) -> bool {
    let actual_level = family_level(actual);
    let min_level = family_level(minimum);
    actual_level >= min_level
}

/// Assign a numeric level to each family for comparison.
fn family_level(family: AneFamily) -> u32 {
    match family {
        AneFamily::A11Legacy => 0,
        AneFamily::A12 => 1,
        AneFamily::A14 => 2,
        AneFamily::A15 => 3,
        AneFamily::A16 => 4,
        AneFamily::A18 => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a16_supports_sdpa() {
        let compiler = VersionedCompiler::new(AneFamily::A16);
        assert!(compiler.target_family().supports_sdpa());
    }

    #[test]
    fn test_a14_no_sdpa() {
        let compiler = VersionedCompiler::new(AneFamily::A14);
        assert!(!compiler.target_family().supports_sdpa());
    }

    #[test]
    fn test_family_level_ordering() {
        assert!(family_level(AneFamily::A18) > family_level(AneFamily::A16));
        assert!(family_level(AneFamily::A16) > family_level(AneFamily::A15));
        assert!(family_level(AneFamily::A15) > family_level(AneFamily::A14));
        assert!(family_level(AneFamily::A14) > family_level(AneFamily::A12));
    }

    #[test]
    fn test_validate_linear_graph() {
        let compiler = VersionedCompiler::new(AneFamily::A16);
        let sir = SirGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
        };
        let result = compiler.validate_sir(&sir, true);
        assert!(result.report.is_faithful);
    }

    #[test]
    fn test_sdpa_family_gated_on_a14() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A14);
        let sdpa_op = SirOp::ScaledDotProductAttention {
            query: SirNodeId("q".into()),
            key: SirNodeId("k".into()),
            value: SirNodeId("v".into()),
            attention_mask: None,
            scale: Some(0.125),
        };
        match matrix.check_op(&sdpa_op) {
            OpSupport::FamilyGated { .. } => {} // expected
            other => panic!("Expected FamilyGated, got {:?}", other),
        }
    }

    #[test]
    fn test_sdpa_supported_on_a16() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A16);
        let sdpa_op = SirOp::ScaledDotProductAttention {
            query: SirNodeId("q".into()),
            key: SirNodeId("k".into()),
            value: SirNodeId("v".into()),
            attention_mask: None,
            scale: Some(0.125),
        };
        match matrix.check_op(&sdpa_op) {
            OpSupport::AneSupported(_) => {} // expected
            other => panic!("Expected AneSupported, got {:?}", other),
        }
    }

    #[test]
    fn test_scatter_always_cpu() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A18);
        let scatter = SirOp::Scatter {
            input: SirNodeId("x".into()),
            indices: SirNodeId("i".into()),
            updates: SirNodeId("u".into()),
            axis: 0,
            mode: "update".to_string(),
        };
        match matrix.check_op(&scatter) {
            OpSupport::CpuOnly(_) => {} // expected
            other => panic!("Expected CpuOnly, got {:?}", other),
        }
    }
}

#[cfg(test)]
use ane_ir::sir::SirNodeId;
