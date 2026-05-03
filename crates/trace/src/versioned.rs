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
        Self { target_family: family, hw_limits, op_support }
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
                            violation_type: ViolationType::FamilyGated { required: minimum_family },
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
            AneFamily::A13 => {
                report.warnings.push(
                    "A13: Broadcast supports full dtype (unlike A11/A12), but uses A14Minus elementwise/reduction converters."
                        .to_string(),
                );
                report.warnings.push(
                    "A13: SDPA not supported. Attention must decompose to MatMul+Softmax+MatMul."
                        .to_string(),
                );
                report.warnings.push(
                    "A13: LayerNorm not supported on ANE. Use RMSNorm decomposition instead."
                        .to_string(),
                );
                report.warnings.push(
                    "A13: ReduceMin only supports FP types; non-FP ReduceMin falls back to CPU."
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
                    "A15: SDPA may not be reliable. Consider decomposing attention.".to_string(),
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
                report.warnings.push(
                    "A18: ArgMinMax (reduce_argmax/argmin) has no ANEC converter on LSE_7; falls back to CPU.".to_string(),
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
    ///
    /// Returns 0.0 for empty graphs (no ops) rather than the misleading
    /// 100.0 that was previously returned — an empty graph has zero ANE
    /// utilization by definition, and claiming 100% masks the fact that
    /// the SIR may not have been populated correctly.
    pub fn ane_utilization(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
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

            // ─── Elementwise binary: ANE-supported on PE ──────────
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
            SirOp::FloorDiv { .. } => OpSupport::CpuOnly(
                "FloorDiv (integer division) has no FP16 ANEC converter".to_string(),
            ),
            SirOp::Mod { .. } => OpSupport::CpuOnly(
                "Modulo has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Pow { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "Pow has no ANEC converter on A11/A12; falls back to CPU".to_string(),
                    )
                } else {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A14,
                        reason: "Pow elementwise requires A14+ ANEC converter".to_string(),
                    }
                }
            }
            SirOp::Equal { .. }
            | SirOp::NotEqual { .. }
            | SirOp::Greater { .. }
            | SirOp::GreaterEqual { .. }
            | SirOp::Less { .. }
            | SirOp::LessEqual { .. } => {
                OpSupport::AneSupported(AneEngineSupport::PE)
            }

            // ─── Elementwise unary: ANE-supported on PE ───────────
            SirOp::Relu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Relu6 { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::LeakyRelu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::SigmoidHard { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ThresholdedRelu { .. } => OpSupport::CpuOnly(
                "ThresholdedRelu has no direct ANEC converter; decompose to Relu+Where".to_string(),
            ),
            SirOp::ClampedRelu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::LinearActivation { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Prelu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Softsign { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Gelu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Silu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ScaledTanh { .. } => OpSupport::CpuOnly(
                "ScaledTanh has no direct ANEC converter; decompose to Tanh+Mul+Add".to_string(),
            ),
            SirOp::Elu { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Softplus { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::SoftplusParametric { .. } => OpSupport::CpuOnly(
                "Parametric Softplus has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Sigmoid { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Tanh { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Clip { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Square { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Threshold { .. } => OpSupport::CpuOnly(
                "Threshold has no direct ANEC converter; decompose to Where+Const".to_string(),
            ),
            SirOp::Sqrt { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Rsqrt { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Inverse { .. } => OpSupport::CpuOnly(
                "Inverse has no direct ANEC converter; decompose to RealDiv(Const(1), x)".to_string(),
            ),
            SirOp::Exp { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Exp2 { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Log { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Ceil { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Floor { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Round { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sign { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Cos { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Sin { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Tan { .. } => OpSupport::CpuOnly(
                "Tan has no direct ANEC converter; decompose to Sin/RealDiv/Cos".to_string(),
            ),
            SirOp::Acos { .. } | SirOp::Asin { .. } | SirOp::Atan { .. } => OpSupport::CpuOnly(
                "Inverse trig ops have no ANEC converter; fall back to CPU".to_string(),
            ),
            SirOp::Cosh { .. } => OpSupport::CpuOnly(
                "Cosh has no direct ANEC converter; decompose to Exp-based expression".to_string(),
            ),
            SirOp::Sinh { .. } => OpSupport::CpuOnly(
                "Sinh has no direct ANEC converter; decompose to Exp-based expression".to_string(),
            ),
            SirOp::Atanh { .. } => OpSupport::CpuOnly(
                "Atanh has no direct ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Erf { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Abs { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Neg { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::LogicalNot { .. } => OpSupport::CpuOnly(
                "LogicalNot has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Cast { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Select { .. } => OpSupport::CpuOnly(
                "mb.select is ANE-illegal in practice despite ConvertSelect in per-op matrix; decompose to cond*a + (1-cond)*b".to_string(),
            ),
            // mb.where: ANE-illegal for same reason as select.
            // Decompose to arithmetic: where(cond, x, y) → cond*x + (1-cond)*y
            SirOp::Where { .. } => OpSupport::CpuOnly(
                "mb.where is ANE-illegal; decompose to cond*x + (1-cond)*y".to_string(),
            ),

            // ─── Reduction: PE, with family restrictions ───────────
            SirOp::ReduceMean { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceSum { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceMax { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ReduceMin { .. } => {
                if !self.family.supports_reducemin_all_dtypes() {
                    OpSupport::FamilyGated {
                        minimum_family: AneFamily::A14,
                        reason: "ReduceMin requires A14+ for non-FP types (A11/A12/A13 are FP-only)".to_string(),
                    }
                } else {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                }
            }
            SirOp::ReduceProd { .. } => OpSupport::CpuOnly(
                "ReduceProd has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::ReduceSumSquare { .. } => {
                // ReduceSumSquare = ReduceSum(x*x), decomposes to Square + ReduceSum
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::ReduceL2Norm { .. } => {
                // ReduceL2Norm = Sqrt(ReduceSum(x*x)), decomposes to ANE-faithful ops
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::ReduceL1Norm { .. } => {
                // ReduceL1Norm = ReduceSum(Abs(x)), decomposes to ANE-faithful ops
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::ReduceLogSumExp { .. } => OpSupport::CpuOnly(
                "ReduceLogSumExp has no direct ANEC converter; decompose to ReduceMax+Exp+ReduceSum+Log".to_string(),
            ),
            SirOp::ReduceLogSum { .. } => OpSupport::CpuOnly(
                "ReduceLogSum has no direct ANEC converter; decompose to ReduceSum+Log".to_string(),
            ),
            SirOp::ReduceArgmax { .. } => {
                // T-32: ArgMinMax has ConvertReductionArg converters for LSE_0-6
                // (A11Legacy through A16) but NO LSE_7 converter for A18/M4.
                if self.family.supports_argminmax() {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                } else {
                    OpSupport::CpuOnly(
                        "ReduceArgmax has no ANEC converter on A18 (LSE_7); supported on A11Legacy through A16 only".to_string(),
                    )
                }
            }
            SirOp::ReduceArgmin { .. } => {
                // T-32: Same as ReduceArgmax — no LSE_7 converter on A18.
                if self.family.supports_argminmax() {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                } else {
                    OpSupport::CpuOnly(
                        "ReduceArgmin has no ANEC converter on A18 (LSE_7); supported on A11Legacy through A16 only".to_string(),
                    )
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
            SirOp::BatchNorm { .. } => OpSupport::CpuOnly(
                "BatchNorm decomposes to InstanceNorm + broadcast; may partially fall back"
                    .to_string(),
            ),
            SirOp::L2Norm { .. } => {
                // L2Norm = x / Sqrt(ReduceSum(x*x) + epsilon), decomposes to ANE-faithful ops
                OpSupport::AneSupported(AneEngineSupport::PE)
            }
            SirOp::LocalResponseNorm { .. } => OpSupport::CpuOnly(
                "LocalResponseNorm has no ANEC converter; deprecated op falls back to CPU"
                    .to_string(),
            ),

            // ─── Pooling: NE ───────────────────────────────────────
            SirOp::MaxPool { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::AvgPool { .. } => OpSupport::AneSupported(AneEngineSupport::NE),
            SirOp::L2Pool { .. } => OpSupport::CpuOnly(
                "L2Pool has no direct ANEC converter; decompose to Square+AvgPool+Sqrt".to_string(),
            ),

            // ─── Image Resizing ────────────────────────────────────
            SirOp::Resize { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "Resize has no ANEC converter on A11/A12; falls back to CPU".to_string(),
                    )
                } else {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                }
            }
            SirOp::ResizeNearestNeighbor { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "ResizeNearestNeighbor has no ANEC converter on A11/A12".to_string(),
                    )
                } else {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                }
            }
            SirOp::ResizeBilinear { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "ResizeBilinear has no ANEC converter on A11/A12".to_string(),
                    )
                } else {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                }
            }
            SirOp::UpsampleNearestNeighbor { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "UpsampleNearestNeighbor has no ANEC converter on A11/A12".to_string(),
                    )
                } else {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                }
            }
            SirOp::UpsampleBilinear { .. } => {
                if matches!(self.family, AneFamily::A11Legacy | AneFamily::A12) {
                    OpSupport::CpuOnly(
                        "UpsampleBilinear has no ANEC converter on A11/A12".to_string(),
                    )
                } else {
                    OpSupport::AneSupported(AneEngineSupport::NE)
                }
            }
            SirOp::CropResize { .. } => OpSupport::CpuOnly(
                "CropResize has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Affine { .. } => OpSupport::CpuOnly(
                "Affine transform has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Resample { .. } => OpSupport::CpuOnly(
                "Resample has no ANEC converter; falls back to CPU".to_string(),
            ),

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
            SirOp::ReshapeLike { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Transpose { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Concat { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Split { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::ExpandDims { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Squeeze { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Flatten2d { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::SliceByIndex { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::SliceBySize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Pad { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Stack { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Reverse { .. } => OpSupport::CpuOnly(
                "Reverse has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::ReverseSequence { .. } => OpSupport::CpuOnly(
                "ReverseSequence has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::SlidingWindows { .. } => OpSupport::CpuOnly(
                "SlidingWindows has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::DepthToSpace { .. } => {
                // DepthToSpace = Reshape + Transpose + Reshape, all ANE-faithful
                OpSupport::AneSupported(AneEngineSupport::Transpose)
            }
            SirOp::SpaceToDepth { .. } => {
                // SpaceToDepth = Reshape + Transpose + Reshape, all ANE-faithful
                OpSupport::AneSupported(AneEngineSupport::Transpose)
            }
            SirOp::PixelShuffle { .. } => {
                // PixelShuffle = DepthToSpace variant, ANE-faithful
                OpSupport::AneSupported(AneEngineSupport::Transpose)
            }
            SirOp::PixelUnshuffle { .. } => {
                // PixelUnshuffle = SpaceToDepth variant, ANE-faithful
                OpSupport::AneSupported(AneEngineSupport::Transpose)
            }
            SirOp::BatchToSpace { .. } => OpSupport::CpuOnly(
                "BatchToSpace has no direct ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::SpaceToBatch { .. } => OpSupport::CpuOnly(
                "SpaceToBatch has no direct ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Tile { .. } => {
                // Tile is ANE-illegal as a native op. However:
                //   1. GQA Tile ops are now eliminated at the SIR builder level
                //      via split-based per-head attention (matching the reference
                //      model pkhairkh/qwen3-coreml-palettized).
                //   2. Any remaining standalone Tile ops are decomposed by the
                //      legality rewrite pass to Reshape + broadcast Mul + Reshape,
                //      all ANE-faithful ops.
                //   3. A panic in the fallback path ensures Tile never survives
                //      to AIR/MIR undecomposed.
                OpSupport::AneSupported(AneEngineSupport::PE)
            }

            // ─── Gather/Scatter ────────────────────────────────────
            SirOp::Gather { axis, .. } => {
                // Gather on ANE requires constant axis and specific constraints
                if *axis >= 0 {
                    OpSupport::AneSupported(AneEngineSupport::PE)
                } else {
                    OpSupport::CpuOnly(
                        "Gather with negative axis may not be ANE-compatible".to_string(),
                    )
                }
            }
            SirOp::GatherAlongAxis { .. } => OpSupport::CpuOnly(
                "GatherAlongAxis has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::GatherNd { .. } => {
                OpSupport::CpuOnly("GatherNd has no ANEC converter".to_string())
            }
            SirOp::Scatter { .. } => {
                OpSupport::CpuOnly("Scatter ops have no ANEC converter".to_string())
            }
            SirOp::ScatterAlongAxis { .. } => OpSupport::CpuOnly(
                "ScatterAlongAxis has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::ScatterNd { .. } => {
                OpSupport::Unsupported("ScatterNd is never ANE-compatible".to_string())
            }
            SirOp::NonMaximumSuppression { .. } => OpSupport::CpuOnly(
                "NMS has no ANEC converter; falls back to CPU".to_string(),
            ),

            // ─── Quantization ──────────────────────────────────────
            SirOp::Quantize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Dequantize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),

            // ─── Constexpr / Compression: compile-time, not runtime ─
            SirOp::ConstexprAffineDequantize { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ConstexprBlockwiseShiftScale { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ConstexprLutToDense { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ConstexprSparseToDense { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ConstexprCast { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::ConstexprLutToSparse { .. } => OpSupport::CpuOnly(
                "ConstexprLutToSparse has no ANEC converter; compression-only, CPU pre-processing".to_string(),
            ),
            SirOp::ConstexprSparseBlockwiseShiftScale { .. } => OpSupport::CpuOnly(
                "ConstexprSparseBlockwiseShiftScale has no ANEC converter; CPU pre-processing".to_string(),
            ),

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
            SirOp::Sampler { .. } => {
                OpSupport::CpuOnly("TopK sampling is CPU-only on most families".to_string())
            }

            // ─── Special structural / utility ops ─────────────────
            SirOp::Const { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Identity { .. } => OpSupport::AneSupported(AneEngineSupport::Transpose),
            SirOp::Fill { .. } => OpSupport::CpuOnly(
                "mb.fill is ANE-illegal; must be replaced with MILConst during lowering".to_string(),
            ),
            SirOp::FillLike { .. } => OpSupport::CpuOnly(
                "mb.fill_like is ANE-illegal; decompose to mul(ref, 0) + add(0, val) at MIR level".to_string(),
            ),
            SirOp::Range1d { .. } => OpSupport::CpuOnly(
                "Range1d has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Shape { .. } => OpSupport::CpuOnly(
                "Shape has no ANEC converter; shape ops are CPU-only".to_string(),
            ),
            SirOp::OneHot { .. } => OpSupport::CpuOnly(
                "OneHot has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::NonZero { .. } => OpSupport::CpuOnly(
                "NonZero has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Argsort { .. } => OpSupport::CpuOnly(
                "Argsort has no ANEC converter; sort ops are CPU-only".to_string(),
            ),
            SirOp::BandPart { .. } => OpSupport::CpuOnly(
                "BandPart has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Crop { .. } => OpSupport::AneSupported(AneEngineSupport::PE),
            SirOp::Topk { .. } => OpSupport::CpuOnly(
                "Topk has no ANEC converter; sort/select ops are CPU-only".to_string(),
            ),
            SirOp::Classify { .. } => OpSupport::CpuOnly(
                "Classify has no ANEC converter; falls back to CPU".to_string(),
            ),
            SirOp::Einsum { .. } => OpSupport::CpuOnly(
                "Einsum has no direct ANEC converter; decompose to MatMul+Transpose for ANE path".to_string(),
            ),
            SirOp::ConvTranspose { .. } => OpSupport::CpuOnly(
                "ConvTranspose has no direct ANEC converter; falls back to CPU".to_string(),
            ),

            // ─── Logical: CPU-only ─────────────────────────────────
            SirOp::LogicalAnd { .. } | SirOp::LogicalOr { .. } | SirOp::LogicalXor { .. } => {
                OpSupport::CpuOnly("Logical ops have no ANEC converter".to_string())
            }

            // ─── Always CPU-only ──────────────────────────────────
            SirOp::Cumsum { .. } => OpSupport::CpuOnly("Cumsum has no ANEC converter".to_string()),
            SirOp::RandomBernoulli { .. }
            | SirOp::RandomNormal { .. }
            | SirOp::RandomUniform { .. }
            | SirOp::RandomCategorical { .. } => {
                OpSupport::Unsupported("Random ops are never ANE-compatible".to_string())
            }
            SirOp::Cond { .. } | SirOp::WhileLoop { .. } => {
                OpSupport::Unsupported("Control flow has no ANEC converter".to_string())
            }
            SirOp::MakeList { .. }
            | SirOp::ListLength { .. }
            | SirOp::ListWrite { .. }
            | SirOp::ListRead { .. }
            | SirOp::ListGather { .. }
            | SirOp::ListScatter { .. } => {
                OpSupport::Unsupported("List ops have no ANEC converter; control-flow dependent".to_string())
            }
            SirOp::Rnn { .. } | SirOp::Gru { .. } | SirOp::Lstm { .. } => {
                OpSupport::CpuOnly("RNN/LSTM/GRU have no direct ANEC converter".to_string())
            }

            // ─── Elementwise ops handled individually ────────────────
        }
    }
}

/// Map a SirOp to a human-readable name.
///
/// Uses the Debug representation to extract the variant name robustly,
/// so that even newly-added variants get a meaningful name without
/// having to update this function.
fn op_name_for_sir(op: &SirOp) -> String {
    // Fast path: common ops with explicit short names
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
        // Fallback: extract variant name from Debug repr ("VariantName { ... }")
        _ => {
            let debug = format!("{:?}", op);
            let name = debug.split('{').next().unwrap_or("Unknown").trim();
            // Must return a String to match the outer .to_string() call;
            // boxing the &str keeps the types aligned across all arms.
            return name.to_string();
        }
    }.to_string()
}

/// Get the default revision for a family.
fn family_to_default_revision(family: AneFamily) -> AneRevision {
    match family {
        AneFamily::A11Legacy => AneRevision::V4,
        AneFamily::A12 => AneRevision::V5,
        AneFamily::A13 => AneRevision::V6,
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
/// A13 sits between A12 and A14 — it has full-dtype broadcast (unlike A12)
/// but A14Minus converters and FP-only ReduceMin (unlike A14).
fn family_level(family: AneFamily) -> u32 {
    match family {
        AneFamily::A11Legacy => 0,
        AneFamily::A12 => 1,
        AneFamily::A13 => 2,
        AneFamily::A14 => 3,
        AneFamily::A15 => 4,
        AneFamily::A16 => 5,
        AneFamily::A18 => 6,
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
        assert!(family_level(AneFamily::A14) > family_level(AneFamily::A13));
        assert!(family_level(AneFamily::A13) > family_level(AneFamily::A12));
        assert!(family_level(AneFamily::A12) > family_level(AneFamily::A11Legacy));
    }

    #[test]
    fn test_a13_constraint_profile() {
        let compiler = VersionedCompiler::new(AneFamily::A13);
        assert!(!compiler.target_family().broadcast_fp16_only());
        assert!(compiler.target_family().uses_a14minus_converters());
        assert!(!compiler.target_family().supports_sdpa());
        assert!(!compiler.target_family().supports_layernorm());
        assert!(!compiler.target_family().supports_reducemin_all_dtypes());
    }

    #[test]
    fn test_reducemin_family_gated_on_a13() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A13);
        let reducemin = SirOp::ReduceMin {
            input: SirNodeId("x".into()),
            axes: vec![1],
            keep_dims: false,
        };
        match matrix.check_op(&reducemin) {
            OpSupport::FamilyGated { .. } => {} // expected — A13 is FP-only
            other => panic!("Expected FamilyGated for ReduceMin on A13, got {:?}", other),
        }
    }

    #[test]
    fn test_reducemin_supported_on_a14() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A14);
        let reducemin = SirOp::ReduceMin {
            input: SirNodeId("x".into()),
            axes: vec![1],
            keep_dims: false,
        };
        match matrix.check_op(&reducemin) {
            OpSupport::AneSupported(_) => {} // expected — A14 supports all dtypes
            other => panic!("Expected AneSupported for ReduceMin on A14, got {:?}", other),
        }
    }

    #[test]
    fn test_validate_linear_graph() {
        let compiler = VersionedCompiler::new(AneFamily::A16);
        let sir = SirGraph { nodes: vec![], inputs: vec![], outputs: vec![] };
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

    // ─── T-32: ArgMinMax A18 guard tests (SIR-level) ──────────────

    #[test]
    fn test_argmax_supported_on_a16() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A16);
        let argmax = SirOp::ReduceArgmax {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmax) {
            OpSupport::AneSupported(AneEngineSupport::PE) => {} // expected
            other => panic!("Expected AneSupported(PE) for ArgMax on A16, got {:?}", other),
        }
    }

    #[test]
    fn test_argmin_supported_on_a16() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A16);
        let argmin = SirOp::ReduceArgmin {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmin) {
            OpSupport::AneSupported(AneEngineSupport::PE) => {} // expected
            other => panic!("Expected AneSupported(PE) for ArgMin on A16, got {:?}", other),
        }
    }

    #[test]
    fn test_argmax_cpu_only_on_a18() {
        // T-32: A18 (LSE_7) has no ConvertReductionArg converter.
        let matrix = OpSupportMatrix::for_family(AneFamily::A18);
        let argmax = SirOp::ReduceArgmax {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmax) {
            OpSupport::CpuOnly(reason) => {
                assert!(reason.contains("LSE_7"), "Expected LSE_7 in reason: {}", reason);
                assert!(reason.contains("A18"), "Expected A18 in reason: {}", reason);
            }
            other => panic!("Expected CpuOnly for ArgMax on A18, got {:?}", other),
        }
    }

    #[test]
    fn test_argmin_cpu_only_on_a18() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A18);
        let argmin = SirOp::ReduceArgmin {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmin) {
            OpSupport::CpuOnly(reason) => {
                assert!(reason.contains("LSE_7"), "Expected LSE_7 in reason: {}", reason);
            }
            other => panic!("Expected CpuOnly for ArgMin on A18, got {:?}", other),
        }
    }

    #[test]
    fn test_argmax_supported_on_a11_legacy() {
        // Even the oldest family (A11Legacy, LSE_0) has a converter.
        let matrix = OpSupportMatrix::for_family(AneFamily::A11Legacy);
        let argmax = SirOp::ReduceArgmax {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmax) {
            OpSupport::AneSupported(AneEngineSupport::PE) => {} // expected
            other => panic!("Expected AneSupported(PE) for ArgMax on A11Legacy, got {:?}", other),
        }
    }

    #[test]
    fn test_argmax_supported_on_a14() {
        let matrix = OpSupportMatrix::for_family(AneFamily::A14);
        let argmax = SirOp::ReduceArgmax {
            input: SirNodeId("x".into()),
            axis: 1,
            keep_dims: false,
        };
        match matrix.check_op(&argmax) {
            OpSupport::AneSupported(AneEngineSupport::PE) => {} // expected
            other => panic!("Expected AneSupported(PE) for ArgMax on A14, got {:?}", other),
        }
    }
}

#[cfg(test)]
use ane_ir::sir::SirNodeId;
