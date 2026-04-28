//! Pass Knowledge Query
//!
//! Trait that compilation passes use to query the knowledge store
//! for legality rules, risk data, precision hazard information,
//! and compute plan placement data.
//! This trait is defined here (in ane-passes) to avoid circular
//! dependencies — the concrete implementation lives in ane-knowledge
//! and is wired in ane-cli.

use ane_ir::kir::KnowledgeScope;

/// Result of querying legality for a specific op pattern.
#[derive(Debug, Clone)]
pub struct LegalityInfo {
    /// Whether the op is believed to be ANE-legal.
    pub ane_legal: bool,
    /// Confidence of this legality claim (0.0–1.0).
    pub confidence: f32,
    /// Number of evidence observations supporting this claim.
    pub evidence_count: usize,
    /// The ID of the knowledge unit that provided this info (if any).
    pub source_id: Option<String>,
}

/// Result of querying risk data for a specific op pattern.
#[derive(Debug, Clone)]
pub struct RiskInfo {
    /// Fallback risk score (0.0 = no risk, 1.0 = certain fallback).
    pub fallback_risk: f32,
    /// Drift risk score (0.0 = no risk, 1.0 = certain drift).
    pub drift_risk: f32,
    /// Confidence of this risk assessment.
    pub confidence: f32,
    /// Number of evidence observations supporting this assessment.
    pub evidence_count: usize,
    /// The ID of the knowledge unit that provided this info (if any).
    pub source_id: Option<String>,
}

/// Result of querying precision hazard knowledge for a specific op.
///
/// A precision hazard indicates that an operation is known to produce
/// unacceptable quality degradation at a given precision, and that a
/// higher precision (e.g., fp32 instead of fp16) should be used.
///
/// This is the first concrete adaptation mechanism: the compiler
/// changes its precision decision because stored empirical knowledge
/// indicates that the default precision is unsafe.
#[derive(Debug, Clone)]
pub struct PrecisionHazardInfo {
    /// The op pattern this hazard applies to (e.g., "LinearProjection").
    pub op_pattern: String,
    /// The dtype that is known to be hazardous (e.g., "fp16").
    pub hazardous_dtype: String,
    /// The recommended safe dtype (e.g., "fp32").
    pub recommended_dtype: String,
    /// Confidence of this hazard assessment (0.0–1.0).
    pub confidence: f32,
    /// Number of evidence observations supporting this assessment.
    pub evidence_count: usize,
    /// The ID of the knowledge unit that provided this info.
    pub source_id: Option<String>,
    /// Human-readable description of the hazard.
    pub description: Option<String>,
}

/// Result of querying compute plan placement for a specific op.
///
/// When MLComputePlan shows that an op was NOT placed on the
/// NeuralEngine, this is strong evidence of fallback risk.
/// Because compute plan data is deterministic for a given
/// hardware+OS combination, it carries high confidence (0.9).
#[derive(Debug, Clone)]
pub struct ComputePlanPlacementInfo {
    /// The op pattern this placement applies to.
    pub op_pattern: String,
    /// Whether the compute planner placed this op on the NeuralEngine.
    pub ane_placed: bool,
    /// The preferred compute device class (e.g., "NeuralEngine", "CPU", "GPU").
    pub preferred_device: String,
    /// Confidence of this placement observation (0.9 for compute plan data).
    pub confidence: f32,
    /// Number of evidence observations supporting this assessment.
    pub evidence_count: usize,
    /// The ID of the knowledge unit that provided this info.
    pub source_id: Option<String>,
}

/// A no-op knowledge query that returns defaults (used when no store is available).
pub struct NoKnowledge;

impl PassKnowledgeQuery for NoKnowledge {
    fn query_legality(
        &self,
        _op_pattern: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<LegalityInfo> {
        None
    }

    fn query_risk(&self, _op_pattern: &str, _scope: Option<&KnowledgeScope>) -> Option<RiskInfo> {
        None
    }

    fn query_precision_hazard(
        &self,
        _op_pattern: &str,
        _current_dtype: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<PrecisionHazardInfo> {
        None
    }

    fn query_compute_plan_placement(
        &self,
        _op_pattern: &str,
        _scope: Option<&KnowledgeScope>,
    ) -> Option<ComputePlanPlacementInfo> {
        None
    }
}

/// Trait that compilation passes use to query the knowledge store.
///
/// This is the seam between the pass pipeline and the knowledge system.
/// The concrete implementation wraps `KnowledgeStore` from `ane-knowledge`.
/// When no knowledge store is available, `NoKnowledge` provides defaults.
pub trait PassKnowledgeQuery {
    /// Query legality information for a given op pattern.
    ///
    /// Returns `None` if no knowledge is available for this op pattern.
    /// The `scope` parameter allows filtering by device class / OS / opset.
    fn query_legality(
        &self,
        op_pattern: &str,
        scope: Option<&KnowledgeScope>,
    ) -> Option<LegalityInfo>;

    /// Query risk information for a given op pattern.
    ///
    /// Returns `None` if no risk knowledge is available for this op pattern.
    /// The `scope` parameter allows filtering by device class / OS / opset.
    fn query_risk(&self, op_pattern: &str, scope: Option<&KnowledgeScope>) -> Option<RiskInfo>;

    /// Query precision hazard knowledge for a given op pattern and current dtype.
    ///
    /// Returns `None` if no precision hazard knowledge is available.
    /// When a hazard is found, the pass should consider overriding the
    /// default precision to the recommended dtype.
    ///
    /// This is the first concrete adaptation mechanism in the pipeline:
    /// stored empirical knowledge about precision hazards changes the
    /// compiler's precision decision.
    fn query_precision_hazard(
        &self,
        op_pattern: &str,
        current_dtype: &str,
        scope: Option<&KnowledgeScope>,
    ) -> Option<PrecisionHazardInfo>;

    /// Query compute plan placement for a given op pattern.
    ///
    /// Returns `None` if no compute plan placement knowledge is available.
    /// When the compute planner did NOT place an op on the NeuralEngine,
    /// this provides strong evidence of fallback risk.
    ///
    /// This is the Sprint 35 adaptation mechanism: deterministic
    /// compute plan evidence increases fallback_risk for ops that
    /// the planner assigns to CPU or GPU.
    fn query_compute_plan_placement(
        &self,
        op_pattern: &str,
        scope: Option<&KnowledgeScope>,
    ) -> Option<ComputePlanPlacementInfo>;
}
