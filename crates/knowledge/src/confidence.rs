//! Confidence Computation
//!
//! Functions for computing and updating confidence scores
//! for knowledge units based on evidence accumulation.

use ane_ir::kir::{KnowledgeUnit, EvidenceSource};

/// Compute a raw confidence score from evidence parameters.
pub fn compute_confidence(
    evidence_count: usize,
    evidence_source: &EvidenceSource,
    cross_validated: bool,
) -> f32 {
    let base = match evidence_source {
        EvidenceSource::RealModelRun => 0.9,
        EvidenceSource::CrossValidated => 1.0,
        EvidenceSource::SyntheticRun => 0.6,
        EvidenceSource::CompileFailure => 0.8,
        EvidenceSource::LoadFailure => 0.8,
        EvidenceSource::RuntimeAnomaly => 0.7,
        EvidenceSource::ManualEntry => 0.5,
        EvidenceSource::ComputePlan => 0.9,
    };

    let evidence_bonus = (evidence_count as f32).ln().max(0.0) * 0.05;
    let cross_bonus = if cross_validated { 0.1 } else { 0.0 };

    (base + evidence_bonus + cross_bonus).min(1.0)
}

/// Update a knowledge unit's confidence after new evidence arrives.
pub fn update_confidence(unit: &mut KnowledgeUnit, new_evidence_count: usize) {
    let new_conf = compute_confidence(
        new_evidence_count,
        &unit.evidence_source,
        matches!(unit.evidence_source, EvidenceSource::CrossValidated),
    );
    unit.confidence = new_conf;
    unit.evidence_count = new_evidence_count;
}

/// Decay confidence over time (simulated temporal decay).
pub fn decay_confidence(current: f32, halflife_days: f32, elapsed_days: f32) -> f32 {
    current * 0.5f32.powf(elapsed_days / halflife_days)
}
