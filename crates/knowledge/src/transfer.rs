//! Synthetic Transfer
//!
//! Manages the transfer and annotation of knowledge gained
//! from synthetic profiling to real-model compilation contexts.
//!
//! Synthetic knowledge can be transferred to real-model contexts
//! with reduced confidence. This module provides transfer safety
//! checks and confidence adjustment.

use ane_ir::kir::{EvidenceSource, KnowledgeType, KnowledgeUnit};
use anyhow::{bail, Result};

use crate::store::KnowledgeEntry;
use crate::util::{payload_ane_legal, payload_ane_placed, payload_fallback_engine, payload_num_partitions, payload_quality_impact, payload_survival_rate};

/// Synthetic transfer annotation and validation.
pub struct SyntheticTransfer {
    /// Confidence scaling factor for synthetic-to-real transfer.
    /// Per SPEC.md section 6.6: operator-level transfer scales by 0.7,
    /// pattern-level transfer scales by 0.5-0.8.
    operator_transfer_scale: f32,
    pattern_transfer_scale_min: f32,
    pattern_transfer_scale_max: f32,
}

impl Default for SyntheticTransfer {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticTransfer {
    /// Create a new synthetic transfer handler with default scaling.
    pub fn new() -> Self {
        Self {
            operator_transfer_scale: 0.7,
            pattern_transfer_scale_min: 0.5,
            pattern_transfer_scale_max: 0.8,
        }
    }

    /// Check if a synthetic knowledge unit is safe to transfer
    /// to a real-model compilation context.
    ///
    /// Transfer is safe if:
    /// - The entry is from a synthetic source
    /// - The knowledge type is operator-level (LegalityRule, SurvivalMatrixEntry)
    /// - The confidence is not too high (synthetic knowledge should not be trusted blindly)
    ///
    /// Transfer is NOT safe for:
    /// - Topology-level knowledge (shard templates, state topology)
    /// - Device fingerprint knowledge
    /// - Knowledge already from real-model runs
    pub fn is_transfer_safe(&self, entry: &KnowledgeEntry) -> bool {
        // Only synthetic entries can be transferred
        if !matches!(entry.unit.evidence_source, EvidenceSource::SyntheticRun) {
            return false;
        }

        // Topology-level knowledge does NOT transfer
        if matches!(
            entry.unit.knowledge_type,
            KnowledgeType::ShardTemplateKnowledge
                | KnowledgeType::StateTopologyOutcome
                | KnowledgeType::DeviceFingerprint
        ) {
            return false;
        }

        // High-confidence synthetic claims should be treated with caution
        // but are still transferable (just with reduced confidence)
        true
    }

    /// Compute the scaled confidence for a synthetic-to-real transfer.
    ///
    /// Per SPEC.md section 6.6:
    /// - Operator-level: scale by 0.7
    /// - Pattern-level: scale by 0.5-0.8 (depending on similarity)
    pub fn transfer_confidence(&self, entry: &KnowledgeEntry) -> f32 {
        match entry.unit.knowledge_type {
            KnowledgeType::LegalityRule | KnowledgeType::SurvivalMatrixEntry => {
                // Operator-level transfer: scale by 0.7
                entry.unit.confidence * self.operator_transfer_scale
            }
            KnowledgeType::MotifCatalog | KnowledgeType::FallbackSignature => {
                // Pattern-level transfer: use mid-range scale
                let scale =
                    (self.pattern_transfer_scale_min + self.pattern_transfer_scale_max) / 2.0;
                entry.unit.confidence * scale
            }
            KnowledgeType::PrecisionHazard => {
                // Precision hazards from synthetic runs have moderate transfer
                entry.unit.confidence * 0.6
            }
            _ => {
                // Default: conservative scaling
                entry.unit.confidence * 0.5
            }
        }
    }

    /// Validate that transferred knowledge still holds
    /// after real-model evidence arrives.
    ///
    /// Compares a synthetic-derived knowledge unit against a real-model
    /// observation of the same pattern. Returns a validation result
    /// indicating whether the synthetic knowledge was consistent.
    pub fn validate_against_real(
        &self,
        synthetic: &KnowledgeEntry,
        real: &KnowledgeEntry,
    ) -> Result<TransferValidation> {
        // Must be the same knowledge type
        if synthetic.unit.knowledge_type != real.unit.knowledge_type {
            bail!(
                "Cannot validate synthetic '{}' against real '{}' — different knowledge types",
                synthetic.unit.id,
                real.unit.id
            );
        }

        // Check if claims agree
        let is_consistent = self.claims_agree(&synthetic.unit, &real.unit);
        let confidence_delta = real.unit.confidence - self.transfer_confidence(synthetic);

        let recommendation = if is_consistent {
            if confidence_delta.abs() < 0.1 {
                TransferRecommendation::Keep
            } else {
                TransferRecommendation::UpdateConfidence(real.unit.confidence * 0.9)
            }
        } else if synthetic.unit.confidence > 0.7 {
            // High-confidence synthetic claim contradicted by real evidence
            TransferRecommendation::EscalateForReview
        } else {
            TransferRecommendation::Deprecate
        };

        Ok(TransferValidation { is_consistent, confidence_delta, recommendation })
    }

    /// Check if two knowledge units make agreeing claims (using typed accessors).
    ///
    /// T-112: Previously defaulted to `true` for 7/8 knowledge types, preventing
    /// contradiction detection. Now implements field-level comparison for all
    /// knowledge types using typed payload accessors. Returns `true` if claims
    /// agree (or if insufficient data to determine disagreement), `false` if
    /// claims explicitly contradict.
    fn claims_agree(&self, a: &KnowledgeUnit, b: &KnowledgeUnit) -> bool {
        match a.knowledge_type {
            KnowledgeType::LegalityRule => {
                // Legality: ane_legal must agree
                match (payload_ane_legal(&a.payload), payload_ane_legal(&b.payload)) {
                    (Some(av), Some(bv)) => av == bv,
                    _ => true, // Can't determine disagreement
                }
            }
            KnowledgeType::PrecisionHazard => {
                // PrecisionHazard: quality_impact must not be opposite
                let a_impact = payload_quality_impact(&a.payload);
                let b_impact = payload_quality_impact(&b.payload);
                match (a_impact, b_impact) {
                    (Some("negligible"), Some("severe"))
                    | (Some("severe"), Some("negligible")) => false,
                    _ => true,
                }
            }
            KnowledgeType::SurvivalMatrixEntry => {
                // SurvivalMatrix: survival_rate should not diverge significantly.
                // If both entries specify a rate and they differ by more than 0.5,
                // they disagree. A survival rate of 0.1 vs 0.9 is a contradiction.
                match (payload_survival_rate(&a.payload), payload_survival_rate(&b.payload)) {
                    (Some(av), Some(bv)) => (av - bv).abs() <= 0.5,
                    _ => true,
                }
            }
            KnowledgeType::FallbackSignature => {
                // FallbackSignature: fallback_engine must agree if both specify one.
                match (payload_fallback_engine(&a.payload), payload_fallback_engine(&b.payload)) {
                    (Some(av), Some(bv)) => av == bv,
                    _ => true,
                }
            }
            KnowledgeType::ShardTemplateKnowledge => {
                // ShardTemplate: num_partitions must agree if both specify it.
                match (payload_num_partitions(&a.payload), payload_num_partitions(&b.payload)) {
                    (Some(av), Some(bv)) => av == bv,
                    _ => true,
                }
            }
            KnowledgeType::StateTopologyOutcome => {
                // StateTopology: ane_placed must agree (whether ops were placed on ANE).
                match (payload_ane_placed(&a.payload), payload_ane_placed(&b.payload)) {
                    (Some(av), Some(bv)) => av == bv,
                    _ => true,
                }
            }
            KnowledgeType::MotifCatalog => {
                // MotifCatalog: motifs are additive (catalog entries), so they
                // agree by default unless they contradict on op_pattern scope.
                // Two motif entries for different patterns don't conflict.
                true
            }
            KnowledgeType::DeviceFingerprint => {
                // DeviceFingerprint: entries describe device capabilities and
                // don't contradict each other (different devices have different
                // fingerprints, but that's additive not contradictory).
                true
            }
            KnowledgeType::SyntheticTransferAnnotation => {
                // Transfer annotations are metadata about transfer decisions,
                // not claims about model behavior. They can't contradict.
                true
            }
        }
    }
}

/// Result of validating synthetic knowledge against real evidence.
#[derive(Debug)]
pub struct TransferValidation {
    /// Whether the synthetic and real evidence are consistent.
    pub is_consistent: bool,
    /// Difference between real confidence and transferred synthetic confidence.
    pub confidence_delta: f32,
    /// Recommended action.
    pub recommendation: TransferRecommendation,
}

/// Recommendation after transfer validation.
#[derive(Debug, PartialEq)]
pub enum TransferRecommendation {
    /// Synthetic knowledge is consistent — keep as-is.
    Keep,
    /// Synthetic knowledge contradicts real evidence — deprecate.
    Deprecate,
    /// Update the confidence score of the synthetic entry.
    UpdateConfidence(f32),
    /// High-confidence contradiction — escalate for manual review.
    EscalateForReview,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ConflictStatus, EntryOrigin, EntryProvenance, EntrySource};
    use ane_ir::kir::KnowledgeScope;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_synthetic_entry(id: &str, ane_legal: bool, confidence: f32) -> KnowledgeEntry {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(ane_legal));
        payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));

        KnowledgeEntry {
            unit: Arc::new(KnowledgeUnit {
                id: id.to_string(),
                version: 1,
                timestamp: chrono::Utc::now().to_rfc3339(),
                knowledge_type: KnowledgeType::LegalityRule,
                confidence,
                evidence_source: EvidenceSource::SyntheticRun,
                evidence_count: 5,
                scope: KnowledgeScope {
                    device_classes: vec!["M2".to_string()],
                    os_versions: vec!["macOS_15".to_string()],
                    opset_versions: vec!["iOS18".to_string()],
                },
                conflict_priority: 0,
                payload,
            }),
            provenance: EntryProvenance {
                origin: EntryOrigin::RunObservation,
                inserted_at: chrono::Utc::now().to_rfc3339(),
                updated_at: None,
                source_path: None,
            },
            source: EntrySource::Observation,
            conflict_status: ConflictStatus::NoConflict,
            revision: 0,
        }
    }

    fn make_real_entry(id: &str, ane_legal: bool, confidence: f32) -> KnowledgeEntry {
        let mut entry = make_synthetic_entry(id, ane_legal, confidence);
        // Arc<KnowledgeUnit> is immutable, so we need to create a new Arc
        // with the modified unit.
        let mut unit = (*entry.unit).clone();
        unit.evidence_source = EvidenceSource::RealModelRun;
        entry.unit = Arc::new(unit);
        entry
    }

    #[test]
    fn test_transfer_safe_for_legality() {
        let transfer = SyntheticTransfer::new();
        let entry = make_synthetic_entry("synth_1", true, 0.5);
        assert!(transfer.is_transfer_safe(&entry));
    }

    #[test]
    fn test_transfer_unsafe_for_shard_template() {
        let transfer = SyntheticTransfer::new();
        let mut entry = make_synthetic_entry("synth_shard", true, 0.5);
        let mut unit = (*entry.unit).clone();
        unit.knowledge_type = KnowledgeType::ShardTemplateKnowledge;
        entry.unit = Arc::new(unit);
        assert!(!transfer.is_transfer_safe(&entry));
    }

    #[test]
    fn test_transfer_unsafe_for_real_model() {
        let transfer = SyntheticTransfer::new();
        let entry = make_real_entry("real_1", true, 0.5);
        assert!(!transfer.is_transfer_safe(&entry));
    }

    #[test]
    fn test_transfer_confidence_scaled() {
        let transfer = SyntheticTransfer::new();
        let entry = make_synthetic_entry("synth_1", true, 0.6);
        let scaled = transfer.transfer_confidence(&entry);
        // Operator-level: 0.6 * 0.7 = 0.42
        assert!((scaled - 0.42).abs() < 0.01);
    }

    #[test]
    fn test_validate_consistent() {
        let transfer = SyntheticTransfer::new();
        let synthetic = make_synthetic_entry("synth_1", true, 0.6);
        let real = make_real_entry("real_1", true, 0.8);
        let result = transfer.validate_against_real(&synthetic, &real).unwrap();
        assert!(result.is_consistent);
        assert!(matches!(
            result.recommendation,
            TransferRecommendation::Keep | TransferRecommendation::UpdateConfidence(_)
        ));
    }

    #[test]
    fn test_validate_contradicted() {
        let transfer = SyntheticTransfer::new();
        let synthetic = make_synthetic_entry("synth_1", true, 0.3);
        let real = make_real_entry("real_1", false, 0.8);
        let result = transfer.validate_against_real(&synthetic, &real).unwrap();
        assert!(!result.is_consistent);
        assert!(matches!(result.recommendation, TransferRecommendation::Deprecate));
    }

    #[test]
    fn test_validate_high_conf_contradiction_escalates() {
        let transfer = SyntheticTransfer::new();
        let synthetic = make_synthetic_entry("synth_1", true, 0.9);
        let real = make_real_entry("real_1", false, 0.8);
        let result = transfer.validate_against_real(&synthetic, &real).unwrap();
        assert!(!result.is_consistent);
        assert_eq!(result.recommendation, TransferRecommendation::EscalateForReview);
    }

    // ─── T-112: claims_agree field-level comparison tests ────────────

    /// T-112: Verify that SurvivalMatrixEntry claims_agree detects
    /// diverging survival_rate values.
    #[test]
    fn test_t112_claims_agree_survival_matrix_diverging() {
        let transfer = SyntheticTransfer::new();
        let mut payload_a = HashMap::new();
        payload_a.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_a.insert("survival_rate".to_string(), serde_json::json!(0.9));
        let mut payload_b = HashMap::new();
        payload_b.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_b.insert("survival_rate".to_string(), serde_json::json!(0.1));

        let unit_a = KnowledgeUnit {
            id: "surv_a".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::SurvivalMatrixEntry,
            confidence: 0.8,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_a,
        };
        let unit_b = KnowledgeUnit {
            id: "surv_b".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::SurvivalMatrixEntry,
            confidence: 0.7,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_b,
        };
        // 0.9 vs 0.1 — differ by 0.8 > 0.5 threshold → disagree
        assert!(!transfer.claims_agree(&unit_a, &unit_b));
    }

    /// T-112: Verify that FallbackSignature claims_agree detects
    /// different fallback_engine values.
    #[test]
    fn test_t112_claims_agree_fallback_signature_disagree() {
        let transfer = SyntheticTransfer::new();
        let mut payload_a = HashMap::new();
        payload_a.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_a.insert("fallback_engine".to_string(), serde_json::json!("GPU"));
        let mut payload_b = HashMap::new();
        payload_b.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_b.insert("fallback_engine".to_string(), serde_json::json!("CPU"));

        let unit_a = KnowledgeUnit {
            id: "fb_a".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::FallbackSignature,
            confidence: 0.8,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_a,
        };
        let unit_b = KnowledgeUnit {
            id: "fb_b".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::FallbackSignature,
            confidence: 0.7,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_b,
        };
        assert!(!transfer.claims_agree(&unit_a, &unit_b));
    }

    /// T-112: Verify that ShardTemplateKnowledge claims_agree detects
    /// different num_partitions values.
    #[test]
    fn test_t112_claims_agree_shard_template_disagree() {
        let transfer = SyntheticTransfer::new();
        let mut payload_a = HashMap::new();
        payload_a.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_a.insert("num_partitions".to_string(), serde_json::json!(3));
        let mut payload_b = HashMap::new();
        payload_b.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));
        payload_b.insert("num_partitions".to_string(), serde_json::json!(5));

        let unit_a = KnowledgeUnit {
            id: "tmpl_a".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::ShardTemplateKnowledge,
            confidence: 0.8,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_a,
        };
        let unit_b = KnowledgeUnit {
            id: "tmpl_b".to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::ShardTemplateKnowledge,
            confidence: 0.7,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count: 5,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload: payload_b,
        };
        assert!(!transfer.claims_agree(&unit_a, &unit_b));
    }
}
