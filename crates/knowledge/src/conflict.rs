//! Conflict Detection
//!
//! Detects and resolves conflicts between knowledge units
//! that make contradictory claims within overlapping scopes.
//!
//! Conflicts are detected automatically when entries are inserted
//! via the knowledge store. The ConflictDetector can also be used
//! standalone to scan a set of entries.

use ane_ir::kir::{KnowledgeUnit, KnowledgeType};
use anyhow::Result;

use crate::store::KnowledgeEntry;

/// A detected conflict between knowledge entries.
#[derive(Debug, Clone)]
pub struct Conflict {
    pub unit_a_id: String,
    pub unit_b_id: String,
    pub conflict_type: ConflictType,
    pub resolution: ConflictResolution,
}

/// The type of conflict detected.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Two entries make opposite legality claims for overlapping scopes.
    ContradictoryLegality,
    /// Two entries have overlapping scopes but may not directly contradict.
    OverlappingScope,
    /// Confidence scores diverge significantly for the same claim.
    ConfidenceDivergence,
    /// Version mismatch between entries that should agree.
    VersionMismatch,
}

/// How the conflict should be resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictResolution {
    /// Entry with higher confidence wins.
    HigherConfidenceWins,
    /// Entry with higher priority wins.
    HigherPriorityWins,
    /// Newer entry wins.
    NewerWins,
    /// Manual review is required (high-confidence conflict).
    ManualReviewRequired,
}

/// Conflict detector implementation.
///
/// Scans knowledge entries for contradictions. The detector is deliberately
/// conservative: it only flags clear contradictions, not merely different
/// observations about different scopes.
pub struct ConflictDetector {
    /// Minimum confidence difference to consider as "divergence".
    confidence_threshold: f32,
}

impl ConflictDetector {
    /// Create a new conflict detector with default settings.
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.4,
        }
    }

    /// Detect conflicts among a set of knowledge entries.
    ///
    /// Compares all pairs of entries of the same knowledge type
    /// that have overlapping scopes. Returns conflicts where
    /// contradictory claims are detected.
    pub fn detect(&self, entries: &[KnowledgeEntry]) -> Result<Vec<Conflict>> {
        let mut conflicts = Vec::new();

        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let a = &entries[i];
                let b = &entries[j];

                // Only compare entries of the same knowledge type
                if a.unit.knowledge_type != b.unit.knowledge_type {
                    continue;
                }

                // Check for scope overlap
                if !scopes_overlap(&a.unit, &b.unit) {
                    continue;
                }

                // Check for specific contradiction types
                if let Some(conflict) = self.check_pair(a, b) {
                    conflicts.push(conflict);
                }
            }
        }

        Ok(conflicts)
    }

    /// Check a pair of entries for conflicts.
    fn check_pair(&self, a: &KnowledgeEntry, b: &KnowledgeEntry) -> Option<Conflict> {
        // Check for contradictory legality claims
        if a.unit.knowledge_type == KnowledgeType::LegalityRule {
            let a_legal = a.unit.payload.get("ane_legal").and_then(|v| v.as_bool());
            let b_legal = b.unit.payload.get("ane_legal").and_then(|v| v.as_bool());

            if let (Some(a_val), Some(b_val)) = (a_legal, b_legal) {
                if a_val != b_val {
                    return Some(Conflict {
                        unit_a_id: a.unit.id.clone(),
                        unit_b_id: b.unit.id.clone(),
                        conflict_type: ConflictType::ContradictoryLegality,
                        resolution: self.resolve_legality_conflict(a, b),
                    });
                }
            }
        }

        // Check for confidence divergence (same claim, very different confidence)
        let conf_diff = (a.unit.confidence - b.unit.confidence).abs();
        if conf_diff > self.confidence_threshold {
            // Check if they're making similar claims
            if self.same_claim(&a.unit, &b.unit) {
                return Some(Conflict {
                    unit_a_id: a.unit.id.clone(),
                    unit_b_id: b.unit.id.clone(),
                    conflict_type: ConflictType::ConfidenceDivergence,
                    resolution: ConflictResolution::HigherConfidenceWins,
                });
            }
        }

        None
    }

    /// Check if two units are making the same core claim
    /// (even if confidence differs).
    fn same_claim(&self, a: &KnowledgeUnit, b: &KnowledgeUnit) -> bool {
        // Same op pattern and same ane_legal value
        let a_pattern = a.payload.get("op_pattern").and_then(|v| v.as_str());
        let b_pattern = b.payload.get("op_pattern").and_then(|v| v.as_str());
        let a_legal = a.payload.get("ane_legal").and_then(|v| v.as_bool());
        let b_legal = b.payload.get("ane_legal").and_then(|v| v.as_bool());

        match (a_pattern, b_pattern, a_legal, b_legal) {
            (Some(ap), Some(bp), Some(al), Some(bl)) => ap == bp && al == bl,
            _ => false,
        }
    }

    /// Determine resolution for a legality conflict.
    fn resolve_legality_conflict(&self, a: &KnowledgeEntry, b: &KnowledgeEntry) -> ConflictResolution {
        // If either entry is high-confidence, require manual review
        if a.unit.confidence >= 0.8 || b.unit.confidence >= 0.8 {
            return ConflictResolution::ManualReviewRequired;
        }

        // If confidence is similar, use higher priority
        if (a.unit.confidence - b.unit.confidence).abs() < 0.1 {
            if a.unit.conflict_priority != b.unit.conflict_priority {
                return ConflictResolution::HigherPriorityWins;
            }
            return ConflictResolution::NewerWins;
        }

        ConflictResolution::HigherConfidenceWins
    }
}

/// Check if two knowledge units have overlapping scopes.
fn scopes_overlap(a: &KnowledgeUnit, b: &KnowledgeUnit) -> bool {
    let devices_overlap = a.scope.device_classes.iter().any(|d| b.scope.device_classes.contains(d))
        || a.scope.device_classes.contains(&"unknown".to_string())
        || b.scope.device_classes.contains(&"unknown".to_string());
    let os_overlap = a.scope.os_versions.iter().any(|v| b.scope.os_versions.contains(v))
        || a.scope.os_versions.contains(&"unknown".to_string())
        || b.scope.os_versions.contains(&"unknown".to_string());
    let opset_overlap = a.scope.opset_versions.iter().any(|v| b.scope.opset_versions.contains(v));

    devices_overlap && os_overlap && opset_overlap
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::kir::{EvidenceSource, KnowledgeScope};
    use crate::store::{EntryProvenance, EntryOrigin, EntrySource, ConflictStatus};
    use std::collections::HashMap;

    fn make_entry(id: &str, ane_legal: bool, confidence: f32, devices: Vec<&str>) -> KnowledgeEntry {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(ane_legal));
        payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));

        KnowledgeEntry {
            unit: KnowledgeUnit {
                id: id.to_string(),
                version: 1,
                timestamp: chrono::Utc::now().to_rfc3339(),
                knowledge_type: KnowledgeType::LegalityRule,
                confidence,
                evidence_source: EvidenceSource::SyntheticRun,
                evidence_count: 1,
                scope: KnowledgeScope {
                    device_classes: devices.into_iter().map(|d| d.to_string()).collect(),
                    os_versions: vec!["macOS_15".to_string()],
                    opset_versions: vec!["iOS18".to_string()],
                },
                conflict_priority: 0,
                payload,
            },
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

    #[test]
    fn test_no_conflicts_same_claim() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry("a", true, 0.5, vec!["M2"]),
            make_entry("b", true, 0.6, vec!["M2"]),
        ];
        let conflicts = detector.detect(&entries).unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_contradictory_legality() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry("a", true, 0.5, vec!["M2"]),
            make_entry("b", false, 0.5, vec!["M2"]),
        ];
        let conflicts = detector.detect(&entries).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].conflict_type, ConflictType::ContradictoryLegality);
    }

    #[test]
    fn test_no_conflict_different_scopes() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry("a", true, 0.5, vec!["M2"]),
            make_entry("b", false, 0.5, vec!["M4"]), // M4 != M2
        ];
        let conflicts = detector.detect(&entries).unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_high_confidence_requires_manual_review() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry("a", true, 0.9, vec!["M2"]),
            make_entry("b", false, 0.5, vec!["M2"]),
        ];
        let conflicts = detector.detect(&entries).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].resolution, ConflictResolution::ManualReviewRequired);
    }

    #[test]
    fn test_low_conflict_auto_resolved() {
        let detector = ConflictDetector::new();
        let entries = vec![
            make_entry("a", true, 0.3, vec!["M2"]),
            make_entry("b", false, 0.5, vec!["M2"]),
        ];
        let conflicts = detector.detect(&entries).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].resolution, ConflictResolution::HigherConfidenceWins);
    }
}
