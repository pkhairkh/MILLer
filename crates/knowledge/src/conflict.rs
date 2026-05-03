//! Conflict Detection
//!
//! Detects and resolves conflicts between knowledge units
//! that make contradictory claims within overlapping scopes.
//!
//! Conflicts are detected automatically when entries are inserted
//! via the knowledge store. The ConflictDetector can also be used
//! standalone to scan a set of entries.

use ane_ir::kir::{KnowledgeType, KnowledgeUnit};
use anyhow::Result;

use crate::store::KnowledgeEntry;
use crate::util::{payload_ane_legal, payload_op_pattern, scopes_overlap};

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

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflictDetector {
    /// Create a new conflict detector with default settings.
    pub fn new() -> Self {
        Self { confidence_threshold: 0.4 }
    }

    /// Detect conflicts among a set of knowledge entries.
    ///
    /// Compares all pairs of entries of the same knowledge type
    /// that have overlapping scopes. Returns conflicts where
    /// contradictory claims are detected.
    ///
    /// The algorithm skips entries with different knowledge types
    /// early (O(1) per pair), and skips non-overlapping scopes
    /// before checking for contradictions. When a conflict is
    /// found, the pair is not re-examined.
    pub fn detect(&self, entries: &[KnowledgeEntry]) -> Result<Vec<Conflict>> {
        let mut conflicts = Vec::new();

        // Group entries by knowledge type for O(n·k) instead of O(n²)
        // where k is the average number of entries per type.
        let mut type_groups: std::collections::HashMap<KnowledgeType, Vec<usize>> =
            std::collections::HashMap::new();
        for (i, entry) in entries.iter().enumerate() {
            type_groups.entry(entry.unit.knowledge_type).or_default().push(i);
        }

        for indices in type_groups.values() {
            for i_pos in 0..indices.len() {
                let i = indices[i_pos];
                for &j in &indices[i_pos + 1..] {
                    let a = &entries[i];
                    let b = &entries[j];

                    // Check for scope overlap
                    if !scopes_overlap(&a.unit.scope, &b.unit.scope) {
                        continue;
                    }

                    // Check for specific contradiction types
                    if let Some(conflict) = self.check_pair(a, b) {
                        conflicts.push(conflict);
                    }
                }
            }
        }

        Ok(conflicts)
    }

    /// Check a pair of entries for conflicts.
    fn check_pair(&self, a: &KnowledgeEntry, b: &KnowledgeEntry) -> Option<Conflict> {
        // Check for contradictory legality claims
        if a.unit.knowledge_type == KnowledgeType::LegalityRule {
            let a_legal = payload_ane_legal(&a.unit.payload);
            let b_legal = payload_ane_legal(&b.unit.payload);

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
        // Same op pattern and same ane_legal value (using typed accessors)
        let a_pattern = payload_op_pattern(&a.payload);
        let b_pattern = payload_op_pattern(&b.payload);
        let a_legal = payload_ane_legal(&a.payload);
        let b_legal = payload_ane_legal(&b.payload);

        match (a_pattern, b_pattern, a_legal, b_legal) {
            (Some(ap), Some(bp), Some(al), Some(bl)) => ap == bp && al == bl,
            _ => false,
        }
    }

    /// Determine resolution for a legality conflict.
    fn resolve_legality_conflict(
        &self,
        a: &KnowledgeEntry,
        b: &KnowledgeEntry,
    ) -> ConflictResolution {
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

// scopes_overlap is now provided by crate::util (takes &KnowledgeScope)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ConflictStatus, EntryOrigin, EntryProvenance, EntrySource};
    use ane_ir::kir::{EvidenceSource, KnowledgeScope};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_entry(
        id: &str,
        ane_legal: bool,
        confidence: f32,
        devices: Vec<&str>,
    ) -> KnowledgeEntry {
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
                evidence_count: 1,
                scope: KnowledgeScope {
                    device_classes: devices.into_iter().map(|d| d.to_string()).collect(),
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

    #[test]
    fn test_no_conflicts_same_claim() {
        let detector = ConflictDetector::new();
        let entries =
            vec![make_entry("a", true, 0.5, vec!["M2"]), make_entry("b", true, 0.6, vec!["M2"])];
        let conflicts = detector.detect(&entries).unwrap();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn test_contradictory_legality() {
        let detector = ConflictDetector::new();
        let entries =
            vec![make_entry("a", true, 0.5, vec!["M2"]), make_entry("b", false, 0.5, vec!["M2"])];
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
        let entries =
            vec![make_entry("a", true, 0.9, vec!["M2"]), make_entry("b", false, 0.5, vec!["M2"])];
        let conflicts = detector.detect(&entries).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].resolution, ConflictResolution::ManualReviewRequired);
    }

    #[test]
    fn test_low_conflict_auto_resolved() {
        let detector = ConflictDetector::new();
        let entries =
            vec![make_entry("a", true, 0.3, vec!["M2"]), make_entry("b", false, 0.5, vec!["M2"])];
        let conflicts = detector.detect(&entries).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].resolution, ConflictResolution::HigherConfidenceWins);
    }
}
