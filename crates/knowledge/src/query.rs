//! Knowledge Query
//!
//! Trait and implementations for querying the knowledge store
//! with type-safe, scoped, and confidence-filtered lookups.

use ane_ir::kir::{KnowledgeUnit, KnowledgeType, KnowledgeScope, EvidenceSource};
use anyhow::Result;

use crate::store::KnowledgeStore;

/// A query against the knowledge store.
#[derive(Debug, Clone)]
pub struct KnowledgeQuery {
    /// Filter by knowledge type.
    pub knowledge_type: Option<KnowledgeType>,
    /// Filter by scope (device class, OS version, opset).
    pub scope: Option<KnowledgeScope>,
    /// Minimum confidence threshold.
    pub min_confidence: Option<f32>,
    /// Filter by evidence source.
    pub evidence_source: Option<EvidenceSource>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

/// Trait for querying knowledge units.
pub trait KnowledgeQueryable {
    /// Execute a query and return matching knowledge units.
    fn query(&self, query: &KnowledgeQuery) -> Result<Vec<KnowledgeUnit>>;

    /// Find the single best-matching knowledge unit for a query.
    fn query_best(&self, query: &KnowledgeQuery) -> Result<Option<KnowledgeUnit>> {
        let mut results = self.query(query)?;
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().next())
    }
}

impl KnowledgeQuery {
    /// Create a new empty query.
    pub fn new() -> Self {
        Self {
            knowledge_type: None,
            scope: None,
            min_confidence: None,
            evidence_source: None,
            limit: None,
        }
    }

    /// Filter by knowledge type.
    pub fn with_type(mut self, kt: KnowledgeType) -> Self {
        self.knowledge_type = Some(kt);
        self
    }

    /// Filter by minimum confidence.
    pub fn with_min_confidence(mut self, conf: f32) -> Self {
        self.min_confidence = Some(conf);
        self
    }

    /// Filter by evidence source.
    pub fn with_evidence_source(mut self, source: EvidenceSource) -> Self {
        self.evidence_source = Some(source);
        self
    }

    /// Filter by scope.
    pub fn with_scope(mut self, scope: KnowledgeScope) -> Self {
        self.scope = Some(scope);
        self
    }

    /// Limit number of results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Implement the queryable trait for KnowledgeStore.
///
/// Queries filter the in-memory index by type, scope, confidence,
/// and evidence source. Scope matching is conservative: an entry
/// with "unknown" scope matches any query scope.
impl KnowledgeQueryable for KnowledgeStore {
    fn query(&self, query: &KnowledgeQuery) -> Result<Vec<KnowledgeUnit>> {
        let mut results: Vec<KnowledgeUnit> = self.index_values()
            .filter(|entry| {
                // Filter by knowledge type
                if let Some(ref kt) = query.knowledge_type {
                    if entry.unit.knowledge_type != *kt {
                        return false;
                    }
                }

                // Filter by minimum confidence
                if let Some(min_conf) = query.min_confidence {
                    if entry.unit.confidence < min_conf {
                        return false;
                    }
                }

                // Filter by evidence source
                if let Some(ref source) = query.evidence_source {
                    if entry.unit.evidence_source != *source {
                        return false;
                    }
                }

                // Filter by scope (if specified, check for overlap)
                if let Some(ref scope) = query.scope {
                    if !scopes_match(&entry.unit.scope, scope) {
                        return false;
                    }
                }

                true
            })
            .map(|entry| entry.unit.clone())
            .collect();

        // Sort by confidence descending
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Apply limit
        if let Some(limit) = query.limit {
            results.truncate(limit);
        }

        Ok(results)
    }
}

/// Check if an entry's scope matches a query scope.
///
/// An entry matches if its scope overlaps with the query scope.
/// Entries with "unknown" scope match any query (conservative).
fn scopes_match(entry_scope: &KnowledgeScope, query_scope: &KnowledgeScope) -> bool {
    // "unknown" entries match any query
    if entry_scope.device_classes.contains(&"unknown".to_string()) {
        return true;
    }

    // Check device class overlap
    let device_match = entry_scope.device_classes.iter()
        .any(|d| query_scope.device_classes.contains(d));

    // Check OS version overlap
    let os_match = entry_scope.os_versions.iter()
        .any(|v| query_scope.os_versions.contains(v))
        || entry_scope.os_versions.contains(&"unknown".to_string())
        || query_scope.os_versions.is_empty();

    // Check opset version overlap
    let opset_match = entry_scope.opset_versions.iter()
        .any(|v| query_scope.opset_versions.contains(v))
        || query_scope.opset_versions.is_empty();

    device_match && os_match && opset_match
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::KnowledgeStore;
    use ane_ir::kir::KnowledgeType;
    use std::collections::HashMap;

    fn make_unit(id: &str, kt: KnowledgeType, confidence: f32, source: EvidenceSource) -> KnowledgeUnit {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(true));
        payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));

        KnowledgeUnit {
            id: id.to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: kt,
            confidence,
            evidence_source: source,
            evidence_count: 1,
            scope: KnowledgeScope {
                device_classes: vec!["M2".to_string()],
                os_versions: vec!["macOS_15".to_string()],
                opset_versions: vec!["iOS18".to_string()],
            },
            conflict_priority: 0,
            payload,
        }
    }

    #[test]
    fn test_query_by_type() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let unit_a = make_unit("a", KnowledgeType::LegalityRule, 0.5, EvidenceSource::SyntheticRun);
        let unit_b = make_unit("b", KnowledgeType::PrecisionHazard, 0.7, EvidenceSource::RealModelRun);
        store.insert_observation(unit_a).unwrap();
        store.insert_observation(unit_b).unwrap();

        let results = store.query(&KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_query_by_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let unit_low = make_unit("low", KnowledgeType::LegalityRule, 0.2, EvidenceSource::SyntheticRun);
        let unit_high = make_unit("high", KnowledgeType::LegalityRule, 0.8, EvidenceSource::RealModelRun);
        store.insert_observation(unit_low).unwrap();
        store.insert_observation(unit_high).unwrap();

        let results = store.query(&KnowledgeQuery::new().with_min_confidence(0.5)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "high");
    }

    #[test]
    fn test_query_best() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let unit_a = make_unit("a", KnowledgeType::LegalityRule, 0.3, EvidenceSource::SyntheticRun);
        let unit_b = make_unit("b", KnowledgeType::LegalityRule, 0.9, EvidenceSource::RealModelRun);
        store.insert_observation(unit_a).unwrap();
        store.insert_observation(unit_b).unwrap();

        let best = store.query_best(&KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule)).unwrap();
        assert!(best.is_some());
        assert_eq!(best.unwrap().id, "b");
    }

    #[test]
    fn test_query_by_evidence_source() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let unit_a = make_unit("a", KnowledgeType::LegalityRule, 0.5, EvidenceSource::SyntheticRun);
        let unit_b = make_unit("b", KnowledgeType::LegalityRule, 0.7, EvidenceSource::RealModelRun);
        store.insert_observation(unit_a).unwrap();
        store.insert_observation(unit_b).unwrap();

        let results = store.query(&KnowledgeQuery::new().with_evidence_source(EvidenceSource::RealModelRun)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }
}
