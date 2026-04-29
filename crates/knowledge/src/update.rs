//! Update Pipeline
//!
//! Pipeline for ingesting new knowledge units into the store,
//! including validation, confidence recomputation, and conflict detection.
//!
//! The update pipeline validates incoming observations, computes initial
//! confidence scores, and inserts them into the knowledge store.
//! If the observation conflicts with an existing entry, the conflict
//! is recorded (not auto-resolved for high-confidence entries).

use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
use anyhow::{bail, Result};

use crate::store::KnowledgeStore;
use crate::ComputePlanObservation;

/// Pipeline for updating the knowledge store.
///
/// The pipeline validates observations before insertion:
/// 1. Structural validation (required fields present)
/// 2. Confidence sanity check (confidence in [0, 1])
/// 3. Evidence count sanity (at least 1)
/// 4. Insertion into the store (with conflict detection)
pub struct UpdatePipeline<'a> {
    store: &'a mut KnowledgeStore,
}

impl<'a> UpdatePipeline<'a> {
    /// Create a new update pipeline backed by the given store.
    pub fn new(store: &'a mut KnowledgeStore) -> Self {
        Self { store }
    }

    /// Ingest a new knowledge unit through the pipeline.
    ///
    /// Validation rules:
    /// - ID must not be empty
    /// - Confidence must be in [0.0, 1.0]
    /// - Evidence count must be >= 1
    /// - Knowledge type must be specified
    ///
    /// If validation passes, the unit is inserted as an observation.
    /// If it conflicts with an existing entry, the conflict is recorded
    /// in the entry's conflict_status field.
    pub fn ingest(&mut self, unit: KnowledgeUnit) -> Result<()> {
        // Validation
        self.validate(&unit)?;

        // Insert into store (conflict detection happens inside)
        self.store.insert_observation(unit)?;

        Ok(())
    }

    /// Batch ingest multiple knowledge units.
    pub fn ingest_batch(&mut self, units: Vec<KnowledgeUnit>) -> Result<()> {
        for unit in units {
            self.ingest(unit)?;
        }
        Ok(())
    }

    /// Validate a knowledge unit before insertion.
    fn validate(&self, unit: &KnowledgeUnit) -> Result<()> {
        if unit.id.is_empty() {
            bail!("Knowledge unit ID must not be empty");
        }
        if unit.confidence < 0.0 || unit.confidence > 1.0 {
            bail!(
                "Knowledge unit confidence must be in [0.0, 1.0], got {} for '{}'",
                unit.confidence,
                unit.id
            );
        }
        if unit.evidence_count == 0 {
            bail!("Knowledge unit evidence_count must be >= 1, got 0 for '{}'", unit.id);
        }
        Ok(())
    }
}

/// Compute an initial confidence score for a new observation
/// based on its evidence source.
///
/// These base values are deliberately conservative:
/// - Single observations never start above 0.5
/// - Deterministic failures (compile/load) start higher
/// - Cross-validated observations start highest
/// - Compute plan observations start at 0.9 (deterministic for given hardware+OS)
pub fn initial_confidence(source: &EvidenceSource, evidence_count: usize) -> f32 {
    let base = match source {
        EvidenceSource::SyntheticRun => 0.2,
        EvidenceSource::RealModelRun => 0.35,
        EvidenceSource::CompileFailure => 0.7,
        EvidenceSource::LoadFailure => 0.8,
        EvidenceSource::RuntimeAnomaly => 0.4,
        EvidenceSource::ManualEntry => 0.5,
        EvidenceSource::CrossValidated => 0.6,
        EvidenceSource::ComputePlan => 0.9,
    };

    // Slight bonus for multiple evidence points
    let bonus = if evidence_count > 1 { (evidence_count as f32).ln().max(0.0) * 0.02 } else { 0.0 };

    (base + bonus).min(1.0)
}

/// Update a confidence score using Bayesian-like update rule.
///
/// Given existing confidence `c_old` and new evidence with weight `w`:
/// `c_new = c_old + w * (1.0 - c_old) * agreement_factor`
///
/// Where `agreement_factor` is +1 for agreeing evidence and -0.5 for
/// disagreeing evidence (asymmetric: disagreement reduces confidence less
/// than agreement increases it).
pub fn update_confidence_bayesian(c_old: f32, evidence_weight: f32, agrees: bool) -> f32 {
    let agreement_factor = if agrees { 1.0 } else { -0.5 };
    let c_new = c_old + evidence_weight * (1.0 - c_old) * agreement_factor;
    c_new.clamp(0.0, 1.0)
}

/// Ingest compute plan observations into the knowledge store.
///
/// Each observation from the Python bridge's `compute_plan_harvest`
/// command is validated and converted into a `KnowledgeUnit` of type
/// `SurvivalMatrixEntry` with `EvidenceSource::ComputePlan`.
///
/// Validation rules:
/// - `op_pattern` must not be empty
/// - `ane_placed` must be a boolean (always true or false)
/// - `confidence` must be in [0.0, 1.0]
///
/// The generated `KnowledgeUnit` stores the op_pattern, device_class,
/// and ane_placed flag in its payload for downstream query by passes.
pub fn ingest_compute_plan_observations(
    store: &mut KnowledgeStore,
    observations: Vec<ComputePlanObservation>,
) -> Result<usize> {
    let mut ingested = 0;

    for obs in observations {
        // Validate required fields
        if obs.op_pattern.is_empty() {
            bail!("Compute plan observation op_pattern must not be empty");
        }
        if obs.confidence < 0.0 || obs.confidence > 1.0 {
            bail!(
                "Compute plan observation confidence must be in [0.0, 1.0], got {} for '{}'",
                obs.confidence,
                obs.op_pattern
            );
        }
        if obs.evidence_count == 0 {
            bail!(
                "Compute plan observation evidence_count must be >= 1, got 0 for '{}'",
                obs.op_pattern
            );
        }

        // Build the payload
        let mut payload = std::collections::HashMap::new();
        payload.insert("op_pattern".to_string(), serde_json::json!(obs.op_pattern));
        payload.insert("device_class".to_string(), serde_json::json!(obs.device_class));
        payload.insert("ane_placed".to_string(), serde_json::json!(obs.ane_placed));
        payload.insert("evidence_source".to_string(), serde_json::json!("compute_plan"));

        // Build the KnowledgeUnit
        let id = format!("cp_{}", obs.op_pattern);
        let unit = KnowledgeUnit {
            id,
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::SurvivalMatrixEntry,
            confidence: obs.confidence,
            evidence_source: EvidenceSource::ComputePlan,
            evidence_count: obs.evidence_count,
            scope: KnowledgeScope {
                device_classes: vec!["unknown".to_string()],
                os_versions: vec!["unknown".to_string()],
                opset_versions: vec!["unknown".to_string()],
            },
            conflict_priority: 0,
            payload,
        };

        // Insert into the store
        store.insert_observation(unit)?;
        ingested += 1;
    }

    Ok(ingested)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::kir::{KnowledgeScope, KnowledgeType};
    use std::collections::HashMap;

    fn make_unit(id: &str, confidence: f32, evidence_count: usize) -> KnowledgeUnit {
        let mut payload = HashMap::new();
        payload.insert("ane_legal".to_string(), serde_json::json!(true));

        KnowledgeUnit {
            id: id.to_string(),
            version: 1,
            timestamp: chrono::Utc::now().to_rfc3339(),
            knowledge_type: KnowledgeType::LegalityRule,
            confidence,
            evidence_source: EvidenceSource::SyntheticRun,
            evidence_count,
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
    fn test_ingest_valid_unit() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();
        let mut pipeline = UpdatePipeline::new(&mut store);

        let unit = make_unit("test_1", 0.3, 1);
        assert!(pipeline.ingest(unit).is_ok());
        assert!(store.get("test_1").is_some());
    }

    #[test]
    fn test_reject_empty_id() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();
        let mut pipeline = UpdatePipeline::new(&mut store);

        let unit = make_unit("", 0.3, 1);
        let result = pipeline.ingest(unit);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ID must not be empty"));
    }

    #[test]
    fn test_reject_bad_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();
        let mut pipeline = UpdatePipeline::new(&mut store);

        let unit = make_unit("bad_conf", 1.5, 1);
        let result = pipeline.ingest(unit);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("confidence must be in"));
    }

    #[test]
    fn test_reject_zero_evidence_count() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();
        let mut pipeline = UpdatePipeline::new(&mut store);

        let unit = make_unit("no_evidence", 0.3, 0);
        let result = pipeline.ingest(unit);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("evidence_count must be >= 1"));
    }

    #[test]
    fn test_initial_confidence() {
        assert_eq!(initial_confidence(&EvidenceSource::SyntheticRun, 1), 0.2);
        assert_eq!(initial_confidence(&EvidenceSource::RealModelRun, 1), 0.35);
        assert_eq!(initial_confidence(&EvidenceSource::CompileFailure, 1), 0.7);
        assert_eq!(initial_confidence(&EvidenceSource::LoadFailure, 1), 0.8);
        assert_eq!(initial_confidence(&EvidenceSource::CrossValidated, 1), 0.6);
        assert_eq!(initial_confidence(&EvidenceSource::ComputePlan, 1), 0.9);

        // Multiple evidence should give a slight bonus
        assert!(initial_confidence(&EvidenceSource::SyntheticRun, 5) > 0.2);
    }

    #[test]
    fn test_bayesian_confidence_update() {
        // Agreement increases confidence
        let c = update_confidence_bayesian(0.5, 0.3, true);
        assert!(c > 0.5);
        assert!(c <= 1.0);

        // Disagreement decreases confidence (less than agreement increases)
        let c = update_confidence_bayesian(0.5, 0.3, false);
        assert!(c < 0.5);
        assert!(c >= 0.0);

        // Confidence stays in bounds
        let c = update_confidence_bayesian(0.99, 1.0, true);
        assert!(c <= 1.0);

        let c = update_confidence_bayesian(0.01, 1.0, false);
        assert!(c >= 0.0);
    }

    #[test]
    fn test_ingest_compute_plan_observations() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let observations = vec![
            ComputePlanObservation {
                op_pattern: "linear_1".to_string(),
                device_class: "NeuralEngine".to_string(),
                ane_placed: true,
                confidence: 0.9,
                evidence_count: 1,
            },
            ComputePlanObservation {
                op_pattern: "reshape_1".to_string(),
                device_class: "CPU".to_string(),
                ane_placed: false,
                confidence: 0.9,
                evidence_count: 1,
            },
        ];

        let count = ingest_compute_plan_observations(&mut store, observations).unwrap();
        assert_eq!(count, 2);

        // Verify the ANE-placed entry
        let ane_entry = store.get("cp_linear_1").unwrap();
        assert_eq!(ane_entry.unit.knowledge_type, KnowledgeType::SurvivalMatrixEntry);
        assert_eq!(ane_entry.unit.evidence_source, EvidenceSource::ComputePlan);
        assert_eq!(ane_entry.unit.confidence, 0.9);
        assert_eq!(ane_entry.unit.payload.get("ane_placed").unwrap().as_bool(), Some(true));
        assert_eq!(ane_entry.unit.payload.get("op_pattern").unwrap().as_str(), Some("linear_1"));

        // Verify the CPU-placed entry
        let cpu_entry = store.get("cp_reshape_1").unwrap();
        assert_eq!(cpu_entry.unit.knowledge_type, KnowledgeType::SurvivalMatrixEntry);
        assert_eq!(cpu_entry.unit.payload.get("ane_placed").unwrap().as_bool(), Some(false));
    }

    #[test]
    fn test_reject_compute_plan_empty_op_pattern() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let observations = vec![ComputePlanObservation {
            op_pattern: "".to_string(),
            device_class: "CPU".to_string(),
            ane_placed: false,
            confidence: 0.9,
            evidence_count: 1,
        }];

        let result = ingest_compute_plan_observations(&mut store, observations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("op_pattern must not be empty"));
    }

    #[test]
    fn test_reject_compute_plan_bad_confidence() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = KnowledgeStore::open(&tmp.path().join("store").to_string_lossy()).unwrap();

        let observations = vec![ComputePlanObservation {
            op_pattern: "linear_1".to_string(),
            device_class: "NeuralEngine".to_string(),
            ane_placed: true,
            confidence: 1.5,
            evidence_count: 1,
        }];

        let result = ingest_compute_plan_observations(&mut store, observations);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("confidence must be in"));
    }
}
