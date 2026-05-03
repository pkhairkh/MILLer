//! #[should_panic] tests for invariant violations
//!
//! Tests that verify invalid operations panic appropriately:
//! - Knowledge store: invalid operations that should fail
//! - IR construction: invalid graph configurations

use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
use ane_knowledge::store::KnowledgeStore;
use ane_knowledge::update::UpdatePipeline;
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

// ─── Knowledge Store should_panic tests ──────────────────────────

#[test]
#[should_panic(expected = "Cannot overwrite seed entry")]
fn test_overwrite_seed_with_observation_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let store_path_str = store_path.to_string_lossy().to_string();
    let mut store = KnowledgeStore::open(&store_path_str).unwrap();

    // Create a seed directory with a seed file
    let seeds_dir = tmp.path().join("seeds");
    std::fs::create_dir_all(&seeds_dir).unwrap();
    let seed_json = serde_json::json!({
        "entries": [
            {
                "id": "seed_1",
                "version": 1,
                "timestamp": "2025-01-01T00:00:00Z",
                "knowledge_type": "LegalityRule",
                "confidence": 0.9,
                "evidence_source": "SyntheticRun",
                "evidence_count": 5,
                "scope": {
                    "device_classes": ["M2"],
                    "os_versions": ["macOS_15"],
                    "opset_versions": ["iOS18"]
                },
                "conflict_priority": 0,
                "payload": {"ane_legal": true, "op_pattern": "mb.matmul"}
            }
        ]
    });
    std::fs::write(seeds_dir.join("seed.json"), serde_json::to_string_pretty(&seed_json).unwrap())
        .unwrap();

    // Load seeds — this makes "seed_1" a seed entry
    store.load_seeds_from_directory(&seeds_dir.to_string_lossy()).unwrap();

    // Try to overwrite with observation — should return error, not panic
    // We wrap in unwrap() to convert the error to a panic
    let obs_unit = make_unit("seed_1", 0.3, 1);
    store.insert_observation(obs_unit).unwrap();
}

#[test]
#[should_panic(expected = "ID must not be empty")]
fn test_ingest_empty_id_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let mut pipeline = UpdatePipeline::new(&mut store);

    let unit = make_unit("", 0.5, 1);
    pipeline.ingest(unit).unwrap();
}

#[test]
#[should_panic(expected = "confidence must be in")]
fn test_ingest_out_of_bounds_confidence_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let mut pipeline = UpdatePipeline::new(&mut store);

    let unit = make_unit("bad_conf", 1.5, 1);
    pipeline.ingest(unit).unwrap();
}

#[test]
#[should_panic(expected = "evidence_count must be >= 1")]
fn test_ingest_zero_evidence_count_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let mut pipeline = UpdatePipeline::new(&mut store);

    let unit = make_unit("no_evidence", 0.5, 0);
    pipeline.ingest(unit).unwrap();
}

#[test]
#[should_panic(expected = "Store directory exists but has no store_index.json")]
fn test_open_corrupted_store_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("corrupted_store");

    // Create directory but no store_index.json
    std::fs::create_dir_all(&store_path).unwrap();

    // This should fail with an error about missing store_index.json
    KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
}

// ─── Additional should_panic tests (T-18) ─────────────────────────

#[test]
#[should_panic(expected = "confidence must be in")]
fn test_ingest_negative_confidence_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let mut pipeline = UpdatePipeline::new(&mut store);

    // Negative confidence is out of bounds [0.0, 1.0]
    let unit = make_unit("neg_conf", -0.5, 1);
    pipeline.ingest(unit).unwrap();
}

#[test]
#[should_panic(expected = "Failed to create store directory")]
fn test_open_store_on_file_path_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("not_a_dir");

    // Create a regular file at the path where a store directory should go
    std::fs::write(&file_path, "this is a file, not a directory").unwrap();

    // The inner "seeds" subdirectory creation should fail because the
    // parent path is a file, not a directory.
    // Note: KnowledgeStore::open first checks if path exists — it does
    // (it's a file), so it tries load_existing, which will fail because
    // there's no store_index.json. But let's check a path that forces
    // create_new: use a sub-path that doesn't exist yet but whose parent
    // is a file.
    let impossible_path = file_path.join("sub_store");
    KnowledgeStore::open(&impossible_path.to_string_lossy()).unwrap();
}

#[test]
#[should_panic(expected = "confidence must be in")]
fn test_ingest_batch_with_invalid_unit_panics() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let mut pipeline = UpdatePipeline::new(&mut store);

    // Batch with a valid unit followed by an invalid one
    let valid_unit = make_unit("valid_1", 0.5, 1);
    let invalid_unit = make_unit("bad_in_batch", 2.0, 1); // confidence > 1.0
    pipeline.ingest_batch(vec![valid_unit, invalid_unit]).unwrap();
}
