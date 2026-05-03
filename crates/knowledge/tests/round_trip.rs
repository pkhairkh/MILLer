//! Integration test: Knowledge store round-trip cycle
//!
//! Tests the full store → query → persist → reload cycle,
//! verifying that knowledge entries survive persistence and
//! can be queried after reload.

use ane_ir::kir::{EvidenceSource, KnowledgeScope, KnowledgeType, KnowledgeUnit};
use ane_knowledge::query::{KnowledgeQuery, KnowledgeQueryable};
use ane_knowledge::snapshot::{SnapshotExport, SnapshotImport};
use ane_knowledge::store::KnowledgeStore;
use ane_knowledge::update::UpdatePipeline;
use std::collections::HashMap;

fn make_unit(id: &str, kt: KnowledgeType, confidence: f32) -> KnowledgeUnit {
    let mut payload = HashMap::new();
    payload.insert("ane_legal".to_string(), serde_json::json!(true));
    payload.insert("op_pattern".to_string(), serde_json::json!("mb.matmul"));

    KnowledgeUnit {
        id: id.to_string(),
        version: 1,
        timestamp: chrono::Utc::now().to_rfc3339(),
        knowledge_type: kt,
        confidence,
        evidence_source: EvidenceSource::SyntheticRun,
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
fn test_store_query_persist_reload_cycle() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("test_store");
    let store_path_str = store_path.to_string_lossy().to_string();

    // Step 1: Create store and insert entries
    {
        let mut store = KnowledgeStore::open(&store_path_str).unwrap();

        let unit_a = make_unit("obs_a", KnowledgeType::LegalityRule, 0.5);
        let unit_b = make_unit("obs_b", KnowledgeType::PrecisionHazard, 0.7);
        let unit_c = make_unit("obs_c", KnowledgeType::LegalityRule, 0.9);

        store.insert_observation(unit_a).unwrap();
        store.insert_observation(unit_b).unwrap();
        store.insert_observation(unit_c).unwrap();

        // Step 2: Query before persist
        let results =
            store.query(&KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule)).unwrap();
        assert_eq!(results.len(), 2, "Should find 2 LegalityRule entries");

        let best = store
            .query_best(&KnowledgeQuery::new().with_type(KnowledgeType::LegalityRule))
            .unwrap();
        assert!(best.is_some());
        assert_eq!(best.unwrap().confidence, 0.9, "Best should be highest confidence");
    }

    // Step 3: Reload store from disk
    {
        let store = KnowledgeStore::open(&store_path_str).unwrap();
        assert_eq!(store.list_ids().len(), 3, "All 3 entries must survive reload");

        // Step 4: Query after reload
        let results =
            store.query(&KnowledgeQuery::new().with_type(KnowledgeType::PrecisionHazard)).unwrap();
        assert_eq!(results.len(), 1, "PrecisionHazard entry must survive reload");
        assert_eq!(results[0].id, "obs_b");
    }
}

#[test]
fn test_snapshot_export_import_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let snapshot_path = tmp.path().join("snapshot.json").to_string_lossy().to_string();
    let import_store_path = tmp.path().join("imported_store");

    // Step 1: Create and populate store
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();
    let unit = make_unit("snapshot_obs", KnowledgeType::LegalityRule, 0.8);
    store.insert_observation(unit).unwrap();

    // Step 2: Export snapshot
    SnapshotExport::export_store(&store, &snapshot_path).unwrap();

    // Step 3: Import snapshot into a new store
    let snapshot = SnapshotImport::import_json(&snapshot_path).unwrap();
    assert_eq!(snapshot.observations.len(), 1);
    assert_eq!(snapshot.observations[0].unit.id, "snapshot_obs");

    let mut imported_store = KnowledgeStore::open(&import_store_path.to_string_lossy()).unwrap();
    let stats = SnapshotImport::import_into_store(&mut imported_store, &snapshot).unwrap();
    assert_eq!(stats.observations_imported, 1);

    // Step 4: Verify the imported store has the entry
    let entry = imported_store.get("snapshot_obs");
    assert!(entry.is_some(), "Imported entry must be queryable");
    assert_eq!(entry.unwrap().unit.confidence, 0.8);
}

#[test]
fn test_update_pipeline_and_query_cycle() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

    // Step 1: Ingest through update pipeline
    let mut pipeline = UpdatePipeline::new(&mut store);
    let unit = make_unit("pipeline_obs", KnowledgeType::LegalityRule, 0.4);
    pipeline.ingest(unit).unwrap();

    // Step 2: Query through the same store
    let results = store.query(&KnowledgeQuery::new().with_min_confidence(0.3)).unwrap();
    assert!(!results.is_empty(), "Pipeline-ingested entry must be queryable");

    // Step 3: Verify persistence by reloading
    let store_path_str = store_path.to_string_lossy().to_string();
    drop(store);
    let reloaded_store = KnowledgeStore::open(&store_path_str).unwrap();
    assert!(reloaded_store.get("pipeline_obs").is_some());
}

#[test]
fn test_confidence_filtered_query_after_reload() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let store_path_str = store_path.to_string_lossy().to_string();

    // Insert entries with different confidence levels
    {
        let mut store = KnowledgeStore::open(&store_path_str).unwrap();
        let low = make_unit("low_conf", KnowledgeType::LegalityRule, 0.2);
        let mid = make_unit("mid_conf", KnowledgeType::LegalityRule, 0.5);
        let high = make_unit("high_conf", KnowledgeType::LegalityRule, 0.9);
        store.insert_observation(low).unwrap();
        store.insert_observation(mid).unwrap();
        store.insert_observation(high).unwrap();
    }

    // Reload and query with confidence filter
    let store = KnowledgeStore::open(&store_path_str).unwrap();
    let results = store.query(&KnowledgeQuery::new().with_min_confidence(0.5)).unwrap();
    assert_eq!(results.len(), 2, "Only entries with confidence >= 0.5 should match");
}

#[test]
fn test_store_counts_after_reload() {
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("store");
    let store_path_str = store_path.to_string_lossy().to_string();

    // Insert 3 observations
    {
        let mut store = KnowledgeStore::open(&store_path_str).unwrap();
        for i in 0..3 {
            let unit = make_unit(&format!("obs_{}", i), KnowledgeType::LegalityRule, 0.5);
            store.insert_observation(unit).unwrap();
        }
        assert_eq!(store.counts(), (0, 3));
    }

    // Reload and verify counts
    let store = KnowledgeStore::open(&store_path_str).unwrap();
    assert_eq!(store.counts(), (0, 3), "Counts must persist across reload");
}
