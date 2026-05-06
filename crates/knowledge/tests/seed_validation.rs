//! T-129: Seed file validation integration test
//!
//! Validates that each seed file in the knowledge/ directory can be loaded
//! without error through its current load path. This tests the *current*
//! load paths, not schema conformance — seed files use flat formats that
//! don't yet match the KnowledgeEntry schema.

use ane_knowledge::shard_template::load_shard_template_seeds;
use ane_knowledge::store::KnowledgeStore;
use std::fs;
use std::path::Path;

/// Resolve the knowledge/ directory relative to the crate.
/// Cargo runs integration tests from the crate root, so we check
/// both workspace-relative and crate-relative paths.
fn knowledge_dir() -> Option<String> {
    let candidates = ["knowledge", "../../knowledge", "../../../knowledge"];
    for candidate in &candidates {
        if Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// List all JSON seed files in the knowledge/ directory.
fn list_seed_files(dir: &str) -> Vec<String> {
    let mut files = Vec::new();
    let path = Path::new(dir);
    if !path.exists() {
        return files;
    }
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    files.push(name.to_string());
                }
            }
        }
    }
    files.sort();
    files
}

// ─── Test: All seed files are valid JSON ─────────────────────────────

#[test]
fn test_all_seed_files_are_valid_json() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    let seed_files = list_seed_files(&dir);
    assert!(!seed_files.is_empty(), "Expected at least one seed file in knowledge/");

    for filename in &seed_files {
        let path = Path::new(&dir).join(filename);
        let json_str = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", filename, e));
        let parsed: serde_json::Value = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {}", filename, e));

        // Every seed file must have a "version" field at the top level
        assert!(
            parsed.get("version").is_some(),
            "{}: missing top-level 'version' field",
            filename
        );
    }
}

// ─── Test: Entries-pattern files have valid structure ─────────────────

#[test]
fn test_entries_pattern_files_have_valid_structure() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    // Files that use the entries[] pattern
    let entries_files = [
        "legality_seed.json",
        "precision_hazard_seed.json",
        "shard_template_seed.json",
        "decode_step_shard_template_seed.json",
    ];

    for filename in &entries_files {
        let path = Path::new(&dir).join(filename);
        if !path.exists() {
            continue;
        }

        let json_str = fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

        let entries = parsed.get("entries").and_then(|e| e.as_array());
        assert!(
            entries.is_some(),
            "{}: missing top-level 'entries' array",
            filename
        );

        let entries = entries.unwrap();
        assert!(
            !entries.is_empty(),
            "{}: 'entries' array is empty",
            filename
        );

        // Each entry must have 'id' and 'knowledge_type'
        for (i, entry) in entries.iter().enumerate() {
            assert!(
                entry.get("id").and_then(|v| v.as_str()).is_some(),
                "{}: entries[{}] missing 'id' string field",
                filename,
                i
            );
            assert!(
                entry.get("knowledge_type").and_then(|v| v.as_str()).is_some(),
                "{}: entries[{}] missing 'knowledge_type' string field",
                filename,
                i
            );
        }
    }
}

// ─── Test: Converted-format files have valid entries structure ────────
//
// These seed files were converted from flat format to the entries[]
// KnowledgeUnit schema. Each entry must have id, knowledge_type, and
// a payload object with type-specific fields.

#[test]
fn test_flat_format_files_have_valid_structure() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    // cpu_only_ops_seed.json: entries[] with CpuOnlyOps knowledge_type
    let cpu_path = Path::new(&dir).join("cpu_only_ops_seed.json");
    if cpu_path.exists() {
        let json_str = fs::read_to_string(&cpu_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let entries = parsed.get("entries").and_then(|e| e.as_array());
        assert!(entries.is_some(), "cpu_only_ops_seed.json: missing 'entries' array");
        let entries = entries.unwrap();
        assert!(!entries.is_empty(), "cpu_only_ops_seed.json: 'entries' array is empty");
        for (i, entry) in entries.iter().enumerate() {
            assert!(
                entry.get("id").and_then(|v| v.as_str()).is_some(),
                "cpu_only_ops_seed.json: entries[{}] missing 'id'",
                i
            );
            assert_eq!(
                entry.get("knowledge_type").and_then(|v| v.as_str()),
                Some("CpuOnlyOps"),
                "cpu_only_ops_seed.json: entries[{}] knowledge_type should be CpuOnlyOps",
                i
            );
            // mil_name and reason_code are in payload
            let payload = entry.get("payload").and_then(|v| v.as_object());
            assert!(payload.is_some(), "cpu_only_ops_seed.json: entries[{}] missing 'payload'", i);
            let payload = payload.unwrap();
            assert!(
                payload.get("mil_name").and_then(|v| v.as_str()).is_some(),
                "cpu_only_ops_seed.json: entries[{}].payload missing 'mil_name'",
                i
            );
            assert!(
                payload.get("reason_code").and_then(|v| v.as_str()).is_some(),
                "cpu_only_ops_seed.json: entries[{}].payload missing 'reason_code'",
                i
            );
        }
    }

    // ane_hw_limits_seed.json: entries[] with AneHwLimits knowledge_type
    let hw_path = Path::new(&dir).join("ane_hw_limits_seed.json");
    if hw_path.exists() {
        let json_str = fs::read_to_string(&hw_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let entries = parsed.get("entries").and_then(|e| e.as_array());
        assert!(entries.is_some(), "ane_hw_limits_seed.json: missing 'entries' array");
        let entries = entries.unwrap();
        assert!(!entries.is_empty(), "ane_hw_limits_seed.json: 'entries' array is empty");
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry.get("knowledge_type").and_then(|v| v.as_str()),
                Some("AneHwLimits"),
                "ane_hw_limits_seed.json: entries[{}] knowledge_type should be AneHwLimits",
                i
            );
            let payload = entry.get("payload").and_then(|v| v.as_object());
            assert!(payload.is_some(), "ane_hw_limits_seed.json: entries[{}] missing 'payload'", i);
            let payload = payload.unwrap();
            assert!(
                payload.get("revision").and_then(|v| v.as_str()).is_some(),
                "ane_hw_limits_seed.json: entries[{}].payload missing 'revision'",
                i
            );
            assert!(
                payload.get("family").and_then(|v| v.as_str()).is_some(),
                "ane_hw_limits_seed.json: entries[{}].payload missing 'family'",
                i
            );
        }
    }

    // ane_op_family_matrix.json: entries[] with AneOpFamilyMatrix knowledge_type
    let matrix_path = Path::new(&dir).join("ane_op_family_matrix.json");
    if matrix_path.exists() {
        let json_str = fs::read_to_string(&matrix_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let entries = parsed.get("entries").and_then(|e| e.as_array());
        assert!(entries.is_some(), "ane_op_family_matrix.json: missing 'entries' array");
        let entries = entries.unwrap();
        assert!(!entries.is_empty(), "ane_op_family_matrix.json: 'entries' array is empty");
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry.get("knowledge_type").and_then(|v| v.as_str()),
                Some("AneOpFamilyMatrix"),
                "ane_op_family_matrix.json: entries[{}] knowledge_type should be AneOpFamilyMatrix",
                i
            );
            let payload = entry.get("payload").and_then(|v| v.as_object());
            assert!(payload.is_some(), "ane_op_family_matrix.json: entries[{}] missing 'payload'", i);
            let payload = payload.unwrap();
            assert!(
                payload.get("mil_name").and_then(|v| v.as_str()).is_some(),
                "ane_op_family_matrix.json: entries[{}].payload missing 'mil_name'",
                i
            );
            assert!(
                payload.get("families").and_then(|v| v.as_object()).is_some(),
                "ane_op_family_matrix.json: entries[{}].payload missing 'families' object",
                i
            );
        }
    }

    // palettization_constraints_seed.json: entries[] with PalettizationConstraints knowledge_type
    let pal_path = Path::new(&dir).join("palettization_constraints_seed.json");
    if pal_path.exists() {
        let json_str = fs::read_to_string(&pal_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let entries = parsed.get("entries").and_then(|e| e.as_array());
        assert!(entries.is_some(), "palettization_constraints_seed.json: missing 'entries' array");
        let entries = entries.unwrap();
        assert!(!entries.is_empty(), "palettization_constraints_seed.json: 'entries' array is empty");
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(
                entry.get("knowledge_type").and_then(|v| v.as_str()),
                Some("PalettizationConstraints"),
                "palettization_constraints_seed.json: entries[{}] knowledge_type should be PalettizationConstraints",
                i
            );
        }
    }
}

// ─── Test: Shard template seeds load via dedicated loader ─────────────

#[test]
fn test_shard_template_seeds_load_successfully() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    let templates = load_shard_template_seeds(&dir)
        .expect("load_shard_template_seeds should not error");

    // Should load at least the Qwen3 three-shard and decode-step templates
    assert!(
        templates.len() >= 2,
        "Expected at least 2 shard template seeds, got {}",
        templates.len()
    );

    // Verify the Qwen3 template
    let qwen3 = templates
        .iter()
        .find(|t| t.seed_id == "shard_qwen3_three_shard_v1")
        .expect("Expected shard_qwen3_three_shard_v1 entry");
    assert_eq!(qwen3.template.template_id, "qwen3-three-shard-v1");
    assert_eq!(qwen3.template.partition_spec.len(), 3);
    assert!(qwen3.known_good);

    // Verify the decode-step template
    let decode_step = templates
        .iter()
        .find(|t| t.seed_id == "shard_decode_step_three_shard_v1")
        .expect("Expected shard_decode_step_three_shard_v1 entry");
    assert_eq!(decode_step.template.template_id, "decode-step-three-shard-v1");
    assert!(!decode_step.known_good);
}

// ─── Test: Knowledge store can load seeds from directory ──────────────

#[test]
fn test_store_load_seeds_from_directory_no_error() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("test_store");

    let mut store = KnowledgeStore::open(&store_path.to_string_lossy()).unwrap();

    // load_seeds_from_directory should not error, even though
    // many seed entries don't yet match the KnowledgeUnit schema
    // and will be silently skipped.
    let loaded = store
        .load_seeds_from_directory(&dir)
        .expect("load_seeds_from_directory should not error");

    // After the seed file migration to KnowledgeUnit schema, all entries
    // should load successfully. loaded should be > 0.
    eprintln!(
        "Loaded {} seed entries from knowledge/ (some may be skipped due to schema mismatch)",
        loaded
    );
}

// ─── Test: Expected seed files are present ────────────────────────────

#[test]
fn test_expected_seed_files_present() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    let expected_files = [
        "legality_seed.json",
        "precision_hazard_seed.json",
        "shard_template_seed.json",
        "decode_step_shard_template_seed.json",
        "cpu_only_ops_seed.json",
        "ane_hw_limits_seed.json",
        "ane_op_family_matrix.json",
        "palettization_constraints_seed.json",
    ];

    for filename in &expected_files {
        let path = Path::new(&dir).join(filename);
        assert!(
            path.exists(),
            "Expected seed file '{}' not found in knowledge/",
            filename
        );
    }
}

// ─── Test: precision_hazard_seed.json uses op_pattern in payload ─────

#[test]
fn test_precision_hazard_uses_op_pattern_field() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    let path = Path::new(&dir).join("precision_hazard_seed.json");
    if !path.exists() {
        return;
    }

    let json_str = fs::read_to_string(&path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    let entries = parsed.get("entries").and_then(|e| e.as_array()).unwrap();
    for (i, entry) in entries.iter().enumerate() {
        // op_pattern is now inside the payload object
        let payload = entry.get("payload").and_then(|v| v.as_object());
        assert!(
            payload.is_some(),
            "precision_hazard_seed.json: entries[{}] missing 'payload' object",
            i
        );
        let payload = payload.unwrap();
        // Must use "op_pattern", not the old "op" field name
        assert!(
            payload.get("op_pattern").is_some(),
            "precision_hazard_seed.json: entries[{}].payload missing 'op_pattern' field (was 'op' before T-129)",
            i
        );
        assert!(
            entry.get("op").is_none(),
            "precision_hazard_seed.json: entries[{}] still uses old 'op' field name (should be 'op_pattern' per T-129)",
            i
        );
    }
}

// ─── M-013: Test that default knowledge/ directory seeds load at runtime ─

/// M-013: Verify that seed loading works end-to-end with the default
/// knowledge/ directory. This test simulates the CLI startup path:
/// open a store, load seeds from the default knowledge/ directory,
/// and verify that seed data is queryable. This is the test that
/// catches the M-013 regression where seeds were validated in tests
/// but never loaded at runtime.
#[test]
fn test_m013_default_knowledge_dir_seed_loading() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    // Simulate the CLI startup path: create a store and load seeds
    let tmp = tempfile::tempdir().unwrap();
    let store_path = tmp.path().join("m013_test_store");

    let mut store = KnowledgeStore::open(&store_path.to_string_lossy())
        .expect("Failed to open knowledge store");

    // Load seeds from the default knowledge/ directory
    let loaded = store
        .load_seeds_from_directory(&dir)
        .expect("load_seeds_from_directory should not error");

    // After the KnowledgeUnit schema migration, seed entries should load.
    // Even if some entries are skipped due to schema mismatch, the call
    // itself must succeed and return > 0 for the three M-013 seed files.
    assert!(
        loaded > 0,
        "M-013: Expected at least one seed entry from default knowledge/, got 0. \
         Seed files exist but none loaded — check KnowledgeUnit schema conformance."
    );

    // Verify the store has seed entries (not observations)
    let (seeds, observations) = store.counts();
    assert_eq!(
        seeds, loaded,
        "M-013: Store seed count should match loaded count"
    );
    assert_eq!(
        observations, 0,
        "M-013: No observations should exist in a freshly seeded store"
    );

    // Verify that the seed IDs are queryable — the key M-013 fix is that
    // seeds are now functional at runtime, not just decorative.
    let seed_ids = store.list_seed_ids();
    assert!(
        !seed_ids.is_empty(),
        "M-013: Seed IDs should be queryable after loading"
    );

    // Verify at least one seed entry can be retrieved by ID
    let first_id = &seed_ids[0];
    let entry = store.get(first_id).expect("M-013: Seed entry should be retrievable by ID");
    assert!(
        entry.unit.id == *first_id,
        "M-013: Retrieved entry ID should match requested ID"
    );

    eprintln!(
        "M-013: Successfully loaded {} seed entries from default knowledge/ directory",
        loaded
    );
}
