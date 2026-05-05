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

// ─── Test: Flat-format files have valid structure ────────────────────

#[test]
fn test_flat_format_files_have_valid_structure() {
    let Some(dir) = knowledge_dir() else {
        eprintln!("Skipping: no knowledge/ directory found");
        return;
    };

    // cpu_only_ops_seed.json: must have cpu_only_ops[] array
    let cpu_path = Path::new(&dir).join("cpu_only_ops_seed.json");
    if cpu_path.exists() {
        let json_str = fs::read_to_string(&cpu_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let ops = parsed.get("cpu_only_ops").and_then(|e| e.as_array());
        assert!(ops.is_some(), "cpu_only_ops_seed.json: missing 'cpu_only_ops' array");
        let ops = ops.unwrap();
        assert!(!ops.is_empty(), "cpu_only_ops_seed.json: 'cpu_only_ops' array is empty");
        for (i, op) in ops.iter().enumerate() {
            assert!(
                op.get("mil_name").and_then(|v| v.as_str()).is_some(),
                "cpu_only_ops_seed.json: cpu_only_ops[{}] missing 'mil_name'",
                i
            );
            assert!(
                op.get("reason_code").and_then(|v| v.as_str()).is_some(),
                "cpu_only_ops_seed.json: cpu_only_ops[{}] missing 'reason_code'",
                i
            );
        }
    }

    // ane_hw_limits_seed.json: must have hw_limits[] array
    let hw_path = Path::new(&dir).join("ane_hw_limits_seed.json");
    if hw_path.exists() {
        let json_str = fs::read_to_string(&hw_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let limits = parsed.get("hw_limits").and_then(|e| e.as_array());
        assert!(limits.is_some(), "ane_hw_limits_seed.json: missing 'hw_limits' array");
        let limits = limits.unwrap();
        assert!(!limits.is_empty(), "ane_hw_limits_seed.json: 'hw_limits' array is empty");
        for (i, limit) in limits.iter().enumerate() {
            assert!(
                limit.get("revision").and_then(|v| v.as_str()).is_some(),
                "ane_hw_limits_seed.json: hw_limits[{}] missing 'revision'",
                i
            );
            assert!(
                limit.get("family").and_then(|v| v.as_str()).is_some(),
                "ane_hw_limits_seed.json: hw_limits[{}] missing 'family'",
                i
            );
        }
    }

    // ane_op_family_matrix.json: must have ane_landing_ops[] array
    let matrix_path = Path::new(&dir).join("ane_op_family_matrix.json");
    if matrix_path.exists() {
        let json_str = fs::read_to_string(&matrix_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let ops = parsed.get("ane_landing_ops").and_then(|e| e.as_array());
        assert!(ops.is_some(), "ane_op_family_matrix.json: missing 'ane_landing_ops' array");
        let ops = ops.unwrap();
        assert!(!ops.is_empty(), "ane_op_family_matrix.json: 'ane_landing_ops' array is empty");
        for (i, op) in ops.iter().enumerate() {
            assert!(
                op.get("mil_name").and_then(|v| v.as_str()).is_some(),
                "ane_op_family_matrix.json: ane_landing_ops[{}] missing 'mil_name'",
                i
            );
            assert!(
                op.get("families").and_then(|v| v.as_object()).is_some(),
                "ane_op_family_matrix.json: ane_landing_ops[{}] missing 'families' object",
                i
            );
        }
    }

    // palettization_constraints_seed.json: must have expected sub-objects
    let pal_path = Path::new(&dir).join("palettization_constraints_seed.json");
    if pal_path.exists() {
        let json_str = fs::read_to_string(&pal_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(
            parsed.get("conv_palette_minimums").is_some(),
            "palettization_constraints_seed.json: missing 'conv_palette_minimums'"
        );
        assert!(
            parsed.get("hard_rejections").is_some(),
            "palettization_constraints_seed.json: missing 'hard_rejections'"
        );
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

    // Currently, entries in the seed files lack required KnowledgeUnit
    // fields (version, timestamp, conflict_priority, payload), so they
    // are skipped. This is expected — the test just verifies no error.
    //
    // When the migration is complete (entries wrapped in KnowledgeEntry
    // format), loaded should be > 0.
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

// ─── Test: precision_hazard_seed.json uses op_pattern (not op) ────────

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
        // Must use "op_pattern", not the old "op" field name
        assert!(
            entry.get("op_pattern").is_some(),
            "precision_hazard_seed.json: entries[{}] missing 'op_pattern' field (was 'op' before T-129)",
            i
        );
        assert!(
            entry.get("op").is_none(),
            "precision_hazard_seed.json: entries[{}] still uses old 'op' field name (should be 'op_pattern' per T-129)",
            i
        );
    }
}
