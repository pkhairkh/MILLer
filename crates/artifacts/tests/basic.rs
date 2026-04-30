//! Basic integration tests for the artifacts crate.
//!
//! Tests manifest generation, content hashing, and serialization
//! through the public API.

use ane_artifacts::hashing::{hash_bytes, hash_file, verify_hash};
use ane_artifacts::manifest::{
    ArtifactManifest, FunctionDescriptor, HandoffEntry, MirOpEntry, PackageEntry, StateEntry,
    TensorSpec,
};

// ─── Content Hashing Tests ────────────────────────────────────────

#[test]
fn test_hash_bytes_deterministic_and_prefixed() {
    let hash_a = hash_bytes(b"test content");
    let hash_b = hash_bytes(b"test content");
    assert_eq!(hash_a, hash_b, "Same input must produce identical hashes");
    assert!(
        hash_a.starts_with("sha256:"),
        "Hash must be prefixed with 'sha256:', got: {}",
        hash_a
    );
    // SHA-256 hex digest is 64 characters after the prefix
    let hex_part = &hash_a[7..];
    assert_eq!(hex_part.len(), 64, "SHA-256 hex digest must be 64 chars");
}

#[test]
fn test_hash_bytes_known_empty_input() {
    let hash = hash_bytes(&[]);
    // Well-known SHA-256 of empty input
    assert_eq!(
        hash,
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
}

#[test]
fn test_hash_bytes_different_inputs_diverge() {
    let hash_a = hash_bytes(b"alpha");
    let hash_b = hash_bytes(b"beta");
    assert_ne!(hash_a, hash_b, "Different inputs must produce different hashes");
}

#[test]
fn test_hash_file_and_verify_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("artifact.bin");
    std::fs::write(&file_path, b"some artifact data").unwrap();

    let path_str = file_path.to_string_lossy().to_string();
    let hash = hash_file(&path_str).unwrap();
    assert!(hash.starts_with("sha256:"));

    // Verify matches
    assert!(verify_hash(&path_str, &hash).unwrap(), "Hash verification must succeed");

    // Wrong hash must not match
    let wrong = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    assert!(!verify_hash(&path_str, wrong).unwrap(), "Wrong hash must not match");
}

// ─── Manifest Generation Tests ────────────────────────────────────

#[test]
fn test_manifest_generation_with_packages_and_state() {
    let manifest = ArtifactManifest {
        version: "1.0.0".into(),
        model_id: "gpt2_prefill".into(),
        task_hash: "sha256:abcdef1234567890".into(),
        created_at: "2025-06-15T12:00:00Z".into(),
        packages: vec![PackageEntry {
            name: "prefill_pkg".into(),
            role: "prefill".into(),
            path: Some("/out/prefill.mlpackage".into()),
            content_hash: Some("sha256:pkg_hash".into()),
            size_bytes: 2048,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorSpec {
                    name: "input_ids".into(),
                    shape: vec![1, 64],
                    dtype: "int32".into(),
                }],
                outputs: vec![TensorSpec {
                    name: "logits".into(),
                    shape: vec![1, 64, 50257],
                    dtype: "fp16".into(),
                }],
                stateful: false,
                emission_status: "emitted".into(),
                mir_ops: vec![
                    MirOpEntry { op_type: "Linear".into() },
                    MirOpEntry { op_type: "Gelu".into() },
                ],
            }],
        }],
        state_declarations: vec![StateEntry {
            state_id: "kv_cache".into(),
            shape: vec![1, 12, 64, 128],
            dtype: "fp16".into(),
            owner_package: "prefill_pkg".into(),
        }],
        handoffs: vec![],
        compiler_version: "0.2.0".into(),
        implementation_status: "host_compiled".into(),
        verification_scope: "host_compile_only".into(),
        environment_limitations: vec!["no_apple_hardware".into()],
    };

    // Serialize to JSON and back
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let back: ArtifactManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(back.version, "1.0.0");
    assert_eq!(back.model_id, "gpt2_prefill");
    assert_eq!(back.packages.len(), 1);
    assert_eq!(back.packages[0].functions.len(), 1);
    assert_eq!(back.packages[0].functions[0].mir_ops.len(), 2);
    assert_eq!(back.packages[0].functions[0].mir_ops[0].op_type, "Linear");
    assert_eq!(back.state_declarations.len(), 1);
    assert_eq!(back.state_declarations[0].state_id, "kv_cache");
}

#[test]
fn test_manifest_generation_with_handoffs() {
    let manifest = ArtifactManifest {
        version: "1.0.0".into(),
        model_id: "multi_fn_model".into(),
        task_hash: "sha256:handoff_test".into(),
        created_at: "2025-06-15T12:00:00Z".into(),
        packages: vec![
            PackageEntry {
                name: "embed_pkg".into(),
                role: "embedding".into(),
                path: Some("/out/embed.mlpackage".into()),
                content_hash: None,
                size_bytes: 512,
                functions: vec![FunctionDescriptor {
                    name: "embedding".into(),
                    inputs: vec![],
                    outputs: vec![TensorSpec {
                        name: "hidden".into(),
                        shape: vec![1, 128],
                        dtype: "fp16".into(),
                    }],
                    stateful: false,
                    emission_status: "emitted".into(),
                    mir_ops: vec![],
                }],
            },
            PackageEntry {
                name: "decode_pkg".into(),
                role: "decode".into(),
                path: Some("/out/decode.mlpackage".into()),
                content_hash: None,
                size_bytes: 1024,
                functions: vec![FunctionDescriptor {
                    name: "decode_step".into(),
                    inputs: vec![TensorSpec {
                        name: "hidden_in".into(),
                        shape: vec![1, 128],
                        dtype: "fp16".into(),
                    }],
                    outputs: vec![],
                    stateful: true,
                    emission_status: "emitted".into(),
                    mir_ops: vec![],
                }],
            },
        ],
        state_declarations: vec![],
        handoffs: vec![HandoffEntry {
            from_package: "embed_pkg".into(),
            to_package: "decode_pkg".into(),
            tensor_name: "hidden".into(),
            shape: vec![1, 128],
            dtype: "fp16".into(),
        }],
        compiler_version: "0.2.0".into(),
        implementation_status: "host_compiled".into(),
        verification_scope: "host_compile_only".into(),
        environment_limitations: vec![],
    };

    let json = serde_json::to_string(&manifest).unwrap();
    let back: ArtifactManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(back.handoffs.len(), 1);
    assert_eq!(back.handoffs[0].from_package, "embed_pkg");
    assert_eq!(back.handoffs[0].to_package, "decode_pkg");
    assert_eq!(back.handoffs[0].tensor_name, "hidden");
}

#[test]
fn test_manifest_json_contains_required_fields() {
    let manifest = ArtifactManifest {
        version: "1.0.0".into(),
        model_id: "field_test".into(),
        task_hash: "sha256:field_check".into(),
        created_at: "2025-01-01T00:00:00Z".into(),
        packages: vec![],
        state_declarations: vec![],
        handoffs: vec![],
        compiler_version: "0.1.0".into(),
        implementation_status: "host_compiled".into(),
        verification_scope: "host_compile_only".into(),
        environment_limitations: vec!["no_apple_hardware".into()],
    };

    let value = serde_json::to_value(&manifest).unwrap();

    // All required fields must be present
    for field in &[
        "version",
        "model_id",
        "task_hash",
        "created_at",
        "packages",
        "state_declarations",
        "handoffs",
        "compiler_version",
        "implementation_status",
        "verification_scope",
        "environment_limitations",
    ] {
        assert!(value.get(field).is_some(), "Missing required field: {}", field);
    }

    // environment_limitations should be a non-empty array
    let limits = value.get("environment_limitations").unwrap().as_array().unwrap();
    assert!(!limits.is_empty());
}
