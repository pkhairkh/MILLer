//! Unit tests for the artifacts crate
//!
//! Tests manifest generation, content hashing, and serialization.

use crate::hashing::{hash_bytes, verify_hash};
use crate::manifest::{
    ArtifactManifest, FunctionDescriptor, HandoffEntry, MirOpEntry, PackageEntry, StateEntry,
    TensorSpec,
};

#[test]
fn test_hash_bytes_empty() {
    let hash = hash_bytes(&[]);
    assert!(hash.starts_with("sha256:"), "Hash must be prefixed with sha256:");
    // SHA-256 of empty input is well-known
    assert_eq!(hash, "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

#[test]
fn test_hash_bytes_hello() {
    let hash = hash_bytes(b"hello");
    assert!(hash.starts_with("sha256:"));
    // Verify deterministic
    let hash2 = hash_bytes(b"hello");
    assert_eq!(hash, hash2, "Same input must produce same hash");
}

#[test]
fn test_hash_bytes_different_inputs() {
    let hash_a = hash_bytes(b"input_a");
    let hash_b = hash_bytes(b"input_b");
    assert_ne!(hash_a, hash_b, "Different inputs must produce different hashes");
}

#[test]
fn test_manifest_serialization_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let model_path = tmp.path().join("main.mlpackage").to_string_lossy().to_string();

    let manifest = ArtifactManifest {
        version: "1.0.0".into(),
        model_id: "test_model".into(),
        task_hash: "sha256:abc123".into(),
        created_at: "2025-01-01T00:00:00Z".into(),
        packages: vec![PackageEntry {
            name: "main_pkg".into(),
            role: "prefill".into(),
            path: Some(model_path.into()),
            content_hash: Some("sha256:def456".into()),
            size_bytes: 1024,
            functions: vec![FunctionDescriptor {
                name: "main".into(),
                inputs: vec![TensorSpec {
                    name: "x".into(),
                    shape: vec![1, 128],
                    dtype: "fp16".into(),
                }],
                outputs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![1, 128],
                    dtype: "fp16".into(),
                }],
                stateful: false,
                emission_status: "emitted".into(),
                mir_ops: vec![MirOpEntry { op_type: "Linear".into() }],
            }],
        }],
        state_declarations: vec![StateEntry {
            state_id: "kv_cache".into(),
            shape: vec![1, 32, 128],
            dtype: "fp16".into(),
            owner_package: "main_pkg".into(),
        }],
        handoffs: vec![HandoffEntry {
            from_package: "pkg_a".into(),
            to_package: "pkg_b".into(),
            tensor_name: "hidden".into(),
            shape: vec![1, 128],
            dtype: "fp16".into(),
        }],
        compiler_version: "0.1.0".into(),
        implementation_status: "host_compiled".into(),
        verification_scope: "host_compile_only".into(),
        environment_limitations: vec!["no_apple_hardware".into()],
    };

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&manifest).unwrap();

    // Deserialize back
    let back: ArtifactManifest = serde_json::from_str(&json).unwrap();

    assert_eq!(back.version, manifest.version);
    assert_eq!(back.model_id, manifest.model_id);
    assert_eq!(back.task_hash, manifest.task_hash);
    assert_eq!(back.packages.len(), 1);
    assert_eq!(back.packages[0].name, "main_pkg");
    assert_eq!(back.packages[0].functions.len(), 1);
    assert_eq!(back.packages[0].functions[0].name, "main");
    assert_eq!(back.state_declarations.len(), 1);
    assert_eq!(back.handoffs.len(), 1);
}

#[test]
fn test_manifest_json_fields_present() {
    let manifest = ArtifactManifest {
        version: "1.0.0".into(),
        model_id: "test_model".into(),
        task_hash: "sha256:abc123".into(),
        created_at: "2025-01-01T00:00:00Z".into(),
        packages: vec![],
        state_declarations: vec![],
        handoffs: vec![],
        compiler_version: "0.1.0".into(),
        implementation_status: "host_compiled".into(),
        verification_scope: "host_compile_only".into(),
        environment_limitations: vec![],
    };

    let json = serde_json::to_value(&manifest).unwrap();
    // Verify key fields are present in the JSON
    assert!(json.get("version").is_some());
    assert!(json.get("model_id").is_some());
    assert!(json.get("task_hash").is_some());
    assert!(json.get("packages").is_some());
    assert!(json.get("state_declarations").is_some());
    assert!(json.get("handoffs").is_some());
    assert!(json.get("compiler_version").is_some());
    assert!(json.get("implementation_status").is_some());
    assert!(json.get("verification_scope").is_some());
    assert!(json.get("environment_limitations").is_some());
}

#[test]
fn test_verify_hash_with_file() {
    let tmp = tempfile::tempdir().unwrap();
    let file_path = tmp.path().join("test_file.bin");
    std::fs::write(&file_path, b"test content").unwrap();

    let hash = crate::hashing::hash_file(&file_path.to_string_lossy()).unwrap();
    assert!(hash.starts_with("sha256:"));

    // Verify the hash matches
    let matches = verify_hash(&file_path.to_string_lossy(), &hash).unwrap();
    assert!(matches, "Hash verification must succeed for same content");

    // Wrong hash must not match
    let wrong_hash = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let matches = verify_hash(&file_path.to_string_lossy(), wrong_hash).unwrap();
    assert!(!matches, "Wrong hash must not match");
}
