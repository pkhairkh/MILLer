//! Basic integration tests for the report crate.
//!
//! Tests JSON and markdown report generation through the public API,
//! verifying that output formats are well-formed and contain expected content.

use ane_report::json_report::JsonReporter;
use ane_report::markdown::MarkdownReporter;

// ─── Helpers ──────────────────────────────────────────────────────

fn sample_manifest() -> serde_json::Value {
    serde_json::json!({
        "model_id": "test_model",
        "task_hash": "sha256:abc123",
        "task_family": "LinearProjection",
        "version": "1.0.0",
        "compiler_version": "0.1.0",
        "created_at": 1700000000_u64,
        "bridge_status": "success",
        "packages": [
            {
                "name": "main_pkg",
                "role": "prefill",
                "path": "/tmp/main.mlpackage",
                "content_hash": "sha256:def456",
                "functions": [
                    {
                        "name": "main",
                        "stateful": false,
                        "inputs": [{"name": "x", "dtype": "fp16", "shape": [1, 128]}],
                        "outputs": [{"name": "output", "dtype": "fp16", "shape": [1, 128]}]
                    }
                ]
            }
        ]
    })
}

fn sample_bridge_result() -> serde_json::Value {
    serde_json::json!({
        "content_hash": "sha256:xyz789",
        "package_files": [
            {"path": "model.mlmodel", "size_bytes": 2048},
            {"path": "weights.npy", "size_bytes": 4096}
        ],
        "coremltools_version": "7.2",
        "compute_plan": {"available": true}
    })
}

// ─── JSON Report Tests ────────────────────────────────────────────

#[test]
fn test_json_compilation_report_contains_key_fields() {
    let reporter = JsonReporter::new();
    let manifest = sample_manifest();
    let bridge = sample_bridge_result();

    let report = reporter.generate_compilation_report(&manifest, Some(&bridge)).unwrap();

    assert_eq!(report.report_type, "compilation");
    assert_eq!(report.version, "1.0.0");
    assert!(!report.timestamp.is_empty());

    let data = &report.data;
    assert_eq!(data["model_id"], "test_model");
    assert_eq!(data["task_hash"], "sha256:abc123");
    assert_eq!(data["status"], "success");
    assert!(data.get("bridge_result").is_some());
    assert!(data.get("residuals").is_some());

    // Bridge result should have computed fields
    let bridge_data = data.get("bridge_result").unwrap();
    assert_eq!(bridge_data["file_count"], 2);
    assert_eq!(bridge_data["total_size_bytes"], 6144);
}

#[test]
fn test_json_knowledge_report_structure() {
    let reporter = JsonReporter::new();
    let knowledge_update = serde_json::json!({
        "task_name": "llama_prefill",
        "task_hash": "sha256:llama123",
        "source": "compile_run",
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.linear",
                "ane_legal": true,
                "confidence": 0.9
            },
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.gelu",
                "ane_legal": true,
                "confidence": 0.7
            }
        ]
    });

    let report = reporter.generate_knowledge_report(&knowledge_update).unwrap();

    assert_eq!(report.report_type, "knowledge");
    assert_eq!(report.data["task_name"], "llama_prefill");
    assert_eq!(report.data["observation_count"], 2);
}

#[test]
fn test_json_report_serializes_to_valid_json() {
    let reporter = JsonReporter::new();
    let manifest = sample_manifest();
    let report = reporter.generate_compilation_report(&manifest, None).unwrap();

    let json_str = serde_json::to_string_pretty(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(parsed["report_type"], "compilation");
    assert_eq!(parsed["version"], "1.0.0");
    assert!(parsed.get("timestamp").is_some());
    assert!(parsed.get("data").is_some());
}

#[test]
fn test_json_diagnostics_report_passthrough() {
    let reporter = JsonReporter::new();
    let error_data = serde_json::json!({
        "status": "error",
        "error_message": "Bridge subprocess exited with code 1",
        "stderr": "ImportError: No module named coremltools"
    });

    let report = reporter.generate_diagnostics_report(&error_data).unwrap();

    assert_eq!(report.report_type, "diagnostics");
    assert_eq!(report.data["status"], "error");
    assert_eq!(report.data["error_message"], "Bridge subprocess exited with code 1");
}

// ─── Markdown Report Tests ───────────────────────────────────────

#[test]
fn test_markdown_compilation_report_contains_model_and_sections() {
    let reporter = MarkdownReporter::new();
    let manifest = sample_manifest();
    let bridge = sample_bridge_result();

    let md = reporter.format_compilation_report(&manifest, Some(&bridge));

    // Must have title referencing model
    assert!(md.contains("# Compilation Report"), "Missing title");
    assert!(md.contains("test_model"), "Missing model_id");

    // Must have standard sections
    assert!(md.contains("## Task Identity"), "Missing Task Identity section");
    assert!(md.contains("## Compilation Status"), "Missing Compilation Status section");
    assert!(md.contains("## Packages"), "Missing Packages section");
    assert!(md.contains("## Residuals"), "Missing Residuals section");
    assert!(md.contains("## Bridge Result Details"), "Missing Bridge Result section");
}

#[test]
fn test_markdown_knowledge_report_observations() {
    let reporter = MarkdownReporter::new();
    let knowledge_update = serde_json::json!({
        "task_name": "gpt2_decode",
        "task_hash": "sha256:gpt2hash",
        "source": "compile_run",
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.matmul",
                "ane_legal": true,
                "confidence": 0.85,
                "evidence_source": "SyntheticRun",
                "evidence_count": 3
            }
        ],
        "compilation_result": {
            "status": "success",
            "mlpackage_produced": true,
            "content_hash": "sha256:pkg_hash"
        }
    });

    let md = reporter.format_knowledge_report(&knowledge_update);

    assert!(md.contains("# Knowledge Report"), "Missing title");
    assert!(md.contains("gpt2_decode"), "Missing task name");
    assert!(md.contains("## Observations"), "Missing Observations section");
    assert!(md.contains("LegalityRule"), "Missing knowledge type");
    assert!(md.contains("mb.matmul"), "Missing op pattern");
    assert!(md.contains("## Compilation Result"), "Missing Compilation Result section");
}

#[test]
fn test_markdown_diagnostics_report_error_formatting() {
    let reporter = MarkdownReporter::new();
    let error_data = serde_json::json!({
        "status": "error",
        "error_message": "coremltools convert failed:\nValueError: unknown op",
        "stderr": "Traceback (most recent call last):\n  File \"bridge.py\", line 42"
    });

    let md = reporter.format_diagnostics_report(&error_data);

    assert!(md.contains("# Diagnostics Report"), "Missing title");
    assert!(md.contains("## Error Details"), "Missing Error Details section");
    assert!(md.contains("coremltools convert failed"), "Missing error message");
}
