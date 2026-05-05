//! Unit tests for the report crate
//!
//! Tests JSON and markdown report generation, verifying that
//! output formats are well-formed and contain expected fields.

use crate::json_report::{JsonReport, JsonReporter};
use crate::markdown::MarkdownReporter;

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

// ─── JSON Report Tests ───────────────────────────────────────────

#[test]
fn test_json_compilation_report_structure() {
    let reporter = JsonReporter::new();
    let manifest = sample_manifest();
    let bridge = sample_bridge_result();

    let report = reporter.generate_compilation_report(&manifest, Some(&bridge)).unwrap();

    assert_eq!(report.report_type, "compilation");
    assert_eq!(report.version, "1.0.0");
    assert!(!report.timestamp.is_empty());

    // Verify key data fields
    let data = &report.data;
    assert_eq!(data["model_id"], "test_model");
    assert_eq!(data["task_hash"], "sha256:abc123");
    assert_eq!(data["status"], "success");
    assert!(data.get("bridge_result").is_some());
}

#[test]
fn test_json_compilation_report_without_bridge() {
    let reporter = JsonReporter::new();
    let manifest = sample_manifest();

    let report = reporter.generate_compilation_report(&manifest, None).unwrap();

    assert_eq!(report.report_type, "compilation");
    // Should not have bridge_result when None
    assert!(report.data.get("bridge_result").is_none());
    // But residuals should still be present
    assert!(report.data.get("residuals").is_some());
}

#[test]
fn test_json_knowledge_report() {
    let reporter = JsonReporter::new();
    let knowledge_update = serde_json::json!({
        "task_name": "test_task",
        "task_hash": "sha256:abc",
        "source": "compile_run",
        "observations": [
            {"knowledge_type": "LegalityRule", "op_pattern": "mb.linear", "ane_legal": true, "confidence": 0.8}
        ]
    });

    let report = reporter.generate_knowledge_report(&knowledge_update).unwrap();

    assert_eq!(report.report_type, "knowledge");
    assert_eq!(report.data["task_name"], "test_task");
    assert_eq!(report.data["observation_count"], 1);
}

#[test]
fn test_json_diagnostics_report() {
    let reporter = JsonReporter::new();
    let error_data = serde_json::json!({
        "status": "error",
        "error_message": "Bridge failed",
        "stderr": "traceback..."
    });

    let report = reporter.generate_diagnostics_report(&error_data).unwrap();

    assert_eq!(report.report_type, "diagnostics");
    assert_eq!(report.data["status"], "error");
}

#[test]
fn test_json_report_serialization() {
    let reporter = JsonReporter::new();
    let manifest = sample_manifest();
    let report = reporter.generate_compilation_report(&manifest, None).unwrap();

    // Must serialize to valid JSON
    let json = serde_json::to_string_pretty(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["report_type"], "compilation");
    assert_eq!(parsed["version"], "1.0.0");
}

// ─── Markdown Report Tests ───────────────────────────────────────

#[test]
fn test_markdown_compilation_report_has_title() {
    let reporter = MarkdownReporter::new();
    let manifest = sample_manifest();
    let bridge = sample_bridge_result();

    let md = reporter.format_compilation_report(&manifest, Some(&bridge));

    assert!(md.contains("# Compilation Report"), "Must have title");
    assert!(md.contains("test_model"), "Must mention model_id");
}

#[test]
fn test_markdown_compilation_report_has_sections() {
    let reporter = MarkdownReporter::new();
    let manifest = sample_manifest();

    let md = reporter.format_compilation_report(&manifest, None);

    assert!(md.contains("## Task Identity"), "Must have Task Identity section");
    assert!(md.contains("## Compilation Status"), "Must have Compilation Status section");
    assert!(md.contains("## Packages"), "Must have Packages section");
    assert!(md.contains("## Residuals"), "Must have Residuals section");
}

#[test]
fn test_markdown_compilation_report_with_bridge() {
    let reporter = MarkdownReporter::new();
    let manifest = sample_manifest();
    let bridge = sample_bridge_result();

    let md = reporter.format_compilation_report(&manifest, Some(&bridge));

    assert!(md.contains("## Bridge Result Details"), "Must have Bridge Result section");
    assert!(md.contains("model.mlmodel"), "Must list package files");
}

#[test]
fn test_markdown_knowledge_report() {
    let reporter = MarkdownReporter::new();
    let knowledge_update = serde_json::json!({
        "task_name": "test_task",
        "task_hash": "sha256:abc",
        "source": "compile_run",
        "observations": [
            {
                "knowledge_type": "LegalityRule",
                "op_pattern": "mb.linear",
                "ane_legal": true,
                "confidence": 0.8,
                "evidence_source": "SyntheticRun",
                "evidence_count": 1
            }
        ]
    });

    let md = reporter.format_knowledge_report(&knowledge_update);

    assert!(md.contains("# Knowledge Report"), "Must have title");
    assert!(md.contains("test_task"), "Must mention task name");
    assert!(md.contains("## Observations"), "Must have Observations section");
    assert!(md.contains("LegalityRule"), "Must mention knowledge type");
}

#[test]
fn test_markdown_diagnostics_report() {
    let reporter = MarkdownReporter::new();
    let error_data = serde_json::json!({
        "status": "error",
        "error_message": "Something went wrong\nwith details",
        "stderr": "traceback line 1\ntraceback line 2"
    });

    let md = reporter.format_diagnostics_report(&error_data);

    assert!(md.contains("# Diagnostics Report"), "Must have title");
    assert!(md.contains("error"), "Must mention error status");
    assert!(md.contains("## Error Details"), "Must have error details section");
}
