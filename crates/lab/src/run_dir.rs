//! Lab Run Directory Structure
//!
//! Defines the layout for lab run output directories and provides
//! utilities for writing and reading run artifacts.
//!
//! The canonical run directory layout is:
//!
//! ```text
//! <output_dir>/
//!   run_<timestamp>_<task_hash_prefix>/
//!     run.json              — LabRun record (the primary artifact)
//!     manifest.json         — Artifact manifest from compilation
//!     mir.json              — MIR dump from compilation
//!     mlpackage/            — The compiled .mlpackage directory
//!     knowledge/            — Knowledge update artifacts
//!       update_<task>.json  — Knowledge update from this run
//!     inspection.json       — Host-side inspection result (if performed)
//!     timing.json           — Timing result (if profiling was performed)
//!     fallback.json         — Fallback suspicion result (if assessed)
//!     baseline.json         — Baseline reference output (FP32 computation)
//!     drift.json            — Drift report (baseline vs actual comparison)
//! ```

use crate::harness::LabRun;
use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// The subdirectory names within a lab run directory.
pub mod layout {
    pub const RUN_JSON: &str = "run.json";
    pub const MANIFEST_JSON: &str = "manifest.json";
    pub const MIR_JSON: &str = "mir.json";
    pub const MLPACKAGE_DIR: &str = "mlpackage";
    pub const KNOWLEDGE_DIR: &str = "knowledge";
    pub const INSPECTION_JSON: &str = "inspection.json";
    pub const TIMING_JSON: &str = "timing.json";
    pub const FALLBACK_JSON: &str = "fallback.json";
    pub const BASELINE_JSON: &str = "baseline.json";
    pub const DRIFT_JSON: &str = "drift.json";
}

/// Writer for lab run directories.
///
/// Creates the directory structure and writes artifacts in the canonical layout.
pub struct LabRunWriter {
    /// Root directory for all lab runs.
    output_dir: PathBuf,
}

impl LabRunWriter {
    /// Create a new lab run writer that places runs under the given output directory.
    pub fn new(output_dir: &Path) -> Self {
        Self { output_dir: output_dir.to_path_buf() }
    }

    /// Create the run directory for a given run ID.
    ///
    /// The directory is created at `<output_dir>/<run_id>/`.
    /// Returns the path to the created directory.
    pub fn create_run_directory(&self, run_id: &str) -> Result<PathBuf> {
        let run_dir = self.output_dir.join(run_id);
        fs::create_dir_all(&run_dir)?;

        // Create subdirectories
        fs::create_dir_all(run_dir.join(layout::MLPACKAGE_DIR))?;
        fs::create_dir_all(run_dir.join(layout::KNOWLEDGE_DIR))?;

        Ok(run_dir)
    }

    /// Write the LabRun record to the run directory.
    pub fn write_run_record(&self, run_dir: &Path, run: &LabRun) -> Result<()> {
        let path = run_dir.join(layout::RUN_JSON);
        run.write_to_file(&path)
    }

    /// Write the artifact manifest to the run directory.
    pub fn write_manifest(&self, run_dir: &Path, manifest: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::MANIFEST_JSON);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(manifest)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write the MIR dump to the run directory.
    pub fn write_mir(&self, run_dir: &Path, mir: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::MIR_JSON);
        let json = serde_json::to_string_pretty(mir)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write a knowledge update to the run directory.
    pub fn write_knowledge_update(
        &self,
        run_dir: &Path,
        task_name: &str,
        update: &serde_json::Value,
    ) -> Result<()> {
        let knowledge_dir = run_dir.join(layout::KNOWLEDGE_DIR);
        fs::create_dir_all(&knowledge_dir)?;
        let path = knowledge_dir.join(format!("update_{}.json", task_name));
        let json = serde_json::to_string_pretty(update)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write an inspection result to the run directory.
    pub fn write_inspection(&self, run_dir: &Path, inspection: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::INSPECTION_JSON);
        let json = serde_json::to_string_pretty(inspection)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write a timing result to the run directory.
    pub fn write_timing(&self, run_dir: &Path, timing: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::TIMING_JSON);
        let json = serde_json::to_string_pretty(timing)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write a fallback suspicion result to the run directory.
    pub fn write_fallback(&self, run_dir: &Path, fallback: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::FALLBACK_JSON);
        let json = serde_json::to_string_pretty(fallback)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write a baseline result to the run directory.
    pub fn write_baseline(&self, run_dir: &Path, baseline: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::BASELINE_JSON);
        let json = serde_json::to_string_pretty(baseline)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Write a drift report to the run directory.
    pub fn write_drift(&self, run_dir: &Path, drift: &serde_json::Value) -> Result<()> {
        let path = run_dir.join(layout::DRIFT_JSON);
        let json = serde_json::to_string_pretty(drift)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Validate that a run directory contains the expected minimum artifacts.
    ///
    /// Returns a list of issues found (empty if valid).
    pub fn validate_run_directory(&self, run_dir: &Path) -> Vec<String> {
        let mut issues = Vec::new();

        if !run_dir.exists() {
            issues.push("Run directory does not exist".to_string());
            return issues;
        }

        if !run_dir.join(layout::RUN_JSON).exists() {
            issues.push("run.json is missing".to_string());
        }

        if !run_dir.join(layout::MANIFEST_JSON).exists() {
            issues.push("manifest.json is missing".to_string());
        }

        // mlpackage directory is optional in error cases but expected in success
        // Don't flag it as an issue since the run may have failed compilation

        // knowledge directory should exist even if empty
        if !run_dir.join(layout::KNOWLEDGE_DIR).exists() {
            issues.push("knowledge/ directory is missing".to_string());
        }

        issues
    }
}

/// Generate a run ID from a timestamp and task hash prefix.
///
/// Format: `run_<YYYYMMDD_HHMMSS>_<hash_prefix>`
/// The hash prefix is the first 8 characters of the task hash.
pub fn generate_run_id(task_hash: &str) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let hash_prefix = if task_hash.starts_with("sha256:") {
        &task_hash[7..15.min(task_hash.len())] // Skip "sha256:" prefix, take 8 chars
    } else {
        &task_hash[..8.min(task_hash.len())]
    };
    format!("run_{}_{}", timestamp, hash_prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        CompileStepResult, EnvironmentSummary, InspectionStepResult, LabRunBuilder,
        VerificationScope,
    };

    fn make_minimal_run() -> LabRun {
        LabRunBuilder::new(
            "run_test_001".to_string(),
            "sha256:abcdef1234567890".to_string(),
            "test_task".to_string(),
            VerificationScope::HostOnlyInspection,
            EnvironmentSummary::detect(1),
        )
        .compile_result(CompileStepResult {
            success: true,
            error: None,
            output_path: Some("/tmp/out.mlpackage".to_string()),
            content_hash: Some("sha256:abc".to_string()),
            file_count: Some(3),
            coremltools_version: None,
        })
        .inspect_result(InspectionStepResult {
            package_present: true,
            manifest_readable: true,
            model_loadable: false,
            model_load_failure_reason: None,
            function_count: None,
            input_specs: vec![],
            output_specs: vec![],
            warnings: vec![],
            structure_inspection_available: None,
            structure_inspection_failure_reason: None,
            structure_op_names: vec![],
            structure_op_count: None,
            structure_function_count: None,
            structure_state_declarations: vec![],
            op_fidelity_score: None,
            missing_ops: vec![],
            extra_ops: vec![],
            inspection_method: "none".to_string(),
        })
        .build()
    }

    #[test]
    fn test_layout_constants() {
        assert_eq!(layout::RUN_JSON, "run.json");
        assert_eq!(layout::MANIFEST_JSON, "manifest.json");
        assert_eq!(layout::MIR_JSON, "mir.json");
        assert_eq!(layout::MLPACKAGE_DIR, "mlpackage");
        assert_eq!(layout::KNOWLEDGE_DIR, "knowledge");
        assert_eq!(layout::INSPECTION_JSON, "inspection.json");
        assert_eq!(layout::TIMING_JSON, "timing.json");
        assert_eq!(layout::FALLBACK_JSON, "fallback.json");
        assert_eq!(layout::BASELINE_JSON, "baseline.json");
        assert_eq!(layout::DRIFT_JSON, "drift.json");
    }

    #[test]
    fn test_lab_run_writer_new() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        assert_eq!(writer.output_dir, tmp.path());
    }

    #[test]
    fn test_create_run_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_001").unwrap();

        assert!(run_dir.exists());
        assert!(run_dir.join(layout::MLPACKAGE_DIR).exists());
        assert!(run_dir.join(layout::KNOWLEDGE_DIR).exists());
    }

    #[test]
    fn test_write_run_record() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_002").unwrap();

        let run = make_minimal_run();
        writer.write_run_record(&run_dir, &run).unwrap();

        let run_json_path = run_dir.join(layout::RUN_JSON);
        assert!(run_json_path.exists(), "run.json should exist after write");

        let content = std::fs::read_to_string(&run_json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["run_id"], "run_test_001");
    }

    #[test]
    fn test_write_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_003").unwrap();

        let manifest = serde_json::json!({
            "model_name": "test_model",
            "version": "1.0"
        });
        writer.write_manifest(&run_dir, &manifest).unwrap();

        let path = run_dir.join(layout::MANIFEST_JSON);
        assert!(path.exists(), "manifest.json should exist after write");

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["model_name"], "test_model");
    }

    #[test]
    fn test_write_mir() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_004").unwrap();

        let mir = serde_json::json!({
            "ops": ["linear", "relu"],
            "op_count": 2
        });
        writer.write_mir(&run_dir, &mir).unwrap();

        let path = run_dir.join(layout::MIR_JSON);
        assert!(path.exists(), "mir.json should exist after write");

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["op_count"], 2);
    }

    #[test]
    fn test_write_knowledge_update() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_005").unwrap();

        let update = serde_json::json!({
            "observation": "model compiles successfully",
            "confidence": 0.9
        });
        writer.write_knowledge_update(&run_dir, "compile_task", &update).unwrap();

        let path = run_dir.join(layout::KNOWLEDGE_DIR).join("update_compile_task.json");
        assert!(path.exists(), "knowledge/update_compile_task.json should exist after write");

        let content = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed["confidence"], 0.9);
    }

    #[test]
    fn test_write_inspection() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_006").unwrap();

        let inspection = serde_json::json!({
            "package_present": true,
            "manifest_readable": true
        });
        writer.write_inspection(&run_dir, &inspection).unwrap();

        let path = run_dir.join(layout::INSPECTION_JSON);
        assert!(path.exists(), "inspection.json should exist after write");
    }

    #[test]
    fn test_write_timing() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_007").unwrap();

        let timing = serde_json::json!({
            "p50_ms": 1.5,
            "p90_ms": 2.0
        });
        writer.write_timing(&run_dir, &timing).unwrap();

        let path = run_dir.join(layout::TIMING_JSON);
        assert!(path.exists(), "timing.json should exist after write");
    }

    #[test]
    fn test_write_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_008").unwrap();

        let fallback = serde_json::json!({
            "suspicion_level": "no_conclusion",
            "explanation": "insufficient evidence"
        });
        writer.write_fallback(&run_dir, &fallback).unwrap();

        let path = run_dir.join(layout::FALLBACK_JSON);
        assert!(path.exists(), "fallback.json should exist after write");
    }

    #[test]
    fn test_write_baseline() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_009").unwrap();

        let baseline = serde_json::json!({
            "output_hash": "sha256:abc",
            "precision": "fp32"
        });
        writer.write_baseline(&run_dir, &baseline).unwrap();

        let path = run_dir.join(layout::BASELINE_JSON);
        assert!(path.exists(), "baseline.json should exist after write");
    }

    #[test]
    fn test_write_drift() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_010").unwrap();

        let drift = serde_json::json!({
            "drift_detected": false,
            "max_abs_diff": 0.001
        });
        writer.write_drift(&run_dir, &drift).unwrap();

        let path = run_dir.join(layout::DRIFT_JSON);
        assert!(path.exists(), "drift.json should exist after write");
    }

    #[test]
    fn test_validate_run_directory_missing_required() {
        let tmp = tempfile::tempdir().unwrap();
        // Create an empty directory (no run.json, no manifest.json)
        let empty_dir = tmp.path().join("empty_run");
        std::fs::create_dir_all(&empty_dir).unwrap();

        let writer = LabRunWriter::new(tmp.path());
        let issues = writer.validate_run_directory(&empty_dir);

        assert!(
            issues.iter().any(|i| i.contains("run.json")),
            "Should report missing run.json, got: {:?}",
            issues
        );
        assert!(
            issues.iter().any(|i| i.contains("manifest.json")),
            "Should report missing manifest.json, got: {:?}",
            issues
        );
    }

    #[test]
    fn test_validate_run_directory_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let run_dir = writer.create_run_directory("run_valid").unwrap();

        // Write run.json and manifest.json to make it valid
        let run = make_minimal_run();
        writer.write_run_record(&run_dir, &run).unwrap();
        writer.write_manifest(&run_dir, &serde_json::json!({})).unwrap();

        let issues = writer.validate_run_directory(&run_dir);
        assert!(issues.is_empty(), "Valid directory should have no issues, got: {:?}", issues);
    }

    #[test]
    fn test_validate_run_directory_nonexistent() {
        let tmp = tempfile::tempdir().unwrap();
        let writer = LabRunWriter::new(tmp.path());
        let nonexistent = tmp.path().join("does_not_exist");

        let issues = writer.validate_run_directory(&nonexistent);
        assert!(
            issues.iter().any(|i| i.contains("does not exist")),
            "Should report nonexistent directory, got: {:?}",
            issues
        );
    }

    #[test]
    fn test_generate_run_id_format() {
        let run_id = generate_run_id("abcdef1234567890");
        assert!(run_id.starts_with("run_"), "Run ID should start with 'run_', got: {}", run_id);
        // Should contain a timestamp (YYYYMMDD_HHMMSS) and hash prefix
        // Format: run_YYYYMMDD_HHMMSS_<8-char-hash-prefix>
        let parts: Vec<&str> = run_id.splitn(4, '_').collect();
        assert!(
            parts.len() >= 3,
            "Run ID should have at least 3 underscore-separated parts, got: {}",
            run_id
        );
        // The hash prefix part should be "abcdef12" (first 8 chars of the hash)
        assert!(
            run_id.ends_with("abcdef12"),
            "Run ID should end with hash prefix 'abcdef12', got: {}",
            run_id
        );
    }

    #[test]
    fn test_generate_run_id_with_sha256_prefix() {
        let run_id = generate_run_id("sha256:abcdef1234567890");
        assert!(run_id.starts_with("run_"), "Run ID should start with 'run_', got: {}", run_id);
        // When hash starts with "sha256:", it should skip that prefix
        // and use the next 8 chars as the hash prefix
        assert!(
            run_id.ends_with("abcdef12"),
            "Run ID should end with 'abcdef12' (after skipping sha256: prefix), got: {}",
            run_id
        );
    }
}
