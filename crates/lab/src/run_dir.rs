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
        &task_hash[7..15] // Skip "sha256:" prefix, take 8 chars
    } else {
        &task_hash[..8.min(task_hash.len())]
    };
    format!("run_{}_{}", timestamp, hash_prefix)
}
