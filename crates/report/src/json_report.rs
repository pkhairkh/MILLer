//! JSON Report Generation
//!
//! Generates machine-readable JSON reports for integration
//! with other tools and CI systems.
//!
//! JSON reports contain the same information as markdown reports but in
//! a structured, machine-parseable format. Each report has a type,
//! version, timestamp, and data payload.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;

/// A structured JSON report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonReport {
    /// Report type (e.g., "compilation", "knowledge", "diagnostics").
    pub report_type: String,
    /// Report schema version.
    pub version: String,
    /// ISO-8601 timestamp of report generation.
    pub timestamp: String,
    /// Report data payload.
    pub data: serde_json::Value,
}

/// JSON report generator.
pub struct JsonReporter {
    // Future: configuration for output format, filtering, etc.
}

impl JsonReporter {
    /// Create a new JSON reporter.
    pub fn new() -> Self {
        Self {}
    }

    /// Generate a compilation JSON report from manifest and optional bridge result.
    ///
    /// The report captures the full compilation outcome: task identity,
    /// packages produced, function descriptors, bridge status, and residuals.
    pub fn generate_compilation_report(
        &self,
        manifest: &serde_json::Value,
        bridge_result: Option<&serde_json::Value>,
    ) -> Result<JsonReport> {
        let timestamp = Self::current_timestamp();
        let mut data = serde_json::Map::new();

        // Task identity
        data.insert("model_id".into(), manifest["model_id"].clone());
        data.insert("task_hash".into(), manifest["task_hash"].clone());
        data.insert("task_family".into(), manifest["task_family"].clone());
        data.insert("manifest_version".into(), manifest["version"].clone());

        // Compilation status
        data.insert("status".into(), manifest["bridge_status"].clone());
        if let Some(err) = manifest.get("bridge_error") {
            data.insert("error".into(), err.clone());
        }

        // Packages
        data.insert("packages".into(), manifest["packages"].clone());

        // Bridge result details
        if let Some(br) = bridge_result {
            let mut bridge_data = serde_json::Map::new();
            if let Some(hash) = br.get("content_hash") {
                bridge_data.insert("content_hash".into(), hash.clone());
            }
            if let Some(files) = br.get("package_files") {
                bridge_data.insert("package_files".into(), files.clone());
                // Compute total size
                let total_size: u64 = files
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|f| f.get("size_bytes").and_then(|v| v.as_u64()))
                            .sum()
                    })
                    .unwrap_or(0);
                bridge_data.insert("total_size_bytes".into(), serde_json::Value::Number(total_size.into()));
                bridge_data.insert("file_count".into(), serde_json::Value::Number(
                    files.as_array().map(|a| a.len()).unwrap_or(0).into()
                ));
            }
            if let Some(ct_ver) = br.get("coremltools_version") {
                bridge_data.insert("coremltools_version".into(), ct_ver.clone());
            }
            if let Some(cp) = br.get("compute_plan") {
                bridge_data.insert("compute_plan".into(), cp.clone());
            }
            data.insert("bridge_result".into(), serde_json::Value::Object(bridge_data));
        }

        // Residuals
        data.insert(
            "residuals".into(),
            serde_json::json!([
                "Device-specific ANE placement not verified (requires Apple hardware)",
                "Numerical drift not measured (requires Apple hardware for predict())",
                "Fallback suspicion not assessed (requires compute plan on Apple hardware)"
            ]),
        );

        Ok(JsonReport {
            report_type: "compilation".into(),
            version: "1.0.0".into(),
            timestamp,
            data: serde_json::Value::Object(data),
        })
    }

    /// Generate a knowledge JSON report from a knowledge update.
    ///
    /// Captures all observations produced during compilation, including
    /// legality rules, confidence scores, and evidence sources.
    pub fn generate_knowledge_report(
        &self,
        knowledge_update: &serde_json::Value,
    ) -> Result<JsonReport> {
        let timestamp = Self::current_timestamp();
        let mut data = serde_json::Map::new();

        data.insert("task_name".into(), knowledge_update["task_name"].clone());
        data.insert("task_hash".into(), knowledge_update["task_hash"].clone());
        data.insert("source".into(), knowledge_update["source"].clone());
        data.insert("observations".into(), knowledge_update["observations"].clone());

        if let Some(comp_result) = knowledge_update.get("compilation_result") {
            data.insert("compilation_result".into(), comp_result.clone());
        }

        if let Some(residuals) = knowledge_update.get("residuals") {
            data.insert("residuals".into(), residuals.clone());
        }

        // Summary statistics
        let observations = knowledge_update
            .get("observations")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .unwrap_or(0);
        data.insert(
            "observation_count".into(),
            serde_json::Value::Number(observations.into()),
        );

        Ok(JsonReport {
            report_type: "knowledge".into(),
            version: "1.0.0".into(),
            timestamp,
            data: serde_json::Value::Object(data),
        })
    }

    /// Generate a diagnostics JSON report from error data.
    ///
    /// Captures error messages, stderr output, and failure context
    /// for debugging and CI integration.
    pub fn generate_diagnostics_report(
        &self,
        error_data: &serde_json::Value,
    ) -> Result<JsonReport> {
        let timestamp = Self::current_timestamp();

        Ok(JsonReport {
            report_type: "diagnostics".into(),
            version: "1.0.0".into(),
            timestamp,
            data: error_data.clone(),
        })
    }

    /// Write a JSON report to a file.
    pub fn write_to_file(&self, report: &JsonReport, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(report)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Get the current timestamp as an ISO-8601 string.
    fn current_timestamp() -> String {
        chrono::Utc::now().to_rfc3339()
    }
}
