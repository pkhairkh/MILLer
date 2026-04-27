//! Markdown Report Generation
//!
//! Generates human-readable markdown reports from compilation results,
//! profiling data, and knowledge store queries.
//!
//! The compilation report includes:
//! - Task identity and metadata
//! - Compilation status and duration
//! - Artifact inventory (packages, functions, hashes)
//! - Bridge result details
//! - Knowledge updates produced
//! - Residuals and limitations

use anyhow::Result;
use serde_json::Value;
use std::fs;

/// Markdown report generator.
pub struct MarkdownReporter {
    // Future: template configuration, style options, etc.
}

impl MarkdownReporter {
    /// Create a new markdown reporter.
    pub fn new() -> Self {
        Self {}
    }

    /// Generate a compilation report from a manifest JSON value and
    /// optional bridge result.
    ///
    /// The manifest contains the authoritative record of what was produced.
    /// The bridge result contains the Python-side emission details.
    pub fn generate_compilation_report(
        &self,
        manifest: &Value,
        bridge_result: Option<&Value>,
        output_path: &str,
    ) -> Result<()> {
        let report = self.format_compilation_report(manifest, bridge_result);
        fs::write(output_path, &report)?;
        Ok(())
    }

    /// Generate a compilation report and return it as a string.
    pub fn format_compilation_report(
        &self,
        manifest: &Value,
        bridge_result: Option<&Value>,
    ) -> String {
        let mut lines: Vec<String> = Vec::new();

        // Title
        let model_id = manifest.get("model_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        lines.push(format!("# Compilation Report: {}", model_id));
        lines.push(String::new());

        // Task identity
        lines.push("## Task Identity".to_string());
        lines.push(String::new());
        if let Some(hash) = manifest.get("task_hash").and_then(|v| v.as_str()) {
            lines.push(format!("- **Task Hash**: `{}`", hash));
        }
        if let Some(family) = manifest.get("task_family").and_then(|v| v.as_str()) {
            lines.push(format!("- **Task Family**: {}", family));
        }
        if let Some(version) = manifest.get("version").and_then(|v| v.as_str()) {
            lines.push(format!("- **Manifest Version**: {}", version));
        }
        if let Some(compiler) = manifest.get("compiler_version").and_then(|v| v.as_str()) {
            lines.push(format!("- **Compiler Version**: {}", compiler));
        }
        if let Some(ts) = manifest.get("created_at").and_then(|v| v.as_u64()) {
            lines.push(format!("- **Created At**: {} (epoch)", ts));
        }
        lines.push(String::new());

        // Compilation status
        lines.push("## Compilation Status".to_string());
        lines.push(String::new());
        let status = manifest.get("bridge_status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let status_icon = if status == "success" { "OK" } else { "FAILED" };
        lines.push(format!("**Status**: {} ({})", status, status_icon));

        if let Some(err) = manifest.get("bridge_error").and_then(|v| v.as_str()) {
            lines.push(format!("- **Error**: {}", err));
        }
        if let Some(ct_ver) = manifest.get("coremltools_version").and_then(|v| v.as_str()) {
            lines.push(format!("- **coremltools**: v{}", ct_ver));
        }
        lines.push(String::new());

        // Packages
        lines.push("## Packages".to_string());
        lines.push(String::new());
        if let Some(packages) = manifest.get("packages").and_then(|v| v.as_array()) {
            if packages.is_empty() {
                lines.push("No packages produced.".to_string());
            } else {
                for pkg in packages {
                    let name = pkg.get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unnamed");
                    let role = pkg.get("role")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let path = pkg.get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("N/A");

                    lines.push(format!("### {}", name));
                    lines.push(format!("- **Role**: {}", role));
                    lines.push(format!("- **Path**: `{}`", path));

                    if let Some(hash) = pkg.get("content_hash").and_then(|v| v.as_str()) {
                        lines.push(format!("- **Content Hash**: `{}`", hash));
                    }

                    // Functions
                    if let Some(functions) = pkg.get("functions").and_then(|v| v.as_array()) {
                        lines.push(String::new());
                        lines.push("#### Functions".to_string());
                        lines.push(String::new());
                        for func in functions {
                            let fname = func.get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unnamed");
                            let stateful = func.get("stateful")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            lines.push(format!("- **{}** (stateful: {})", fname, stateful));

                            if let Some(inputs) = func.get("inputs").and_then(|v| v.as_array()) {
                                for input in inputs {
                                    let iname = input.get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?");
                                    let dtype = input.get("dtype")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?");
                                    let shape = input.get("shape")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_u64())
                                                .map(|n| n.to_string())
                                                .collect::<Vec<_>>()
                                                .join("x")
                                        })
                                        .unwrap_or_else(|| "?".into());
                                    lines.push(format!("  - Input `{}`: {} [{}]", iname, dtype, shape));
                                }
                            }

                            if let Some(outputs) = func.get("outputs").and_then(|v| v.as_array()) {
                                for output in outputs {
                                    let oname = output.get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?");
                                    let dtype = output.get("dtype")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?");
                                    let shape = output.get("shape")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|v| v.as_u64())
                                                .map(|n| n.to_string())
                                                .collect::<Vec<_>>()
                                                .join("x")
                                        })
                                        .unwrap_or_else(|| "?".into());
                                    lines.push(format!("  - Output `{}`: {} [{}]", oname, dtype, shape));
                                }
                            }
                        }
                    }
                    lines.push(String::new());
                }
            }
        } else {
            lines.push("No package data available.".to_string());
        }
        lines.push(String::new());

        // Bridge result details (if available)
        if let Some(br) = bridge_result {
            lines.push("## Bridge Result Details".to_string());
            lines.push(String::new());

            if let Some(files) = br.get("package_files").and_then(|v| v.as_array()) {
                lines.push(format!("**Package Files**: {} entries", files.len()));
                lines.push(String::new());
                lines.push("| File | Size |".to_string());
                lines.push("|------|------|".to_string());
                for file in files {
                    let fpath = file.get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    let fsize = file.get("size_bytes")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    lines.push(format!("| {} | {} bytes |", fpath, fsize));
                }
                lines.push(String::new());
            }

            if let Some(cp) = br.get("compute_plan") {
                let available = cp.get("available")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                lines.push(format!("**Compute Plan**: {}", if available { "available" } else { "unavailable" }));
                if !available {
                    if let Some(reason) = cp.get("reason").and_then(|v| v.as_str()) {
                        lines.push(format!("- Reason: {}", reason));
                    }
                }
                lines.push(String::new());
            }
        }

        // Residuals
        lines.push("## Residuals".to_string());
        lines.push(String::new());
        lines.push("- Device-specific ANE placement not verified (requires Apple hardware)".to_string());
        lines.push("- Numerical drift not measured (requires Apple hardware for predict())".to_string());
        lines.push("- Fallback suspicion not assessed (requires compute plan on Apple hardware)".to_string());

        lines.join("\n")
    }

    /// Generate a knowledge summary report from knowledge update JSON.
    ///
    /// Produces a markdown summary of the knowledge observations generated
    /// during compilation.
    pub fn generate_knowledge_report(
        &self,
        knowledge_update: &Value,
        output_path: &str,
    ) -> Result<()> {
        let report = self.format_knowledge_report(knowledge_update);
        fs::write(output_path, &report)?;
        Ok(())
    }

    /// Format a knowledge report as markdown.
    pub fn format_knowledge_report(&self, knowledge_update: &Value) -> String {
        let mut lines: Vec<String> = Vec::new();

        let task_name = knowledge_update.get("task_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        lines.push(format!("# Knowledge Report: {}", task_name));
        lines.push(String::new());

        if let Some(hash) = knowledge_update.get("task_hash").and_then(|v| v.as_str()) {
            lines.push(format!("- **Task Hash**: `{}`", hash));
        }
        if let Some(source) = knowledge_update.get("source").and_then(|v| v.as_str()) {
            lines.push(format!("- **Source**: {}", source));
        }
        lines.push(String::new());

        // Observations
        if let Some(observations) = knowledge_update.get("observations").and_then(|v| v.as_array()) {
            lines.push("## Observations".to_string());
            lines.push(String::new());
            for (i, obs) in observations.iter().enumerate() {
                let ktype = obs.get("knowledge_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");
                let op_pattern = obs.get("op_pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let ane_legal = obs.get("ane_legal")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let confidence = obs.get("confidence")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let evidence_source = obs.get("evidence_source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let evidence_count = obs.get("evidence_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                lines.push(format!("### Observation {}: {}", i + 1, ktype));
                lines.push(format!("- **Op Pattern**: `{}`", op_pattern));
                lines.push(format!("- **ANE Legal**: {}", ane_legal));
                lines.push(format!("- **Confidence**: {:.2}", confidence));
                lines.push(format!("- **Evidence**: {} ({} observations)", evidence_source, evidence_count));
                if let Some(ctx) = obs.get("context").and_then(|v| v.as_str()) {
                    lines.push(format!("- **Context**: {}", ctx));
                }
                lines.push(String::new());
            }
        }

        // Compilation result
        if let Some(comp_result) = knowledge_update.get("compilation_result") {
            lines.push("## Compilation Result".to_string());
            lines.push(String::new());
            let status = comp_result.get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let mlpackage = comp_result.get("mlpackage_produced")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            lines.push(format!("- **Status**: {}", status));
            lines.push(format!("- **mlpackage Produced**: {}", mlpackage));
            if let Some(hash) = comp_result.get("content_hash").and_then(|v| v.as_str()) {
                lines.push(format!("- **Content Hash**: `{}`", hash));
            }
            lines.push(String::new());
        }

        // Residuals from the knowledge update
        if let Some(residuals) = knowledge_update.get("residuals").and_then(|v| v.as_array()) {
            lines.push("## Residuals".to_string());
            lines.push(String::new());
            for res in residuals {
                if let Some(text) = res.as_str() {
                    lines.push(format!("- {}", text));
                }
            }
            lines.push(String::new());
        }

        lines.join("\n")
    }

    /// Generate a diagnostics report from compilation error data.
    ///
    /// This is a minimal implementation that formats error information.
    pub fn generate_diagnostics_report(
        &self,
        error_data: &Value,
        output_path: &str,
    ) -> Result<()> {
        let report = self.format_diagnostics_report(error_data);
        fs::write(output_path, &report)?;
        Ok(())
    }

    /// Format a diagnostics report as markdown.
    pub fn format_diagnostics_report(&self, error_data: &Value) -> String {
        let mut lines: Vec<String> = Vec::new();

        lines.push("# Diagnostics Report".to_string());
        lines.push(String::new());

        let status = error_data.get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        lines.push(format!("**Status**: {}", status));

        if let Some(err) = error_data.get("error_message").and_then(|v| v.as_str()) {
            lines.push(String::new());
            lines.push("## Error Details".to_string());
            lines.push(String::new());
            lines.push(format!("```"));
            lines.push(err.to_string());
            lines.push(format!("```"));
        }

        if let Some(stderr) = error_data.get("stderr").and_then(|v| v.as_str()) {
            if !stderr.is_empty() {
                lines.push(String::new());
                lines.push("## Bridge stderr".to_string());
                lines.push(String::new());
                lines.push("```".to_string());
                for line in stderr.lines().take(50) {
                    lines.push(line.to_string());
                }
                lines.push("```".to_string());
            }
        }

        lines.push(String::new());
        lines.join("\n")
    }
}
