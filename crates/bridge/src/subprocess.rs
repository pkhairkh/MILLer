//! Python Bridge Subprocess
//!
//! Manages the Python subprocess that executes MIL Builder
//! commands via JSON command/result file exchange.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

/// Which emission path was used for the mlpackage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmissionPath {
    /// Python subprocess via coremltools.
    PythonBridge,
    /// Direct Rust protobuf emission (proto-direct, Sprint 41).
    ProtoDirect,
}

/// Python subprocess bridge for MIL Builder.
pub struct PythonBridge {
    /// Path to the Python bridge script.
    pub bridge_script_path: PathBuf,
    /// Path to the Python interpreter.
    pub python_path: String,
    /// Timeout in seconds for bridge execution (T-77, I-52).
    ///
    /// If the Python subprocess does not exit within this duration,
    /// it is killed and a timeout error is returned. Defaults to 300
    /// seconds (5 minutes).
    pub timeout_secs: u64,
}

impl PythonBridge {
    /// Create a new Python bridge pointing at the given bridge.py.
    pub fn new(python_path: &str, bridge_script_path: &str) -> Self {
        Self {
            bridge_script_path: PathBuf::from(bridge_script_path),
            python_path: python_path.to_string(),
            timeout_secs: 300,
        }
    }

    /// Detect the project root by walking up from the bridge script location.
    pub fn detect_project_root(&self) -> Option<PathBuf> {
        let mut dir = self.bridge_script_path.parent()?;
        loop {
            if dir.join("Cargo.toml").exists() {
                return Some(dir.to_path_buf());
            }
            dir = dir.parent()?;
        }
    }

    /// Execute a raw JSON payload against the bridge.
    /// This is the primary path for the vertical slice: Rust serializes
    /// the payload, Python reads it, executes, writes result.
    ///
    /// # Timeout (T-77, I-52)
    ///
    /// The subprocess is spawned and polled until it exits or the timeout
    /// (`timeout_secs`) elapses. On timeout, the child is killed and a
    /// timeout error is returned. This prevents a hung Python subprocess
    /// from blocking the compiler indefinitely.
    pub fn execute_raw_payload(&self, payload: &serde_json::Value) -> Result<BridgeResult> {
        let tmp_dir = tempfile::tempdir()?;
        let cmd_path = tmp_dir.path().join("command.json");
        let res_path = tmp_dir.path().join("result.json");

        // Write command
        let cmd_json = serde_json::to_string_pretty(payload)?;
        fs::write(&cmd_path, &cmd_json)?;

        // Spawn Python subprocess with timeout enforcement (T-77, I-52).
        // Previously, .output() blocked indefinitely if the Python process hung.
        let mut child = Command::new(&self.python_path)
            .arg(&self.bridge_script_path)
            .arg(&cmd_path)
            .arg(&res_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let deadline = Instant::now() + Duration::from_secs(self.timeout_secs);

        // Poll for completion with timeout
        let output = loop {
            match child.try_wait()? {
                Some(_status) => {
                    // Process exited — collect output
                    let output = child.wait_with_output()?;
                    break OutputWithStatus {
                        status: output.status,
                        stdout: output.stdout,
                        stderr: output.stderr,
                    };
                }
                None if Instant::now() >= deadline => {
                    // Timeout — kill the subprocess
                    log::warn!(
                        "Python bridge timed out after {}s — killing subprocess",
                        self.timeout_secs
                    );
                    let _ = child.kill();
                    let _ = child.wait(); // Reap the zombie process

                    return Ok(BridgeResult {
                        status: "error".into(),
                        error_message: Some(format!(
                            "Python bridge timed out after {} seconds",
                            self.timeout_secs
                        )),
                        output_path: None,
                        coremltools_version: None,
                        content_hash: None,
                        package_files: vec![],
                        compute_plan: None,
                        function_descriptors: vec![],
                        metadata: serde_json::Value::Null,
                        stderr: String::new(),
                        emission_path: EmissionPath::PythonBridge,
                    });
                }
                None => {
                    // Still running — sleep briefly before polling again
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        };

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Ok(BridgeResult {
                status: "error".into(),
                error_message: Some(format!(
                    "Python bridge exited with {}: {}",
                    output.status, stderr
                )),
                output_path: None,
                coremltools_version: None,
                content_hash: None,
                package_files: vec![],
                compute_plan: None,
                function_descriptors: vec![],
                metadata: serde_json::Value::Null,
                stderr,
                emission_path: EmissionPath::PythonBridge,
            });
        }

        // Read result
        if !res_path.exists() {
            return Ok(BridgeResult {
                status: "error".into(),
                error_message: Some("Python bridge produced no result file".into()),
                output_path: None,
                coremltools_version: None,
                content_hash: None,
                package_files: vec![],
                compute_plan: None,
                function_descriptors: vec![],
                metadata: serde_json::Value::Null,
                stderr,
                emission_path: EmissionPath::PythonBridge,
            });
        }

        let res_json = fs::read_to_string(&res_path)?;
        let mut result: BridgeResult = serde_json::from_str(&res_json)?;
        // Ensure the emission path is set correctly for results deserialized
        // from the Python bridge (they won't have the field in older JSON)
        result.emission_path = EmissionPath::PythonBridge;

        // T-D-01 (M-020): Run structural verification with default settings.
        // Issues are logged as warnings but do not block the result, maintaining
        // backward compatibility.
        let verifier = BridgeVerifier::new();
        let issues = verifier.verify(&result);
        for issue in &issues {
            log::warn!("M-020: Bridge verification issue: {:?}", issue);
        }

        Ok(result)
    }

    /// Execute a raw JSON payload against the bridge and verify the result
    /// with strict checks.
    ///
    /// Unlike [`execute_raw_payload`], this method uses [`BridgeVerifier::strict`],
    /// which includes on-disk verification of the mlpackage directory.
    /// Verification issues are logged as warnings but do **not** cause a
    /// hard error — callers that need strict enforcement should inspect the
    /// returned `BridgeResult` themselves or use a separate verifier pass.
    pub fn execute_and_verify(&self, payload: &serde_json::Value) -> Result<BridgeResult> {
        let result = self.execute_raw_payload(payload)?;
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        if !issues.is_empty() {
            for issue in &issues {
                log::warn!("M-020: Bridge verification issue: {:?}", issue);
            }
        }
        Ok(result)
    }
}

/// Helper struct to carry process output alongside exit status.
/// Used by `execute_raw_payload` to unify the spawn+timeout path with
/// the same fields as `std::process::Output`.
struct OutputWithStatus {
    status: std::process::ExitStatus,
    #[allow(dead_code)] // stdout captured but not currently read; reserved for future use
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// A single file entry in the mlpackage output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageFileEntry {
    /// Relative path within the mlpackage.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// A function descriptor returned by the Python bridge.
/// Matches the structure emitted by mil_emitter._resolve_function_descriptors().
///
/// This is the bridge-layer representation of a function descriptor.
/// The artifacts-layer `FunctionDescriptor` (in manifest.rs) adds
/// an `emission_status` field and uses typed `TensorSpec` instead
/// of raw JSON values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeFunctionDescriptor {
    /// Function name (e.g., "main", "encode", "decode").
    pub name: String,
    /// Input tensor specifications.
    /// Each entry has {name, shape, dtype}.
    pub inputs: Vec<serde_json::Value>,
    /// Output tensor specifications.
    /// Each entry has {name, shape, dtype}.
    pub outputs: Vec<serde_json::Value>,
    /// Whether this function uses persistent state.
    pub stateful: bool,
}

/// Result from a raw bridge execution.
///
/// Fields correspond 1:1 to what the Python bridge returns in
/// mil_emitter.emit_linear_projection() and _error_result().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeResult {
    /// "success" or "error".
    pub status: String,
    /// Error message if status == "error".
    pub error_message: Option<String>,
    /// Path to the saved .mlpackage directory.
    pub output_path: Option<String>,
    /// coremltools version string (e.g., "9.0").
    pub coremltools_version: Option<String>,
    /// SHA-256 content hash of the mlpackage directory.
    /// Format: "sha256:<hex>". Matches what Python's _hash_directory returns.
    pub content_hash: Option<String>,
    /// File inventory of the mlpackage directory.
    /// Each entry has {path: str, size_bytes: int}.
    pub package_files: Vec<PackageFileEntry>,
    /// Compute plan availability info.
    /// None if the bridge didn't attempt compute plan inspection.
    /// Some({available: bool, reason?: str}) otherwise.
    pub compute_plan: Option<serde_json::Value>,
    /// Function descriptors for the package.
    /// Populated by the Python emitter's _resolve_function_descriptors().
    pub function_descriptors: Vec<BridgeFunctionDescriptor>,
    /// Additional metadata from the Python side.
    pub metadata: serde_json::Value,
    /// Captured stderr from the Python subprocess.
    #[serde(default)]
    pub stderr: String,
    /// Which emission path was used.
    #[serde(default = "default_emission_path")]
    pub emission_path: EmissionPath,
}

/// Default emission path for deserialization of older results that
/// don't include the `emission_path` field.
fn default_emission_path() -> EmissionPath {
    EmissionPath::PythonBridge
}

// ---------------------------------------------------------------------------
// T-D-01 (M-020): Structural verification for Python bridge results
// ---------------------------------------------------------------------------

/// Verification issues found by [`BridgeVerifier`].
///
/// Each variant represents a specific structural inconsistency between
/// the `status == "success"` claim and the actual data returned by the
/// Python bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeVerificationIssue {
    /// A "success" result has no `output_path` set.
    SuccessWithNoOutputPath,
    /// A "success" result has an empty `package_files` list.
    SuccessWithNoPackageFiles,
    /// A "success" result refers to a directory that does not exist on disk.
    SuccessWithMissingDirectory {
        /// The missing directory path.
        path: String,
    },
    /// A "success" result refers to an mlpackage that has no manifest file.
    SuccessWithMissingManifest {
        /// The path where the manifest was expected.
        path: String,
    },
    /// A "success" result refers to an mlpackage that has no model .mlmodel file.
    SuccessWithMissingModelFile {
        /// The path where the model file was expected.
        path: String,
    },
    /// A function descriptor has no input specifications.
    FunctionDescriptorMissingInputs {
        /// Name of the function with missing inputs.
        name: String,
    },
    /// A function descriptor has no output specifications.
    FunctionDescriptorMissingOutputs {
        /// Name of the function with missing outputs.
        name: String,
    },
    /// A content_hash that does not match the expected "sha256:<hex>" format.
    ContentHashFormatInvalid {
        /// The invalid hash string.
        hash: String,
    },
}

/// T-D-01 (M-020): Structural verification for Python bridge results.
///
/// The Python bridge trusts `BridgeResult.status == "success"` as semantic
/// legality without structural verification. This verifier adds post-hoc
/// structural checks to catch common failure modes:
/// - Missing output path for "success" results
/// - Empty package files for "success" results
/// - Inconsistent function descriptors
/// - Output mlpackage directory missing expected structure
/// - Malformed content hash
///
/// Verification issues are **not** hard errors by default — they are
/// returned as a list and callers decide how to act on them. The
/// [`PythonBridge::execute_and_verify`] method logs them as warnings;
/// strict consumers can treat any non-empty list as an error.
pub struct BridgeVerifier {
    /// Whether to require `output_path` for success results.
    pub require_output_path: bool,
    /// Whether to require non-empty `package_files` for success results.
    pub require_package_files: bool,
    /// Whether to verify the mlpackage directory structure on disk.
    pub verify_disk_structure: bool,
}

impl BridgeVerifier {
    /// Create a verifier with default (moderate) settings.
    ///
    /// Enables `require_output_path` and `require_package_files`,
    /// but does **not** verify on-disk structure (to avoid I/O in
    /// hot paths and CI environments without the actual artifacts).
    pub fn new() -> Self {
        Self {
            require_output_path: true,
            require_package_files: true,
            verify_disk_structure: false,
        }
    }

    /// Create a verifier with all checks enabled.
    pub fn strict() -> Self {
        Self { require_output_path: true, require_package_files: true, verify_disk_structure: true }
    }

    /// Create a verifier with minimal checks.
    ///
    /// Only verifies structural invariants that are almost
    /// certainly bugs (e.g. a function descriptor with zero
    /// outputs). Does not require `output_path` or `package_files`
    /// and does not touch the filesystem.
    pub fn lenient() -> Self {
        Self {
            require_output_path: false,
            require_package_files: false,
            verify_disk_structure: false,
        }
    }

    /// Verify a [`BridgeResult`] for structural consistency.
    ///
    /// Returns a list of verification issues (empty = pass).
    /// Only "success" results are checked; error results are
    /// considered structurally valid by definition.
    pub fn verify(&self, result: &BridgeResult) -> Vec<BridgeVerificationIssue> {
        let mut issues = Vec::new();

        // Error results are structurally valid by definition.
        if result.status != "success" {
            return issues;
        }

        // --- output_path checks ---
        if self.require_output_path && result.output_path.is_none() {
            issues.push(BridgeVerificationIssue::SuccessWithNoOutputPath);
        }

        // --- package_files checks ---
        if self.require_package_files && result.package_files.is_empty() {
            issues.push(BridgeVerificationIssue::SuccessWithNoPackageFiles);
        }

        // --- content_hash format check ---
        if let Some(ref hash) = result.content_hash {
            if !hash.starts_with("sha256:") || hash.len() != 7 + 64 {
                issues
                    .push(BridgeVerificationIssue::ContentHashFormatInvalid { hash: hash.clone() });
            }
        }

        // --- function descriptor checks ---
        for fd in &result.function_descriptors {
            if fd.inputs.is_empty() {
                issues.push(BridgeVerificationIssue::FunctionDescriptorMissingInputs {
                    name: fd.name.clone(),
                });
            }
            if fd.outputs.is_empty() {
                issues.push(BridgeVerificationIssue::FunctionDescriptorMissingOutputs {
                    name: fd.name.clone(),
                });
            }
        }

        // --- on-disk structure checks ---
        if self.verify_disk_structure {
            if let Some(ref path) = result.output_path {
                issues.extend(self.verify_mlpackage_structure(path));
            }
        }

        issues
    }

    /// Verify the mlpackage directory structure on disk.
    ///
    /// Checks that:
    /// 1. The directory exists.
    /// 2. It contains a manifest.json (or .mlmodel/manifest.json for
    ///    older-style packages).
    /// 3. It contains at least one `.mlmodel` file.
    fn verify_mlpackage_structure(&self, path: &str) -> Vec<BridgeVerificationIssue> {
        let mut issues = Vec::new();
        let pkg_path = std::path::Path::new(path);

        if !pkg_path.exists() {
            issues.push(BridgeVerificationIssue::SuccessWithMissingDirectory {
                path: path.to_string(),
            });
            return issues;
        }

        // Check for manifest — could be at top level or inside .mlmodel subdirectory
        let manifest_top = pkg_path.join("manifest.json");
        let has_manifest_in_subdir = pkg_path
            .read_dir()
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| e.path().join("manifest.json").exists())
            })
            .unwrap_or(false);

        if !manifest_top.exists() && !has_manifest_in_subdir {
            issues.push(BridgeVerificationIssue::SuccessWithMissingManifest {
                path: path.to_string(),
            });
        }

        // Check for at least one .mlmodel file or directory
        let has_model = pkg_path
            .read_dir()
            .map(|entries| {
                entries.filter_map(|e| e.ok()).any(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.ends_with(".mlmodel")
                })
            })
            .unwrap_or(false);

        if !has_model {
            issues.push(BridgeVerificationIssue::SuccessWithMissingModelFile {
                path: path.to_string(),
            });
        }

        issues
    }
}

impl Default for BridgeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a minimal success BridgeResult.
    fn success_result() -> BridgeResult {
        BridgeResult {
            status: "success".into(),
            error_message: None,
            output_path: Some("/tmp/model.mlpackage".into()),
            coremltools_version: Some("9.0".into()),
            content_hash: Some(format!("sha256:{}", "a".repeat(64))),
            package_files: vec![PackageFileEntry {
                path: "model.mlmodel/weights/weight.bin".into(),
                size_bytes: 1024,
            }],
            compute_plan: None,
            function_descriptors: vec![BridgeFunctionDescriptor {
                name: "main".into(),
                inputs: vec![serde_json::json!({"name": "x", "shape": [1, 10], "dtype": "fp16"})],
                outputs: vec![serde_json::json!({"name": "y", "shape": [1, 5], "dtype": "fp16"})],
                stateful: false,
            }],
            metadata: serde_json::Value::Null,
            stderr: String::new(),
            emission_path: EmissionPath::PythonBridge,
        }
    }

    /// Helper to build a minimal error BridgeResult.
    fn error_result() -> BridgeResult {
        BridgeResult {
            status: "error".into(),
            error_message: Some("something went wrong".into()),
            output_path: None,
            coremltools_version: None,
            content_hash: None,
            package_files: vec![],
            compute_plan: None,
            function_descriptors: vec![],
            metadata: serde_json::Value::Null,
            stderr: "traceback...".into(),
            emission_path: EmissionPath::PythonBridge,
        }
    }

    // ----- Strict mode tests -----

    #[test]
    fn test_strict_success_with_no_output_path() {
        let mut result = success_result();
        result.output_path = None;
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::SuccessWithNoOutputPath));
    }

    #[test]
    fn test_strict_success_with_no_package_files() {
        let mut result = success_result();
        result.package_files = vec![];
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::SuccessWithNoPackageFiles));
    }

    #[test]
    fn test_error_result_passes_verification() {
        let result = error_result();
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        assert!(issues.is_empty(), "Error results should have no verification issues");
    }

    // ----- Lenient mode tests -----

    #[test]
    fn test_lenient_skips_output_path_and_package_files_checks() {
        let mut result = success_result();
        result.output_path = None;
        result.package_files = vec![];
        let verifier = BridgeVerifier::lenient();
        let issues = verifier.verify(&result);
        // Lenient mode should not flag missing output_path or package_files
        assert!(!issues.contains(&BridgeVerificationIssue::SuccessWithNoOutputPath));
        assert!(!issues.contains(&BridgeVerificationIssue::SuccessWithNoPackageFiles));
    }

    #[test]
    fn test_lenient_still_checks_function_descriptors() {
        let mut result = success_result();
        result.function_descriptors = vec![BridgeFunctionDescriptor {
            name: "bad_fn".into(),
            inputs: vec![],
            outputs: vec![],
            stateful: false,
        }];
        let verifier = BridgeVerifier::lenient();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::FunctionDescriptorMissingInputs {
            name: "bad_fn".into(),
        }));
        assert!(issues.contains(&BridgeVerificationIssue::FunctionDescriptorMissingOutputs {
            name: "bad_fn".into(),
        }));
    }

    // ----- Content hash format tests -----

    #[test]
    fn test_invalid_content_hash_format() {
        let mut result = success_result();
        result.content_hash = Some("md5:abc".into());
        let verifier = BridgeVerifier::new();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::ContentHashFormatInvalid {
            hash: "md5:abc".into(),
        }));
    }

    #[test]
    fn test_valid_content_hash_passes() {
        let result = success_result(); // has valid sha256:<64 hex chars>
        let verifier = BridgeVerifier::new();
        let issues = verifier.verify(&result);
        assert!(!issues
            .iter()
            .any(|i| matches!(i, BridgeVerificationIssue::ContentHashFormatInvalid { .. })));
    }

    // ----- Function descriptor tests -----

    #[test]
    fn test_function_descriptor_missing_inputs() {
        let mut result = success_result();
        result.function_descriptors = vec![BridgeFunctionDescriptor {
            name: "main".into(),
            inputs: vec![],
            outputs: vec![serde_json::json!({"name": "y"})],
            stateful: false,
        }];
        let verifier = BridgeVerifier::new();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::FunctionDescriptorMissingInputs {
            name: "main".into(),
        }));
    }

    #[test]
    fn test_function_descriptor_missing_outputs() {
        let mut result = success_result();
        result.function_descriptors = vec![BridgeFunctionDescriptor {
            name: "main".into(),
            inputs: vec![serde_json::json!({"name": "x"})],
            outputs: vec![],
            stateful: false,
        }];
        let verifier = BridgeVerifier::new();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::FunctionDescriptorMissingOutputs {
            name: "main".into(),
        }));
    }

    // ----- Full valid result passes all checks -----

    #[test]
    fn test_fully_valid_success_result_passes() {
        let result = success_result();
        let verifier = BridgeVerifier::new(); // default: no disk checks
        let issues = verifier.verify(&result);
        assert!(issues.is_empty(), "A fully valid result should have no issues, got: {:?}", issues);
    }

    // ----- Strict mode includes disk checks -----

    #[test]
    fn test_strict_flags_missing_directory() {
        let mut result = success_result();
        result.output_path = Some("/tmp/nonexistent_mlpackage_999.mlpackage".into());
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        assert!(issues.contains(&BridgeVerificationIssue::SuccessWithMissingDirectory {
            path: "/tmp/nonexistent_mlpackage_999.mlpackage".into(),
        }));
    }

    // ----- Default mode excludes disk checks -----

    #[test]
    fn test_default_does_not_check_disk() {
        let mut result = success_result();
        result.output_path = Some("/tmp/nonexistent_mlpackage_999.mlpackage".into());
        let verifier = BridgeVerifier::new(); // default: no disk checks
        let issues = verifier.verify(&result);
        assert!(!issues
            .iter()
            .any(|i| matches!(i, BridgeVerificationIssue::SuccessWithMissingDirectory { .. })));
    }

    // ----- Disk structure: valid mlpackage on disk -----

    #[test]
    fn test_strict_valid_mlpackage_on_disk() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let pkg_path = tmp_dir.path().join("model.mlpackage");
        let mlmodel_dir = pkg_path.join("model.mlmodel");
        fs::create_dir_all(&mlmodel_dir).unwrap();
        fs::write(mlmodel_dir.join("manifest.json"), "{}").unwrap();
        fs::write(mlmodel_dir.join("model.mlmodel"), "fake").unwrap();

        let mut result = success_result();
        result.output_path = Some(pkg_path.to_string_lossy().to_string());
        let verifier = BridgeVerifier::strict();
        let issues = verifier.verify(&result);
        assert!(!issues
            .iter()
            .any(|i| matches!(i, BridgeVerificationIssue::SuccessWithMissingDirectory { .. })));
        assert!(!issues
            .iter()
            .any(|i| matches!(i, BridgeVerificationIssue::SuccessWithMissingManifest { .. })));
        assert!(!issues
            .iter()
            .any(|i| matches!(i, BridgeVerificationIssue::SuccessWithMissingModelFile { .. })));
    }
}
