//! Python Bridge Subprocess
//!
//! Manages the Python subprocess that executes MIL Builder
//! commands via JSON command/result file exchange.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

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
    /// Timeout in seconds for bridge execution.
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
    pub fn execute_raw_payload(&self, payload: &serde_json::Value) -> Result<BridgeResult> {
        let tmp_dir = tempfile::tempdir()?;
        let cmd_path = tmp_dir.path().join("command.json");
        let res_path = tmp_dir.path().join("result.json");

        // Write command
        let cmd_json = serde_json::to_string_pretty(payload)?;
        fs::write(&cmd_path, &cmd_json)?;

        // Run Python subprocess
        let output = Command::new(&self.python_path)
            .arg(&self.bridge_script_path)
            .arg(&cmd_path)
            .arg(&res_path)
            .output()?;

        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Ok(BridgeResult {
                status: "error".into(),
                error_message: Some(format!("Python bridge exited with {}: {}", output.status, stderr)),
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
        Ok(result)
    }
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
