//! Artifact Manifest
//!
//! Manifest describing all artifacts in a deployment package,
//! their relationships, and metadata.
//!
//! The manifest supports both single-function and multifunction packages.
//! Current emission always produces single-function packages; the
//! multifunction schema is a formalized seam for future use.
//!
//! Truth fields (implementation_status, verification_scope,
//! environment_limitations) ensure that a manifest cannot be misread
//! as proving device/runtime success when only host-side compilation
//! was performed.

use serde::{Deserialize, Serialize};

/// A complete deployment manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManifest {
    /// Manifest schema version.
    pub version: String,
    /// Model identifier (from task spec name).
    pub model_id: String,
    /// Deterministic task identity hash (sha256:<hex>).
    /// This is the primary identity of the manifest — same task config
    /// always produces the same task_hash, enabling artifact identity
    /// verification and cache invalidation.
    pub task_hash: String,
    /// ISO-8601 timestamp of manifest creation.
    /// Informational only; task_hash is the deterministic identity.
    pub created_at: String,
    /// List of packages in this deployment.
    pub packages: Vec<PackageEntry>,
    /// State declarations.
    pub state_declarations: Vec<StateEntry>,
    /// Handoff specifications.
    pub handoffs: Vec<HandoffEntry>,
    /// Compiler version that produced this manifest.
    pub compiler_version: String,
    /// What is implemented vs schema-only in this artifact.
    /// Values: "host_compiled" | "device_verified" | "partial"
    pub implementation_status: String,
    /// What verification scope was actually performed.
    /// E.g. "host_compile_only" means no device/runtime verification.
    pub verification_scope: String,
    /// Known environment limitations that affect artifact validity.
    /// E.g. "no_apple_hardware" means ANE placement is not verified.
    pub environment_limitations: Vec<String>,
}

/// Entry for a single package in the manifest.
///
/// Each package contains one or more named functions. For single-function
/// packages (the current default), `functions` contains one entry.
/// For multifunction packages (future), `functions` contains multiple
/// entries that share the package's weight storage within a single mlpackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageEntry {
    pub name: String,
    pub role: String,
    pub path: Option<String>,
    pub content_hash: Option<String>,
    pub size_bytes: u64,
    /// Named functions within this package.
    /// For single-function packages: one entry named "main".
    /// For multifunction packages: multiple named functions.
    /// This is the architectural seam for multifunction support.
    pub functions: Vec<FunctionDescriptor>,
}

/// A named function within a package.
///
/// Describes the I/O contract and status of a single function.
/// In a multifunction mlpackage, each function has its own I/O signature
/// but shares the package's weight storage.
///
/// This model formalizes the function descriptor that flows through:
/// Rust bridge payload → Python emitter → bridge result → manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDescriptor {
    /// Function name (e.g., "main", "encode", "decode").
    pub name: String,
    /// Input tensor specifications.
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications.
    pub outputs: Vec<TensorSpec>,
    /// Whether this function uses persistent state.
    pub stateful: bool,
    /// Emission status: "emitted" if successfully built and saved,
    /// "seam_only" if the schema exists but emission is not implemented.
    pub emission_status: String,
    /// MIR op type list for this function, populated during compilation.
    /// Each entry is a JSON object with an "op_type" key (e.g., {"op_type": "Linear"}).
    /// Used by the `verify` command to auto-populate `--mir-ops` for op-fidelity
    /// verification without requiring manual specification.
    /// Empty if not populated by the compiler (backward-compatible default).
    #[serde(default)]
    pub mir_ops: Vec<MirOpEntry>,
}

/// A single MIR op entry in the manifest, used for verification.
///
/// This captures the op type name from the MIR graph during compilation
/// so that the `verify` command can compare expected vs actual MIL ops
/// without requiring the user to manually specify `--mir-ops` on the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirOpEntry {
    /// MIR op type name (e.g., "Linear", "Gelu", "ScaledDotProductAttention").
    pub op_type: String,
}

/// Tensor shape and dtype specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Tensor shape (concrete dimensions).
    pub shape: Vec<usize>,
    /// Data type (e.g., "fp16", "fp32").
    pub dtype: String,
}

/// Entry for a state declaration in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry {
    pub state_id: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub owner_package: String,
}

/// Entry for a handoff in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffEntry {
    pub from_package: String,
    pub to_package: String,
    pub tensor_name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}
