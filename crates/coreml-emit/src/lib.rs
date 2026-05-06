//! Direct Protobuf-Based ML Package Emission (Sprint 41)
//!
//! This crate implements direct emission of Core ML `.mlpackage` artifacts
//! from Rust, bypassing the Python bridge entirely. The emission path is:
//!
//! ```text
//! Rust MIR → CoreMlModel (proto representation) → .mlpackage on disk
//! ```
//!
//! This replaces the previous path:
//! ```text
//! Rust MIR → JSON payload → Python subprocess → coremltools → .mlpackage on disk
//! ```
//!
//! ## Key Advantages
//!
//! 1. **True weight sharing**: By controlling the protobuf serialization directly,
//!    we can emit multiple functions that share the same weight tensor by referencing
//!    the same offset in `weight.bin`. coremltools 9.0's `add_function()` cannot do
//!    this — it duplicates constants per function boundary.
//!
//! 2. **No Python dependency**: The compiler can emit valid mlpackages without
//!    requiring coremltools, numpy, or a Python runtime.
//!
//! 3. **Deterministic emission**: Protobuf serialization is deterministic by default,
//!    ensuring bit-for-bit reproducible mlpackage output.
//!
//! 4. **Full structural control**: We can construct any valid Core ML model
//!    structure, including those that coremltools doesn't expose through its API.
//!
//! ## Current Scope
//!
//! The initial implementation supports:
//! - Single-function linear projection (proof of concept)
//! - Multi-function packages with shared weights
//! - Stateful decode-step models (iOS 18+)
//! - All 29 MIR ops through the proto representation
//!
//! ## mlpackage Directory Structure
//!
//! An `.mlpackage` directory follows Apple's Core ML package format:
//!
//! ```text
//! model.mlpackage/
//! ├── Manifest.json           — Package metadata (Apple schema)
//! └── Data/
//!     └── com.apple.CoreML/
//!         ├── model.mlmodel   — Protobuf model definition
//!         └── weights/
//!             └── weight.bin  — Concatenated weight data
//! ```
//!
//! The `model.mlmodel` file contains a serialized `Model` protobuf message
//! as defined in `Model.proto`. The `weight.bin` file contains the raw weight
//! data for all constant tensors, referenced by offset from the protobuf.
//!
//! ## Validation
//!
//! Emitted mlpackages can be validated by:
//! 1. Loading with `ct.models.MLModel(path)` in coremltools
//! 2. Structural inspection via `MLModelStructure.load_from_path()`
//! 3. On-device `predict()` execution (requires Apple hardware)
//!
//! **Note**: Rust toolchain is not available in the current environment,
//! so this crate cannot be compiled here. It is designed for macOS CI
//! where the full Rust toolchain and Core ML runtime are available.

pub mod emitter;
pub mod mir_to_proto;
pub mod package;
pub mod weights;

pub use emitter::ProtoEmitter;
pub use emitter::{compilation_count, COMPILATION_LIMIT, COMPILATION_WARNING_THRESHOLD}; // T-120
pub use mir_to_proto::convert_mir_to_proto;
// M-028: ValidationPolicy is now re-exported from placement_validate
pub use mir_to_proto::ValidationPolicy;
// M-028: MIN_IOSURFACE_BYTES is now re-exported from placement_validate
pub use mir_to_proto::MIN_IOSURFACE_BYTES;
pub use package::MlPackageWriter;
pub use weights::WeightBinBuilder;

/// Errors produced by the emission layer during MIR-to-proto conversion.
///
/// T-P2-10 / T-P3-01: These typed error variants enable programmatic error
/// handling by callers — they can match on specific error kinds rather than
/// parsing string messages from `anyhow::Error`.
#[derive(Debug, thiserror::Error)]
pub enum EmissionError {
    /// A graph input or output descriptor is missing shape/dtype information.
    ///
    /// Previously, missing I/O descriptors silently defaulted to empty shape
    /// and Float16 dtype, producing models that compile but have wrong I/O
    /// types (e.g., Int32 inputs marked as Float16).
    #[error(
        "Missing {kind} descriptor for '{name}' in function '{function}'. \
             Cannot determine shape/dtype for Core ML proto emission. \
             All graph {kind}s must have corresponding {kind}_descs entries."
    )]
    MissingIODescriptor {
        /// Whether this is an "input" or "output" descriptor.
        kind: String,
        /// The I/O name that was not found.
        name: String,
        /// The function name where the descriptor is missing.
        function: String,
    },

    /// An output tensor's total byte size is below the ANE's minimum
    /// IOSurface allocation threshold (~49 KB). The ANE silently fails
    /// with a 0x1d runtime error for undersized buffers.
    #[error(
        "Output '{name}' in function '{function}' is {actual_bytes} bytes, \
             below the ANE minimum IOSurface size of {min_bytes} bytes. \
             The ANE silently fails with 0x1d for undersized output buffers."
    )]
    UndersizedIOSurface {
        /// The output tensor name.
        name: String,
        /// The function name.
        function: String,
        /// Actual byte size of the output tensor.
        actual_bytes: usize,
        /// Minimum required byte size (MIN_IOSURFACE_BYTES).
        min_bytes: usize,
    },

    /// An output tensor in a multi-output function has a different byte size
    /// from other outputs. The ANE requires all outputs in a function to have
    /// the same IOSurface size (Orion constraints #2 and #18).
    #[error(
        "Non-uniform output IOSurface sizes in function '{function}'. \
             Output '{name}' is {actual_bytes} bytes but other outputs are \
             {expected_bytes} bytes. The ANE requires uniform IOSurface sizes \
             for multi-output functions."
    )]
    NonUniformSurface {
        /// The output tensor name with the non-uniform size.
        name: String,
        /// The function name.
        function: String,
        /// Actual byte size of the output tensor.
        actual_bytes: usize,
        /// Expected byte size (from the first output).
        expected_bytes: usize,
    },

    /// An output tensor does not follow the ANE's canonical [1,C,1,S] flat
    /// buffer layout convention (Orion constraint #20). The ANE expects
    /// 4D tensors with the layout [1,channels,1,spatial].
    #[error(
        "Output '{name}' in function '{function}' has shape {shape:?} which \
             does not follow the ANE's canonical [1,C,1,S] flat buffer layout \
             convention (Orion #20). The ANE expects 4D output tensors."
    )]
    InvalidFlatBufferLayout {
        /// The output tensor name.
        name: String,
        /// The function name.
        function: String,
        /// The output tensor shape.
        shape: Vec<usize>,
    },
}
