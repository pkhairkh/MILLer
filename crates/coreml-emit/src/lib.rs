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
pub use package::MlPackageWriter;
pub use weights::WeightBinBuilder;
