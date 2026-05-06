//! Core ML Protobuf Definitions (Sprint 41)
//!
//! This crate provides Rust type definitions for the Core ML model
//! protobuf format, enabling direct Rust-to-mlpackage emission without
//! the Python bridge subprocess.
//!
//! ## Architecture
//!
//! The proto definitions are organized into three files:
//! - `DataStructures.proto` — Tensor types, feature descriptions, weight data
//! - `MIL.proto` — ML Program operations (29 ops matching our MIR enum)
//! - `Model.proto` — Top-level Model message with description and MLProgram
//!
//! ## Why This Exists
//!
//! Previously, all Core ML interaction went through the Python bridge:
//! Rust → JSON payload → Python subprocess → coremltools → .mlpackage
//!
//! This creates several structural limitations:
//! 1. **No weight sharing across functions**: coremltools 9.0's `add_function()`
//!    duplicates constants per function boundary (Sprint 42 finding).
//! 2. **Subprocess overhead**: Each compile step spawns a Python process.
//! 3. **Python dependency**: The compiler requires coremltools at runtime.
//! 4. **Limited structural control**: We can't manipulate the protobuf directly.
//!
//! With proto-direct emission, the path becomes:
//! Rust MIR → Core ML protobuf → .mlpackage on disk
//!
//! This eliminates the Python subprocess for emission and gives us full
//! control over the serialized model format, enabling:
//! - True weight sharing across functions (shared weight tensor references)
//! - Deterministic emission without Python dependency
//! - Future FFI integration with the Core ML C API
//!
//! ## Proto Origin
//!
//! These definitions are based on Apple's public Core ML specification:
//! - https://developer.apple.com/documentation/coreml
//! - coremltools source: https://github.com/apple/coremltools
//!
//! The subset covers the 29 MIL operations in our MIR enum plus the
//! structural messages needed for valid mlpackage construction.
//!
//! ## Build
//!
//! The `build.rs` script compiles the .proto files using prost-build.
//! Generated Rust code appears in `target/` during build. The types
//! are re-exported from this module via `include!` of the generated code.

/// Prost-generated Core ML protobuf types (legacy custom format).
///
/// This module uses a custom `coreml` protobuf package that does
/// not match Apple's actual wire format. Use [`apple_proto`] for
/// producing protobuf that Core ML's runtime can decode. This module
/// is kept for backward compatibility with existing tests.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/coreml.rs"));
}

/// Prost-generated Apple-compatible Core ML protobuf types.
///
/// These match Apple's actual wire format exactly:
/// - `apple_proto::Model` — top-level model (package `CoreML.Specification`)
/// - `apple_proto::mil_spec::*` — MIL operations (package `CoreML.Specification.MILSpec`)
///
/// This is the format that Core ML's runtime can actually decode.
pub mod apple_proto {
    pub mod mil_spec {
        include!(concat!(env!("OUT_DIR"), "/core_ml.specification.mil_spec.rs"));
    }
    include!(concat!(env!("OUT_DIR"), "/core_ml.specification.rs"));
}

use serde::{Deserialize, Serialize};

// ─── Core ML Types (hand-written for environments without proto compilation) ─

/// Data type for tensor elements.
/// Mirrors `ArrayDataType` from DataStructures.proto plus `DataType` from MIL.proto.
///
/// T-35 (I-14): Added Int4, UInt4, E4M3, E5M2, UInt16 for proper
/// quantization and float8 constraint enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreMlDataType {
    Unknown,
    Float32,
    Float16,
    Float64,
    Int32,
    UInt8,
    Int8,
    Bool,
    /// 4-bit signed integer (MIL proto DataType::INT4 = 25).
    Int4,
    /// 4-bit unsigned integer (MIL proto DataType::UINT4 = 35).
    UInt4,
    /// 8-bit float E4M3 (MIL proto DataType::FLOAT8E4M3FN = 40).
    E4M3,
    /// 8-bit float E5M2 (MIL proto DataType::FLOAT8E5M2 = 41).
    E5M2,
    /// 16-bit unsigned integer (MIL proto DataType::UINT16 = 32).
    UInt16,
}

impl CoreMlDataType {
    /// Convert from MIR dtype to Core ML data type.
    pub fn from_mir_dtype(mir_dtype: &crate::mir_compat::MilDtypeCompat) -> Self {
        match mir_dtype {
            crate::mir_compat::MilDtypeCompat::Fp16 => CoreMlDataType::Float16,
            crate::mir_compat::MilDtypeCompat::Fp32 => CoreMlDataType::Float32,
            crate::mir_compat::MilDtypeCompat::Int32 => CoreMlDataType::Int32,
            crate::mir_compat::MilDtypeCompat::UInt8 => CoreMlDataType::UInt8,
            crate::mir_compat::MilDtypeCompat::Int4 => CoreMlDataType::Int4,
            crate::mir_compat::MilDtypeCompat::UInt4 => CoreMlDataType::UInt4,
            crate::mir_compat::MilDtypeCompat::E4M3 => CoreMlDataType::E4M3,
            crate::mir_compat::MilDtypeCompat::E5M2 => CoreMlDataType::E5M2,
            crate::mir_compat::MilDtypeCompat::UInt16 => CoreMlDataType::UInt16,
        }
    }

    /// Size in bytes per element.
    pub fn element_size(&self) -> usize {
        match self {
            // T-71: Float64 (double precision) is 8 bytes per element, not 4.
            // Previously Float32 and Float64 shared a match arm returning 4,
            // causing weight entries with half the required bytes for Float64.
            CoreMlDataType::Float32 => 4,
            CoreMlDataType::Float64 => 8,
            CoreMlDataType::Float16 => 2,
            CoreMlDataType::Int32 => 4,
            CoreMlDataType::UInt8 | CoreMlDataType::Int8 | CoreMlDataType::Bool => 1,
            CoreMlDataType::Int4 | CoreMlDataType::UInt4 => 1, // 4-bit types stored as 1 byte
            CoreMlDataType::E4M3 | CoreMlDataType::E5M2 => 1,  // 8-bit float types
            CoreMlDataType::UInt16 => 2,
            CoreMlDataType::Unknown => 0,
        }
    }

    /// Core ML protobuf enum value.
    pub fn proto_value(&self) -> i32 {
        match self {
            CoreMlDataType::Unknown => 0,
            CoreMlDataType::Float32 => 1,
            CoreMlDataType::Float16 => 2,
            CoreMlDataType::Float64 => 3,
            CoreMlDataType::Int32 => 4,
            CoreMlDataType::UInt8 => 5,
            CoreMlDataType::Int8 => 6,
            CoreMlDataType::Bool => 7,
            // T-35: new dtypes use Apple MIL proto DataType values
            CoreMlDataType::Int4 => 25,   // DataType::INT4
            CoreMlDataType::UInt4 => 35,  // DataType::UINT4
            CoreMlDataType::E4M3 => 40,   // DataType::FLOAT8E4M3FN
            CoreMlDataType::E5M2 => 41,   // DataType::FLOAT8E5M2
            CoreMlDataType::UInt16 => 32, // DataType::UINT16
        }
    }
}

/// Compute unit preference for Core ML model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreMlComputeUnit {
    Unknown,
    CpuOnly,
    CpuAndGpu,
    CpuAndNe,
    All,
}

impl CoreMlComputeUnit {
    /// Convert from MIR compute unit hint.
    pub fn from_mir_hint(hint: &crate::mir_compat::ComputeUnitHintCompat) -> Self {
        match hint {
            crate::mir_compat::ComputeUnitHintCompat::CPUAndNE => CoreMlComputeUnit::CpuAndNe,
            crate::mir_compat::ComputeUnitHintCompat::CPUAndGPU => CoreMlComputeUnit::CpuAndGpu,
            crate::mir_compat::ComputeUnitHintCompat::CPUOnly => CoreMlComputeUnit::CpuOnly,
            crate::mir_compat::ComputeUnitHintCompat::All => CoreMlComputeUnit::All,
        }
    }

    /// Core ML protobuf enum value.
    pub fn proto_value(&self) -> i32 {
        match self {
            CoreMlComputeUnit::Unknown => 0,
            CoreMlComputeUnit::CpuOnly => 1,
            CoreMlComputeUnit::CpuAndGpu => 2,
            CoreMlComputeUnit::CpuAndNe => 3,
            CoreMlComputeUnit::All => 4,
        }
    }

    /// String representation used in bridge payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            CoreMlComputeUnit::Unknown => "ALL",
            CoreMlComputeUnit::CpuOnly => "CPU_ONLY",
            CoreMlComputeUnit::CpuAndGpu => "CPU_AND_GPU",
            CoreMlComputeUnit::CpuAndNe => "CPU_AND_NE",
            CoreMlComputeUnit::All => "ALL",
        }
    }
}

/// Specification version for the Core ML model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecVersion {
    /// ML Program format (iOS 15+, macOS 12+)
    V7,
    /// ML Program with state support (iOS 18+, macOS 15+)
    V8,
    /// ML Program with expanded state support (iOS 18+, macOS 15+)
    V9,
    /// ML Program with multi-function + state support (iOS 17+, macOS 14+)
    V10,
}

impl SpecVersion {
    /// Protobuf specification version number.
    pub fn proto_value(&self) -> i32 {
        match self {
            SpecVersion::V7 => 7,
            SpecVersion::V8 => 8,
            SpecVersion::V9 => 9,
            SpecVersion::V10 => 10,
        }
    }

    /// Whether this version supports stateful models (mb.read_state / mb.write_state).
    pub fn supports_state(&self) -> bool {
        matches!(self, SpecVersion::V8 | SpecVersion::V9 | SpecVersion::V10)
    }
}

/// A weight tensor entry in the weight.bin file.
///
/// This represents a single weight tensor that will be serialized into the
/// `Data/com.apple.CoreML/weights/weight.bin` file inside the mlpackage.
/// The weight data is stored at a specific offset with a known size,
/// and the model protobuf references it by offset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    /// Unique name for this weight tensor (e.g., "shared_projection_weight").
    pub name: String,
    /// Offset in bytes into the weight.bin file.
    pub offset: u64,
    /// Size in bytes of the weight data.
    pub size: u64,
    /// Shape of the weight tensor.
    pub shape: Vec<u64>,
    /// Data type of the weight tensor.
    pub dtype: CoreMlDataType,
    /// Raw weight data.
    pub data: Vec<u8>,
}

/// A shared weight reference that can be used across function boundaries.
///
/// This is the key primitive that coremltools 9.0 lacks: when two functions
/// reference the same weight, they should share the same offset in weight.bin
/// rather than each getting their own copy. This struct captures that sharing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedWeightRef {
    /// The weight entry being shared.
    pub weight: WeightEntry,
    /// Names of the functions that reference this weight.
    pub referencing_functions: Vec<String>,
}

/// Manifest.json content for an mlpackage.
///
/// Every .mlpackage directory contains a Manifest.json file that describes
/// the package structure using Apple's required schema:
/// - `fileFormatVersion`: Must be `"1.0.0"`
/// - `itemInfoEntries`: Map of UUID → item info (path, name, author, description)
/// - `rootModelIdentifier`: UUID of the model.mlmodel entry
///
/// Reference: coremltools ModelPackage.cpp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    /// File format version. Must be `"1.0.0"` (Apple's only supported value).
    #[serde(rename = "fileFormatVersion")]
    pub file_format_version: String,
    /// Map of UUID string → item info entries.
    #[serde(rename = "itemInfoEntries")]
    pub item_info_entries: std::collections::HashMap<String, ManifestItemInfo>,
    /// UUID of the root model specification entry.
    #[serde(rename = "rootModelIdentifier")]
    pub root_model_identifier: String,
}

/// A single item entry in the Manifest.json `itemInfoEntries` map.
///
/// Each entry MUST have exactly 4 keys: `path`, `name`, `author`, `description`.
/// The `path` is relative to the `Data/` directory inside the mlpackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestItemInfo {
    /// Relative path under `Data/` directory (e.g., `"com.apple.CoreML/model.mlmodel"`).
    pub path: String,
    /// Item name (e.g., `"model.mlmodel"` or `"weights"`).
    pub name: String,
    /// Author identifier (e.g., `"com.apple.CoreML"`).
    pub author: String,
    /// Human-readable description.
    pub description: String,
}

/// Legacy manifest metadata — kept for backward compatibility with existing
/// conversion functions that populate user-defined metadata. This data is
/// now placed inside the Apple protobuf `Metadata.userDefined` map instead
/// of the Manifest.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifestMetadata {
    /// Author string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Short description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_description: Option<String>,
    /// License string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Version string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// User-defined metadata.
    #[serde(rename = "userDefined", default, skip_serializing_if = "HashMap::is_empty")]
    pub user_defined: std::collections::HashMap<String, String>,
}

use std::collections::HashMap;

// ─── MIR Compatibility Layer ─────────────────────────────────────────────────
// These types mirror the MIR types from ane-ir so that this crate can
// consume MIR graphs without a direct dependency on ane-ir (which would
// create a circular dependency). Instead, the coreml-emit crate converts
// from ane-ir::mir to these compat types.

pub mod mir_compat {
    use serde::{Deserialize, Serialize};

    /// MIR dtype compatibility type.
    ///
    /// T-35 (I-14): Added Int4, UInt4, E4M3, E5M2, UInt16 for proper
    /// quantization and float8 constraint enforcement. These map directly
    /// to Apple's MIL proto DataType enum values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum MilDtypeCompat {
        Fp16,
        Fp32,
        Int32,
        UInt8,
        /// 4-bit signed integer. Used for palettized/quantized weights.
        Int4,
        /// 4-bit unsigned integer. Used for palettized weights.
        UInt4,
        /// 8-bit floating point (4-bit exponent, 3-bit mantissa).
        E4M3,
        /// 8-bit floating point (5-bit exponent, 2-bit mantissa).
        E5M2,
        /// 16-bit unsigned integer.
        UInt16,
    }

    /// Compute unit hint compatibility type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum ComputeUnitHintCompat {
        CPUAndNE,
        CPUAndGPU,
        CPUOnly,
        All,
    }

    /// A single operation in the MIR graph (compatibility representation).
    ///
    /// **Note**: This type should eventually be replaced by direct `MirOp` usage
    /// with a `ToProto` trait, eliminating the manual compat-layer duplication.
    /// See T-17 (W-12, W-16) for the unification roadmap. Every variant here
    /// should map 1:1 to a `MirOp` variant; ops without a specialized compat
    /// representation fall into `Unsupported` as a transitional measure.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum MirOpCompat {
        Const {
            name: String,
            data: Vec<u8>,
            dtype: MilDtypeCompat,
            shape: Vec<usize>,
        },
        Linear {
            name: String,
            x: String,
            weight_name: String,
            bias_name: Option<String>,
        },
        MatMul {
            name: String,
            x: String,
            y: String,
            transpose_y: bool,
        },
        Add {
            name: String,
            x: String,
            y: String,
        },
        Mul {
            name: String,
            x: String,
            y: String,
        },
        Sub {
            name: String,
            x: String,
            y: String,
        },
        Abs {
            name: String,
            x: String,
        },
        Maximum {
            name: String,
            x: String,
            y: String,
        },
        Minimum {
            name: String,
            x: String,
            y: String,
        },
        Reshape {
            name: String,
            x: String,
            /// INT32 shape vector. Core ML's ios19.reshape rejects INT64 shape
            /// tensors ("Expected { tensor<int32, [?]>, tensor<int16, [?]>,
            /// tensor<int8, [?]> }; got tensor<int64, [N]>").
            shape: Vec<i32>,
        },
        Transpose {
            name: String,
            x: String,
            /// INT32 permutation vector. Core ML's ios19.transpose rejects
            /// INT64 perm tensors, same dtype restriction as reshape.shape.
            perm: Vec<i32>,
        },
        SliceByIndex {
            name: String,
            x: String,
            /// INT32 begin indices. Core ML's ios19.slice_by_index rejects
            /// INT64 for begin/end, same dtype restriction as reshape.shape.
            begin: Vec<i32>,
            end: Vec<i32>,
            /// INT32 stride values. Empty = all 1s (default).
            stride: Vec<i32>,
            /// Boolean mask for begin: mask[N]=true means "ignore begin[N], start at 0".
            /// Emitted as tensor<bool, [rank]> per Core ML ios19 schema.
            begin_mask: Vec<bool>,
            /// Boolean mask for end: mask[N]=true means "ignore end[N], go to full extent".
            /// Emitted as tensor<bool, [rank]> per Core ML ios19 schema.
            end_mask: Vec<bool>,
            /// Boolean squeeze mask: mask[N]=true means "remove dimension N from output".
            /// Emitted as tensor<bool, [rank]> per Core ML ios19 schema.
            squeeze_mask: Vec<bool>,
        },
        SliceUpdate {
            name: String,
            x: String,
            update: String,
            /// INT32 begin/end indices. Same INT64 rejection as slice_by_index.
            begin: Vec<i32>,
            end: Vec<i32>,
        },
        Concat {
            name: String,
            values: Vec<String>,
            axis: i64,
        },
        /// Stack: concatenates tensors along a new axis dimension.
        /// Semantically equivalent to: unsqueeze each input at `axis`, then concat along `axis`.
        /// Core ML MIL program op type: "stack".
        Stack {
            name: String,
            values: Vec<String>,
            axis: i64,
        },
        Softmax {
            name: String,
            x: String,
            axis: i64,
        },
        Gelu {
            name: String,
            x: String,
            mode: String,
        },
        ScaledDotProductAttention {
            name: String,
            query: String,
            key: String,
            value: String,
            attention_mask: Option<String>,
            scale: Option<f32>,
        },
        ReadState {
            name: String,
            state_id: String,
            shape: Vec<usize>,
            dtype: MilDtypeCompat,
        },
        CoremlUpdateState {
            name: String,
            state_id: String,
            value: String,
        },
        Gather {
            name: String,
            x: String,
            indices: String,
            axis: i64,
        },
        ReduceMean {
            name: String,
            x: String,
            axes: Vec<i64>,
            keep_dims: bool,
        },
        // Sprint 54: ReduceSum compat (was previously bailing in mir_to_compat)
        ReduceSum {
            name: String,
            x: String,
            axes: Vec<i64>,
            keep_dims: bool,
        },
        // Sprint 54: Conv compat (was previously bailing in mir_to_compat)
        Conv {
            name: String,
            x: String,
            weight: String,
            pad_type: String,
            groups: i64,
            /// T-98 (V-110): Per-op scale factor for quantized/palettized conv weights.
            kernel_scale: Option<f32>,
            /// T-98 (V-110): Zero-point offset for quantized conv weights.
            kernel_zero_point: Option<i32>,
            /// T-98 (V-110): Name of the palettized LUT weight for this conv.
            kernel_palettized_lut: Option<String>,
        },
        // Sprint 54: StateWrite compat (was previously bailing in mir_to_compat)
        // Note: This is the generic state-write op, distinct from CoremlUpdateState
        // which is the iOS 18+ specific state mutation op.
        StateWrite {
            name: String,
            state_ref: String,
            value: String,
        },
        Rsqrt {
            name: String,
            x: String,
        },
        RealDiv {
            name: String,
            x: String,
            y: String,
        },
        LayerNorm {
            name: String,
            x: String,
            weight_name: String,
            bias_name: Option<String>,
            epsilon: f32,
            axes: Vec<i64>,
        },
        Topk {
            name: String,
            x: String,
            k: i64,
            axis: i64,
        },
        Cos {
            name: String,
            x: String,
        },
        Sin {
            name: String,
            x: String,
        },
        Cast {
            name: String,
            x: String,
            dtype: MilDtypeCompat,
        },
        Split {
            name: String,
            x: String,
            axis: i64,
            num_splits: i64,
        },
        // Sprint 50: P2 MIR ops
        Exp {
            name: String,
            x: String,
        },
        Sigmoid {
            name: String,
            x: String,
        },
        Tanh {
            name: String,
            x: String,
        },
        Relu {
            name: String,
            x: String,
        },
        Where {
            name: String,
            condition: String,
            x: String,
            y: String,
        },
        /// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
        /// Core ML MIL program op type: "silu".
        Silu {
            name: String,
            x: String,
        },
        /// Identity / passthrough op. Core ML MIL program op type: "identity".
        /// Carries the output dtype so the MIL operation's output type matches
        /// the input (critical for integer inputs like input_ids).
        Identity {
            name: String,
            x: String,
            dtype: MilDtypeCompat,
        },
        /// Placeholder op for graph inputs. Core ML MIL program op type: "placeholder".
        /// This is the correct way to declare a graph input in Core ML's MIL format.
        /// Unlike Identity (which references another SSA value), Placeholder takes
        /// no inputs and produces the named output tensor, declaring it as a function
        /// parameter. Carries shape and dtype for the output type declaration.
        Placeholder {
            name: String,
            dtype: MilDtypeCompat,
        },
        /// Tile / repeat op. Core ML MIL program op type: "tile".
        /// Repeats the input tensor along each dimension according to `reps`.
        /// For example, GQA K/V expansion uses reps=[1, n_rep, 1, 1] to
        /// replicate KV heads to match the number of query heads.
        Tile {
            name: String,
            x: String,
            /// INT32 repetition counts. Core ML's ios19.tile rejects INT64
            /// reps tensors, same dtype restriction as reshape.shape.
            reps: Vec<i32>,
        },
        /// Fill: creates a tensor of the given shape filled with a scalar value.
        /// Core ML MIL program op type: "fill".
        /// This is the primary op for generating constant tensors during
        /// Tile decomposition (e.g., ones tensors for broadcast Mul in GQA).
        Fill {
            name: String,
            /// INT32 shape vector. Core ML's ios19.fill rejects INT64 shape
            /// tensors, same dtype restriction as reshape.shape.
            shape: Vec<i32>,
            value: f32,
            dtype: MilDtypeCompat,
        },
        /// FillLike: creates a tensor with the same shape as a reference tensor,
        /// filled with a scalar value. Core ML MIL program op type: "fill_like".
        FillLike {
            name: String,
            ref_tensor: String,
            value: f32,
            dtype: MilDtypeCompat,
        },
        /// Neg: arithmetic negation. Lowered to mul(x, -1) because
        /// Core ML has no "neg" op in the ML Program op set.
        /// Needed for RoPE rotate_half: -x[..., d//2:].
        Neg {
            name: String,
            x: String,
        },
        /// ExpandDims: insert singleton dimensions. Core ML MIL op type: "expand_dims".
        /// Used for adding head/sequence dimensions before broadcast ops in attention.
        ExpandDims {
            name: String,
            x: String,
            /// INT32 axis vector. Core ML's ios19.expand_dims rejects INT64
            /// axis tensors, same dtype restriction as reshape.shape.
            axis: Vec<i32>,
        },
        /// Squeeze: remove singleton dimensions. Core ML MIL op type: "squeeze".
        /// Used for collapsing dimensions after reduction ops in attention output.
        Squeeze {
            name: String,
            x: String,
            /// INT32 axis vector. Core ML's ios19.squeeze rejects INT64 axis
            /// tensors, same dtype restriction as reshape.shape.
            axis: Vec<i32>,
        },
        /// Sqrt: element-wise square root. Core ML MIL op type: "sqrt".
        /// Used in RMSNorm (alternative to Rsqrt) and scaling computations.
        Sqrt {
            name: String,
            x: String,
        },
        /// Pow: element-wise power. Core ML MIL op type: "pow".
        /// Used in RoPE frequency computation and attention scaling.
        Pow {
            name: String,
            x: String,
            y: String,
        },
        /// Clip: clamp values to [min, max]. Core ML MIL op type: "clip".
        /// Used for gradient clipping and attention score clamping.
        Clip {
            name: String,
            x: String,
            min_val: f32,
            max_val: f32,
        },
        /// Equal: element-wise equality comparison. Core ML MIL op type: "equal".
        /// Used for attention masking (padding mask generation).
        Equal {
            name: String,
            x: String,
            y: String,
        },
        /// NotEqual: element-wise inequality. Core ML MIL op type: "not_equal".
        NotEqual {
            name: String,
            x: String,
            y: String,
        },
        /// Greater: element-wise greater-than. Core ML MIL op type: "greater".
        /// Used for attention causal masking.
        Greater {
            name: String,
            x: String,
            y: String,
        },
        /// GreaterEqual: element-wise greater-or-equal. Core ML MIL op type: "greater_equal".
        GreaterEqual {
            name: String,
            x: String,
            y: String,
        },
        /// Less: element-wise less-than. Core ML MIL op type: "less".
        /// Used for attention causal masking.
        Less {
            name: String,
            x: String,
            y: String,
        },
        /// LessEqual: element-wise less-or-equal. Core ML MIL op type: "less_equal".
        LessEqual {
            name: String,
            x: String,
            y: String,
        },
        /// LogicalNot: element-wise logical NOT. Core ML MIL op type: "logical_not".
        /// Used for inverting attention masks.
        LogicalNot {
            name: String,
            x: String,
        },
        /// LogicalAnd: element-wise logical AND. Core ML MIL op type: "logical_and".
        LogicalAnd {
            name: String,
            x: String,
            y: String,
        },
        /// LogicalOr: element-wise logical OR. Core ML MIL op type: "logical_or".
        LogicalOr {
            name: String,
            x: String,
            y: String,
        },
        /// Pad: tensor padding. Core ML MIL op type: "pad".
        /// Used for attention padding and convolution boundary handling.
        Pad {
            name: String,
            x: String,
            /// INT32 padding amounts. Core ML's ios19.pad rejects INT64 pad
            /// tensors, same dtype restriction as reshape.shape.
            pad_amounts: Vec<i32>,
            mode: String,
            constant_value: f32,
        },
        /// ReduceMax: max reduction. Core ML MIL op type: "reduce_max".
        /// Used for max pooling and attention score normalization.
        ReduceMax {
            name: String,
            x: String,
            axes: Vec<i64>,
            keep_dims: bool,
        },
        /// ReduceMin: min reduction. Core ML MIL op type: "reduce_min".
        ReduceMin {
            name: String,
            x: String,
            axes: Vec<i64>,
            keep_dims: bool,
        },
        /// ReduceProd: product reduction. Core ML MIL op type: "reduce_prod".
        ReduceProd {
            name: String,
            x: String,
            axes: Vec<i64>,
            keep_dims: bool,
        },
        /// Select: conditional select. Core ML MIL op type: "select".
        /// Used for conditional masking and blending.
        Select {
            name: String,
            condition: String,
            x: String,
            y: String,
        },
        /// LeakyRelu: leaky ReLU activation. Core ML MIL op type: "leaky_relu".
        /// Used in some model architectures.
        LeakyRelu {
            name: String,
            x: String,
            alpha: f32,
        },
        /// FloorDiv: integer division. Core ML MIL op type: "floor_div".
        FloorDiv {
            name: String,
            x: String,
            y: String,
        },
        /// Mod: modulo. Core ML MIL op type: "mod".
        Mod {
            name: String,
            x: String,
            y: String,
        },
        /// Ceil: ceiling. Core ML MIL op type: "ceil".
        Ceil {
            name: String,
            x: String,
        },
        /// Floor: floor. Core ML MIL op type: "floor".
        Floor {
            name: String,
            x: String,
        },
        /// Round: rounding. Core ML MIL op type: "round".
        Round {
            name: String,
            x: String,
        },
        /// Sign: sign function. Core ML MIL op type: "sign".
        Sign {
            name: String,
            x: String,
        },
        /// Log: natural logarithm. Core ML MIL op type: "log".
        Log {
            name: String,
            x: String,
        },
        // ─── Constexpr / Weight Compression (T-39: I-18) ────────────
        //
        // These represent Core ML's constexpr weight compression ops.
        // The Python bridge can emit palettized/quantized/compressed weights;
        // the proto-direct path previously could not (I-18).
        //
        // Each constexpr op produces a constant tensor output and references
        // weight data by name (resolved to MIL Const op outputs during emission).
        // Unlike regular MirOpCompat variants where inputs are SSA value names,
        // these `String` fields are weight-name references to MILConst nodes.
        /// Constexpr affine dequantize: decompresses quantized weights using
        /// per-channel scale and zero_point. Core ML MIL op type:
        /// "constexpr_affine_dequantize".
        ConstexprAffineDequantize {
            name: String,
            /// Weight name reference for the quantized data tensor.
            quantized_data: String,
            scale: f32,
            zero_point: i32,
            axis: i64,
        },
        /// Constexpr blockwise shift-scale: decompresses weights using
        /// per-block scale and offset. Core ML MIL op type:
        /// "constexpr_blockwise_shift_scale".
        /// T-25 (I-04): Hard-rejected by is_blockwise_scale_supported() on ANE.
        ConstexprBlockwiseShiftScale {
            name: String,
            /// Weight name reference for the data tensor.
            data: String,
            /// Weight name reference for the scale tensor.
            scale: String,
            /// Weight name reference for the offset tensor.
            offset: String,
            block_size: Vec<i64>,
        },
        /// Constexpr LUT-to-dense: decompresses palettized (LUT-based) weights.
        /// Core ML MIL op type: "constexpr_lut_to_dense".
        /// This is the primary op for palettized weight emission.
        ConstexprLutToDense {
            name: String,
            /// Weight name reference for the indices (palette lookups).
            indices: String,
            /// Weight name reference for the LUT (lookup table / palette).
            lut: String,
            num_bits: i64,
        },
        /// Constexpr sparse-to-dense: decompresses sparse weights to dense form.
        /// Core ML MIL op type: "constexpr_sparse_to_dense".
        ConstexprSparseToDense {
            name: String,
            /// Weight name reference for the nonzero data.
            nonzero_data: String,
            shape: Vec<i64>,
            default_value: f32,
        },
        /// Constexpr cast: changes the data type of a constant tensor.
        /// Core ML MIL op type: "constexpr_cast".
        ConstexprCast {
            name: String,
            /// Weight name reference for the source data tensor.
            data: String,
            dtype: MilDtypeCompat,
        },
        /// Constexpr LUT-to-sparse: compresses a dense tensor into a
        /// palettized + sparse representation. Core ML MIL op type:
        /// "constexpr_lut_to_sparse".
        ConstexprLutToSparse {
            name: String,
            /// Weight name reference for the source data tensor.
            data: String,
            num_bits: i64,
        },
        /// Constexpr sparse blockwise shift-scale: applies blockwise
        /// quantization with sparse structure. Core ML MIL op type:
        /// "constexpr_sparse_blockwise_shift_scale".
        /// T-25 (I-04): Hard-rejected by is_blockwise_scale_supported() on ANE.
        ConstexprSparseBlockwiseShiftScale {
            name: String,
            /// Weight name reference for the data tensor.
            data: String,
            /// Weight name reference for the scale tensor.
            scale: String,
            /// Weight name reference for the offset tensor.
            offset: String,
            block_size: Vec<i64>,
            block_axis: i64,
        },
        // ─── Pooling (T-66 / I-40) ──────────────────────────────────
        /// Max pooling. Core ML MIL op type: "max_pool".
        MaxPool {
            name: String,
            x: String,
            /// INT32 kernel sizes [kh, kw].
            kernel_sizes: Vec<i32>,
            /// INT32 strides [sh, sw].
            strides: Vec<i32>,
            /// Padding type: "valid" or "same".
            pad_type: String,
            /// INT32 padding amounts (if explicit).
            pad_amounts: Vec<i32>,
        },
        /// Average pooling. Core ML MIL op type: "avg_pool".
        AvgPool {
            name: String,
            x: String,
            kernel_sizes: Vec<i32>,
            strides: Vec<i32>,
            pad_type: String,
            pad_amounts: Vec<i32>,
            count_include_padding: bool,
        },
        /// L2 pooling. Core ML MIL op type: "l2_pool".
        L2Pool {
            name: String,
            x: String,
            kernel_sizes: Vec<i32>,
            strides: Vec<i32>,
            pad_type: String,
            pad_amounts: Vec<i32>,
        },
        // ─── Spatial Rearrangement (T-66 / I-40) ────────────────────
        /// Depth-to-space rearrangement. Core ML MIL op type: "depth_to_space".
        DepthToSpace {
            name: String,
            x: String,
            block_size: i64,
        },
        /// Space-to-depth rearrangement. Core ML MIL op type: "space_to_depth".
        SpaceToDepth {
            name: String,
            x: String,
            block_size: i64,
        },
        /// Pixel shuffle (sub-pixel rearrangement). Core ML MIL op type: "pixel_shuffle".
        PixelShuffle {
            name: String,
            x: String,
            upscale_factor: i64,
        },
        /// Pixel unshuffle (inverse of pixel shuffle). Core ML MIL op type: "pixel_unshuffle".
        PixelUnshuffle {
            name: String,
            x: String,
            downscale_factor: i64,
        },
        // ─── Normalization (T-66 / I-40) ────────────────────────────
        /// Batch normalization. Core ML MIL op type: "batch_norm".
        BatchNorm {
            name: String,
            x: String,
            mean: String,
            variance: String,
            gamma: Option<String>,
            beta: Option<String>,
            epsilon: f32,
        },
        /// Instance normalization. Core ML MIL op type: "instance_norm".
        InstanceNorm {
            name: String,
            x: String,
            gamma: Option<String>,
            beta: Option<String>,
            epsilon: f32,
        },
        /// L2 normalization. Core ML MIL op type: "l2_norm".
        L2Norm {
            name: String,
            x: String,
            epsilon: f32,
            axes: Vec<i64>,
        },
        // ─── Quantize / Dequantize (T-66 / I-40) ────────────────────
        /// Quantize: quantizes a float tensor to integer. Core ML MIL op type: "quantize".
        Quantize {
            name: String,
            x: String,
            scale: f32,
            zero_point: i32,
            axis: i64,
            output_dtype: MilDtypeCompat,
        },
        /// Dequantize: dequantizes an integer tensor to float. Core ML MIL op type: "dequantize".
        Dequantize {
            name: String,
            x: String,
            scale: f32,
            zero_point: i32,
            axis: i64,
            output_dtype: MilDtypeCompat,
        },
        // ─── F-OPS-01: ANEC ops promoted from CPU-only ──────────────────
        /// ANEC-internal: fused conv+activation operation.
        /// Decomposed to conv + activation during proto emission since
        /// Core ML proto doesn't have a fused conv+activate op.
        AnecFusedConvActivate {
            name: String,
            x: String,
            weight: String,
            activation: String,
            strides: Vec<i32>,
            pad_amounts: Vec<i32>,
            dilations: Vec<i32>,
            groups: i64,
        },
        /// ANEC-internal: scaled elementwise operation.
        /// Decomposed to mul(x, scale_a) + mul(y, scale_b) during proto
        /// emission since Core ML proto doesn't have a native scaled
        /// elementwise op.
        AnecScaledElementwise {
            name: String,
            x: String,
            y: String,
            scale_a: f32,
            scale_b: f32,
        },
        /// Catch-all for MIL ops that don't have specialized compat representations.
        /// The proto emission layer handles these by emitting the appropriate
        /// MIL builder call based on the op_kind string.
        ///
        /// T-P5-10: `inputs` carries the SSA input reference names so that
        /// `input_names()` returns correct data for weight materialization.
        /// Previously this was missing, causing a silent gap where weights
        /// referenced only by Unsupported ops would be excluded from the
        /// model's weight.bin.
        Unsupported {
            op_kind: String,
            name: String,
            /// Serialized JSON of the op's parameters for flexible emission
            params_json: String,
            /// SSA input reference names (weight / tensor names this op reads).
            /// Populated from `MirOp::proto_input_refs()` at construction time
            /// so that `input_names()` never silently drops weight references.
            inputs: Vec<String>,
        },
    }

    impl MirOpCompat {
        /// Returns the SSA output name of this operation.
        ///
        /// Every variant has a `name: String` field. Uses a macro to avoid
        /// listing all variants in a hand-written match.
        pub fn output_name(&self) -> String {
            macro_rules! name_from_variants {
                ($($variant:ident),* $(,)?) => {
                    match self {
                        $(MirOpCompat::$variant { name, .. } => name.clone(),)*
                    }
                };
            }
            name_from_variants!(
                Const,
                Linear,
                MatMul,
                Add,
                Mul,
                Sub,
                Abs,
                Maximum,
                Minimum,
                Reshape,
                Transpose,
                SliceByIndex,
                SliceUpdate,
                Concat,
                Stack,
                Softmax,
                Gelu,
                ScaledDotProductAttention,
                ReadState,
                CoremlUpdateState,
                Gather,
                ReduceMean,
                ReduceSum,
                Conv,
                StateWrite,
                Rsqrt,
                RealDiv,
                LayerNorm,
                Topk,
                Cos,
                Sin,
                Cast,
                Split,
                Exp,
                Sigmoid,
                Tanh,
                Relu,
                Where,
                Silu,
                Identity,
                Placeholder,
                Tile,
                Fill,
                FillLike,
                Neg,
                ExpandDims,
                Squeeze,
                Sqrt,
                Pow,
                Clip,
                Equal,
                NotEqual,
                Greater,
                GreaterEqual,
                Less,
                LessEqual,
                LogicalNot,
                LogicalAnd,
                LogicalOr,
                Pad,
                ReduceMax,
                ReduceMin,
                ReduceProd,
                Select,
                LeakyRelu,
                FloorDiv,
                Mod,
                Ceil,
                Floor,
                Round,
                Sign,
                Log,
                ConstexprAffineDequantize,
                ConstexprBlockwiseShiftScale,
                ConstexprLutToDense,
                ConstexprSparseToDense,
                ConstexprCast,
                ConstexprLutToSparse,
                ConstexprSparseBlockwiseShiftScale,
                MaxPool,
                AvgPool,
                L2Pool,
                DepthToSpace,
                SpaceToDepth,
                PixelShuffle,
                PixelUnshuffle,
                BatchNorm,
                InstanceNorm,
                L2Norm,
                Quantize,
                Dequantize,
                AnecFusedConvActivate,
                AnecScaledElementwise,
                Unsupported,
            )
        }

        /// Returns all SSA input reference names.
        ///
        /// Replaces the `compat_input_names()` free function in `mir_to_compat.rs`.
        /// The match arms are copied exactly from that function.
        pub fn input_names(&self) -> Vec<String> {
            match self {
                MirOpCompat::Const { .. } | MirOpCompat::Placeholder { .. } => vec![],
                MirOpCompat::Linear { x, weight_name, bias_name, .. } => {
                    let mut names = vec![x.clone(), weight_name.clone()];
                    if let Some(bias) = bias_name {
                        names.push(bias.clone());
                    }
                    names
                }
                MirOpCompat::MatMul { x, y, .. }
                | MirOpCompat::Add { x, y, .. }
                | MirOpCompat::Mul { x, y, .. }
                | MirOpCompat::Sub { x, y, .. }
                | MirOpCompat::Maximum { x, y, .. }
                | MirOpCompat::Minimum { x, y, .. }
                | MirOpCompat::RealDiv { x, y, .. } => vec![x.clone(), y.clone()],
                MirOpCompat::Abs { x, .. }
                | MirOpCompat::Reshape { x, .. }
                | MirOpCompat::Transpose { x, .. }
                | MirOpCompat::Softmax { x, .. }
                | MirOpCompat::Gelu { x, .. }
                | MirOpCompat::ReduceMean { x, .. }
                | MirOpCompat::ReduceSum { x, .. }
                | MirOpCompat::Rsqrt { x, .. }
                | MirOpCompat::Topk { x, .. }
                | MirOpCompat::Cos { x, .. }
                | MirOpCompat::Sin { x, .. }
                | MirOpCompat::Cast { x, .. }
                | MirOpCompat::Split { x, .. }
                | MirOpCompat::Exp { x, .. }
                | MirOpCompat::Sigmoid { x, .. }
                | MirOpCompat::Tanh { x, .. }
                | MirOpCompat::Relu { x, .. }
                | MirOpCompat::Silu { x, .. }
                | MirOpCompat::Identity { x, .. }
                | MirOpCompat::Tile { x, .. } => vec![x.clone()],
                // T-80 (I-55): In Core ML MIL, `fill` takes `shape` as an input
                // tensor (not an attribute). Returning the shape input name allows
                // SSA validation and dead-reference detection to account for the
                // Fill op's shape dependency. The convention is `<name>_shape`.
                MirOpCompat::Fill { name, .. } => vec![format!("{}_shape", name)],
                MirOpCompat::FillLike { ref_tensor, .. } => vec![ref_tensor.clone()],
                MirOpCompat::Neg { x, .. } => vec![x.clone()],
                MirOpCompat::Gather { x, indices, .. } => vec![x.clone(), indices.clone()],
                MirOpCompat::SliceByIndex { x, .. } => vec![x.clone()],
                MirOpCompat::SliceUpdate { x, update, .. } => vec![x.clone(), update.clone()],
                MirOpCompat::Concat { values, .. } => values.clone(),
                MirOpCompat::Stack { values, .. } => values.clone(),
                MirOpCompat::ScaledDotProductAttention {
                    query,
                    key,
                    value,
                    attention_mask,
                    ..
                } => {
                    let mut names = vec![query.clone(), key.clone(), value.clone()];
                    if let Some(mask) = attention_mask {
                        names.push(mask.clone());
                    }
                    names
                }
                MirOpCompat::ReadState { state_id, .. } => vec![state_id.clone()],
                MirOpCompat::CoremlUpdateState { state_id, value, .. } => {
                    vec![state_id.clone(), value.clone()]
                }
                MirOpCompat::Conv { x, weight, .. } => vec![x.clone(), weight.clone()],
                MirOpCompat::StateWrite { state_ref, value, .. } => {
                    vec![state_ref.clone(), value.clone()]
                }
                MirOpCompat::LayerNorm { x, weight_name, bias_name, .. } => {
                    let mut names = vec![x.clone(), weight_name.clone()];
                    if let Some(bias) = bias_name {
                        names.push(bias.clone());
                    }
                    names
                }
                MirOpCompat::Where { condition, x, y, .. } => {
                    vec![condition.clone(), x.clone(), y.clone()]
                }
                // Unary ops
                MirOpCompat::Sqrt { x, .. }
                | MirOpCompat::LogicalNot { x, .. }
                | MirOpCompat::Ceil { x, .. }
                | MirOpCompat::Floor { x, .. }
                | MirOpCompat::Round { x, .. }
                | MirOpCompat::Sign { x, .. }
                | MirOpCompat::Log { x, .. } => vec![x.clone()],
                // Binary ops
                MirOpCompat::Pow { x, y, .. }
                | MirOpCompat::Equal { x, y, .. }
                | MirOpCompat::NotEqual { x, y, .. }
                | MirOpCompat::Greater { x, y, .. }
                | MirOpCompat::GreaterEqual { x, y, .. }
                | MirOpCompat::Less { x, y, .. }
                | MirOpCompat::LessEqual { x, y, .. }
                | MirOpCompat::LogicalAnd { x, y, .. }
                | MirOpCompat::LogicalOr { x, y, .. }
                | MirOpCompat::FloorDiv { x, y, .. }
                | MirOpCompat::Mod { x, y, .. } => vec![x.clone(), y.clone()],
                // Axis-based ops
                MirOpCompat::ExpandDims { x, .. } | MirOpCompat::Squeeze { x, .. } => {
                    vec![x.clone()]
                }
                // Clip: unary with scalar params
                MirOpCompat::Clip { x, .. } => vec![x.clone()],
                // Pad: unary with scalar/vector params
                MirOpCompat::Pad { x, .. } => vec![x.clone()],
                // Reduce ops: unary with axes
                MirOpCompat::ReduceMax { x, .. }
                | MirOpCompat::ReduceMin { x, .. }
                | MirOpCompat::ReduceProd { x, .. } => vec![x.clone()],
                // Select: ternary (like Where)
                MirOpCompat::Select { condition, x, y, .. } => {
                    vec![condition.clone(), x.clone(), y.clone()]
                }
                // LeakyRelu: unary with alpha param
                MirOpCompat::LeakyRelu { x, .. } => vec![x.clone()],
                // T-39: Constexpr weight compression ops.
                // These use weight-name references as inputs (resolved to MIL Const ops).
                MirOpCompat::ConstexprAffineDequantize { quantized_data, .. } => {
                    vec![quantized_data.clone()]
                }
                MirOpCompat::ConstexprBlockwiseShiftScale { data, scale, offset, .. } => {
                    vec![data.clone(), scale.clone(), offset.clone()]
                }
                MirOpCompat::ConstexprLutToDense { indices, lut, .. } => {
                    vec![indices.clone(), lut.clone()]
                }
                MirOpCompat::ConstexprSparseToDense { nonzero_data, .. } => {
                    vec![nonzero_data.clone()]
                }
                MirOpCompat::ConstexprCast { data, .. } => vec![data.clone()],
                MirOpCompat::ConstexprLutToSparse { data, .. } => vec![data.clone()],
                MirOpCompat::ConstexprSparseBlockwiseShiftScale { data, scale, offset, .. } => {
                    vec![data.clone(), scale.clone(), offset.clone()]
                }
                // T-66 (I-40): Pooling ops
                MirOpCompat::MaxPool { x, .. }
                | MirOpCompat::AvgPool { x, .. }
                | MirOpCompat::L2Pool { x, .. } => vec![x.clone()],
                // T-66 (I-40): Spatial rearrangement ops
                MirOpCompat::DepthToSpace { x, .. }
                | MirOpCompat::SpaceToDepth { x, .. }
                | MirOpCompat::PixelShuffle { x, .. }
                | MirOpCompat::PixelUnshuffle { x, .. } => vec![x.clone()],
                // T-66 (I-40): Normalization ops
                MirOpCompat::L2Norm { x, .. } => vec![x.clone()],
                MirOpCompat::BatchNorm { x, mean, variance, gamma, beta, .. } => {
                    let mut names = vec![x.clone(), mean.clone(), variance.clone()];
                    if let Some(g) = gamma {
                        names.push(g.clone());
                    }
                    if let Some(b) = beta {
                        names.push(b.clone());
                    }
                    names
                }
                MirOpCompat::InstanceNorm { x, gamma, beta, .. } => {
                    let mut names = vec![x.clone()];
                    if let Some(g) = gamma {
                        names.push(g.clone());
                    }
                    if let Some(b) = beta {
                        names.push(b.clone());
                    }
                    names
                }
                // T-66 (I-40): Quantize / Dequantize ops
                MirOpCompat::Quantize { x, .. } | MirOpCompat::Dequantize { x, .. } => {
                    vec![x.clone()]
                }
                // F-OPS-01: ANEC ops promoted from CPU-only
                MirOpCompat::AnecFusedConvActivate { x, weight, .. } => {
                    vec![x.clone(), weight.clone()]
                }
                MirOpCompat::AnecScaledElementwise { x, y, .. } => {
                    vec![x.clone(), y.clone()]
                }
                // T-P5-10: Unsupported ops now carry their input references
                // in the `inputs` field, preventing silent weight materialization gaps.
                MirOpCompat::Unsupported { inputs, .. } => inputs.clone(),
            }
        }

        /// Remaps all input references using the provided function, keeping
        /// non-input fields unchanged.
        ///
        /// Replaces the `remap_compat_inputs()` free function in `mir_to_compat.rs`.
        /// Instead of an alias map, takes a generic closure `f` that transforms
        /// each input String field.
        pub fn remap_inputs(self, f: impl Fn(String) -> String) -> MirOpCompat {
            match self {
                MirOpCompat::Linear { name, x, weight_name, bias_name } => MirOpCompat::Linear {
                    name,
                    x: f(x),
                    weight_name: f(weight_name),
                    bias_name: bias_name.map(&f),
                },
                MirOpCompat::MatMul { name, x, y, transpose_y } => {
                    MirOpCompat::MatMul { name, x: f(x), y: f(y), transpose_y }
                }
                MirOpCompat::Add { name, x, y } => MirOpCompat::Add { name, x: f(x), y: f(y) },
                MirOpCompat::Mul { name, x, y } => MirOpCompat::Mul { name, x: f(x), y: f(y) },
                MirOpCompat::Sub { name, x, y } => MirOpCompat::Sub { name, x: f(x), y: f(y) },
                MirOpCompat::Maximum { name, x, y } => {
                    MirOpCompat::Maximum { name, x: f(x), y: f(y) }
                }
                MirOpCompat::Minimum { name, x, y } => {
                    MirOpCompat::Minimum { name, x: f(x), y: f(y) }
                }
                MirOpCompat::RealDiv { name, x, y } => {
                    MirOpCompat::RealDiv { name, x: f(x), y: f(y) }
                }
                MirOpCompat::Abs { name, x } => MirOpCompat::Abs { name, x: f(x) },
                MirOpCompat::Reshape { name, x, shape } => {
                    MirOpCompat::Reshape { name, x: f(x), shape }
                }
                MirOpCompat::Transpose { name, x, perm } => {
                    MirOpCompat::Transpose { name, x: f(x), perm }
                }
                MirOpCompat::SliceByIndex {
                    name,
                    x,
                    begin,
                    end,
                    stride,
                    begin_mask,
                    end_mask,
                    squeeze_mask,
                } => MirOpCompat::SliceByIndex {
                    name,
                    x: f(x),
                    begin,
                    end,
                    stride,
                    begin_mask,
                    end_mask,
                    squeeze_mask,
                },
                MirOpCompat::SliceUpdate { name, x, update, begin, end } => {
                    MirOpCompat::SliceUpdate { name, x: f(x), update: f(update), begin, end }
                }
                MirOpCompat::Concat { name, values, axis } => {
                    MirOpCompat::Concat { name, values: values.into_iter().map(&f).collect(), axis }
                }
                MirOpCompat::Stack { name, values, axis } => {
                    MirOpCompat::Stack { name, values: values.into_iter().map(&f).collect(), axis }
                }
                MirOpCompat::Softmax { name, x, axis } => {
                    MirOpCompat::Softmax { name, x: f(x), axis }
                }
                MirOpCompat::Gelu { name, x, mode } => MirOpCompat::Gelu { name, x: f(x), mode },
                MirOpCompat::ScaledDotProductAttention {
                    name,
                    query,
                    key,
                    value,
                    attention_mask,
                    scale,
                } => MirOpCompat::ScaledDotProductAttention {
                    name,
                    query: f(query),
                    key: f(key),
                    value: f(value),
                    attention_mask: attention_mask.map(&f),
                    scale,
                },
                MirOpCompat::CoremlUpdateState { name, state_id, value } => {
                    MirOpCompat::CoremlUpdateState { name, state_id: f(state_id), value: f(value) }
                }
                MirOpCompat::Gather { name, x, indices, axis } => {
                    MirOpCompat::Gather { name, x: f(x), indices: f(indices), axis }
                }
                MirOpCompat::ReduceMean { name, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMean { name, x: f(x), axes, keep_dims }
                }
                MirOpCompat::ReduceSum { name, x, axes, keep_dims } => {
                    MirOpCompat::ReduceSum { name, x: f(x), axes, keep_dims }
                }
                MirOpCompat::Conv {
                    name,
                    x,
                    weight,
                    pad_type,
                    groups,
                    kernel_scale,
                    kernel_zero_point,
                    kernel_palettized_lut,
                } => MirOpCompat::Conv {
                    name,
                    x: f(x),
                    weight: f(weight),
                    pad_type,
                    groups,
                    kernel_scale,
                    kernel_zero_point,
                    kernel_palettized_lut,
                },
                MirOpCompat::StateWrite { name, state_ref, value } => {
                    MirOpCompat::StateWrite { name, state_ref: f(state_ref), value: f(value) }
                }
                MirOpCompat::Rsqrt { name, x } => MirOpCompat::Rsqrt { name, x: f(x) },
                MirOpCompat::LayerNorm { name, x, weight_name, bias_name, epsilon, axes } => {
                    MirOpCompat::LayerNorm {
                        name,
                        x: f(x),
                        weight_name: f(weight_name),
                        bias_name: bias_name.map(&f),
                        epsilon,
                        axes,
                    }
                }
                MirOpCompat::Topk { name, x, k, axis } => {
                    MirOpCompat::Topk { name, x: f(x), k, axis }
                }
                MirOpCompat::Cos { name, x } => MirOpCompat::Cos { name, x: f(x) },
                MirOpCompat::Sin { name, x } => MirOpCompat::Sin { name, x: f(x) },
                MirOpCompat::Cast { name, x, dtype } => MirOpCompat::Cast { name, x: f(x), dtype },
                MirOpCompat::Split { name, x, axis, num_splits } => {
                    MirOpCompat::Split { name, x: f(x), axis, num_splits }
                }
                MirOpCompat::Exp { name, x } => MirOpCompat::Exp { name, x: f(x) },
                MirOpCompat::Sigmoid { name, x } => MirOpCompat::Sigmoid { name, x: f(x) },
                MirOpCompat::Tanh { name, x } => MirOpCompat::Tanh { name, x: f(x) },
                MirOpCompat::Relu { name, x } => MirOpCompat::Relu { name, x: f(x) },
                MirOpCompat::Where { name, condition, x, y } => {
                    MirOpCompat::Where { name, condition: f(condition), x: f(x), y: f(y) }
                }
                MirOpCompat::Silu { name, x } => MirOpCompat::Silu { name, x: f(x) },
                MirOpCompat::Identity { name, x, dtype } => {
                    MirOpCompat::Identity { name, x: f(x), dtype }
                }
                MirOpCompat::Tile { name, x, reps } => MirOpCompat::Tile { name, x: f(x), reps },
                MirOpCompat::Fill { name, shape, value, dtype } => {
                    // Fill has no tensor inputs — nothing to remap
                    MirOpCompat::Fill { name, shape, value, dtype }
                }
                MirOpCompat::FillLike { name, ref_tensor, value, dtype } => {
                    MirOpCompat::FillLike { name, ref_tensor: f(ref_tensor), value, dtype }
                }
                MirOpCompat::Neg { name, x } => MirOpCompat::Neg { name, x: f(x) },
                // Unary ops
                MirOpCompat::Sqrt { name, x } => MirOpCompat::Sqrt { name, x: f(x) },
                MirOpCompat::LogicalNot { name, x } => MirOpCompat::LogicalNot { name, x: f(x) },
                MirOpCompat::Ceil { name, x } => MirOpCompat::Ceil { name, x: f(x) },
                MirOpCompat::Floor { name, x } => MirOpCompat::Floor { name, x: f(x) },
                MirOpCompat::Round { name, x } => MirOpCompat::Round { name, x: f(x) },
                MirOpCompat::Sign { name, x } => MirOpCompat::Sign { name, x: f(x) },
                MirOpCompat::Log { name, x } => MirOpCompat::Log { name, x: f(x) },
                // Binary ops
                MirOpCompat::Pow { name, x, y } => MirOpCompat::Pow { name, x: f(x), y: f(y) },
                MirOpCompat::Equal { name, x, y } => MirOpCompat::Equal { name, x: f(x), y: f(y) },
                MirOpCompat::NotEqual { name, x, y } => {
                    MirOpCompat::NotEqual { name, x: f(x), y: f(y) }
                }
                MirOpCompat::Greater { name, x, y } => {
                    MirOpCompat::Greater { name, x: f(x), y: f(y) }
                }
                MirOpCompat::GreaterEqual { name, x, y } => {
                    MirOpCompat::GreaterEqual { name, x: f(x), y: f(y) }
                }
                MirOpCompat::Less { name, x, y } => MirOpCompat::Less { name, x: f(x), y: f(y) },
                MirOpCompat::LessEqual { name, x, y } => {
                    MirOpCompat::LessEqual { name, x: f(x), y: f(y) }
                }
                MirOpCompat::LogicalAnd { name, x, y } => {
                    MirOpCompat::LogicalAnd { name, x: f(x), y: f(y) }
                }
                MirOpCompat::LogicalOr { name, x, y } => {
                    MirOpCompat::LogicalOr { name, x: f(x), y: f(y) }
                }
                MirOpCompat::FloorDiv { name, x, y } => {
                    MirOpCompat::FloorDiv { name, x: f(x), y: f(y) }
                }
                MirOpCompat::Mod { name, x, y } => MirOpCompat::Mod { name, x: f(x), y: f(y) },
                // Axis-based ops
                MirOpCompat::ExpandDims { name, x, axis } => {
                    MirOpCompat::ExpandDims { name, x: f(x), axis }
                }
                MirOpCompat::Squeeze { name, x, axis } => {
                    MirOpCompat::Squeeze { name, x: f(x), axis }
                }
                // Clip
                MirOpCompat::Clip { name, x, min_val, max_val } => {
                    MirOpCompat::Clip { name, x: f(x), min_val, max_val }
                }
                // Pad
                MirOpCompat::Pad { name, x, pad_amounts, mode, constant_value } => {
                    MirOpCompat::Pad { name, x: f(x), pad_amounts, mode, constant_value }
                }
                // Reduce ops
                MirOpCompat::ReduceMax { name, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMax { name, x: f(x), axes, keep_dims }
                }
                MirOpCompat::ReduceMin { name, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMin { name, x: f(x), axes, keep_dims }
                }
                MirOpCompat::ReduceProd { name, x, axes, keep_dims } => {
                    MirOpCompat::ReduceProd { name, x: f(x), axes, keep_dims }
                }
                // Select
                MirOpCompat::Select { name, condition, x, y } => {
                    MirOpCompat::Select { name, condition: f(condition), x: f(x), y: f(y) }
                }
                // LeakyRelu
                MirOpCompat::LeakyRelu { name, x, alpha } => {
                    MirOpCompat::LeakyRelu { name, x: f(x), alpha }
                }
                // T-39: Constexpr weight compression ops — remap weight-name references.
                MirOpCompat::ConstexprAffineDequantize {
                    name,
                    quantized_data,
                    scale,
                    zero_point,
                    axis,
                } => MirOpCompat::ConstexprAffineDequantize {
                    name,
                    quantized_data: f(quantized_data),
                    scale,
                    zero_point,
                    axis,
                },
                MirOpCompat::ConstexprBlockwiseShiftScale {
                    name,
                    data,
                    scale,
                    offset,
                    block_size,
                } => MirOpCompat::ConstexprBlockwiseShiftScale {
                    name,
                    data: f(data),
                    scale: f(scale),
                    offset: f(offset),
                    block_size,
                },
                MirOpCompat::ConstexprLutToDense { name, indices, lut, num_bits } => {
                    MirOpCompat::ConstexprLutToDense {
                        name,
                        indices: f(indices),
                        lut: f(lut),
                        num_bits,
                    }
                }
                MirOpCompat::ConstexprSparseToDense {
                    name,
                    nonzero_data,
                    shape,
                    default_value,
                } => MirOpCompat::ConstexprSparseToDense {
                    name,
                    nonzero_data: f(nonzero_data),
                    shape,
                    default_value,
                },
                MirOpCompat::ConstexprCast { name, data, dtype } => {
                    MirOpCompat::ConstexprCast { name, data: f(data), dtype }
                }
                MirOpCompat::ConstexprLutToSparse { name, data, num_bits } => {
                    MirOpCompat::ConstexprLutToSparse { name, data: f(data), num_bits }
                }
                MirOpCompat::ConstexprSparseBlockwiseShiftScale {
                    name,
                    data,
                    scale,
                    offset,
                    block_size,
                    block_axis,
                } => MirOpCompat::ConstexprSparseBlockwiseShiftScale {
                    name,
                    data: f(data),
                    scale: f(scale),
                    offset: f(offset),
                    block_size,
                    block_axis,
                },
                // T-66 (I-40): Pooling ops
                MirOpCompat::MaxPool { name, x, kernel_sizes, strides, pad_type, pad_amounts } => {
                    MirOpCompat::MaxPool {
                        name,
                        x: f(x),
                        kernel_sizes,
                        strides,
                        pad_type,
                        pad_amounts,
                    }
                }
                MirOpCompat::AvgPool {
                    name,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                    count_include_padding,
                } => MirOpCompat::AvgPool {
                    name,
                    x: f(x),
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                    count_include_padding,
                },
                MirOpCompat::L2Pool { name, x, kernel_sizes, strides, pad_type, pad_amounts } => {
                    MirOpCompat::L2Pool {
                        name,
                        x: f(x),
                        kernel_sizes,
                        strides,
                        pad_type,
                        pad_amounts,
                    }
                }
                // T-66 (I-40): Spatial rearrangement ops
                MirOpCompat::DepthToSpace { name, x, block_size } => {
                    MirOpCompat::DepthToSpace { name, x: f(x), block_size }
                }
                MirOpCompat::SpaceToDepth { name, x, block_size } => {
                    MirOpCompat::SpaceToDepth { name, x: f(x), block_size }
                }
                MirOpCompat::PixelShuffle { name, x, upscale_factor } => {
                    MirOpCompat::PixelShuffle { name, x: f(x), upscale_factor }
                }
                MirOpCompat::PixelUnshuffle { name, x, downscale_factor } => {
                    MirOpCompat::PixelUnshuffle { name, x: f(x), downscale_factor }
                }
                // T-66 (I-40): Normalization ops
                MirOpCompat::BatchNorm { name, x, mean, variance, gamma, beta, epsilon } => {
                    MirOpCompat::BatchNorm {
                        name,
                        x: f(x),
                        mean: f(mean),
                        variance: f(variance),
                        gamma: gamma.map(&f),
                        beta: beta.map(&f),
                        epsilon,
                    }
                }
                MirOpCompat::InstanceNorm { name, x, gamma, beta, epsilon } => {
                    MirOpCompat::InstanceNorm {
                        name,
                        x: f(x),
                        gamma: gamma.map(&f),
                        beta: beta.map(&f),
                        epsilon,
                    }
                }
                MirOpCompat::L2Norm { name, x, epsilon, axes } => {
                    MirOpCompat::L2Norm { name, x: f(x), epsilon, axes }
                }
                // T-66 (I-40): Quantize / Dequantize ops
                MirOpCompat::Quantize { name, x, scale, zero_point, axis, output_dtype } => {
                    MirOpCompat::Quantize { name, x: f(x), scale, zero_point, axis, output_dtype }
                }
                MirOpCompat::Dequantize { name, x, scale, zero_point, axis, output_dtype } => {
                    MirOpCompat::Dequantize { name, x: f(x), scale, zero_point, axis, output_dtype }
                }
                // F-OPS-01: ANEC ops promoted from CPU-only
                MirOpCompat::AnecFusedConvActivate {
                    name,
                    x,
                    weight,
                    activation,
                    strides,
                    pad_amounts,
                    dilations,
                    groups,
                } => MirOpCompat::AnecFusedConvActivate {
                    name,
                    x: f(x),
                    weight: f(weight),
                    activation,
                    strides,
                    pad_amounts,
                    dilations,
                    groups,
                },
                MirOpCompat::AnecScaledElementwise { name, x, y, scale_a, scale_b } => {
                    MirOpCompat::AnecScaledElementwise { name, x: f(x), y: f(y), scale_a, scale_b }
                }
                // T-P5-10: Remap input references in Unsupported ops
                MirOpCompat::Unsupported { op_kind, name, params_json, inputs } => {
                    MirOpCompat::Unsupported {
                        op_kind,
                        name,
                        params_json,
                        inputs: inputs.into_iter().map(&f).collect(),
                    }
                }
                // Variants with no remappable inputs: pass through unchanged.
                other => other,
            }
        }

        /// Replaces the `name` field with `new_name`, keeping all other fields unchanged.
        ///
        /// Replaces the `rename_compat_output()` free function in `mir_to_compat.rs`.
        /// The match arms are copied exactly from that function.
        pub fn rename_output(self, new_name: String) -> MirOpCompat {
            match self {
                MirOpCompat::Const { name: _, data, dtype, shape } => {
                    MirOpCompat::Const { name: new_name, data, dtype, shape }
                }
                MirOpCompat::Linear { name: _, x, weight_name, bias_name } => {
                    MirOpCompat::Linear { name: new_name, x, weight_name, bias_name }
                }
                MirOpCompat::MatMul { name: _, x, y, transpose_y } => {
                    MirOpCompat::MatMul { name: new_name, x, y, transpose_y }
                }
                MirOpCompat::Add { name: _, x, y } => MirOpCompat::Add { name: new_name, x, y },
                MirOpCompat::Mul { name: _, x, y } => MirOpCompat::Mul { name: new_name, x, y },
                MirOpCompat::Sub { name: _, x, y } => MirOpCompat::Sub { name: new_name, x, y },
                MirOpCompat::Abs { name: _, x } => MirOpCompat::Abs { name: new_name, x },
                MirOpCompat::Maximum { name: _, x, y } => {
                    MirOpCompat::Maximum { name: new_name, x, y }
                }
                MirOpCompat::Minimum { name: _, x, y } => {
                    MirOpCompat::Minimum { name: new_name, x, y }
                }
                MirOpCompat::Reshape { name: _, x, shape } => {
                    MirOpCompat::Reshape { name: new_name, x, shape }
                }
                MirOpCompat::Transpose { name: _, x, perm } => {
                    MirOpCompat::Transpose { name: new_name, x, perm }
                }
                MirOpCompat::SliceByIndex {
                    name: _,
                    x,
                    begin,
                    end,
                    stride,
                    begin_mask,
                    end_mask,
                    squeeze_mask,
                } => MirOpCompat::SliceByIndex {
                    name: new_name,
                    x,
                    begin,
                    end,
                    stride,
                    begin_mask,
                    end_mask,
                    squeeze_mask,
                },
                MirOpCompat::SliceUpdate { name: _, x, update, begin, end } => {
                    MirOpCompat::SliceUpdate { name: new_name, x, update, begin, end }
                }
                MirOpCompat::Concat { name: _, values, axis } => {
                    MirOpCompat::Concat { name: new_name, values, axis }
                }
                MirOpCompat::Stack { name: _, values, axis } => {
                    MirOpCompat::Stack { name: new_name, values, axis }
                }
                MirOpCompat::Softmax { name: _, x, axis } => {
                    MirOpCompat::Softmax { name: new_name, x, axis }
                }
                MirOpCompat::Gelu { name: _, x, mode } => {
                    MirOpCompat::Gelu { name: new_name, x, mode }
                }
                MirOpCompat::ScaledDotProductAttention {
                    name: _,
                    query,
                    key,
                    value,
                    attention_mask,
                    scale,
                } => MirOpCompat::ScaledDotProductAttention {
                    name: new_name,
                    query,
                    key,
                    value,
                    attention_mask,
                    scale,
                },
                MirOpCompat::ReadState { name: _, state_id, shape, dtype } => {
                    MirOpCompat::ReadState { name: new_name, state_id, shape, dtype }
                }
                MirOpCompat::CoremlUpdateState { name: _, state_id, value } => {
                    MirOpCompat::CoremlUpdateState { name: new_name, state_id, value }
                }
                MirOpCompat::Gather { name: _, x, indices, axis } => {
                    MirOpCompat::Gather { name: new_name, x, indices, axis }
                }
                MirOpCompat::ReduceMean { name: _, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMean { name: new_name, x, axes, keep_dims }
                }
                MirOpCompat::ReduceSum { name: _, x, axes, keep_dims } => {
                    MirOpCompat::ReduceSum { name: new_name, x, axes, keep_dims }
                }
                MirOpCompat::Conv {
                    name: _,
                    x,
                    weight,
                    pad_type,
                    groups,
                    kernel_scale,
                    kernel_zero_point,
                    kernel_palettized_lut,
                } => MirOpCompat::Conv {
                    name: new_name,
                    x,
                    weight,
                    pad_type,
                    groups,
                    kernel_scale,
                    kernel_zero_point,
                    kernel_palettized_lut,
                },
                MirOpCompat::StateWrite { name: _, state_ref, value } => {
                    MirOpCompat::StateWrite { name: new_name, state_ref, value }
                }
                MirOpCompat::Rsqrt { name: _, x } => MirOpCompat::Rsqrt { name: new_name, x },
                MirOpCompat::RealDiv { name: _, x, y } => {
                    MirOpCompat::RealDiv { name: new_name, x, y }
                }
                MirOpCompat::LayerNorm { name: _, x, weight_name, bias_name, epsilon, axes } => {
                    MirOpCompat::LayerNorm {
                        name: new_name,
                        x,
                        weight_name,
                        bias_name,
                        epsilon,
                        axes,
                    }
                }
                MirOpCompat::Topk { name: _, x, k, axis } => {
                    MirOpCompat::Topk { name: new_name, x, k, axis }
                }
                MirOpCompat::Cos { name: _, x } => MirOpCompat::Cos { name: new_name, x },
                MirOpCompat::Sin { name: _, x } => MirOpCompat::Sin { name: new_name, x },
                MirOpCompat::Cast { name: _, x, dtype } => {
                    MirOpCompat::Cast { name: new_name, x, dtype }
                }
                MirOpCompat::Split { name: _, x, axis, num_splits } => {
                    MirOpCompat::Split { name: new_name, x, axis, num_splits }
                }
                MirOpCompat::Exp { name: _, x } => MirOpCompat::Exp { name: new_name, x },
                MirOpCompat::Sigmoid { name: _, x } => MirOpCompat::Sigmoid { name: new_name, x },
                MirOpCompat::Tanh { name: _, x } => MirOpCompat::Tanh { name: new_name, x },
                MirOpCompat::Relu { name: _, x } => MirOpCompat::Relu { name: new_name, x },
                MirOpCompat::Where { name: _, condition, x, y } => {
                    MirOpCompat::Where { name: new_name, condition, x, y }
                }
                MirOpCompat::Silu { name: _, x } => MirOpCompat::Silu { name: new_name, x },
                MirOpCompat::Identity { name: _, x, dtype } => {
                    MirOpCompat::Identity { name: new_name, x, dtype }
                }
                MirOpCompat::Placeholder { name: _, dtype } => {
                    MirOpCompat::Placeholder { name: new_name, dtype }
                }
                MirOpCompat::Tile { name: _, x, reps } => {
                    MirOpCompat::Tile { name: new_name, x, reps }
                }
                MirOpCompat::Fill { name: _, shape, value, dtype } => {
                    MirOpCompat::Fill { name: new_name, shape, value, dtype }
                }
                MirOpCompat::FillLike { name: _, ref_tensor, value, dtype } => {
                    MirOpCompat::FillLike { name: new_name, ref_tensor, value, dtype }
                }
                MirOpCompat::Neg { name: _, x } => MirOpCompat::Neg { name: new_name, x },
                // Unary ops
                MirOpCompat::Sqrt { name: _, x } => MirOpCompat::Sqrt { name: new_name, x },
                MirOpCompat::LogicalNot { name: _, x } => {
                    MirOpCompat::LogicalNot { name: new_name, x }
                }
                MirOpCompat::Ceil { name: _, x } => MirOpCompat::Ceil { name: new_name, x },
                MirOpCompat::Floor { name: _, x } => MirOpCompat::Floor { name: new_name, x },
                MirOpCompat::Round { name: _, x } => MirOpCompat::Round { name: new_name, x },
                MirOpCompat::Sign { name: _, x } => MirOpCompat::Sign { name: new_name, x },
                MirOpCompat::Log { name: _, x } => MirOpCompat::Log { name: new_name, x },
                // Binary ops
                MirOpCompat::Pow { name: _, x, y } => MirOpCompat::Pow { name: new_name, x, y },
                MirOpCompat::Equal { name: _, x, y } => MirOpCompat::Equal { name: new_name, x, y },
                MirOpCompat::NotEqual { name: _, x, y } => {
                    MirOpCompat::NotEqual { name: new_name, x, y }
                }
                MirOpCompat::Greater { name: _, x, y } => {
                    MirOpCompat::Greater { name: new_name, x, y }
                }
                MirOpCompat::GreaterEqual { name: _, x, y } => {
                    MirOpCompat::GreaterEqual { name: new_name, x, y }
                }
                MirOpCompat::Less { name: _, x, y } => MirOpCompat::Less { name: new_name, x, y },
                MirOpCompat::LessEqual { name: _, x, y } => {
                    MirOpCompat::LessEqual { name: new_name, x, y }
                }
                MirOpCompat::LogicalAnd { name: _, x, y } => {
                    MirOpCompat::LogicalAnd { name: new_name, x, y }
                }
                MirOpCompat::LogicalOr { name: _, x, y } => {
                    MirOpCompat::LogicalOr { name: new_name, x, y }
                }
                MirOpCompat::FloorDiv { name: _, x, y } => {
                    MirOpCompat::FloorDiv { name: new_name, x, y }
                }
                MirOpCompat::Mod { name: _, x, y } => MirOpCompat::Mod { name: new_name, x, y },
                // Axis-based ops
                MirOpCompat::ExpandDims { name: _, x, axis } => {
                    MirOpCompat::ExpandDims { name: new_name, x, axis }
                }
                MirOpCompat::Squeeze { name: _, x, axis } => {
                    MirOpCompat::Squeeze { name: new_name, x, axis }
                }
                // Clip
                MirOpCompat::Clip { name: _, x, min_val, max_val } => {
                    MirOpCompat::Clip { name: new_name, x, min_val, max_val }
                }
                // Pad
                MirOpCompat::Pad { name: _, x, pad_amounts, mode, constant_value } => {
                    MirOpCompat::Pad { name: new_name, x, pad_amounts, mode, constant_value }
                }
                // Reduce ops
                MirOpCompat::ReduceMax { name: _, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMax { name: new_name, x, axes, keep_dims }
                }
                MirOpCompat::ReduceMin { name: _, x, axes, keep_dims } => {
                    MirOpCompat::ReduceMin { name: new_name, x, axes, keep_dims }
                }
                MirOpCompat::ReduceProd { name: _, x, axes, keep_dims } => {
                    MirOpCompat::ReduceProd { name: new_name, x, axes, keep_dims }
                }
                // Select
                MirOpCompat::Select { name: _, condition, x, y } => {
                    MirOpCompat::Select { name: new_name, condition, x, y }
                }
                // LeakyRelu
                MirOpCompat::LeakyRelu { name: _, x, alpha } => {
                    MirOpCompat::LeakyRelu { name: new_name, x, alpha }
                }
                // T-39: Constexpr weight compression ops — rename output name.
                MirOpCompat::ConstexprAffineDequantize {
                    name: _,
                    quantized_data,
                    scale,
                    zero_point,
                    axis,
                } => MirOpCompat::ConstexprAffineDequantize {
                    name: new_name,
                    quantized_data,
                    scale,
                    zero_point,
                    axis,
                },
                MirOpCompat::ConstexprBlockwiseShiftScale {
                    name: _,
                    data,
                    scale,
                    offset,
                    block_size,
                } => MirOpCompat::ConstexprBlockwiseShiftScale {
                    name: new_name,
                    data,
                    scale,
                    offset,
                    block_size,
                },
                MirOpCompat::ConstexprLutToDense { name: _, indices, lut, num_bits } => {
                    MirOpCompat::ConstexprLutToDense { name: new_name, indices, lut, num_bits }
                }
                MirOpCompat::ConstexprSparseToDense {
                    name: _,
                    nonzero_data,
                    shape,
                    default_value,
                } => MirOpCompat::ConstexprSparseToDense {
                    name: new_name,
                    nonzero_data,
                    shape,
                    default_value,
                },
                MirOpCompat::ConstexprCast { name: _, data, dtype } => {
                    MirOpCompat::ConstexprCast { name: new_name, data, dtype }
                }
                MirOpCompat::ConstexprLutToSparse { name: _, data, num_bits } => {
                    MirOpCompat::ConstexprLutToSparse { name: new_name, data, num_bits }
                }
                MirOpCompat::ConstexprSparseBlockwiseShiftScale {
                    name: _,
                    data,
                    scale,
                    offset,
                    block_size,
                    block_axis,
                } => MirOpCompat::ConstexprSparseBlockwiseShiftScale {
                    name: new_name,
                    data,
                    scale,
                    offset,
                    block_size,
                    block_axis,
                },
                // T-66 (I-40): Pooling ops
                MirOpCompat::MaxPool {
                    name: _,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                } => MirOpCompat::MaxPool {
                    name: new_name,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                },
                MirOpCompat::AvgPool {
                    name: _,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                    count_include_padding,
                } => MirOpCompat::AvgPool {
                    name: new_name,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                    count_include_padding,
                },
                MirOpCompat::L2Pool {
                    name: _,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                } => MirOpCompat::L2Pool {
                    name: new_name,
                    x,
                    kernel_sizes,
                    strides,
                    pad_type,
                    pad_amounts,
                },
                // T-66 (I-40): Spatial rearrangement ops
                MirOpCompat::DepthToSpace { name: _, x, block_size } => {
                    MirOpCompat::DepthToSpace { name: new_name, x, block_size }
                }
                MirOpCompat::SpaceToDepth { name: _, x, block_size } => {
                    MirOpCompat::SpaceToDepth { name: new_name, x, block_size }
                }
                MirOpCompat::PixelShuffle { name: _, x, upscale_factor } => {
                    MirOpCompat::PixelShuffle { name: new_name, x, upscale_factor }
                }
                MirOpCompat::PixelUnshuffle { name: _, x, downscale_factor } => {
                    MirOpCompat::PixelUnshuffle { name: new_name, x, downscale_factor }
                }
                // T-66 (I-40): Normalization ops
                MirOpCompat::BatchNorm { name: _, x, mean, variance, gamma, beta, epsilon } => {
                    MirOpCompat::BatchNorm {
                        name: new_name,
                        x,
                        mean,
                        variance,
                        gamma,
                        beta,
                        epsilon,
                    }
                }
                MirOpCompat::InstanceNorm { name: _, x, gamma, beta, epsilon } => {
                    MirOpCompat::InstanceNorm { name: new_name, x, gamma, beta, epsilon }
                }
                MirOpCompat::L2Norm { name: _, x, epsilon, axes } => {
                    MirOpCompat::L2Norm { name: new_name, x, epsilon, axes }
                }
                // T-66 (I-40): Quantize / Dequantize ops
                MirOpCompat::Quantize { name: _, x, scale, zero_point, axis, output_dtype } => {
                    MirOpCompat::Quantize {
                        name: new_name,
                        x,
                        scale,
                        zero_point,
                        axis,
                        output_dtype,
                    }
                }
                MirOpCompat::Dequantize { name: _, x, scale, zero_point, axis, output_dtype } => {
                    MirOpCompat::Dequantize {
                        name: new_name,
                        x,
                        scale,
                        zero_point,
                        axis,
                        output_dtype,
                    }
                }
                // F-OPS-01: ANEC ops promoted from CPU-only
                MirOpCompat::AnecFusedConvActivate {
                    name: _,
                    x,
                    weight,
                    activation,
                    strides,
                    pad_amounts,
                    dilations,
                    groups,
                } => MirOpCompat::AnecFusedConvActivate {
                    name: new_name,
                    x,
                    weight,
                    activation,
                    strides,
                    pad_amounts,
                    dilations,
                    groups,
                },
                MirOpCompat::AnecScaledElementwise { name: _, x, y, scale_a, scale_b } => {
                    MirOpCompat::AnecScaledElementwise { name: new_name, x, y, scale_a, scale_b }
                }
                // T-P5-10: preserve inputs when renaming output
                MirOpCompat::Unsupported { op_kind, name: _, params_json, inputs } => {
                    MirOpCompat::Unsupported { op_kind, name: new_name, params_json, inputs }
                }
            }
        }
    }

    /// A named tensor descriptor for graph I/O.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TensorDescCompat {
        /// Tensor name.
        pub name: String,
        /// Tensor shape (empty if unknown).
        pub shape: Vec<usize>,
        /// Tensor data type.
        pub dtype: MilDtypeCompat,
    }

    /// A MIR graph in compatibility representation.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MirGraphCompat {
        /// Operations in topological order.
        pub ops: Vec<MirOpCompat>,
        /// Input tensor names.
        pub inputs: Vec<String>,
        /// Output tensor names.
        pub outputs: Vec<String>,
        /// Opset version (e.g., "iOS18").
        pub opset_version: String,
        /// Name of the function this graph represents.
        pub function_name: String,
        /// Input tensor descriptors (with shape and dtype).
        /// If empty, shapes/dtypes are inferred from graph ops.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub input_descs: Vec<TensorDescCompat>,
        /// Output tensor descriptors (with shape and dtype).
        /// If empty, shapes/dtypes are inferred from graph ops.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub output_descs: Vec<TensorDescCompat>,
        /// Map from tensor name to shape, built from MIR node metadata.
        /// Used by the Apple proto emitter to set correct output types on
        /// MIL operations (e.g., Linear, MatMul, Silu, etc. all need to
        /// declare their output shape/dtype in the MIL program).
        #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
        pub node_shapes: std::collections::HashMap<String, Vec<usize>>,
    }
}

// ─── From<ane_ir::mir::MirOp> for MirOpCompat (T-17: lightweight unification) ──
// Converts from the canonical MirOp type to the compat layer used for
// protobuf emission. Ops without a specialized MirOpCompat variant map
// to `Unsupported` as a transitional measure until full unification.

impl From<ane_ir::mir::MirOp> for mir_compat::MirOpCompat {
    fn from(op: ane_ir::mir::MirOp) -> mir_compat::MirOpCompat {
        use ane_ir::mir::MirOp;
        use ane_ir::toproto::ToProto;
        use mir_compat::MilDtypeCompat;

        /// Convert ane-ir MilDtype to compat MilDtypeCompat.
        /// Types without a compat representation map to Fp32 (conservative default).
        /// T-35: Int4, UInt4, E4M3, E5M2, UInt16 now have proper compat representations.
        fn convert_dtype(d: ane_ir::mir::MilDtype) -> MilDtypeCompat {
            match d {
                ane_ir::mir::MilDtype::Fp16 => MilDtypeCompat::Fp16,
                ane_ir::mir::MilDtype::Fp32 => MilDtypeCompat::Fp32,
                ane_ir::mir::MilDtype::Int32 => MilDtypeCompat::Int32,
                ane_ir::mir::MilDtype::UInt8 => MilDtypeCompat::UInt8,
                ane_ir::mir::MilDtype::Int4 => MilDtypeCompat::Int4,
                ane_ir::mir::MilDtype::UInt4 => MilDtypeCompat::UInt4,
                ane_ir::mir::MilDtype::E4M3 => MilDtypeCompat::E4M3,
                ane_ir::mir::MilDtype::E5M2 => MilDtypeCompat::E5M2,
                ane_ir::mir::MilDtype::UInt16 => MilDtypeCompat::UInt16,
                // No compat representation; Fp32 is the safest default
                ane_ir::mir::MilDtype::Bool
                | ane_ir::mir::MilDtype::Fp64
                | ane_ir::mir::MilDtype::Int8
                | ane_ir::mir::MilDtype::Int16 => MilDtypeCompat::Fp32,
            }
        }

        /// Extract the String from a MirNodeId (consumes it).
        fn nid(id: ane_ir::mir::MirNodeId) -> String {
            use ane_ir::common::IrNodeId;
            id.as_str().to_string()
        }

        // Pre-serialize the op for use in Unsupported variants (avoids partial-move issues).
        // T-P5-10: Also extract input refs *before* consuming the op in the match,
        // so Unsupported variants carry correct input names for weight materialization.
        let op_json = serde_json::to_string(&op).unwrap_or_default();
        let op_inputs = op.proto_input_refs();

        // T-P5-10: Closure that captures `op_inputs` from the enclosing scope
        // so that `input_names()` returns correct data for weight materialization.
        let unsupported = |op_kind: &str, name: &str, op_json: &str| -> mir_compat::MirOpCompat {
            mir_compat::MirOpCompat::Unsupported {
                op_kind: op_kind.to_string(),
                name: name.to_string(),
                params_json: op_json.to_string(),
                inputs: op_inputs.clone(),
            }
        };
        match op {
            // ─── Constants ───────────────────────────────────────────
            MirOp::MILConst { name, dtype, .. } => mir_compat::MirOpCompat::Const {
                name,
                data: Vec::new(), // weight data loaded separately via WeightBinBuilder
                dtype: convert_dtype(dtype),
                shape: vec![], // shape filled from MIR node metadata
            },

            // ─── Linear / FC ─────────────────────────────────────────
            MirOp::MILLinear { name, x, weight, bias } => mir_compat::MirOpCompat::Linear {
                name,
                x: nid(x),
                weight_name: weight,
                bias_name: bias,
            },
            MirOp::MILMatMul { name, x, y, transpose_y } => {
                mir_compat::MirOpCompat::MatMul { name, x: nid(x), y: nid(y), transpose_y }
            }
            MirOp::MILEinsum { name, .. } => unsupported("einsum", &name, &op_json),

            // ─── Convolution ─────────────────────────────────────────
            MirOp::MILConv { name, x, weight, pad_type, groups, .. } => {
                // T-P5-08: kernel_scale/kernel_zero_point/kernel_palettized_lut
                // are now on MirOpTargetAnnotation::ane_quant, not on MILConv.
                // This path creates MirOpCompat::Conv without quant attributes.
                // Use mir_op_to_compat_with_quant() for the full quant path.
                mir_compat::MirOpCompat::Conv {
                    name,
                    x: nid(x),
                    weight: nid(weight),
                    pad_type,
                    groups: groups as i64,
                    kernel_scale: None,
                    kernel_zero_point: None,
                    kernel_palettized_lut: None,
                }
            }
            MirOp::MILConvTranspose { name, .. } => unsupported("conv_transpose", &name, &op_json),

            // ─── Elementwise Binary ──────────────────────────────────
            MirOp::MILAdd { name, x, y } => {
                mir_compat::MirOpCompat::Add { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILMul { name, x, y } => {
                mir_compat::MirOpCompat::Mul { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILSub { name, x, y } => {
                mir_compat::MirOpCompat::Sub { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILMaximum { name, x, y } => {
                mir_compat::MirOpCompat::Maximum { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILMinimum { name, x, y } => {
                mir_compat::MirOpCompat::Minimum { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILRealDiv { name, x, y } => {
                mir_compat::MirOpCompat::RealDiv { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILFloorDiv { name, x, y } => {
                mir_compat::MirOpCompat::FloorDiv { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILMod { name, x, y } => {
                mir_compat::MirOpCompat::Mod { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILPow { name, x, y } => {
                mir_compat::MirOpCompat::Pow { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILEqual { name, x, y } => {
                mir_compat::MirOpCompat::Equal { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILNotEqual { name, x, y } => {
                mir_compat::MirOpCompat::NotEqual { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILGreater { name, x, y } => {
                mir_compat::MirOpCompat::Greater { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILGreaterEqual { name, x, y } => {
                mir_compat::MirOpCompat::GreaterEqual { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILLess { name, x, y } => {
                mir_compat::MirOpCompat::Less { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILLessEqual { name, x, y } => {
                mir_compat::MirOpCompat::LessEqual { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILLogicalAnd { name, x, y } => {
                mir_compat::MirOpCompat::LogicalAnd { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILLogicalOr { name, x, y } => {
                mir_compat::MirOpCompat::LogicalOr { name, x: nid(x), y: nid(y) }
            }
            MirOp::MILLogicalXor { name, .. } => unsupported("logical_xor", &name, &op_json),

            // ─── Elementwise Unary ───────────────────────────────────
            MirOp::MILAbs { name, x } => mir_compat::MirOpCompat::Abs { name, x: nid(x) },
            MirOp::MILNeg { name, x } => mir_compat::MirOpCompat::Neg { name, x: nid(x) },
            MirOp::MILSigmoid { name, x } => mir_compat::MirOpCompat::Sigmoid { name, x: nid(x) },
            MirOp::MILTanh { name, x } => mir_compat::MirOpCompat::Tanh { name, x: nid(x) },
            MirOp::MILRelu { name, x } => mir_compat::MirOpCompat::Relu { name, x: nid(x) },
            MirOp::MILRelu6 { name, .. } => unsupported("relu6", &name, &op_json),
            MirOp::MILLeakyRelu { name, x, alpha } => {
                mir_compat::MirOpCompat::LeakyRelu { name, x: nid(x), alpha }
            }
            MirOp::MILSigmoidHard { name, .. } => unsupported("sigmoid_hard", &name, &op_json),
            MirOp::MILThresholdedRelu { name, .. } => {
                unsupported("thresholded_relu", &name, &op_json)
            }
            MirOp::MILClampedRelu { name, .. } => unsupported("clamped_relu", &name, &op_json),
            MirOp::MILLinearActivation { name, .. } => {
                unsupported("linear_activation", &name, &op_json)
            }
            MirOp::MILPrelu { name, .. } => unsupported("prelu", &name, &op_json),
            MirOp::MILSoftsign { name, .. } => unsupported("softsign", &name, &op_json),
            MirOp::MILSilu { name, x } => mir_compat::MirOpCompat::Silu { name, x: nid(x) },
            MirOp::MILScaledTanh { name, .. } => unsupported("scaled_tanh", &name, &op_json),
            MirOp::MILElu { name, .. } => unsupported("elu", &name, &op_json),
            MirOp::MILSoftplus { name, .. } => unsupported("softplus", &name, &op_json),
            MirOp::MILSoftplusParametric { name, .. } => {
                unsupported("softplus_parametric", &name, &op_json)
            }
            MirOp::MILGelu { name, x, mode } => {
                mir_compat::MirOpCompat::Gelu { name, x: nid(x), mode }
            }
            MirOp::MILClip { name, x, min_val, max_val } => {
                mir_compat::MirOpCompat::Clip { name, x: nid(x), min_val, max_val }
            }
            MirOp::MILSquare { name, .. } => unsupported("square", &name, &op_json),
            MirOp::MILThreshold { name, .. } => unsupported("threshold", &name, &op_json),
            MirOp::MILSqrt { name, x } => mir_compat::MirOpCompat::Sqrt { name, x: nid(x) },
            MirOp::MILRsqrt { name, x } => mir_compat::MirOpCompat::Rsqrt { name, x: nid(x) },
            MirOp::MILInverse { name, .. } => unsupported("inverse", &name, &op_json),
            MirOp::MILCeil { name, x } => mir_compat::MirOpCompat::Ceil { name, x: nid(x) },
            MirOp::MILFloor { name, x } => mir_compat::MirOpCompat::Floor { name, x: nid(x) },
            MirOp::MILRound { name, x } => mir_compat::MirOpCompat::Round { name, x: nid(x) },
            MirOp::MILExp { name, x } => mir_compat::MirOpCompat::Exp { name, x: nid(x) },
            MirOp::MILExp2 { name, .. } => unsupported("exp2", &name, &op_json),
            MirOp::MILLog { name, x, .. } => mir_compat::MirOpCompat::Log { name, x: nid(x) },
            MirOp::MILSign { name, x } => mir_compat::MirOpCompat::Sign { name, x: nid(x) },
            MirOp::MILCos { name, x } => mir_compat::MirOpCompat::Cos { name, x: nid(x) },
            MirOp::MILSin { name, x } => mir_compat::MirOpCompat::Sin { name, x: nid(x) },
            MirOp::MILTan { name, .. } => unsupported("tan", &name, &op_json),
            MirOp::MILAcos { name, .. } => unsupported("acos", &name, &op_json),
            MirOp::MILAsin { name, .. } => unsupported("asin", &name, &op_json),
            MirOp::MILAtan { name, .. } => unsupported("atan", &name, &op_json),
            MirOp::MILCosh { name, .. } => unsupported("cosh", &name, &op_json),
            MirOp::MILSinh { name, .. } => unsupported("sinh", &name, &op_json),
            MirOp::MILAtanh { name, .. } => unsupported("atanh", &name, &op_json),
            MirOp::MILErf { name, .. } => unsupported("erf", &name, &op_json),
            MirOp::MILLogicalNot { name, x } => {
                mir_compat::MirOpCompat::LogicalNot { name, x: nid(x) }
            }
            MirOp::MILCast { name, x, dtype } => {
                mir_compat::MirOpCompat::Cast { name, x: nid(x), dtype: convert_dtype(dtype) }
            }
            MirOp::MILSelect { name, condition, x, y } => mir_compat::MirOpCompat::Select {
                name,
                condition: nid(condition),
                x: nid(x),
                y: nid(y),
            },
            MirOp::MILWhere { name, condition, x, y } => mir_compat::MirOpCompat::Where {
                name,
                condition: nid(condition),
                x: nid(x),
                y: nid(y),
            },
            MirOp::MILSoftmax { name, x, axis } => {
                mir_compat::MirOpCompat::Softmax { name, x: nid(x), axis: axis as i64 }
            }

            // ─── Reduction ───────────────────────────────────────────
            MirOp::MILReduceSum { name, x, axes, keep_dims } => {
                mir_compat::MirOpCompat::ReduceSum {
                    name,
                    x: nid(x),
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                    keep_dims,
                }
            }
            MirOp::MILReduceMean { name, x, axes, keep_dims } => {
                mir_compat::MirOpCompat::ReduceMean {
                    name,
                    x: nid(x),
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                    keep_dims,
                }
            }
            MirOp::MILReduceMax { name, x, axes, keep_dims } => {
                mir_compat::MirOpCompat::ReduceMax {
                    name,
                    x: nid(x),
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                    keep_dims,
                }
            }
            MirOp::MILReduceMin { name, x, axes, keep_dims } => {
                mir_compat::MirOpCompat::ReduceMin {
                    name,
                    x: nid(x),
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                    keep_dims,
                }
            }
            MirOp::MILReduceProd { name, x, axes, keep_dims } => {
                mir_compat::MirOpCompat::ReduceProd {
                    name,
                    x: nid(x),
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                    keep_dims,
                }
            }
            MirOp::MILReduceSumSquare { name, .. } => {
                unsupported("reduce_sum_square", &name, &op_json)
            }
            MirOp::MILReduceL2Norm { name, .. } => unsupported("reduce_l2_norm", &name, &op_json),
            MirOp::MILReduceL1Norm { name, .. } => unsupported("reduce_l1_norm", &name, &op_json),
            MirOp::MILReduceLogSumExp { name, .. } => {
                unsupported("reduce_log_sum_exp", &name, &op_json)
            }
            MirOp::MILReduceLogSum { name, .. } => unsupported("reduce_log_sum", &name, &op_json),
            MirOp::MILReduceArgmax { name, .. } => unsupported("reduce_argmax", &name, &op_json),
            MirOp::MILReduceArgmin { name, .. } => unsupported("reduce_argmin", &name, &op_json),

            // ─── Normalization ───────────────────────────────────────
            MirOp::MILBatchNorm { name, x, mean, variance, gamma, beta, epsilon } => {
                mir_compat::MirOpCompat::BatchNorm {
                    name,
                    x: nid(x),
                    mean,
                    variance,
                    gamma,
                    beta,
                    epsilon,
                }
            }
            MirOp::MILInstanceNorm { name, x, gamma, beta, epsilon } => {
                mir_compat::MirOpCompat::InstanceNorm { name, x: nid(x), gamma, beta, epsilon }
            }
            MirOp::MILLayerNorm { name, x, weight, bias, epsilon, axes } => {
                mir_compat::MirOpCompat::LayerNorm {
                    name,
                    x: nid(x),
                    weight_name: weight,
                    bias_name: bias,
                    epsilon,
                    axes: axes.into_iter().map(|a| a as i64).collect(),
                }
            }
            MirOp::MILL2Norm { name, x, epsilon, axes } => mir_compat::MirOpCompat::L2Norm {
                name,
                x: nid(x),
                epsilon,
                axes: axes.into_iter().map(|a| a as i64).collect(),
            },
            MirOp::MILLocalResponseNorm { name, .. } => {
                unsupported("local_response_norm", &name, &op_json)
            }

            // ─── Pooling ─────────────────────────────────────────────
            MirOp::MILMaxPool { name, x, kernel_sizes, strides, pad_types, pad_amounts } => {
                mir_compat::MirOpCompat::MaxPool {
                    name,
                    x: nid(x),
                    kernel_sizes: kernel_sizes.into_iter().map(|d| d as i32).collect(),
                    strides: strides.into_iter().map(|d| d as i32).collect(),
                    pad_type: pad_types.first().cloned().unwrap_or_else(|| "valid".to_string()),
                    pad_amounts: pad_amounts.into_iter().map(|d| d as i32).collect(),
                }
            }
            MirOp::MILAvgPool {
                name,
                x,
                kernel_sizes,
                strides,
                pad_types,
                pad_amounts,
                count_include_padding,
            } => mir_compat::MirOpCompat::AvgPool {
                name,
                x: nid(x),
                kernel_sizes: kernel_sizes.into_iter().map(|d| d as i32).collect(),
                strides: strides.into_iter().map(|d| d as i32).collect(),
                pad_type: pad_types.first().cloned().unwrap_or_else(|| "valid".to_string()),
                pad_amounts: pad_amounts.into_iter().map(|d| d as i32).collect(),
                count_include_padding,
            },
            MirOp::MILL2Pool { name, x, kernel_sizes, strides, pad_types, pad_amounts } => {
                mir_compat::MirOpCompat::L2Pool {
                    name,
                    x: nid(x),
                    kernel_sizes: kernel_sizes.into_iter().map(|d| d as i32).collect(),
                    strides: strides.into_iter().map(|d| d as i32).collect(),
                    pad_type: pad_types.first().cloned().unwrap_or_else(|| "valid".to_string()),
                    pad_amounts: pad_amounts.into_iter().map(|d| d as i32).collect(),
                }
            }

            // ─── Image Resizing ──────────────────────────────────────
            MirOp::MILResize { name, .. } => unsupported("resize", &name, &op_json),
            MirOp::MILResizeNearestNeighbor { name, .. } => {
                unsupported("resize_nearest_neighbor", &name, &op_json)
            }
            MirOp::MILResizeBilinear { name, .. } => {
                unsupported("resize_bilinear", &name, &op_json)
            }
            MirOp::MILUpsampleNearestNeighbor { name, .. } => {
                unsupported("upsample_nearest_neighbor", &name, &op_json)
            }
            MirOp::MILUpsampleBilinear { name, .. } => {
                unsupported("upsample_bilinear", &name, &op_json)
            }
            MirOp::MILCropResize { name, .. } => unsupported("crop_resize", &name, &op_json),
            MirOp::MILAffine { name, .. } => unsupported("affine", &name, &op_json),
            MirOp::MILResample { name, .. } => unsupported("resample", &name, &op_json),

            // ─── Tensor Transform ────────────────────────────────────
            MirOp::MILReshape { name, x, shape } => mir_compat::MirOpCompat::Reshape {
                name,
                x: nid(x),
                shape: shape.into_iter().map(|d| d as i32).collect(),
            },
            MirOp::MILReshapeLike { name, .. } => unsupported("reshape_like", &name, &op_json),
            MirOp::MILTranspose { name, x, perm } => mir_compat::MirOpCompat::Transpose {
                name,
                x: nid(x),
                perm: perm.into_iter().map(|d| d as i32).collect(),
            },
            MirOp::MILSplit { name, x, axis, num_splits } => mir_compat::MirOpCompat::Split {
                name,
                x: nid(x),
                axis: axis as i64,
                num_splits: num_splits as i64,
            },
            MirOp::MILConcat { name, values, axis } => mir_compat::MirOpCompat::Concat {
                name,
                values: values.into_iter().map(nid).collect(),
                axis: axis as i64,
            },
            MirOp::MILExpandDims { name, x, axis } => mir_compat::MirOpCompat::ExpandDims {
                name,
                x: nid(x),
                axis: axis.into_iter().map(|a| a as i32).collect(),
            },
            MirOp::MILSqueeze { name, x, axis } => mir_compat::MirOpCompat::Squeeze {
                name,
                x: nid(x),
                axis: axis.into_iter().map(|a| a as i32).collect(),
            },
            MirOp::MILFlatten2d { name, .. } => unsupported("flatten2d", &name, &op_json),
            MirOp::MILReverse { name, .. } => unsupported("reverse", &name, &op_json),
            MirOp::MILReverseSequence { name, .. } => {
                unsupported("reverse_sequence", &name, &op_json)
            }
            MirOp::MILSliceByIndex {
                name,
                x,
                begin,
                end,
                stride,
                begin_mask,
                end_mask,
                squeeze_mask,
            } => mir_compat::MirOpCompat::SliceByIndex {
                name,
                x: nid(x),
                begin: begin.into_iter().map(|v| v as i32).collect(),
                end: end.into_iter().map(|v| v as i32).collect(),
                stride: stride.into_iter().map(|v| v as i32).collect(),
                begin_mask: begin_mask.clone(),
                end_mask: end_mask.clone(),
                squeeze_mask: squeeze_mask.clone(),
            },
            MirOp::MILSliceBySize { name, .. } => unsupported("slice_by_size", &name, &op_json),
            MirOp::MILSliceUpdate { name, x, update, begin, end } => {
                mir_compat::MirOpCompat::SliceUpdate {
                    name,
                    x: nid(x),
                    update: nid(update),
                    begin: begin.into_iter().map(|v| v as i32).collect(),
                    end: end.into_iter().map(|v| v as i32).collect(),
                }
            }
            MirOp::MILSlidingWindows { name, .. } => {
                unsupported("sliding_windows", &name, &op_json)
            }
            MirOp::MILDepthToSpace { name, x, block_size } => {
                mir_compat::MirOpCompat::DepthToSpace {
                    name,
                    x: nid(x),
                    block_size: block_size as i64,
                }
            }
            MirOp::MILSpaceToDepth { name, x, block_size } => {
                mir_compat::MirOpCompat::SpaceToDepth {
                    name,
                    x: nid(x),
                    block_size: block_size as i64,
                }
            }
            MirOp::MILPixelShuffle { name, x, upscale_factor } => {
                mir_compat::MirOpCompat::PixelShuffle {
                    name,
                    x: nid(x),
                    upscale_factor: upscale_factor as i64,
                }
            }
            MirOp::MILPixelUnshuffle { name, x, downscale_factor } => {
                mir_compat::MirOpCompat::PixelUnshuffle {
                    name,
                    x: nid(x),
                    downscale_factor: downscale_factor as i64,
                }
            }
            MirOp::MILBatchToSpace { name, .. } => unsupported("batch_to_space", &name, &op_json),
            MirOp::MILSpaceToBatch { name, .. } => unsupported("space_to_batch", &name, &op_json),
            MirOp::MILPad { name, x, pad_amounts, mode, constant_value } => {
                mir_compat::MirOpCompat::Pad {
                    name,
                    x: nid(x),
                    pad_amounts: pad_amounts.into_iter().map(|v| v as i32).collect(),
                    mode,
                    constant_value,
                }
            }
            MirOp::MILStack { name, values, axis } => mir_compat::MirOpCompat::Stack {
                name,
                values: values.into_iter().map(|v| nid(v)).collect(),
                axis: *axis as i64,
            },
            MirOp::MILTile { name, x, reps } => mir_compat::MirOpCompat::Tile {
                name,
                x: nid(x),
                reps: reps.into_iter().map(|r| r as i32).collect(),
            },
            MirOp::MILCumsum { name, .. } => unsupported("cumsum", &name, &op_json),
            MirOp::MILFill { name, shape, value, dtype } => mir_compat::MirOpCompat::Fill {
                name,
                shape: shape.into_iter().map(|d| d as i32).collect(),
                value,
                dtype: convert_dtype(dtype),
            },
            MirOp::MILFillLike { name, ref_tensor, value, dtype } => {
                mir_compat::MirOpCompat::FillLike {
                    name,
                    ref_tensor: nid(ref_tensor),
                    value,
                    dtype: convert_dtype(dtype),
                }
            }
            MirOp::MILIdentity { name, x } => {
                mir_compat::MirOpCompat::Identity { name, x: nid(x), dtype: MilDtypeCompat::Fp16 }
            }
            MirOp::MILOneHot { name, .. } => unsupported("one_hot", &name, &op_json),
            MirOp::MILNonZero { name, .. } => unsupported("non_zero", &name, &op_json),
            MirOp::MILArgsort { name, .. } => unsupported("argsort", &name, &op_json),
            MirOp::MILBandPart { name, .. } => unsupported("band_part", &name, &op_json),
            MirOp::MILRange1d { name, .. } => unsupported("range1d", &name, &op_json),
            MirOp::MILShape { name, .. } => unsupported("shape", &name, &op_json),
            MirOp::MILCrop { name, .. } => unsupported("crop", &name, &op_json),

            // ─── Scatter / Gather ────────────────────────────────────
            MirOp::MILGather { name, x, indices, axis } => mir_compat::MirOpCompat::Gather {
                name,
                x: nid(x),
                indices: nid(indices),
                axis: axis as i64,
            },
            MirOp::MILGatherAlongAxis { name, .. } => {
                unsupported("gather_along_axis", &name, &op_json)
            }
            MirOp::MILGatherNd { name, .. } => unsupported("gather_nd", &name, &op_json),
            MirOp::MILScatter { name, .. } => unsupported("scatter", &name, &op_json),
            MirOp::MILScatterAlongAxis { name, .. } => {
                unsupported("scatter_along_axis", &name, &op_json)
            }
            MirOp::MILScatterNd { name, .. } => unsupported("scatter_nd", &name, &op_json),
            MirOp::MILNonMaximumSuppression { name, .. } => {
                unsupported("non_maximum_suppression", &name, &op_json)
            }

            // ─── Attention ───────────────────────────────────────────
            MirOp::MILScaledDotProductAttention {
                name,
                query,
                key,
                value,
                attention_mask,
                scale,
            } => mir_compat::MirOpCompat::ScaledDotProductAttention {
                name,
                query: nid(query),
                key: nid(key),
                value: nid(value),
                attention_mask: attention_mask.map(nid),
                scale,
            },

            // ─── Quantization ────────────────────────────────────────
            MirOp::MILQuantize { name, x, scale, zero_point, axis, output_dtype } => {
                mir_compat::MirOpCompat::Quantize {
                    name,
                    x: nid(x),
                    scale,
                    zero_point,
                    axis: axis as i64,
                    output_dtype: convert_dtype(output_dtype),
                }
            }
            MirOp::MILDequantize { name, x, scale, zero_point, axis, output_dtype } => {
                mir_compat::MirOpCompat::Dequantize {
                    name,
                    x: nid(x),
                    scale,
                    zero_point,
                    axis: axis as i64,
                    output_dtype: convert_dtype(output_dtype),
                }
            }

            // ─── Constexpr / Compression ─────────────────────────────
            // T-39 (I-18): Constexpr* variants now have proper MirOpCompat
            // representations instead of being mapped to Unsupported.
            MirOp::MILConstexprAffineDequantize {
                name,
                quantized_data,
                scale,
                zero_point,
                axis,
            } => mir_compat::MirOpCompat::ConstexprAffineDequantize {
                name,
                quantized_data,
                scale,
                zero_point,
                axis: axis as i64,
            },
            MirOp::MILConstexprBlockwiseShiftScale { name, data, scale, offset, block_size } => {
                mir_compat::MirOpCompat::ConstexprBlockwiseShiftScale {
                    name,
                    data,
                    scale,
                    offset,
                    block_size: block_size.into_iter().map(|d| d as i64).collect(),
                }
            }
            MirOp::MILConstexprLutToDense { name, indices, lut, num_bits } => {
                mir_compat::MirOpCompat::ConstexprLutToDense {
                    name,
                    indices,
                    lut,
                    num_bits: num_bits as i64,
                }
            }
            MirOp::MILConstexprSparseToDense { name, nonzero_data, shape, default_value } => {
                mir_compat::MirOpCompat::ConstexprSparseToDense {
                    name,
                    nonzero_data,
                    shape: shape.into_iter().map(|d| d as i64).collect(),
                    default_value,
                }
            }
            MirOp::MILConstexprCast { name, data, dtype } => {
                mir_compat::MirOpCompat::ConstexprCast { name, data, dtype: convert_dtype(dtype) }
            }
            MirOp::MILConstexprLutToSparse { name, data, num_bits } => {
                mir_compat::MirOpCompat::ConstexprLutToSparse {
                    name,
                    data,
                    num_bits: num_bits as i64,
                }
            }
            MirOp::MILConstexprSparseBlockwiseShiftScale {
                name,
                data,
                scale,
                offset,
                block_size,
                block_axis,
            } => mir_compat::MirOpCompat::ConstexprSparseBlockwiseShiftScale {
                name,
                data,
                scale,
                offset,
                block_size: block_size.into_iter().map(|d| d as i64).collect(),
                block_axis: block_axis as i64,
            },

            // ─── Recurrent ───────────────────────────────────────────
            MirOp::MILRnn { name, .. } => unsupported("rnn", &name, &op_json),
            MirOp::MILGru { name, .. } => unsupported("gru", &name, &op_json),
            MirOp::MILLstm { name, .. } => unsupported("lstm", &name, &op_json),

            // ─── Control Flow ────────────────────────────────────────
            MirOp::MILCond { name, .. } => unsupported("cond", &name, &op_json),
            MirOp::MILWhileLoop { name, .. } => unsupported("while_loop", &name, &op_json),
            MirOp::MILMakeList { name, .. } => unsupported("make_list", &name, &op_json),
            MirOp::MILListLength { name, .. } => unsupported("list_length", &name, &op_json),
            MirOp::MILListWrite { name, .. } => unsupported("list_write", &name, &op_json),
            MirOp::MILListRead { name, .. } => unsupported("list_read", &name, &op_json),
            MirOp::MILListGather { name, .. } => unsupported("list_gather", &name, &op_json),
            MirOp::MILListScatter { name, .. } => unsupported("list_scatter", &name, &op_json),

            // ─── Random ──────────────────────────────────────────────
            MirOp::MILRandomBernoulli { name, .. } => {
                unsupported("random_bernoulli", &name, &op_json)
            }
            MirOp::MILRandomNormal { name, .. } => unsupported("random_normal", &name, &op_json),
            MirOp::MILRandomUniform { name, .. } => unsupported("random_uniform", &name, &op_json),
            MirOp::MILRandomCategorical { name, .. } => {
                unsupported("random_categorical", &name, &op_json)
            }

            // ─── State ───────────────────────────────────────────────
            MirOp::MILReadState { name, state_id, shape, dtype } => {
                mir_compat::MirOpCompat::ReadState {
                    name,
                    state_id,
                    shape,
                    dtype: convert_dtype(dtype),
                }
            }
            MirOp::MILCoremlUpdateState { name, state_id, value } => {
                mir_compat::MirOpCompat::CoremlUpdateState { name, state_id, value: nid(value) }
            }
            MirOp::MILStateWrite { name, state_ref, value } => {
                mir_compat::MirOpCompat::StateWrite { name, state_ref, value: nid(value) }
            }

            // ─── Metadata / Misc ─────────────────────────────────────
            MirOp::MILTopk { name, x, k, axis } => {
                mir_compat::MirOpCompat::Topk { name, x: nid(x), k: k as i64, axis: axis as i64 }
            }
            MirOp::MILClassify { name, .. } => unsupported("classify", &name, &op_json),
            // ─── ANEC Internal Ops (T-P4-08) ──────────────────────────────
            // These are ANE-internal operations that do not map to MIL builder
            // calls. They are emitted as no-ops in proto emission because the
            // ANEC binary handles them internally during model compilation.
            MirOp::AnecFusedConvActivate { name, .. } => {
                unsupported("anec_fused_conv_activate", &name, &op_json)
            }
            MirOp::AnecFusedLinearActivate { name, .. } => {
                unsupported("anec_fused_linear_activate", &name, &op_json)
            }
            // Future MirOp variants added via #[non_exhaustive]
            _ => unsupported("unknown_mir_op", "", &op_json),
        }
    }
}

/// A Core ML function definition with its operations and I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlFunction {
    /// Function name (e.g., "main", "embedding", "decode_step").
    pub name: String,
    /// Input tensor descriptions.
    pub inputs: Vec<TensorDesc>,
    /// Output tensor descriptions.
    pub outputs: Vec<TensorDesc>,
    /// State tensor descriptions (for stateful models).
    pub states: Vec<TensorDesc>,
    /// Operations in this function.
    pub operations: Vec<mir_compat::MirOpCompat>,
    /// Map from tensor name to shape, used for emitting correct output types
    /// in the MIL program. Without this, all op outputs default to scalar fp16.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub node_shapes: std::collections::HashMap<String, Vec<usize>>,
}

/// Tensor description for function I/O.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDesc {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// Tensor data type.
    pub dtype: CoreMlDataType,
    /// Whether this is a state tensor.
    pub is_state: bool,
}

/// A complete Core ML model ready for serialization to mlpackage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlModel {
    /// Specification version.
    pub spec_version: SpecVersion,
    /// Model description.
    pub description: ModelDescriptionCompat,
    /// Named functions.
    pub functions: Vec<CoreMlFunction>,
    /// Default function name.
    pub default_function_name: String,
    /// Weight entries (for weight.bin construction).
    pub weights: Vec<WeightEntry>,
    /// Shared weight references (for cross-function weight sharing).
    pub shared_weights: Vec<SharedWeightRef>,
    /// Compute unit hint.
    pub compute_unit: CoreMlComputeUnit,
    /// User-defined metadata.
    pub user_defined_metadata: HashMap<String, String>,
}

/// Model description (compatibility representation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptionCompat {
    /// Input features.
    pub inputs: Vec<TensorDesc>,
    /// Output features.
    pub outputs: Vec<TensorDesc>,
    /// State features (iOS 18+).
    pub states: Vec<TensorDesc>,
}

// ─── Conversion Functions: hand-written compat → prost-generated proto types ──
// Legacy proto module usage is allowed within this crate during the deprecation period.
// External consumers should use apple_proto instead.
// These functions bridge between the hand-written domain types (used throughout
// the compiler) and the prost-generated protobuf message types (used for
// serialization). This is the critical bridge layer for real proto emission.

/// Convert `CoreMlDataType` → `proto::ArrayDataType` (prost enum as i32).
pub fn data_type_to_proto(dt: &CoreMlDataType) -> i32 {
    match dt {
        CoreMlDataType::Unknown => proto::ArrayDataType::Unknown as i32,
        CoreMlDataType::Float32 => proto::ArrayDataType::Float32 as i32,
        CoreMlDataType::Float16 => proto::ArrayDataType::Float16 as i32,
        CoreMlDataType::Float64 => proto::ArrayDataType::Double as i32,
        CoreMlDataType::Int32 => proto::ArrayDataType::Int32 as i32,
        CoreMlDataType::UInt8 => proto::ArrayDataType::Uint8 as i32,
        CoreMlDataType::Int8 => proto::ArrayDataType::Int8 as i32,
        CoreMlDataType::Bool => proto::ArrayDataType::Bool as i32,
        // T-35: new dtypes — legacy proto ArrayDataType doesn't have these,
        // map to Unknown since they should use the MIL proto DataType path
        CoreMlDataType::Int4
        | CoreMlDataType::UInt4
        | CoreMlDataType::E4M3
        | CoreMlDataType::E5M2
        | CoreMlDataType::UInt16 => proto::ArrayDataType::Unknown as i32,
    }
}

/// Convert `CoreMlComputeUnit` → `proto::ComputeUnit` (prost enum as i32).
pub fn compute_unit_to_proto(cu: &CoreMlComputeUnit) -> i32 {
    match cu {
        CoreMlComputeUnit::Unknown => proto::ComputeUnit::Unknown as i32,
        CoreMlComputeUnit::CpuOnly => proto::ComputeUnit::CpuOnly as i32,
        CoreMlComputeUnit::CpuAndGpu => proto::ComputeUnit::CpuAndGpu as i32,
        CoreMlComputeUnit::CpuAndNe => proto::ComputeUnit::CpuAndNe as i32,
        CoreMlComputeUnit::All => proto::ComputeUnit::All as i32,
    }
}

/// Convert `SpecVersion` → `proto::SpecificationVersion` (prost enum as i32).
pub fn spec_version_to_proto(sv: &SpecVersion) -> i32 {
    match sv {
        SpecVersion::V7 => proto::SpecificationVersion::SpecificationVersion7 as i32,
        SpecVersion::V8 => proto::SpecificationVersion::SpecificationVersion8 as i32,
        SpecVersion::V9 => 9, // SpecificationVersion9 not in legacy proto enum; use raw value
        SpecVersion::V10 => 10, // SpecificationVersion10 not in legacy proto enum; use raw value
    }
}

/// Convert `MilDtypeCompat` → `proto::ArrayDataType` (prost enum as i32).
///
/// T-35: New dtypes (Int4, UInt4, E4M3, E5M2, UInt16) use raw proto values
/// since they are not in the legacy `ArrayDataType` enum. These correspond
/// to Apple's MIL proto `DataType` enum values.
pub fn mil_dtype_to_proto(dt: &mir_compat::MilDtypeCompat) -> i32 {
    match dt {
        mir_compat::MilDtypeCompat::Fp16 => proto::ArrayDataType::Float16 as i32,
        mir_compat::MilDtypeCompat::Fp32 => proto::ArrayDataType::Float32 as i32,
        mir_compat::MilDtypeCompat::Int32 => proto::ArrayDataType::Int32 as i32,
        mir_compat::MilDtypeCompat::UInt8 => proto::ArrayDataType::Uint8 as i32,
        // T-35: new dtypes — use raw MIL proto DataType values
        mir_compat::MilDtypeCompat::Int4 => 25, // DataType::INT4
        mir_compat::MilDtypeCompat::UInt4 => 35, // DataType::UINT4
        mir_compat::MilDtypeCompat::E4M3 => 40, // DataType::FLOAT8E4M3FN
        mir_compat::MilDtypeCompat::E5M2 => 41, // DataType::FLOAT8E5M2
        mir_compat::MilDtypeCompat::UInt16 => 32, // DataType::UINT16
    }
}

/// Convert a shape `Vec<u64>` → `proto::ArrayShape` (all fixed dimensions).
pub fn shape_to_proto(shape: &[u64]) -> proto::ArrayShape {
    proto::ArrayShape {
        dimensions: shape
            .iter()
            .map(|&d| proto::DimensionValue {
                dimension_value: Some(proto::dimension_value::DimensionValue::Fixed(d as i64)),
            })
            .collect(),
    }
}

/// Convert `TensorDesc` → `proto::FeatureDescription`.
pub fn tensor_desc_to_proto(td: &TensorDesc) -> proto::FeatureDescription {
    let tensor_desc = proto::TensorFeatureDescription {
        shape: Some(shape_to_proto(&td.shape)),
        data_type: data_type_to_proto(&td.dtype),
        short_description: String::new(),
    };

    let feature_type = if td.is_state {
        proto::feature_description::FeatureType::State(proto::StateFeatureDescription {
            wrapped_types: vec![tensor_desc],
        })
    } else {
        proto::feature_description::FeatureType::Tensor(tensor_desc)
    };

    proto::FeatureDescription {
        name: td.name.clone(),
        short_description: String::new(),
        feature_type: Some(feature_type),
    }
}

/// Convert `WeightEntry` → `proto::WeightData` using `FileReference`
/// (weight data goes into weight.bin; protobuf only references by offset/size).
pub fn weight_entry_to_proto(entry: &WeightEntry) -> proto::WeightData {
    proto::WeightData {
        weight_data: Some(proto::weight_data::WeightData::FileRef(proto::FileReference {
            offset: entry.offset as i64,
            size: entry.size as i64,
        })),
    }
}

/// Build a `proto::WeightData` with inline value from raw const data.
/// Used for const ops where the weight data is embedded directly in the proto.
pub fn weight_data_inline(
    data: &[u8],
    dtype: &mir_compat::MilDtypeCompat,
    shape: &[usize],
) -> proto::WeightData {
    let proto_dtype = mil_dtype_to_proto(dtype);
    let value = match dtype {
        mir_compat::MilDtypeCompat::Fp16
        | mir_compat::MilDtypeCompat::Fp32
        | mir_compat::MilDtypeCompat::UInt8
        | mir_compat::MilDtypeCompat::E4M3
        | mir_compat::MilDtypeCompat::E5M2 => proto::weight_value::Value::FloatValue(data.to_vec()),
        mir_compat::MilDtypeCompat::Int32 => proto::weight_value::Value::IntValue(data.to_vec()),
        // T-35: Int4, UInt4, UInt16 stored as integer data
        mir_compat::MilDtypeCompat::Int4
        | mir_compat::MilDtypeCompat::UInt4
        | mir_compat::MilDtypeCompat::UInt16 => proto::weight_value::Value::IntValue(data.to_vec()),
    };

    proto::WeightData {
        weight_data: Some(proto::weight_data::WeightData::Value(proto::WeightValue {
            value: Some(value),
            data_type: proto_dtype,
            shape: Some(shape_to_proto(&shape.iter().map(|&d| d as u64).collect::<Vec<_>>())),
        })),
    }
}

/// Convert `MirOpCompat` → `proto::MilOperation`.
///
/// This is the core conversion mapping each MIR op variant to its
/// corresponding protobuf MilOperation message. The `name` field in
/// MirOpCompat becomes the output SSA name in the proto MilOperation.
///
/// For `Const` ops, the weight data uses `FileReference` (referencing
/// weight.bin by offset) when a matching `WeightEntry` is found in
/// `weight_entries`, otherwise falls back to inline data.
pub fn mir_op_to_proto_op(
    op: &mir_compat::MirOpCompat,
    weight_entries: &[WeightEntry],
) -> proto::MilOperation {
    let (name, operation) = match op {
        mir_compat::MirOpCompat::Const { name, data, dtype, shape } => {
            // Look up the weight entry by name for FileReference; otherwise inline
            let weight_data = if let Some(entry) = weight_entries.iter().find(|w| w.name == *name) {
                weight_entry_to_proto(entry)
            } else {
                weight_data_inline(data, dtype, shape)
            };
            (
                name.clone(),
                proto::mil_operation::Operation::ConstOp(proto::MilConstOp {
                    value: Some(weight_data),
                }),
            )
        }
        mir_compat::MirOpCompat::Linear { name, x, weight_name, bias_name } => {
            let weight_data = weight_entries
                .iter()
                .find(|w| w.name == *weight_name)
                .map(weight_entry_to_proto)
                .unwrap_or_else(|| proto::WeightData { weight_data: None });

            let (bias_data, has_bias) = if let Some(bname) = bias_name {
                let bd = weight_entries
                    .iter()
                    .find(|w| w.name == *bname)
                    .map(weight_entry_to_proto)
                    .unwrap_or_else(|| proto::WeightData { weight_data: None });
                (Some(bd), true)
            } else {
                (None, false)
            };

            (
                name.clone(),
                proto::mil_operation::Operation::LinearOp(proto::MilLinearOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                    weight: Some(weight_data),
                    bias: bias_data,
                    has_bias,
                }),
            )
        }
        mir_compat::MirOpCompat::MatMul { name, x, y, transpose_y } => (
            name.clone(),
            proto::mil_operation::Operation::MatmulOp(proto::MilMatMulOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
                transpose_y: *transpose_y,
            }),
        ),
        mir_compat::MirOpCompat::Add { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::AddOp(proto::MilAddOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Mul { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::MulOp(proto::MilMulOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Sub { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::SubOp(proto::MilSubOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Abs { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::AbsOp(proto::MilAbsOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Maximum { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::MaximumOp(proto::MilMaximumOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Minimum { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::MinimumOp(proto::MilMinimumOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Reshape { name, x, shape } => (
            name.clone(),
            proto::mil_operation::Operation::ReshapeOp(proto::MilReshapeOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                shape: shape.iter().map(|&d| d as i64).collect(),
            }),
        ),
        mir_compat::MirOpCompat::Transpose { name, x, perm } => (
            name.clone(),
            proto::mil_operation::Operation::TransposeOp(proto::MilTransposeOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                perm: perm.iter().map(|&d| d as i64).collect(),
            }),
        ),
        mir_compat::MirOpCompat::SliceByIndex {
            name,
            x,
            begin,
            end,
            stride,
            begin_mask,
            end_mask,
            squeeze_mask: _,
        } => (
            name.clone(),
            proto::mil_operation::Operation::SliceByIndexOp(proto::MilSliceByIndexOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                begin: begin.iter().map(|&d| d as i64).collect(),
                end: end.iter().map(|&d| d as i64).collect(),
                stride: stride.iter().map(|&d| d as i64).collect(),
                // Legacy proto path: pack bools into bitmasks for the old proto format.
                // The proto-direct path (used by trace-compile) emits bool tensors instead.
                begin_mask: begin_mask
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m)
                    .map(|(i, _)| 1i64 << i)
                    .sum(),
                end_mask: end_mask
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m)
                    .map(|(i, _)| 1i64 << i)
                    .sum(),
            }),
        ),
        mir_compat::MirOpCompat::SliceUpdate { name, x, update, begin, end } => (
            name.clone(),
            proto::mil_operation::Operation::SliceUpdateOp(proto::MilSliceUpdateOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                update: Some(proto::OperandRef { name: update.clone() }),
                begin: begin.iter().map(|&d| d as i64).collect(),
                end: end.iter().map(|&d| d as i64).collect(),
            }),
        ),
        mir_compat::MirOpCompat::Concat { name, values, axis } => (
            name.clone(),
            proto::mil_operation::Operation::ConcatOp(proto::MilConcatOp {
                values: values.iter().map(|v| proto::OperandRef { name: v.clone() }).collect(),
                axis: *axis,
            }),
        ),
        mir_compat::MirOpCompat::Stack { name, values, axis } => (
            name.clone(),
            proto::mil_operation::Operation::StackOp(proto::MilStackOp {
                values: values.iter().map(|v| proto::OperandRef { name: v.clone() }).collect(),
                axis: *axis,
            }),
        ),
        mir_compat::MirOpCompat::Softmax { name, x, axis } => (
            name.clone(),
            proto::mil_operation::Operation::SoftmaxOp(proto::MilSoftmaxOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                axis: *axis,
            }),
        ),
        mir_compat::MirOpCompat::Gelu { name, x, mode } => (
            name.clone(),
            proto::mil_operation::Operation::GeluOp(proto::MilGeluOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                mode: mode.clone(),
            }),
        ),
        mir_compat::MirOpCompat::ScaledDotProductAttention {
            name,
            query,
            key,
            value,
            attention_mask,
            scale,
        } => (
            name.clone(),
            proto::mil_operation::Operation::ScaledDotProductAttentionOp(
                proto::MilScaledDotProductAttentionOp {
                    query: Some(proto::OperandRef { name: query.clone() }),
                    key: Some(proto::OperandRef { name: key.clone() }),
                    value: Some(proto::OperandRef { name: value.clone() }),
                    attn_mask: attention_mask
                        .as_ref()
                        .map(|m| proto::OperandRef { name: m.clone() }),
                    has_attn_mask: attention_mask.is_some(),
                    causal: false,
                    scale: scale.unwrap_or(0.0),
                },
            ),
        ),
        mir_compat::MirOpCompat::ReadState { name, state_id, shape, dtype } => (
            name.clone(),
            proto::mil_operation::Operation::ReadStateOp(proto::MilReadStateOp {
                state_id: state_id.clone(),
                shape: Some(shape_to_proto(&shape.iter().map(|&d| d as u64).collect::<Vec<_>>())),
                dtype: mil_dtype_to_proto(dtype),
            }),
        ),
        mir_compat::MirOpCompat::CoremlUpdateState { name, state_id, value } => (
            name.clone(),
            proto::mil_operation::Operation::CoremlUpdateStateOp(proto::MilCoremlUpdateStateOp {
                state_id: state_id.clone(),
                value: Some(proto::OperandRef { name: value.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Gather { name, x, indices, axis } => (
            name.clone(),
            proto::mil_operation::Operation::GatherOp(proto::MilGatherOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                indices: Some(proto::OperandRef { name: indices.clone() }),
                axis: *axis,
            }),
        ),
        mir_compat::MirOpCompat::ReduceMean { name, x, axes, keep_dims } => (
            name.clone(),
            proto::mil_operation::Operation::ReduceMeanOp(proto::MilReduceMeanOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                axes: axes.clone(),
                keep_dims: *keep_dims,
            }),
        ),
        // Sprint 54: ReduceSum proto emission
        mir_compat::MirOpCompat::ReduceSum { name, x, axes, keep_dims } => (
            name.clone(),
            proto::mil_operation::Operation::ReduceSumOp(proto::MilReduceSumOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                axes: axes.clone(),
                keep_dims: *keep_dims,
            }),
        ),
        // Sprint 54: Conv proto emission
        // T-98: Now includes quantized weight attributes (kernel_scale, kernel_zero_point,
        // kernel_palettized_lut) for the legacy MIL proto format.
        mir_compat::MirOpCompat::Conv { name, x, weight, pad_type, groups, kernel_scale, kernel_zero_point, kernel_palettized_lut } => {
            let weight_data = weight_entries
                .iter()
                .find(|w| w.name == *weight)
                .map(weight_entry_to_proto)
                .unwrap_or_else(|| proto::WeightData { weight_data: None });

            // T-98: Quantized conv attributes are logged in mir_to_compat.rs.
            // The legacy MIL proto (MilConvOp) carries these as dedicated proto fields
            // while the Apple wire format uses named const nodes for them.

            (
                name.clone(),
                proto::mil_operation::Operation::ConvOp(proto::MilConvOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                    weight: Some(weight_data),
                    pad_type: pad_type.clone(),
                    groups: *groups,
                    kernel_scale: kernel_scale.unwrap_or(0.0),
                    kernel_zero_point: kernel_zero_point.unwrap_or(0),
                    kernel_palettized_lut: kernel_palettized_lut.clone().unwrap_or_default(),
                }),
            )
        }
        // Sprint 54: StateWrite proto emission
        mir_compat::MirOpCompat::StateWrite { name, state_ref, value } => (
            name.clone(),
            proto::mil_operation::Operation::StateWriteOp(proto::MilStateWriteOp {
                state_ref: state_ref.clone(),
                value: Some(proto::OperandRef { name: value.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Rsqrt { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::RsqrtOp(proto::MilRsqrtOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::RealDiv { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::RealDivOp(proto::MilRealDivOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::LayerNorm { name, x, weight_name, bias_name, epsilon, axes } => {
            let weight_data = weight_entries
                .iter()
                .find(|w| w.name == *weight_name)
                .map(weight_entry_to_proto)
                .unwrap_or_else(|| proto::WeightData { weight_data: None });

            let (bias_data, has_bias) = if let Some(bname) = bias_name {
                let bd = weight_entries
                    .iter()
                    .find(|w| w.name == *bname)
                    .map(weight_entry_to_proto)
                    .unwrap_or_else(|| proto::WeightData { weight_data: None });
                (Some(bd), true)
            } else {
                (None, false)
            };

            (
                name.clone(),
                proto::mil_operation::Operation::LayerNormOp(proto::MilLayerNormOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                    weight: Some(weight_data),
                    bias: bias_data,
                    has_bias,
                    epsilon: *epsilon,
                    axes: axes.clone(),
                }),
            )
        }
        mir_compat::MirOpCompat::Topk { name, x, k, axis } => (
            name.clone(),
            proto::mil_operation::Operation::TopkOp(proto::MilTopkOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                k: *k,
                axis: *axis,
            }),
        ),
        mir_compat::MirOpCompat::Cos { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::CosOp(proto::MilCosOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Sin { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::SinOp(proto::MilSinOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Cast { name, x, dtype } => (
            name.clone(),
            proto::mil_operation::Operation::CastOp(proto::MilCastOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                dtype: mil_dtype_to_proto(dtype),
            }),
        ),
        mir_compat::MirOpCompat::Split { name, x, axis, num_splits } => (
            name.clone(),
            proto::mil_operation::Operation::SplitOp(proto::MilSplitOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                axis: *axis,
                num_splits: *num_splits,
            }),
        ),
        // Sprint 50: P2 ops
        mir_compat::MirOpCompat::Exp { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::ExpOp(proto::MilExpOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Sigmoid { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::SigmoidOp(proto::MilSigmoidOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Tanh { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::TanhOp(proto::MilTanhOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Relu { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::ReluOp(proto::MilReluOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Where { name, condition, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::WhereOp(proto::MilWhereOp {
                condition: Some(proto::OperandRef { name: condition.clone() }),
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Silu { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::SiluOp(proto::MilSiluOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Identity { name, x, dtype: _ } => (
            name.clone(),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Placeholder { name, dtype: _ } => (
            name.clone(),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: name.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::Tile { name, x, reps: _ } => {
            // Legacy proto has no MilTileOp variant; emit as identity.
            // The Apple wire-format emitter (mir_op_to_apple_ops) handles
            // the real "tile" MIL operation encoding for production use.
            (
                name.clone(),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                }),
            )
        }
        mir_compat::MirOpCompat::Fill { name, .. } => {
            // Legacy proto has no MilFillOp variant; emit as identity.
            // The Apple wire-format emitter (mir_op_to_apple_ops) handles
            // the real "fill" MIL operation encoding for production use.
            (
                name.clone(),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: name.clone() }),
                }),
            )
        }
        mir_compat::MirOpCompat::FillLike { name, ref_tensor, .. } => {
            // Legacy proto has no MilFillLikeOp variant; emit as identity.
            // The Apple wire-format emitter (mir_op_to_apple_ops) handles
            // the real "fill_like" MIL operation encoding for production use.
            (
                name.clone(),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: ref_tensor.clone() }),
                }),
            )
        }
        mir_compat::MirOpCompat::Neg { name, x } => {
            // Legacy proto has no MilNegOp variant; emit as identity.
            // The Apple wire-format emitter (mir_op_to_apple_ops) lowers
            // Neg to mul(x, -1), which is the production path.
            // Legacy format is not used for ML Program packages.
            (
                name.clone(),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                }),
            )
        }
        // New variants: legacy proto has no dedicated op types; emit as identity.
        // The Apple wire-format emitter (mir_op_to_apple_ops) handles the real
        // MIL operation encoding for production use.
        mir_compat::MirOpCompat::ExpandDims { name, x, .. }
        | mir_compat::MirOpCompat::Squeeze { name, x, .. }
        | mir_compat::MirOpCompat::Sqrt { name, x }
        | mir_compat::MirOpCompat::Pow { name, x, .. }
        | mir_compat::MirOpCompat::Clip { name, x, .. }
        | mir_compat::MirOpCompat::Equal { name, x, .. }
        | mir_compat::MirOpCompat::NotEqual { name, x, .. }
        | mir_compat::MirOpCompat::Greater { name, x, .. }
        | mir_compat::MirOpCompat::GreaterEqual { name, x, .. }
        | mir_compat::MirOpCompat::Less { name, x, .. }
        | mir_compat::MirOpCompat::LessEqual { name, x, .. }
        | mir_compat::MirOpCompat::LogicalNot { name, x }
        | mir_compat::MirOpCompat::LogicalAnd { name, x, .. }
        | mir_compat::MirOpCompat::LogicalOr { name, x, .. }
        | mir_compat::MirOpCompat::Pad { name, x, .. }
        | mir_compat::MirOpCompat::ReduceMax { name, x, .. }
        | mir_compat::MirOpCompat::ReduceMin { name, x, .. }
        | mir_compat::MirOpCompat::ReduceProd { name, x, .. }
        | mir_compat::MirOpCompat::Select { name, x, .. }
        | mir_compat::MirOpCompat::LeakyRelu { name, x, .. }
        | mir_compat::MirOpCompat::FloorDiv { name, x, .. }
        | mir_compat::MirOpCompat::Mod { name, x, .. }
        | mir_compat::MirOpCompat::Ceil { name, x }
        | mir_compat::MirOpCompat::Floor { name, x }
        | mir_compat::MirOpCompat::Round { name, x }
        | mir_compat::MirOpCompat::Sign { name, x }
        | mir_compat::MirOpCompat::Log { name, x } => (
            name.clone(),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: x.clone() }),
            }),
        ),
        // T-39 (I-18): Constexpr weight compression ops.
        // These are emitted using the MIL "constexpr_*" op_type naming
        // convention, which the Core ML runtime recognizes for weight
        // compression. The op parameters are serialized as named attributes.
        // The weight-name references are resolved to Const op outputs by name.
        //
        // Note: The proto emission uses a structured naming scheme for now.
        // When the MIL.proto is extended with constexpr_* message types,
        // these can be migrated to typed proto operations. Currently the
        // constexpr ops are emitted with their op_type as a named marker
        // so downstream tooling (Python bridge, Core ML runtime) can
        // recognize and process them.
        mir_compat::MirOpCompat::ConstexprAffineDequantize {
            name,
            quantized_data,
            scale: _,
            zero_point: _,
            axis: _,
        } => (
            format!("{name}__constexpr_affine_dequant"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: quantized_data.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprBlockwiseShiftScale {
            name,
            data,
            scale: _,
            offset: _,
            block_size: _,
        } => (
            format!("{name}__constexpr_blockwise_shift_scale"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: data.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprLutToDense { name, indices, lut: _, num_bits: _ } => (
            format!("{name}__constexpr_lut_to_dense"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: indices.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprSparseToDense {
            name,
            nonzero_data,
            shape: _,
            default_value: _,
        } => (
            format!("{name}__constexpr_sparse_to_dense"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: nonzero_data.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprCast { name, data, dtype: _ } => (
            format!("{name}__constexpr_cast"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: data.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprLutToSparse { name, data, num_bits: _ } => (
            format!("{name}__constexpr_lut_to_sparse"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: data.clone() }),
            }),
        ),
        mir_compat::MirOpCompat::ConstexprSparseBlockwiseShiftScale {
            name,
            data,
            scale: _,
            offset: _,
            block_size: _,
            block_axis: _,
        } => (
            format!("{name}__constexpr_sparse_blockwise_shift_scale"),
            proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                x: Some(proto::OperandRef { name: data.clone() }),
            }),
        ),
        // Unsupported ops are emitted as identity pass-through with a comment
        // marker in the function name. The op_kind and params are preserved
        // for downstream Python emission or manual inspection.
        //
        // T-66 (I-40): Pooling, spatial rearrangement, normalization, and
        // quantize/dequantize ops now have proper MirOpCompat variants but
        // their proto emission is handled through the Apple proto path or
        // the identity fallback here.
        op @ mir_compat::MirOpCompat::MaxPool { .. }
        | op @ mir_compat::MirOpCompat::AvgPool { .. }
        | op @ mir_compat::MirOpCompat::L2Pool { .. }
        | op @ mir_compat::MirOpCompat::DepthToSpace { .. }
        | op @ mir_compat::MirOpCompat::SpaceToDepth { .. }
        | op @ mir_compat::MirOpCompat::PixelShuffle { .. }
        | op @ mir_compat::MirOpCompat::PixelUnshuffle { .. }
        | op @ mir_compat::MirOpCompat::BatchNorm { .. }
        | op @ mir_compat::MirOpCompat::InstanceNorm { .. }
        | op @ mir_compat::MirOpCompat::L2Norm { .. }
        | op @ mir_compat::MirOpCompat::Quantize { .. }
        | op @ mir_compat::MirOpCompat::Dequantize { .. }
        // F-OPS-01: ANEC ops promoted from CPU-only — legacy proto uses
        // identity pass-through; Apple proto path does actual decomposition.
        | op @ mir_compat::MirOpCompat::AnecFusedConvActivate { .. }
        | op @ mir_compat::MirOpCompat::AnecScaledElementwise { .. } => {
            // These ops have proper MirOpCompat representations but the
            // legacy proto format doesn't have dedicated message types.
            // Emit as identity pass-through to preserve graph structure;
            // the Apple proto path handles their actual emission.
            let op_name = op.output_name();
            let op_kind = match op {
                mir_compat::MirOpCompat::MaxPool { .. } => "max_pool",
                mir_compat::MirOpCompat::AvgPool { .. } => "avg_pool",
                mir_compat::MirOpCompat::L2Pool { .. } => "l2_pool",
                mir_compat::MirOpCompat::DepthToSpace { .. } => "depth_to_space",
                mir_compat::MirOpCompat::SpaceToDepth { .. } => "space_to_depth",
                mir_compat::MirOpCompat::PixelShuffle { .. } => "pixel_shuffle",
                mir_compat::MirOpCompat::PixelUnshuffle { .. } => "pixel_unshuffle",
                mir_compat::MirOpCompat::BatchNorm { .. } => "batch_norm",
                mir_compat::MirOpCompat::InstanceNorm { .. } => "instance_norm",
                mir_compat::MirOpCompat::L2Norm { .. } => "l2_norm",
                mir_compat::MirOpCompat::Quantize { .. } => "quantize",
                mir_compat::MirOpCompat::Dequantize { .. } => "dequantize",
                mir_compat::MirOpCompat::AnecFusedConvActivate { .. } => "anec_fused_conv_activate",
                mir_compat::MirOpCompat::AnecScaledElementwise { .. } => "anec_scaled_elementwise",
                _ => unreachable!(),
            };
            (
                format!("{op_name}__unsupported_{op_kind}"),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: op_name }),
                }),
            )
        }
        mir_compat::MirOpCompat::Unsupported { op_kind, name, params_json: _, inputs: _ } => {
            (
                format!("{name}__unsupported_{op_kind}"),
                proto::mil_operation::Operation::IdentityOp(proto::MilIdentityOp {
                    x: Some(proto::OperandRef { name: name.clone() }),
                }),
            )
        }
    };

    proto::MilOperation { name, operation: Some(operation) }
}

/// Convert `MirGraphCompat` → `proto::MilFunction`.
pub fn mir_graph_to_proto_function(
    graph: &mir_compat::MirGraphCompat,
    weight_entries: &[WeightEntry],
) -> proto::MilFunction {
    let operations = graph.ops.iter().map(|op| mir_op_to_proto_op(op, weight_entries)).collect();

    proto::MilFunction {
        block: Some(proto::MilBlock {
            operations,
            input_names: graph.inputs.clone(),
            output_names: graph.outputs.clone(),
        }),
        additional_blocks: vec![],
    }
}

/// Convert `CoreMlModel` → `proto::Model`.
///
/// This is the top-level conversion that produces a complete protobuf `Model`
/// message ready for serialization with `prost::Message::encode_to_vec()`.
pub fn convert_to_proto_model(model: &CoreMlModel, weight_entries: &[WeightEntry]) -> proto::Model {
    // Build function descriptions map for ModelDescription
    let mut functions_map = std::collections::HashMap::new();
    for func in &model.functions {
        functions_map.insert(
            func.name.clone(),
            proto::FunctionDescription {
                input: func.inputs.iter().map(tensor_desc_to_proto).collect(),
                output: func.outputs.iter().map(tensor_desc_to_proto).collect(),
                state: func.states.iter().map(tensor_desc_to_proto).collect(),
            },
        );
    }

    // Build MLProgram functions map
    let mut ml_program_functions = std::collections::HashMap::new();
    for func in &model.functions {
        // Find ops for this function
        let mir_ops: Vec<_> = func.operations.clone();
        let graph = mir_compat::MirGraphCompat {
            ops: mir_ops,
            inputs: func.inputs.iter().map(|td| td.name.clone()).collect(),
            outputs: func.outputs.iter().map(|td| td.name.clone()).collect(),
            opset_version: "iOS18".to_string(),
            function_name: func.name.clone(),
            input_descs: vec![],
            output_descs: vec![],
            node_shapes: std::collections::HashMap::new(),
        };
        ml_program_functions
            .insert(func.name.clone(), mir_graph_to_proto_function(&graph, weight_entries));
    }

    proto::Model {
        specification_version: spec_version_to_proto(&model.spec_version),
        description: Some(proto::ModelDescription {
            input: model.description.inputs.iter().map(tensor_desc_to_proto).collect(),
            output: model.description.outputs.iter().map(tensor_desc_to_proto).collect(),
            state: model.description.states.iter().map(tensor_desc_to_proto).collect(),
            default_function_name: model.default_function_name.clone(),
            functions: functions_map,
        }),
        ml_program: Some(proto::MlProgram {
            functions: ml_program_functions,
            default_function_name: model.default_function_name.clone(),
        }),
        user_defined_metadata: model.user_defined_metadata.clone(),
        deployment_target: None,
        optimization_hints: Some(proto::OptimizationHints {
            preferred_compute_unit: compute_unit_to_proto(&model.compute_unit),
            allow_fp16_accumulation: true,
        }),
        author: String::new(),
        short_description: String::new(),
        license: String::new(),
        version_string: String::new(),
    }
}

// ─── Apple-Compatible Conversion Functions ──────────────────────────────────
// These convert the hand-written domain types into Apple's actual protobuf
// wire format (packages CoreML.Specification / CoreML.Specification.MILSpec).
// This is the format Core ML's runtime expects for .mlpackage models.

/// Convert `MilDtypeCompat` → `apple_proto::mil_spec::DataType` (Apple enum values).
///
/// T-35: Added Int4, UInt4, E4M3, E5M2, UInt16 mapping to Apple's MIL proto DataType.
pub fn mil_dtype_to_apple(dt: &mir_compat::MilDtypeCompat) -> i32 {
    match dt {
        mir_compat::MilDtypeCompat::Fp16 => apple_proto::mil_spec::DataType::Float16 as i32,
        mir_compat::MilDtypeCompat::Fp32 => apple_proto::mil_spec::DataType::Float32 as i32,
        mir_compat::MilDtypeCompat::Int32 => apple_proto::mil_spec::DataType::Int32 as i32,
        mir_compat::MilDtypeCompat::UInt8 => apple_proto::mil_spec::DataType::Uint8 as i32,
        // T-35: new dtypes — Apple MIL proto v2 DataType values
        mir_compat::MilDtypeCompat::Int4 => apple_proto::mil_spec::DataType::Int4 as i32,
        mir_compat::MilDtypeCompat::UInt4 => apple_proto::mil_spec::DataType::Uint4 as i32,
        mir_compat::MilDtypeCompat::E4M3 => apple_proto::mil_spec::DataType::Float8e4m3fn as i32,
        mir_compat::MilDtypeCompat::E5M2 => apple_proto::mil_spec::DataType::Float8e5m2 as i32,
        mir_compat::MilDtypeCompat::UInt16 => apple_proto::mil_spec::DataType::Uint16 as i32,
    }
}

/// Convert `CoreMlDataType` → `apple_proto::mil_spec::DataType` (Apple MIL enum values).
pub fn coreml_dtype_to_apple_mil(dt: &CoreMlDataType) -> i32 {
    match dt {
        CoreMlDataType::Float16 => apple_proto::mil_spec::DataType::Float16 as i32,
        CoreMlDataType::Float32 => apple_proto::mil_spec::DataType::Float32 as i32,
        CoreMlDataType::Int32 => apple_proto::mil_spec::DataType::Int32 as i32,
        CoreMlDataType::UInt8 => apple_proto::mil_spec::DataType::Uint8 as i32,
        CoreMlDataType::Int8 => apple_proto::mil_spec::DataType::Int8 as i32,
        CoreMlDataType::Float64 => apple_proto::mil_spec::DataType::Float64 as i32,
        CoreMlDataType::Unknown => apple_proto::mil_spec::DataType::UnusedType as i32,
        CoreMlDataType::Bool => apple_proto::mil_spec::DataType::Bool as i32,
        // T-35: new dtypes
        CoreMlDataType::Int4 => apple_proto::mil_spec::DataType::Int4 as i32,
        CoreMlDataType::UInt4 => apple_proto::mil_spec::DataType::Uint4 as i32,
        CoreMlDataType::E4M3 => apple_proto::mil_spec::DataType::Float8e4m3fn as i32,
        CoreMlDataType::E5M2 => apple_proto::mil_spec::DataType::Float8e5m2 as i32,
        CoreMlDataType::UInt16 => apple_proto::mil_spec::DataType::Uint16 as i32,
    }
}

/// Convert `MilDtypeCompat` → `apple_proto::mil_spec::DataType` (Apple MIL enum values).
fn compat_dtype_to_apple_mil(dt: &mir_compat::MilDtypeCompat) -> i32 {
    match dt {
        mir_compat::MilDtypeCompat::Fp16 => apple_proto::mil_spec::DataType::Float16 as i32,
        mir_compat::MilDtypeCompat::Fp32 => apple_proto::mil_spec::DataType::Float32 as i32,
        mir_compat::MilDtypeCompat::Int32 => apple_proto::mil_spec::DataType::Int32 as i32,
        mir_compat::MilDtypeCompat::UInt8 => apple_proto::mil_spec::DataType::Uint8 as i32,
        // T-35: new dtypes
        mir_compat::MilDtypeCompat::Int4 => apple_proto::mil_spec::DataType::Int4 as i32,
        mir_compat::MilDtypeCompat::UInt4 => apple_proto::mil_spec::DataType::Uint4 as i32,
        mir_compat::MilDtypeCompat::E4M3 => apple_proto::mil_spec::DataType::Float8e4m3fn as i32,
        mir_compat::MilDtypeCompat::E5M2 => apple_proto::mil_spec::DataType::Float8e5m2 as i32,
        mir_compat::MilDtypeCompat::UInt16 => apple_proto::mil_spec::DataType::Uint16 as i32,
    }
}

/// Convert `CoreMlDataType` → `apple_proto::array_feature_type::ArrayDataType`
/// (Apple enum values: FLOAT32=65568, FLOAT16=65552, INT32=131104).
pub fn coreml_dtype_to_apple_array(dt: &CoreMlDataType) -> i32 {
    match dt {
        CoreMlDataType::Float32 => apple_proto::array_feature_type::ArrayDataType::Float32 as i32,
        CoreMlDataType::Float16 => apple_proto::array_feature_type::ArrayDataType::Float16 as i32,
        CoreMlDataType::Int32 => apple_proto::array_feature_type::ArrayDataType::Int32 as i32,
        CoreMlDataType::UInt8 => apple_proto::array_feature_type::ArrayDataType::Int8 as i32,
        CoreMlDataType::Int8 => apple_proto::array_feature_type::ArrayDataType::Int8 as i32,
        CoreMlDataType::Float64 => apple_proto::array_feature_type::ArrayDataType::Double as i32,
        // T-35: new dtypes — the FeatureTypes proto doesn't have these ArrayDataType
        // variants yet, so map to InvalidArrayDataType as fallback. The MIL proto
        // DataType enum (used in operations) handles them properly via
        // coreml_dtype_to_apple_mil().
        CoreMlDataType::Int4
        | CoreMlDataType::UInt4
        | CoreMlDataType::E4M3
        | CoreMlDataType::E5M2
        | CoreMlDataType::UInt16 => {
            apple_proto::array_feature_type::ArrayDataType::InvalidArrayDataType as i32
        }
        CoreMlDataType::Unknown | CoreMlDataType::Bool => {
            apple_proto::array_feature_type::ArrayDataType::InvalidArrayDataType as i32
        }
    }
}

/// Build an `apple_proto::mil_spec::Dimension` for a constant (known) dimension size.
fn make_constant_dimension(size: u64) -> apple_proto::mil_spec::Dimension {
    apple_proto::mil_spec::Dimension {
        dimension: Some(apple_proto::mil_spec::dimension::Dimension::Constant(
            apple_proto::mil_spec::dimension::ConstantDimension { size },
        )),
    }
}

/// Build an `apple_proto::mil_spec::TensorType` from dtype and shape.
fn make_apple_tensor_type(dtype: i32, shape: &[u64]) -> apple_proto::mil_spec::TensorType {
    apple_proto::mil_spec::TensorType {
        data_type: dtype,
        rank: shape.len() as i64,
        dimensions: shape.iter().map(|&d| make_constant_dimension(d)).collect(),
        attributes: HashMap::new(),
    }
}

/// Build an `apple_proto::mil_spec::ValueType` wrapping a TensorType.
fn make_apple_value_type(dtype: i32, shape: &[u64]) -> apple_proto::mil_spec::ValueType {
    apple_proto::mil_spec::ValueType {
        r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(make_apple_tensor_type(
            dtype, shape,
        ))),
    }
}

/// Sanitize a string for use as a MIL SSA variable identifier.
///
/// MIL (and Python) treat `.` as an attribute accessor, so `lm_head.weight`
/// is parsed as `lm_head` accessing attribute `weight`. To avoid this, we
/// replace dots and other non-identifier characters with underscores.
///
/// This function replaces:
/// - `.` → `_` (most common: HuggingFace weight names like `model.layers.0.weight`)
/// - Any other non-alphanumeric, non-underscore, non-hyphen character → `_`
///
/// The original name is preserved in the `name = string(...)` attribute for
/// debugging/metadata; only the SSA variable identifier is sanitized.
fn sanitize_mil_ident(name: &str) -> String {
    name.replace('.', "_").replace(|c: char| !c.is_alphanumeric() && c != '_' && c != '-', "_")
}

/// Build an `apple_proto::mil_spec::NamedValueType` (name + type pair).
fn make_apple_named_value_type(
    name: &str,
    dtype: i32,
    shape: &[u64],
) -> apple_proto::mil_spec::NamedValueType {
    apple_proto::mil_spec::NamedValueType {
        name: name.to_string(),
        r#type: Some(make_apple_value_type(dtype, shape)),
    }
}

/// Build an `apple_proto::mil_spec::NamedValueType` with `StateType` wrapping
/// a `TensorType`.
///
/// Core ML requires state variables to be declared in MIL function inputs
/// with `ValueType.type = StateType { wrappedType = TensorType {...} }`.
/// Without this, `read_state` and `write_state` ops reference unbound names
/// and the runtime rejects the model.
fn make_apple_state_named_value_type(
    name: &str,
    dtype: i32,
    shape: &[u64],
) -> apple_proto::mil_spec::NamedValueType {
    let inner_value_type = make_apple_value_type(dtype, shape);
    apple_proto::mil_spec::NamedValueType {
        name: name.to_string(),
        r#type: Some(apple_proto::mil_spec::ValueType {
            r#type: Some(apple_proto::mil_spec::value_type::Type::StateType(Box::new(
                apple_proto::mil_spec::StateType { wrapped_type: Some(Box::new(inner_value_type)) },
            ))),
        }),
    }
}

/// Look up the shape of a tensor by name from the node_shapes map,
/// converting `Vec<usize>` to `Vec<u64>`. Returns empty vec if not found.
fn lookup_shape_u64(
    name: &str,
    node_shapes: &std::collections::HashMap<String, Vec<usize>>,
) -> Vec<u64> {
    node_shapes.get(name).map(|s| s.iter().map(|&d| d as u64).collect()).unwrap_or_default()
}

/// Build an `apple_proto::mil_spec::Argument` that references an SSA name.
fn make_name_arg(name: &str) -> apple_proto::mil_spec::Argument {
    apple_proto::mil_spec::Argument {
        arguments: vec![apple_proto::mil_spec::argument::Binding {
            binding: Some(apple_proto::mil_spec::argument::binding::Binding::Name(
                name.to_string(),
            )),
        }],
    }
}

/// Build an `apple_proto::mil_spec::Value` with an immediate bytes tensor value.
/// Used for inline const data (small tensors, biases, etc.).
fn make_immediate_bytes_value(
    raw_data: Vec<u8>,
    dtype: i32,
    shape: &[u64],
) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(dtype, shape)),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Bytes(
                            apple_proto::mil_spec::tensor_value::RepeatedBytes { values: raw_data },
                        )),
                    },
                )),
            },
        )),
    }
}

fn make_immediate_int32_value(values: Vec<i32>, shape: &[u64]) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(apple_proto::mil_spec::DataType::Int32 as i32, shape)),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Ints(
                            apple_proto::mil_spec::tensor_value::RepeatedInts { values },
                        )),
                    },
                )),
            },
        )),
    }
}

/// Create an immediate INT64 vector value using typed `RepeatedLongInts` storage.
///
/// **CAUTION**: This function should NOT be used for Core ML ios19+ MIL
/// operations. Shape/index parameters (reshape.shape, transpose.perm,
/// tile.reps, fill.shape, expand_dims.axis, squeeze.axis, pad.pad,
/// slice_by_index.begin/end, slice_update.begin/end) must use INT32
/// (`make_immediate_int32_value`), because ios19 ops reject INT64:
///
///   ios19.reshape: "Expected { tensor<int32, [?]>, tensor<int16, [?]>,
///     tensor<int8, [?]> }; got tensor<int64, [N]>"
///
/// This function is kept only for potential future use cases where INT64
/// immediate values are genuinely required by a Core ML operation.
#[allow(dead_code)]
fn make_immediate_int64_value(values: Vec<i64>, shape: &[u64]) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(apple_proto::mil_spec::DataType::Int64 as i32, shape)),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::LongInts(
                            apple_proto::mil_spec::tensor_value::RepeatedLongInts { values },
                        )),
                    },
                )),
            },
        )),
    }
}

fn make_immediate_bool_value(value: bool) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(apple_proto::mil_spec::DataType::Bool as i32, &[])),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Bools(
                            apple_proto::mil_spec::tensor_value::RepeatedBools {
                                values: vec![value],
                            },
                        )),
                    },
                )),
            },
        )),
    }
}

/// Create a boolean tensor immediate value (1D), e.g. for `slice_by_index` masks.
///
/// Core ML's `ios19.slice_by_index` requires `begin_mask`, `end_mask`, and
/// `squeeze_mask` to be `tensor<bool, [rank]>`, NOT integer bitmasks.
/// This function creates such a boolean tensor immediate value.
fn make_immediate_bool_tensor_value(values: Vec<bool>) -> apple_proto::mil_spec::Value {
    let rank = values.len() as u64;
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(apple_proto::mil_spec::DataType::Bool as i32, &[rank])),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Bools(
                            apple_proto::mil_spec::tensor_value::RepeatedBools { values },
                        )),
                    },
                )),
            },
        )),
    }
}

fn make_immediate_float32_value(value: f32) -> apple_proto::mil_spec::Value {
    make_immediate_float_value(apple_proto::mil_spec::DataType::Float32 as i32, value)
}

/// Create a float scalar immediate value whose *declared* dtype matches `apple_dtype`.
///
/// Core ML requires the `fill.value` parameter dtype to match the fill output dtype.
/// Crucially, the storage encoding must also be compatible with the declared dtype:
///
/// - **Float32**: uses `TensorValue.floats` (repeated `float` = f32 bits).
/// - **Float16**: uses `TensorValue.bytes` with raw IEEE-754 half-precision bits.
///   Core ML's ML Program parser rejects a `FLOAT16` tensor stored via the `floats`
///   field with: "Tensor storage and type have different number of elements".
fn make_immediate_float_value(apple_dtype: i32, value: f32) -> apple_proto::mil_spec::Value {
    let float16_dtype = apple_proto::mil_spec::DataType::Float16 as i32;

    let tensor_value = if apple_dtype == float16_dtype {
        // Encode as raw f16 bytes. Core ML expects FLOAT16 immediates in the
        // `bytes` field, not the `floats` field.
        let f16_bits = half::f16::from_f32(value).to_bits();
        let raw_bytes = f16_bits.to_le_bytes().to_vec();
        apple_proto::mil_spec::tensor_value::Value::Bytes(
            apple_proto::mil_spec::tensor_value::RepeatedBytes { values: raw_bytes },
        )
    } else {
        // Float32 (or any other float dtype) → standard `floats` storage.
        apple_proto::mil_spec::tensor_value::Value::Floats(
            apple_proto::mil_spec::tensor_value::RepeatedFloats { values: vec![value] },
        )
    };

    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(apple_dtype, &[])),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue { value: Some(tensor_value) },
                )),
            },
        )),
    }
}

fn make_immediate_string_value(value: String) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(apple_proto::mil_spec::ValueType {
            r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(
                apple_proto::mil_spec::TensorType {
                    data_type: apple_proto::mil_spec::DataType::String as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Strings(
                            apple_proto::mil_spec::tensor_value::RepeatedStrings {
                                values: vec![value],
                            },
                        )),
                    },
                )),
            },
        )),
    }
}

/// Build an `apple_proto::mil_spec::Value` referencing weight.bin via BlobFileValue.
///
/// The `fileName` field uses Apple's virtual path convention:
/// `"@model_path/weights/weight.bin"`. The `@model_path` is resolved at runtime
/// by Core ML to the actual weights directory within the mlpackage.
fn make_blob_file_value(offset: u64, dtype: i32, shape: &[u64]) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(make_apple_value_type(dtype, shape)),
        value: Some(apple_proto::mil_spec::value::Value::BlobFileValue(
            apple_proto::mil_spec::value::BlobFileValue {
                file_name: "@model_path/weights/weight.bin".to_string(),
                offset,
            },
        )),
    }
}

/// Build an `apple_proto::mil_spec::Argument` with a compile-time constant value.
fn make_value_arg(value: apple_proto::mil_spec::Value) -> apple_proto::mil_spec::Argument {
    apple_proto::mil_spec::Argument {
        arguments: vec![apple_proto::mil_spec::argument::Binding {
            binding: Some(apple_proto::mil_spec::argument::binding::Binding::Value(value)),
        }],
    }
}

/// Build an `apple_proto::mil_spec::Value` for an `attributes["name"]` entry.
///
/// In Apple's format, every operation has an `attributes["name"]` entry containing
/// the operation's output name as a STRING tensor with an immediate value.
fn make_name_attribute_value(name: &str) -> apple_proto::mil_spec::Value {
    apple_proto::mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(apple_proto::mil_spec::ValueType {
            r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(
                apple_proto::mil_spec::TensorType {
                    data_type: apple_proto::mil_spec::DataType::String as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
            apple_proto::mil_spec::value::ImmediateValue {
                value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                    apple_proto::mil_spec::TensorValue {
                        value: Some(apple_proto::mil_spec::tensor_value::Value::Strings(
                            apple_proto::mil_spec::tensor_value::RepeatedStrings {
                                values: vec![name.to_string()],
                            },
                        )),
                    },
                )),
            },
        )),
    }
}

/// Add `attributes["name"]` to an operation's attributes map.
fn add_name_attribute(attributes: &mut HashMap<String, apple_proto::mil_spec::Value>, name: &str) {
    attributes.insert("name".to_string(), make_name_attribute_value(name));
}

/// Build an `apple_proto::FeatureDescription` for a tensor input/output.
fn make_apple_feature_desc(
    name: &str,
    dtype: &CoreMlDataType,
    shape: &[u64],
) -> apple_proto::FeatureDescription {
    let array_dtype = coreml_dtype_to_apple_array(dtype);

    // Core ML requires non-empty shape constraints on every multiarray feature.
    // Empty shape → fallback to [1]. Dynamic dims (0) get ShapeRange flexibility.
    let effective_shape: Vec<i64> = if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().map(|&d| if d == 0 { 1 } else { d as i64 }).collect()
    };

    let has_dynamic_dims = shape.contains(&0);

    let shape_flexibility = if has_dynamic_dims {
        // Build ShapeRange: each dimension gets a SizeRange.
        // Dynamic dims (0) get range [1, -1] (unbounded upper), static dims get [n, n].
        let size_ranges: Vec<apple_proto::SizeRange> = shape
            .iter()
            .map(|&d| {
                if d == 0 {
                    apple_proto::SizeRange {
                        lower_bound: 1,
                        upper_bound: -1, // -1 means unbounded
                    }
                } else {
                    apple_proto::SizeRange { lower_bound: d, upper_bound: d as i64 }
                }
            })
            .collect();
        Some(apple_proto::array_feature_type::ShapeFlexibility::ShapeRange(
            apple_proto::array_feature_type::ShapeRange { size_ranges },
        ))
    } else {
        None
    };

    apple_proto::FeatureDescription {
        name: name.to_string(),
        short_description: String::new(),
        r#type: Some(apple_proto::FeatureType {
            is_optional: false,
            r#type: Some(apple_proto::feature_type::Type::MultiArrayType(
                apple_proto::ArrayFeatureType {
                    shape: effective_shape,
                    data_type: array_dtype,
                    shape_flexibility,
                },
            )),
        }),
    }
}

/// Build an `apple_proto::FeatureDescription` for a state tensor.
///
/// State features use `StateFeatureType` wrapping `ArrayFeatureType`,
/// matching Apple's wire format for stateful models (e.g., KV-cache).
fn make_apple_state_feature_desc(
    name: &str,
    dtype: &CoreMlDataType,
    shape: &[u64],
) -> apple_proto::FeatureDescription {
    let array_dtype = coreml_dtype_to_apple_array(dtype);

    // Core ML requires non-empty shape constraints on every multiarray feature.
    let effective_shape: Vec<i64> = if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().map(|&d| if d == 0 { 1 } else { d as i64 }).collect()
    };

    let has_dynamic_dims = shape.contains(&0);

    let shape_flexibility = if has_dynamic_dims {
        let size_ranges: Vec<apple_proto::SizeRange> = shape
            .iter()
            .map(|&d| {
                if d == 0 {
                    apple_proto::SizeRange { lower_bound: 1, upper_bound: -1 }
                } else {
                    apple_proto::SizeRange { lower_bound: d, upper_bound: d as i64 }
                }
            })
            .collect();
        Some(apple_proto::array_feature_type::ShapeFlexibility::ShapeRange(
            apple_proto::array_feature_type::ShapeRange { size_ranges },
        ))
    } else {
        None
    };

    apple_proto::FeatureDescription {
        name: name.to_string(),
        short_description: String::new(),
        r#type: Some(apple_proto::FeatureType {
            is_optional: false,
            r#type: Some(apple_proto::feature_type::Type::StateType(
                apple_proto::StateFeatureType {
                    r#type: Some(apple_proto::state_feature_type::Type::ArrayType(
                        apple_proto::ArrayFeatureType {
                            shape: effective_shape,
                            data_type: array_dtype,
                            shape_flexibility,
                        },
                    )),
                },
            )),
        }),
    }
}

/// Convert `MirOpCompat` → `Vec<apple_proto::mil_spec::Operation>` (Apple's generic format).
///
/// In Apple's format, every op is a generic `Operation` with:
/// - `type`: string like "const", "linear", "add", etc.
/// - `inputs`: map of named arguments
/// - `outputs`: list of NamedValueType
/// - `attributes`: map including `"name"` (op output name as STRING) and for
///   `const` ops `"val"` (the value data)
///
/// Returns a Vec because some MIR ops (e.g., Cast) require emitting additional
/// preceding const ops for their parameter values.
///
/// This is fundamentally different from the legacy per-op-type proto messages.
fn mir_op_to_apple_ops(
    op: &mir_compat::MirOpCompat,
    weight_entries: &[WeightEntry],
    node_shapes: &std::collections::HashMap<String, Vec<usize>>,
) -> Vec<apple_proto::mil_spec::Operation> {
    match op {
        mir_compat::MirOpCompat::Const { name, data, dtype, shape } => {
            let apple_dtype = mil_dtype_to_apple(dtype);
            let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

            // If this const has a weight entry, use BlobFileValue; otherwise inline
            let value = if let Some(entry) = weight_entries.iter().find(|w| w.name == *name) {
                make_blob_file_value(entry.offset, apple_dtype, &shape_u64)
            } else {
                make_immediate_bytes_value(data.clone(), apple_dtype, &shape_u64)
            };

            // Apple's format: const ops use attributes["val"] for the value
            // and attributes["name"] for the op name (NOT inputs["value"])
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert("val".to_string(), value);

            vec![apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(name, apple_dtype, &shape_u64)],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Linear { name, x, weight_name, bias_name } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("weight".to_string(), make_name_arg(weight_name));
            if let Some(bname) = bias_name {
                inputs.insert("bias".to_string(), make_name_arg(bname));
            }

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "linear".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::MatMul { name, x, y, transpose_y } => {
            // T-131 (V-129 / Orion #12): Per Orion #12, transpose flags must be
            // emitted as named const nodes (not immediate bools). The ANEC
            // ConvertMatMul converter is family-scoped (8 instantiations) and
            // immediate bool values may not be accepted by all family
            // implementations. Named const nodes are the canonical form used
            // by Core ML's own tools for boolean parameters.
            let transpose_x_name = format!("{name}_transpose_x_0");
            let transpose_y_name = format!("{name}_transpose_y_0");

            // Emit const op for transpose_x (always false)
            let mut tx_const_attrs = HashMap::new();
            add_name_attribute(&mut tx_const_attrs, &transpose_x_name);
            tx_const_attrs.insert("val".to_string(), make_immediate_bool_value(false));
            let tx_const_op = apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &transpose_x_name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: tx_const_attrs,
            };

            // Emit const op for transpose_y
            let mut ty_const_attrs = HashMap::new();
            add_name_attribute(&mut ty_const_attrs, &transpose_y_name);
            ty_const_attrs.insert("val".to_string(), make_immediate_bool_value(*transpose_y));
            let ty_const_op = apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &transpose_y_name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: ty_const_attrs,
            };

            // Emit matmul op referencing const nodes by name
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            inputs.insert("transpose_x".to_string(), make_name_arg(&transpose_x_name));
            inputs.insert("transpose_y".to_string(), make_name_arg(&transpose_y_name));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            let matmul_op = apple_proto::mil_spec::Operation {
                r#type: "matmul".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            };

            vec![tx_const_op, ty_const_op, matmul_op]
        }
        mir_compat::MirOpCompat::Add { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "add".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Mul { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "mul".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Sub { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "sub".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Abs { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "abs".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Maximum { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "maximum".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Minimum { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "minimum".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Reshape { name, x, shape } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            // shape as INT32 immediate — Core ML ios19.reshape rejects INT64
            // ("Expected { tensor<int32, [?]>, tensor<int16, [?]>,
            //   tensor<int8, [?]> }; got tensor<int64, [N]>")
            inputs.insert(
                "shape".to_string(),
                make_value_arg(make_immediate_int32_value(shape.clone(), &[shape.len() as u64])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "reshape".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Transpose { name, x, perm } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            // perm as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "perm".to_string(),
                make_value_arg(make_immediate_int32_value(perm.clone(), &[perm.len() as u64])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "transpose".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::SliceByIndex {
            name,
            x,
            begin,
            end,
            stride,
            begin_mask,
            end_mask,
            squeeze_mask,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            // begin as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "begin".to_string(),
                make_value_arg(make_immediate_int32_value(begin.clone(), &[begin.len() as u64])),
            );

            // end as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "end".to_string(),
                make_value_arg(make_immediate_int32_value(end.clone(), &[end.len() as u64])),
            );

            // stride as INT32 immediate (omit if all 1s — Core ML defaults to 1)
            if !stride.is_empty() && stride.iter().any(|&s| s != 1) {
                inputs.insert(
                    "stride".to_string(),
                    make_value_arg(make_immediate_int32_value(
                        stride.clone(),
                        &[stride.len() as u64],
                    )),
                );
            }

            // begin_mask as tensor<bool, [rank]> — Core ML ios19.schema requires
            // boolean tensors, NOT integer bitmasks. mask[N]=true means "ignore
            // begin[N], start at 0". Only emit when any mask bit is true (Core ML
            // defaults to all-false when omitted).
            if begin_mask.iter().any(|&m| m) {
                inputs.insert(
                    "begin_mask".to_string(),
                    make_value_arg(make_immediate_bool_tensor_value(begin_mask.clone())),
                );
            }

            // end_mask as tensor<bool, [rank]> — same boolean tensor schema.
            // mask[N]=true means "ignore end[N], go to full extent".
            if end_mask.iter().any(|&m| m) {
                inputs.insert(
                    "end_mask".to_string(),
                    make_value_arg(make_immediate_bool_tensor_value(end_mask.clone())),
                );
            }

            // squeeze_mask as tensor<bool, [rank]> — same boolean tensor schema.
            // mask[N]=true means "remove dimension N from the output".
            // Only emit when any squeeze bit is true (Core ML defaults to
            // all-false = no dimensions squeezed).
            if squeeze_mask.iter().any(|&m| m) {
                inputs.insert(
                    "squeeze_mask".to_string(),
                    make_value_arg(make_immediate_bool_tensor_value(squeeze_mask.clone())),
                );
            }

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "slice_by_index".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::SliceUpdate { name, x, update, begin, end } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("update".to_string(), make_name_arg(update));

            // begin as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "begin".to_string(),
                make_value_arg(make_immediate_int32_value(begin.clone(), &[begin.len() as u64])),
            );

            // end as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "end".to_string(),
                make_value_arg(make_immediate_int32_value(end.clone(), &[end.len() as u64])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "slice_update".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Concat { name, values, axis } => {
            let mut inputs = HashMap::new();
            let mut concat_args = apple_proto::mil_spec::Argument { arguments: vec![] };
            for v in values {
                concat_args.arguments.push(apple_proto::mil_spec::argument::Binding {
                    binding: Some(apple_proto::mil_spec::argument::binding::Binding::Name(
                        v.clone(),
                    )),
                });
            }
            inputs.insert("values".to_string(), concat_args);

            // axis as immediate value
            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );

            // interleave as bool scalar — Core ML's ios19.concat requires this
            // parameter to be explicitly serialized (the compiler does not apply
            // the default value of false when it's missing). Non-interleaved
            // concat is the standard behavior for all current use cases
            // (RoPE rotate_half, attention Q/K/V stacking, etc.).
            inputs
                .insert("interleave".to_string(), make_value_arg(make_immediate_bool_value(false)));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "concat".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Stack { name, values, axis } => {
            let mut inputs = HashMap::new();
            let mut stack_args = apple_proto::mil_spec::Argument { arguments: vec![] };
            for v in values {
                stack_args.arguments.push(apple_proto::mil_spec::argument::Binding {
                    binding: Some(apple_proto::mil_spec::argument::binding::Binding::Name(
                        v.clone(),
                    )),
                });
            }
            inputs.insert("values".to_string(), stack_args);

            // axis as immediate value
            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "stack".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Softmax { name, x, axis } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "softmax".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Gelu { name, x, mode } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            // mode as immediate string value
            inputs.insert(
                "mode".to_string(),
                make_value_arg(apple_proto::mil_spec::Value {
                    doc_string: String::new(),
                    r#type: Some(apple_proto::mil_spec::ValueType {
                        r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(
                            apple_proto::mil_spec::TensorType {
                                data_type: apple_proto::mil_spec::DataType::String as i32,
                                rank: 0,
                                dimensions: vec![],
                                attributes: HashMap::new(),
                            },
                        )),
                    }),
                    value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
                        apple_proto::mil_spec::value::ImmediateValue {
                            value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                                apple_proto::mil_spec::TensorValue {
                                    value: Some(apple_proto::mil_spec::tensor_value::Value::Strings(
                                        apple_proto::mil_spec::tensor_value::RepeatedStrings {
                                            values: vec![mode.clone()],
                                        },
                                    )),
                                },
                            )),
                        },
                    )),
                }),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "gelu".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ScaledDotProductAttention {
            name,
            query,
            key,
            value,
            attention_mask,
            scale,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("query".to_string(), make_name_arg(query));
            inputs.insert("key".to_string(), make_name_arg(key));
            inputs.insert("value".to_string(), make_name_arg(value));
            if let Some(mask) = attention_mask {
                inputs.insert("attn_mask".to_string(), make_name_arg(mask));
            }
            if let Some(s) = scale {
                inputs
                    .insert("scale".to_string(), make_value_arg(make_immediate_float32_value(*s)));
            }

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "scaled_dot_product_attention".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ReadState { name, state_id, shape, dtype } => {
            let apple_dtype = mil_dtype_to_apple(dtype);
            let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            let mut inputs = HashMap::new();
            // Core ML schema: mb.read_state(input=k_state)
            // The parameter name is "input", NOT "state".
            // Using "state" causes: Invalid param name 'state'.
            inputs.insert("input".to_string(), make_name_arg(state_id));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "read_state".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(name, apple_dtype, &shape_u64)],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::CoremlUpdateState { name, state_id, value } => {
            let mut inputs = HashMap::new();
            // Core ML serialized MIL schema: write_state(state=k_state, value=updated_k)
            // The Python builder API uses mb.coreml_update_state(), but when
            // serialized to the ML Program protobuf, the op is lowered to "write_state".
            // Using "coreml_update_state" causes: Unknown operator 'coreml_update_state'.
            // The parameter names "state" and "value" are correct for write_state.
            inputs.insert("state".to_string(), make_name_arg(state_id));
            inputs.insert("value".to_string(), make_name_arg(value));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "write_state".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Gather { name, x, indices, axis } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("indices".to_string(), make_name_arg(indices));

            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );
            inputs.insert(
                "validate_indices".to_string(),
                make_value_arg(make_immediate_bool_value(false)),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "gather".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ReduceMean { name, x, axes, keep_dims } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );

            inputs.insert(
                "keep_dims".to_string(),
                make_value_arg(make_immediate_bool_value(*keep_dims)),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "reduce_mean".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ReduceSum { name, x, axes, keep_dims } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );

            inputs.insert(
                "keep_dims".to_string(),
                make_value_arg(make_immediate_bool_value(*keep_dims)),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "reduce_sum".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Conv {
            name,
            x,
            weight,
            pad_type,
            groups,
            kernel_scale,
            kernel_zero_point,
            kernel_palettized_lut,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("weight".to_string(), make_name_arg(weight));

            inputs.insert(
                "pad_type".to_string(),
                make_value_arg(apple_proto::mil_spec::Value {
                    doc_string: String::new(),
                    r#type: Some(apple_proto::mil_spec::ValueType {
                        r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(
                            apple_proto::mil_spec::TensorType {
                                data_type: apple_proto::mil_spec::DataType::String as i32,
                                rank: 0,
                                dimensions: vec![],
                                attributes: HashMap::new(),
                            },
                        )),
                    }),
                    value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
                        apple_proto::mil_spec::value::ImmediateValue {
                            value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                                apple_proto::mil_spec::TensorValue {
                                    value: Some(apple_proto::mil_spec::tensor_value::Value::Strings(
                                        apple_proto::mil_spec::tensor_value::RepeatedStrings {
                                            values: vec![pad_type.clone()],
                                        },
                                    )),
                                },
                            )),
                        },
                    )),
                }),
            );

            inputs.insert(
                "groups".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*groups as i32], &[])),
            );

            // T-98 (V-110): Emit quantized conv weight attributes as named const nodes.
            // ANEC requires kernel_scale, kernel_zero_point, and kernel_palettized_LUT
            // for quantized/palettized conv operations. Following the same named-const-node
            // pattern as Cast dtype and MatMul transpose flags (Orion #12), we emit
            // preceding const operations for each quantization parameter and reference
            // them by name in the conv op's inputs map.
            let mut const_ops: Vec<apple_proto::mil_spec::Operation> = Vec::new();

            if let Some(scale) = kernel_scale {
                let scale_const_name = format!("{name}_kernel_scale_0");
                let mut scale_attrs = HashMap::new();
                add_name_attribute(&mut scale_attrs, &scale_const_name);
                scale_attrs.insert("val".to_string(), make_immediate_float32_value(*scale));
                let scale_const_op = apple_proto::mil_spec::Operation {
                    r#type: "const".to_string(),
                    inputs: HashMap::new(),
                    outputs: vec![make_apple_named_value_type(
                        &scale_const_name,
                        apple_proto::mil_spec::DataType::Float32 as i32,
                        &[],
                    )],
                    blocks: vec![],
                    attributes: scale_attrs,
                };
                inputs.insert("kernel_scale".to_string(), make_name_arg(&scale_const_name));
                const_ops.push(scale_const_op);
            }

            if let Some(zp) = kernel_zero_point {
                let zp_const_name = format!("{name}_kernel_zero_point_0");
                let mut zp_attrs = HashMap::new();
                add_name_attribute(&mut zp_attrs, &zp_const_name);
                zp_attrs.insert("val".to_string(), make_immediate_int32_value(vec![*zp], &[]));
                let zp_const_op = apple_proto::mil_spec::Operation {
                    r#type: "const".to_string(),
                    inputs: HashMap::new(),
                    outputs: vec![make_apple_named_value_type(
                        &zp_const_name,
                        apple_proto::mil_spec::DataType::Int32 as i32,
                        &[],
                    )],
                    blocks: vec![],
                    attributes: zp_attrs,
                };
                inputs.insert("kernel_zero_point".to_string(), make_name_arg(&zp_const_name));
                const_ops.push(zp_const_op);
            }

            if let Some(lut_name) = kernel_palettized_lut {
                let lut_const_name = format!("{name}_kernel_palettized_LUT_0");
                let mut lut_attrs = HashMap::new();
                add_name_attribute(&mut lut_attrs, &lut_const_name);
                // The LUT references a weight entry by name — emit as a named const
                // that resolves to the palettized LUT weight data.
                lut_attrs.insert("val".to_string(), make_immediate_string_value(lut_name.clone()));
                let lut_const_op = apple_proto::mil_spec::Operation {
                    r#type: "const".to_string(),
                    inputs: HashMap::new(),
                    outputs: vec![make_apple_named_value_type(
                        &lut_const_name,
                        apple_proto::mil_spec::DataType::String as i32,
                        &[],
                    )],
                    blocks: vec![],
                    attributes: lut_attrs,
                };
                inputs.insert("kernel_palettized_LUT".to_string(), make_name_arg(&lut_const_name));
                const_ops.push(lut_const_op);
            }

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            let conv_op = apple_proto::mil_spec::Operation {
                r#type: "conv".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            };

            // Return const ops first, then the conv op
            let mut result = const_ops;
            result.push(conv_op);
            result
        }
        mir_compat::MirOpCompat::StateWrite { name, state_ref, value } => {
            let mut inputs = HashMap::new();
            // Core ML serialized MIL schema: write_state(state=..., value=...)
            // Same lowered form as CoremlUpdateState — the Python builder API
            // mb.coreml_update_state() is lowered to write_state when serialized.
            inputs.insert("state".to_string(), make_name_arg(state_ref));
            inputs.insert("value".to_string(), make_name_arg(value));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "write_state".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Rsqrt { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("epsilon".to_string(), make_value_arg(make_immediate_float32_value(0.0)));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "rsqrt".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::RealDiv { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "real_div".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::LayerNorm { name, x, weight_name, bias_name, epsilon, axes } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("gamma".to_string(), make_name_arg(weight_name));
            if let Some(bname) = bias_name {
                inputs.insert("beta".to_string(), make_name_arg(bname));
            }

            inputs.insert(
                "epsilon".to_string(),
                make_value_arg(make_immediate_float32_value(*epsilon)),
            );

            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "layer_norm".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Topk { name, x, k, axis } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            inputs.insert(
                "k".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*k as i32], &[])),
            );

            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "topk".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Cos { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "cos".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Sin { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "sin".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Cast { name, x, dtype } => {
            let apple_dtype = mil_dtype_to_apple(dtype);

            // Apple's format: dtype input references a separate const op by name
            // (not an immediate value). We emit a preceding const op for the dtype string.
            let dtype_str = match dtype {
                mir_compat::MilDtypeCompat::Fp16 => "fp16",
                mir_compat::MilDtypeCompat::Fp32 => "fp32",
                mir_compat::MilDtypeCompat::Int32 => "int32",
                mir_compat::MilDtypeCompat::UInt8 => "uint8",
                // T-35: new dtype strings for Cast op
                mir_compat::MilDtypeCompat::Int4 => "int4",
                mir_compat::MilDtypeCompat::UInt4 => "uint4",
                mir_compat::MilDtypeCompat::E4M3 => "e4m3",
                mir_compat::MilDtypeCompat::E5M2 => "e5m2",
                mir_compat::MilDtypeCompat::UInt16 => "uint16",
            };
            let dtype_const_name = format!("{name}_dtype_0");

            // Emit the dtype const op
            let mut dtype_const_attrs = HashMap::new();
            add_name_attribute(&mut dtype_const_attrs, &dtype_const_name);
            dtype_const_attrs.insert(
                "val".to_string(),
                apple_proto::mil_spec::Value {
                    doc_string: String::new(),
                    r#type: Some(apple_proto::mil_spec::ValueType {
                        r#type: Some(apple_proto::mil_spec::value_type::Type::TensorType(
                            apple_proto::mil_spec::TensorType {
                                data_type: apple_proto::mil_spec::DataType::String as i32,
                                rank: 0,
                                dimensions: vec![],
                                attributes: HashMap::new(),
                            },
                        )),
                    }),
                    value: Some(apple_proto::mil_spec::value::Value::ImmediateValue(
                        apple_proto::mil_spec::value::ImmediateValue {
                            value: Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(
                                apple_proto::mil_spec::TensorValue {
                                    value: Some(apple_proto::mil_spec::tensor_value::Value::Strings(
                                        apple_proto::mil_spec::tensor_value::RepeatedStrings {
                                            values: vec![dtype_str.to_string()],
                                        },
                                    )),
                                },
                            )),
                        },
                    )),
                },
            );

            let dtype_const_op = apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &dtype_const_name,
                    apple_proto::mil_spec::DataType::String as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: dtype_const_attrs,
            };

            // Emit the cast op referencing the dtype by name
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("dtype".to_string(), make_name_arg(&dtype_const_name));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            let cast_op = apple_proto::mil_spec::Operation {
                r#type: "cast".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_dtype,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            };

            vec![dtype_const_op, cast_op]
        }
        mir_compat::MirOpCompat::Split { name, x, axis, num_splits } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            inputs.insert(
                "axis".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*axis as i32], &[])),
            );

            inputs.insert(
                "num_splits".to_string(),
                make_value_arg(make_immediate_int32_value(vec![*num_splits as i32], &[])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "split".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Exp { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "exp".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Sigmoid { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "sigmoid".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Tanh { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "tanh".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Relu { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "relu".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // Where → Select: CoreML MIL has no "where" op; use "select" instead.
        // The MILWhere→MILSelect rewrite in mil_lower.rs should make this
        // unreachable at runtime, but we handle it defensively.
        mir_compat::MirOpCompat::Where { name, condition, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("cond".to_string(), make_name_arg(condition));
            inputs.insert("a".to_string(), make_name_arg(x));
            inputs.insert("b".to_string(), make_name_arg(y));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "select".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Silu { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "silu".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Identity { name, x, dtype } => {
            let mil_dtype = compat_dtype_to_apple_mil(dtype);

            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "identity".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    mil_dtype,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Placeholder { name: _, dtype: _ } => {
            // Placeholder is NOT a Core ML MIL operation. It's a marker for
            // graph inputs that gets stripped during proto emission. Function
            // inputs are declared as block parameters (NamedValueType in
            // Function.inputs / Block.inputs), not as operations. Core ML
            // rejects the "placeholder" operator with "Unknown operator".
            vec![]
        }
        mir_compat::MirOpCompat::Tile { name, x, reps } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            // reps as INT32 immediate — same ios19 dtype restriction as reshape.shape
            inputs.insert(
                "reps".to_string(),
                make_value_arg(make_immediate_int32_value(reps.clone(), &[reps.len() as u64])),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "tile".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Fill { name, shape, value, dtype } => {
            // `fill` is ANE-illegal (no ANE converter).
            // Decompose to ANE-legal ops using the same pattern as FillLike:
            //   fill(shape, val) → const(zeros_of_shape) + add(zeros, val)
            //
            // Since MILFill should never reach the proto emitter (the mil_lower
            // pass replaces AirOp::Fill with MILConst), this path is defensive.
            // If Fill DOES reach here, we decompose it to avoid CPU fallback.
            let apple_dtype = mil_dtype_to_apple(dtype);
            let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

            // Step 1: Create a zeros const of the target shape.
            // We use a `const` op with an immediate zero tensor.
            // For efficiency, use a scalar 0.0 that broadcasts to the target shape.
            let zero_name = format!("{name}__zero");
            let mut const_inputs = HashMap::new();
            const_inputs.insert(
                "val".to_string(),
                make_value_arg(make_immediate_float_value(apple_dtype, 0.0)),
            );
            let mut const_attrs = HashMap::new();
            add_name_attribute(&mut const_attrs, &zero_name);

            let const_op = apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: const_inputs,
                outputs: vec![make_apple_named_value_type(&zero_name, apple_dtype, &shape_u64)],
                blocks: vec![],
                attributes: const_attrs,
            };

            // Step 2: add(zeros, val) → fills the tensor with val via broadcast.
            let mut add_inputs = HashMap::new();
            add_inputs.insert("x".to_string(), make_name_arg(&zero_name));
            add_inputs.insert(
                "y".to_string(),
                make_value_arg(make_immediate_float_value(apple_dtype, *value)),
            );
            let mut add_attrs = HashMap::new();
            add_name_attribute(&mut add_attrs, name);

            let add_op = apple_proto::mil_spec::Operation {
                r#type: "add".to_string(),
                inputs: add_inputs,
                outputs: vec![make_apple_named_value_type(name, apple_dtype, &shape_u64)],
                blocks: vec![],
                attributes: add_attrs,
            };

            vec![const_op, add_op]
        }
        mir_compat::MirOpCompat::FillLike { name, ref_tensor, value, dtype } => {
            // fill_like is ANE-illegal (Core ML cannot provision it on the Neural Engine).
            // Decompose to ANE-legal ops:
            //   fill_like(ref, val) → mul(ref, 0) + add(zero, val)
            //
            // Step 1: zero_tensor = mul(ref_tensor, 0)
            //   Creates a zero tensor with the same shape as ref_tensor.
            // Step 2: eps_tensor = add(zero_tensor, val)
            //   Fills the zero tensor with val via broadcasting.
            //
            // This matches the approach used in the reference CoreML export at
            // https://huggingface.co/pkhairkh/qwen3-coreml-palettized which avoids
            // fill/fill_like entirely for ANE compatibility.
            let apple_dtype = mil_dtype_to_apple(dtype);
            let shape = lookup_shape_u64(name, node_shapes);

            // Step 1: zero = mul(ref_tensor, 0)
            let zero_name = format!("{name}__zero");
            let mut mul_inputs = HashMap::new();
            mul_inputs.insert("x".to_string(), make_name_arg(ref_tensor));
            mul_inputs.insert(
                "y".to_string(),
                make_value_arg(make_immediate_float_value(apple_dtype, 0.0)),
            );

            let mut mul_attrs = HashMap::new();
            add_name_attribute(&mut mul_attrs, &zero_name);

            let mul_op = apple_proto::mil_spec::Operation {
                r#type: "mul".to_string(),
                inputs: mul_inputs,
                outputs: vec![make_apple_named_value_type(&zero_name, apple_dtype, &shape)],
                blocks: vec![],
                attributes: mul_attrs,
            };

            // Step 2: eps = add(zero, epsilon)
            let mut add_inputs = HashMap::new();
            add_inputs.insert("x".to_string(), make_name_arg(&zero_name));
            add_inputs.insert(
                "y".to_string(),
                make_value_arg(make_immediate_float_value(apple_dtype, *value)),
            );

            let mut add_attrs = HashMap::new();
            add_name_attribute(&mut add_attrs, name);

            let add_op = apple_proto::mil_spec::Operation {
                r#type: "add".to_string(),
                inputs: add_inputs,
                outputs: vec![make_apple_named_value_type(name, apple_dtype, &shape)],
                blocks: vec![],
                attributes: add_attrs,
            };

            vec![mul_op, add_op]
        }
        mir_compat::MirOpCompat::Neg { name, x } => {
            // Core ML does NOT have a "neg" op. Lower to mul(x, -1).
            // This matches coremltools' own Torch frontend lowering:
            //   neg(x) → mb.mul(x=x, y=-1)
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            // y = -1.0 as FP16 immediate scalar (broadcast-compatible)
            inputs.insert(
                "y".to_string(),
                make_value_arg(make_immediate_float_value(
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    -1.0,
                )),
            );

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: "mul".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── New unary ops ───
        mir_compat::MirOpCompat::Sqrt { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "sqrt".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::LogicalNot { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "logical_not".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Ceil { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "ceil".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Floor { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "floor".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Round { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "round".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Sign { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "sign".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Log { name, x } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "log".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── New binary ops ───
        mir_compat::MirOpCompat::Pow { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "pow".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Equal { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "equal".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::NotEqual { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "not_equal".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Greater { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "greater".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::GreaterEqual { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "greater_equal".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Less { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "less".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::LessEqual { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "less_equal".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::LogicalAnd { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "logical_and".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::LogicalOr { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "logical_or".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Bool as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::FloorDiv { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "floor_div".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Mod { name, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert("y".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "mod".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── ExpandDims ───
        mir_compat::MirOpCompat::ExpandDims { name, x, axis } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(axis.clone(), &[axis.len() as u64])),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "expand_dims".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── Squeeze ───
        mir_compat::MirOpCompat::Squeeze { name, x, axis } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(axis.clone(), &[axis.len() as u64])),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "squeeze".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── Clip ───
        mir_compat::MirOpCompat::Clip { name, x, min_val, max_val } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "alpha".to_string(),
                make_value_arg(make_immediate_float32_value(*min_val)),
            );
            inputs
                .insert("beta".to_string(), make_value_arg(make_immediate_float32_value(*max_val)));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "clip".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── Pad ───
        mir_compat::MirOpCompat::Pad { name, x, pad_amounts, mode, constant_value } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "pad".to_string(),
                make_value_arg(make_immediate_int32_value(
                    pad_amounts.clone(),
                    &[pad_amounts.len() as u64],
                )),
            );
            inputs.insert(
                "mode".to_string(),
                make_value_arg(make_immediate_string_value(mode.clone())),
            );
            inputs.insert(
                "constant_value".to_string(),
                make_value_arg(make_immediate_float32_value(*constant_value)),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "pad".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── ReduceMax ───
        mir_compat::MirOpCompat::ReduceMax { name, x, axes, keep_dims } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );
            inputs.insert(
                "keep_dims".to_string(),
                make_value_arg(make_immediate_bool_value(*keep_dims)),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "reduce_max".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── ReduceMin ───
        mir_compat::MirOpCompat::ReduceMin { name, x, axes, keep_dims } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );
            inputs.insert(
                "keep_dims".to_string(),
                make_value_arg(make_immediate_bool_value(*keep_dims)),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "reduce_min".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── ReduceProd ───
        mir_compat::MirOpCompat::ReduceProd { name, x, axes, keep_dims } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs.insert(
                "axes".to_string(),
                make_value_arg(make_immediate_int32_value(
                    axes.iter().map(|&v| v as i32).collect(),
                    &[axes.len() as u64],
                )),
            );
            inputs.insert(
                "keep_dims".to_string(),
                make_value_arg(make_immediate_bool_value(*keep_dims)),
            );
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "reduce_prod".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── Select ───
        mir_compat::MirOpCompat::Select { name, condition, x, y } => {
            let mut inputs = HashMap::new();
            inputs.insert("cond".to_string(), make_name_arg(condition));
            inputs.insert("a".to_string(), make_name_arg(x));
            inputs.insert("b".to_string(), make_name_arg(y));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "select".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── LeakyRelu ───
        mir_compat::MirOpCompat::LeakyRelu { name, x, alpha } => {
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(x));
            inputs
                .insert("alpha".to_string(), make_value_arg(make_immediate_float32_value(*alpha)));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            vec![apple_proto::mil_spec::Operation {
                r#type: "leaky_relu".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // T-39 (I-18): Constexpr weight compression ops — Apple proto emission.
        // Emitted as constexpr_* op_type with weight-name inputs and scalar attributes.
        // The primary input (quantized_data/data/indices/nonzero_data) is wired as
        // the "x" input; additional weight references are wired as named inputs.
        mir_compat::MirOpCompat::ConstexprAffineDequantize {
            name,
            quantized_data,
            scale,
            zero_point,
            axis,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("quantized_data".to_string(), make_name_arg(quantized_data));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert("scale".to_string(), make_immediate_float32_value(*scale));
            attributes.insert(
                "zero_point".to_string(),
                make_immediate_int64_value(vec![*zero_point as i64], &[]),
            );
            attributes.insert("axis".to_string(), make_immediate_int64_value(vec![*axis], &[]));
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_affine_dequantize".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprBlockwiseShiftScale {
            name,
            data,
            scale,
            offset,
            block_size,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("data".to_string(), make_name_arg(data));
            inputs.insert("scale".to_string(), make_name_arg(scale));
            inputs.insert("offset".to_string(), make_name_arg(offset));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert(
                "block_size".to_string(),
                make_immediate_int64_value(block_size.clone(), &[block_size.len() as u64]),
            );
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_blockwise_shift_scale".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprLutToDense { name, indices, lut, num_bits } => {
            let mut inputs = HashMap::new();
            inputs.insert("indices".to_string(), make_name_arg(indices));
            inputs.insert("lut".to_string(), make_name_arg(lut));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes
                .insert("num_bits".to_string(), make_immediate_int64_value(vec![*num_bits], &[]));
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_lut_to_dense".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprSparseToDense {
            name,
            nonzero_data,
            shape,
            default_value,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("nonzero_data".to_string(), make_name_arg(nonzero_data));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert(
                "shape".to_string(),
                make_immediate_int64_value(shape.clone(), &[shape.len() as u64]),
            );
            attributes
                .insert("default_value".to_string(), make_immediate_float32_value(*default_value));
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_sparse_to_dense".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprCast { name, data, dtype } => {
            let mut inputs = HashMap::new();
            inputs.insert("data".to_string(), make_name_arg(data));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert(
                "dtype".to_string(),
                make_immediate_int64_value(vec![compat_dtype_to_apple_mil(dtype) as i64], &[]),
            );
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_cast".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    compat_dtype_to_apple_mil(dtype),
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprLutToSparse { name, data, num_bits } => {
            let mut inputs = HashMap::new();
            inputs.insert("data".to_string(), make_name_arg(data));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes
                .insert("num_bits".to_string(), make_immediate_int64_value(vec![*num_bits], &[]));
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_lut_to_sparse".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::ConstexprSparseBlockwiseShiftScale {
            name,
            data,
            scale,
            offset,
            block_size,
            block_axis,
        } => {
            let mut inputs = HashMap::new();
            inputs.insert("data".to_string(), make_name_arg(data));
            inputs.insert("scale".to_string(), make_name_arg(scale));
            inputs.insert("offset".to_string(), make_name_arg(offset));
            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);
            attributes.insert(
                "block_size".to_string(),
                make_immediate_int64_value(block_size.clone(), &[block_size.len() as u64]),
            );
            attributes.insert(
                "block_axis".to_string(),
                make_immediate_int64_value(vec![*block_axis], &[]),
            );
            vec![apple_proto::mil_spec::Operation {
                r#type: "constexpr_sparse_blockwise_shift_scale".to_string(),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        mir_compat::MirOpCompat::Unsupported { op_kind, name, params_json: _, inputs: _ } => {
            // Emit as identity to preserve graph structure
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(name));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, name);

            vec![apple_proto::mil_spec::Operation {
                r#type: format!("identity__unsupported_{op_kind}"),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
        // ─── F-OPS-01: ANEC ops promoted from CPU-only ──────────────
        // AnecFusedConvActivate: decompose to conv + activation.
        // Core ML proto doesn't have a fused conv+activate op, so we emit
        // a conv operation followed by an activation operation.
        mir_compat::MirOpCompat::AnecFusedConvActivate {
            name,
            x,
            weight,
            activation,
            strides,
            pad_amounts,
            dilations,
            groups,
        } => {
            let mut ops: Vec<apple_proto::mil_spec::Operation> = Vec::new();

            // Emit preceding const ops for conv parameters
            let stride_const_name = format!("{name}_stride");
            let stride_shape: Vec<u64> = vec![strides.len() as u64];
            let mut stride_attrs = HashMap::new();
            add_name_attribute(&mut stride_attrs, &stride_const_name);
            stride_attrs.insert(
                "val".to_string(),
                make_immediate_int32_value(strides.to_vec(), &stride_shape),
            );
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &stride_const_name,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    &stride_shape,
                )],
                blocks: vec![],
                attributes: stride_attrs,
            });

            let pad_const_name = format!("{name}_pad");
            let pad_shape: Vec<u64> = vec![pad_amounts.len() as u64];
            let mut pad_attrs = HashMap::new();
            add_name_attribute(&mut pad_attrs, &pad_const_name);
            pad_attrs.insert(
                "val".to_string(),
                make_immediate_int32_value(pad_amounts.to_vec(), &pad_shape),
            );
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &pad_const_name,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    &pad_shape,
                )],
                blocks: vec![],
                attributes: pad_attrs,
            });

            let dil_const_name = format!("{name}_dilation");
            let dil_shape: Vec<u64> = vec![dilations.len() as u64];
            let mut dil_attrs = HashMap::new();
            add_name_attribute(&mut dil_attrs, &dil_const_name);
            dil_attrs.insert(
                "val".to_string(),
                make_immediate_int32_value(dilations.to_vec(), &dil_shape),
            );
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &dil_const_name,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    &dil_shape,
                )],
                blocks: vec![],
                attributes: dil_attrs,
            });

            let groups_const_name = format!("{name}_groups");
            let mut groups_attrs = HashMap::new();
            add_name_attribute(&mut groups_attrs, &groups_const_name);
            groups_attrs
                .insert("val".to_string(), make_immediate_int32_value(vec![*groups as i32], &[]));
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &groups_const_name,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: groups_attrs,
            });

            // Emit the conv operation
            let conv_output_name = format!("{name}_conv");
            let mut conv_inputs = HashMap::new();
            conv_inputs.insert("x".to_string(), make_name_arg(x));
            conv_inputs.insert("weight".to_string(), make_name_arg(weight));
            conv_inputs.insert("stride".to_string(), make_name_arg(&stride_const_name));
            conv_inputs.insert("pad".to_string(), make_name_arg(&pad_const_name));
            conv_inputs.insert("dilation".to_string(), make_name_arg(&dil_const_name));
            conv_inputs.insert("groups".to_string(), make_name_arg(&groups_const_name));

            let mut conv_attrs = HashMap::new();
            add_name_attribute(&mut conv_attrs, &conv_output_name);

            ops.push(apple_proto::mil_spec::Operation {
                r#type: "conv".to_string(),
                inputs: conv_inputs,
                outputs: vec![make_apple_named_value_type(
                    &conv_output_name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes: conv_attrs,
            });

            // Emit the activation operation following the conv
            // The activation type is determined by the `activation` field
            let act_type = match activation.as_str() {
                "relu" => "relu",
                "sigmoid" => "sigmoid",
                "tanh" => "tanh",
                "silu" => "silu",
                "gelu" => "gelu",
                "linear" => "identity", // linear activation = no-op
                other => other,         // fallback: use the activation name directly
            };

            let mut act_inputs = HashMap::new();
            act_inputs.insert("x".to_string(), make_name_arg(&conv_output_name));

            let mut act_attrs = HashMap::new();
            add_name_attribute(&mut act_attrs, name);

            ops.push(apple_proto::mil_spec::Operation {
                r#type: act_type.to_string(),
                inputs: act_inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes: act_attrs,
            });

            ops
        }

        // AnecScaledElementwise: decompose to mul(x, scale_a) + mul(y, scale_b).
        // Core ML proto doesn't have a native scaled elementwise op.
        mir_compat::MirOpCompat::AnecScaledElementwise { name, x, y, scale_a, scale_b } => {
            let mut ops: Vec<apple_proto::mil_spec::Operation> = Vec::new();

            // Emit const for scale_a
            let scale_a_const_name = format!("{name}_scale_a");
            let mut scale_a_attrs = HashMap::new();
            add_name_attribute(&mut scale_a_attrs, &scale_a_const_name);
            scale_a_attrs.insert("val".to_string(), make_immediate_float32_value(*scale_a));
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &scale_a_const_name,
                    apple_proto::mil_spec::DataType::Float32 as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: scale_a_attrs,
            });

            // Emit const for scale_b
            let scale_b_const_name = format!("{name}_scale_b");
            let mut scale_b_attrs = HashMap::new();
            add_name_attribute(&mut scale_b_attrs, &scale_b_const_name);
            scale_b_attrs.insert("val".to_string(), make_immediate_float32_value(*scale_b));
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "const".to_string(),
                inputs: HashMap::new(),
                outputs: vec![make_apple_named_value_type(
                    &scale_b_const_name,
                    apple_proto::mil_spec::DataType::Float32 as i32,
                    &[],
                )],
                blocks: vec![],
                attributes: scale_b_attrs,
            });

            // Emit mul(x, scale_a)
            let mul_a_output_name = format!("{name}_scaled_x");
            let mut mul_a_inputs = HashMap::new();
            mul_a_inputs.insert("x".to_string(), make_name_arg(x));
            mul_a_inputs.insert("y".to_string(), make_name_arg(&scale_a_const_name));
            let mut mul_a_attrs = HashMap::new();
            add_name_attribute(&mut mul_a_attrs, &mul_a_output_name);
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "mul".to_string(),
                inputs: mul_a_inputs,
                outputs: vec![make_apple_named_value_type(
                    &mul_a_output_name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes: mul_a_attrs,
            });

            // Emit mul(y, scale_b)
            let mul_b_output_name = format!("{name}_scaled_y");
            let mut mul_b_inputs = HashMap::new();
            mul_b_inputs.insert("x".to_string(), make_name_arg(y));
            mul_b_inputs.insert("y".to_string(), make_name_arg(&scale_b_const_name));
            let mut mul_b_attrs = HashMap::new();
            add_name_attribute(&mut mul_b_attrs, &mul_b_output_name);
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "mul".to_string(),
                inputs: mul_b_inputs,
                outputs: vec![make_apple_named_value_type(
                    &mul_b_output_name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes: mul_b_attrs,
            });

            // Emit add(scaled_x, scaled_y)
            let mut add_inputs = HashMap::new();
            add_inputs.insert("x".to_string(), make_name_arg(&mul_a_output_name));
            add_inputs.insert("y".to_string(), make_name_arg(&mul_b_output_name));
            let mut add_attrs = HashMap::new();
            add_name_attribute(&mut add_attrs, name);
            ops.push(apple_proto::mil_spec::Operation {
                r#type: "add".to_string(),
                inputs: add_inputs,
                outputs: vec![make_apple_named_value_type(
                    name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(name, node_shapes),
                )],
                blocks: vec![],
                attributes: add_attrs,
            });

            ops
        }

        // T-66 (I-40): Pooling, spatial rearrangement, normalization, and
        // quantize/dequantize ops — Apple proto path fallback until dedicated
        // emission is implemented.
        op @ mir_compat::MirOpCompat::MaxPool { .. }
        | op @ mir_compat::MirOpCompat::AvgPool { .. }
        | op @ mir_compat::MirOpCompat::L2Pool { .. }
        | op @ mir_compat::MirOpCompat::DepthToSpace { .. }
        | op @ mir_compat::MirOpCompat::SpaceToDepth { .. }
        | op @ mir_compat::MirOpCompat::PixelShuffle { .. }
        | op @ mir_compat::MirOpCompat::PixelUnshuffle { .. }
        | op @ mir_compat::MirOpCompat::BatchNorm { .. }
        | op @ mir_compat::MirOpCompat::InstanceNorm { .. }
        | op @ mir_compat::MirOpCompat::L2Norm { .. }
        | op @ mir_compat::MirOpCompat::Quantize { .. }
        | op @ mir_compat::MirOpCompat::Dequantize { .. } => {
            let op_name = op.output_name();
            let op_kind = match op {
                mir_compat::MirOpCompat::MaxPool { .. } => "max_pool",
                mir_compat::MirOpCompat::AvgPool { .. } => "avg_pool",
                mir_compat::MirOpCompat::L2Pool { .. } => "l2_pool",
                mir_compat::MirOpCompat::DepthToSpace { .. } => "depth_to_space",
                mir_compat::MirOpCompat::SpaceToDepth { .. } => "space_to_depth",
                mir_compat::MirOpCompat::PixelShuffle { .. } => "pixel_shuffle",
                mir_compat::MirOpCompat::PixelUnshuffle { .. } => "pixel_unshuffle",
                mir_compat::MirOpCompat::BatchNorm { .. } => "batch_norm",
                mir_compat::MirOpCompat::InstanceNorm { .. } => "instance_norm",
                mir_compat::MirOpCompat::L2Norm { .. } => "l2_norm",
                mir_compat::MirOpCompat::Quantize { .. } => "quantize",
                mir_compat::MirOpCompat::Dequantize { .. } => "dequantize",
                _ => unreachable!(),
            };
            let mut inputs = HashMap::new();
            inputs.insert("x".to_string(), make_name_arg(&op_name));

            let mut attributes = HashMap::new();
            add_name_attribute(&mut attributes, &op_name);

            vec![apple_proto::mil_spec::Operation {
                r#type: format!("identity__unsupported_{op_kind}"),
                inputs,
                outputs: vec![make_apple_named_value_type(
                    &op_name,
                    apple_proto::mil_spec::DataType::Float16 as i32,
                    &lookup_shape_u64(&op_name, node_shapes),
                )],
                blocks: vec![],
                attributes,
            }]
        }
    }
}

// ─── Proto Validation Error Type ──────────────────────────────────

/// Error type for proto validation failures.
///
/// These represent user-facing errors where the generated proto would be
/// rejected by Core ML's runtime — dtype mismatches, shape inconsistencies,
/// incompatible dimensions, etc. Previously these were `panic!()` calls,
/// which is inappropriate for user-facing errors that should be reported
/// gracefully rather than crashing the compiler process.
#[derive(Debug, Clone)]
pub struct ProtoValidationError {
    /// Which validation check failed.
    pub kind: ProtoValidationKind,
    /// Human-readable diagnostic message with remediation hints.
    pub message: String,
    /// Name of the operation that triggered the error (if available).
    pub op_name: Option<String>,
}

/// Category of proto validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoValidationKind {
    /// Fill/FillLike value dtype differs from output dtype.
    FillDtypeMismatch,
    /// FLOAT16 fill value stored via `floats` field instead of `bytes`.
    FillFp16FloatsStorage,
    /// FLOAT32 fill value stored via `bytes` field instead of `floats`.
    FillFp32BytesStorage,
    /// Elementwise binary op output shape doesn't match broadcast result.
    ElementwiseBroadcastShapeMismatch,
    /// MatMul inner dimensions don't match (lhs_cols != rhs_rows).
    MatMulInnerDimMismatch,
    /// MatMul output shape doesn't match inferred batched matmul result.
    MatMulOutputShapeMismatch,
}

impl std::fmt::Display for ProtoValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(op) = &self.op_name {
            write!(f, "[{}] {}: {}", self.kind, op, self.message)
        } else {
            write!(f, "[{}] {}", self.kind, self.message)
        }
    }
}

impl std::fmt::Display for ProtoValidationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtoValidationKind::FillDtypeMismatch => write!(f, "FillDtypeMismatch"),
            ProtoValidationKind::FillFp16FloatsStorage => write!(f, "FillFp16FloatsStorage"),
            ProtoValidationKind::FillFp32BytesStorage => write!(f, "FillFp32BytesStorage"),
            ProtoValidationKind::ElementwiseBroadcastShapeMismatch => {
                write!(f, "ElementwiseBroadcastShapeMismatch")
            }
            ProtoValidationKind::MatMulInnerDimMismatch => write!(f, "MatMulInnerDimMismatch"),
            ProtoValidationKind::MatMulOutputShapeMismatch => {
                write!(f, "MatMulOutputShapeMismatch")
            }
        }
    }
}

impl std::error::Error for ProtoValidationError {}

/// Validate that every `fill` and `fill_like` operation has a value parameter whose
/// declared dtype matches the output dtype.  Core ML's parser rejects the model
/// otherwise with: "In 'fill' operations, tensors parameter value[0], and output at
/// index 0 must have the same data type."
///
/// Also validates that immediate tensor storage is compatible with the declared dtype:
/// - `FLOAT16` tensors must use `bytes` storage (not `floats`), because Core ML's
///   ML Program parser rejects them with "Tensor storage and type have different
///   number of elements".
/// - `FLOAT32` tensors must use `floats` storage.
fn validate_fill_dtype_consistency(
    operations: &[apple_proto::mil_spec::Operation],
) -> Result<(), ProtoValidationError> {
    let float16_dtype = apple_proto::mil_spec::DataType::Float16 as i32;
    let float32_dtype = apple_proto::mil_spec::DataType::Float32 as i32;

    for op in operations {
        if op.r#type != "fill" && op.r#type != "fill_like" {
            continue;
        }
        let op_name = op
            .attributes
            .get("name")
            .and_then(|v| v.value.as_ref())
            .and_then(|iv| match iv {
                apple_proto::mil_spec::value::Value::ImmediateValue(imv) => imv.value.as_ref(),
                _ => None,
            })
            .and_then(|v| match v {
                apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv) => {
                    match tv.value.as_ref() {
                        Some(apple_proto::mil_spec::tensor_value::Value::Strings(rs)) => {
                            rs.values.first().cloned()
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .unwrap_or_else(|| format!("<unnamed {}>", op.r#type));

        // Extract value parameter: inputs["value"] → Argument → first Binding → Value
        let value_entry =
            op.inputs.get("value").and_then(|arg| arg.arguments.first()).and_then(|binding| {
                match &binding.binding {
                    Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => Some(v),
                    _ => None,
                }
            });

        // Extract declared dtype from the value parameter's ValueType
        let value_dtype = value_entry
            .as_ref()
            .and_then(|v| v.r#type.as_ref())
            .and_then(|vt| vt.r#type.as_ref())
            .and_then(|t| match t {
                apple_proto::mil_spec::value_type::Type::TensorType(tt) => Some(tt.data_type),
                _ => None,
            });

        // Extract the storage kind (which oneof variant is set) from the immediate tensor
        let storage_kind = value_entry
            .as_ref()
            .and_then(|v| v.value.as_ref())
            .and_then(|iv| match iv {
                apple_proto::mil_spec::value::Value::ImmediateValue(imv) => imv.value.as_ref(),
                _ => None,
            })
            .and_then(|v| match v {
                apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv) => {
                    tv.value.as_ref()
                }
                _ => None,
            })
            .map(|tv| match tv {
                apple_proto::mil_spec::tensor_value::Value::Floats(_) => "floats",
                apple_proto::mil_spec::tensor_value::Value::Ints(_) => "ints",
                apple_proto::mil_spec::tensor_value::Value::Bools(_) => "bools",
                apple_proto::mil_spec::tensor_value::Value::Strings(_) => "strings",
                apple_proto::mil_spec::tensor_value::Value::LongInts(_) => "longInts",
                apple_proto::mil_spec::tensor_value::Value::Doubles(_) => "doubles",
                apple_proto::mil_spec::tensor_value::Value::Bytes(_) => "bytes",
            });

        // Extract output dtype (first output)
        let output_dtype = op
            .outputs
            .first()
            .and_then(|nvt| nvt.r#type.as_ref())
            .and_then(|vt| vt.r#type.as_ref())
            .and_then(|t| match t {
                apple_proto::mil_spec::value_type::Type::TensorType(tt) => Some(tt.data_type),
                _ => None,
            });

        match (value_dtype, output_dtype) {
            (Some(vd), Some(od)) if vd != od => {
                return Err(ProtoValidationError {
                    kind: ProtoValidationKind::FillDtypeMismatch,
                    message: format!(
                        "fill value dtype ({}) != output dtype ({}) — \
                         Core ML requires them to match. Value is likely encoded as \
                         FLOAT32 but output is FLOAT16; use make_immediate_float_value() \
                         with the output dtype instead of make_immediate_float32_value().",
                        vd, od
                    ),
                    op_name: Some(op_name),
                });
            }
            (None, _) | (_, None) => {
                // Could not extract dtype — this shouldn't happen in practice,
                // but don't error since it might be a valid edge case we haven't seen.
            }
            _ => {} // dtypes match, all good
        }

        // Validate storage/dtype consistency.
        // Core ML rejects FLOAT16 immediates stored via the `floats` field.
        match (value_dtype, storage_kind) {
            (Some(dt), Some("floats")) if dt == float16_dtype => {
                return Err(ProtoValidationError {
                    kind: ProtoValidationKind::FillFp16FloatsStorage,
                    message: "fill value is FLOAT16 but stored via `floats` field — \
                         Core ML rejects this with 'Tensor storage and type have different \
                         number of elements'. Use `bytes` storage with raw f16 bits instead \
                         (see make_immediate_float_value)."
                        .to_string(),
                    op_name: Some(op_name),
                });
            }
            (Some(dt), Some("bytes")) if dt == float32_dtype => {
                return Err(ProtoValidationError {
                    kind: ProtoValidationKind::FillFp32BytesStorage,
                    message: "fill value is FLOAT32 but stored via `bytes` field — \
                         FLOAT32 scalars should use `floats` storage."
                        .to_string(),
                    op_name: Some(op_name),
                });
            }
            _ => {} // storage/dtype pair is valid
        }
    }

    Ok(())
}

/// Validate that elementwise binary ops (mul, add, sub, etc.) declare an output
/// shape consistent with numpy-style broadcasting of their inputs.
///
/// Core ML's `ios19.mul` (etc.) type inference applies standard broadcasting,
/// so the declared output shape must equal the broadcast result. If it doesn't,
/// Core ML rejects the model with:
///   "Output '0' has unexpected type 'ios19.mul'. Expected tensor<fp16, [...]>;
///    got tensor<fp16, [...]>."
fn validate_elementwise_broadcast_shapes(
    operations: &[apple_proto::mil_spec::Operation],
) -> Result<(), ProtoValidationError> {
    let elementwise_types: &[&str] = &[
        "mul",
        "add",
        "sub",
        "real_div",
        "maximum",
        "minimum",
        "pow",
        "floor_div",
        "mod",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
    ];

    for op in operations {
        if !elementwise_types.contains(&op.r#type.as_str()) {
            continue;
        }

        let op_name = op
            .attributes
            .get("name")
            .and_then(|v| v.value.as_ref())
            .and_then(|iv| match iv {
                apple_proto::mil_spec::value::Value::ImmediateValue(imv) => imv.value.as_ref(),
                _ => None,
            })
            .and_then(|v| match v {
                apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv) => {
                    match tv.value.as_ref() {
                        Some(apple_proto::mil_spec::tensor_value::Value::Strings(rs)) => {
                            rs.values.first().cloned()
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .unwrap_or_else(|| format!("<unnamed {}>", op.r#type));

        // Extract input shapes for x and y
        let extract_shape = |arg_name: &str| -> Option<Vec<u64>> {
            op.inputs
                .get(arg_name)
                .and_then(|arg| arg.arguments.first())
                .and_then(|binding| match &binding.binding {
                    Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => Some(v),
                    _ => None,
                })
                .and_then(|v| v.r#type.as_ref())
                .and_then(|vt| vt.r#type.as_ref())
                .and_then(|t| match t {
                    apple_proto::mil_spec::value_type::Type::TensorType(tt) => {
                        if tt.dimensions.is_empty() {
                            None
                        } else {
                            Some(tt.dimensions.clone())
                        }
                    }
                    _ => None,
                })
                .map(|dims| {
                    dims.iter()
                        .filter_map(|d| match &d.dimension {
                            Some(apple_proto::mil_spec::dimension::Dimension::Constant(cd)) => {
                                Some(cd.size)
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|v: &Vec<u64>| !v.is_empty())
        };

        let shape_x = extract_shape("x");
        let shape_y = extract_shape("y");

        // Extract declared output shape
        let output_shape: Option<Vec<u64>> = op
            .outputs
            .first()
            .and_then(|nvt| nvt.r#type.as_ref())
            .and_then(|vt| vt.r#type.as_ref())
            .and_then(|t| match t {
                apple_proto::mil_spec::value_type::Type::TensorType(tt) => {
                    if tt.dimensions.is_empty() {
                        None
                    } else {
                        Some(tt.dimensions.clone())
                    }
                }
                _ => None,
            })
            .map(|dims| {
                dims.iter()
                    .filter_map(|d| match &d.dimension {
                        Some(apple_proto::mil_spec::dimension::Dimension::Constant(cd)) => {
                            Some(cd.size)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        if let (Some(sx), Some(sy), Some(os)) = (&shape_x, &shape_y, &output_shape) {
            // Compute expected broadcast shape
            let max_rank = sx.len().max(sy.len());
            let mut expected = Vec::with_capacity(max_rank);
            let mut compatible = true;
            for i in 0..max_rank {
                let da = if i < max_rank - sx.len() { 1u64 } else { sx[i - (max_rank - sx.len())] };
                let db = if i < max_rank - sy.len() { 1u64 } else { sy[i - (max_rank - sy.len())] };
                if da != db && da != 1 && db != 1 {
                    compatible = false;
                    break;
                }
                expected.push(da.max(db));
            }

            if compatible && expected != *os {
                return Err(ProtoValidationError {
                    kind: ProtoValidationKind::ElementwiseBroadcastShapeMismatch,
                    message: format!(
                        "{} output shape is {:?} but broadcast of x={:?} and y={:?} \
                         produces {:?} — Core ML will reject this. The elementwise \
                         broadcast shape inference must compute the broadcast result, \
                         not just propagate x's shape.",
                        op.r#type, os, sx, sy, expected
                    ),
                    op_name: Some(op_name),
                });
            }
        }
    }

    Ok(())
}

/// Validate matmul output shapes against inferred batched matmul result.
///
/// For matmul ops: [*, M, K] × [*, K, N] → [*, M, N] where * are broadcast
/// batch dims. Core ML rejects matmul ops where the declared output shape
/// doesn't match the inferred result, or where inner dimensions are
/// incompatible (K must match).
fn validate_matmul_shapes(
    operations: &[apple_proto::mil_spec::Operation],
) -> Result<(), ProtoValidationError> {
    for op in operations {
        if op.r#type != "matmul" {
            continue;
        }

        let op_name = op
            .attributes
            .get("name")
            .and_then(|v| v.value.as_ref())
            .and_then(|iv| match iv {
                apple_proto::mil_spec::value::Value::ImmediateValue(imv) => imv.value.as_ref(),
                _ => None,
            })
            .and_then(|v| match v {
                apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv) => {
                    match tv.value.as_ref() {
                        Some(apple_proto::mil_spec::tensor_value::Value::Strings(rs)) => {
                            rs.values.first().cloned()
                        }
                        _ => None,
                    }
                }
                _ => None,
            })
            .unwrap_or_else(|| "<unnamed matmul>".to_string());

        // Reuse the same shape extraction helper pattern
        let extract_shape = |arg_name: &str| -> Option<Vec<u64>> {
            op.inputs
                .get(arg_name)
                .and_then(|arg| arg.arguments.first())
                .and_then(|binding| match &binding.binding {
                    Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => Some(v),
                    _ => None,
                })
                .and_then(|v| v.r#type.as_ref())
                .and_then(|vt| vt.r#type.as_ref())
                .and_then(|t| match t {
                    apple_proto::mil_spec::value_type::Type::TensorType(tt) => {
                        if tt.dimensions.is_empty() {
                            None
                        } else {
                            Some(tt.dimensions.clone())
                        }
                    }
                    _ => None,
                })
                .map(|dims| {
                    dims.iter()
                        .filter_map(|d| match &d.dimension {
                            Some(apple_proto::mil_spec::dimension::Dimension::Constant(cd)) => {
                                Some(cd.size)
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                })
                .filter(|v: &Vec<u64>| !v.is_empty())
        };

        let shape_x = extract_shape("x");
        let shape_y = extract_shape("y");

        // Extract declared output shape
        let output_shape: Option<Vec<u64>> = op
            .outputs
            .first()
            .and_then(|nvt| nvt.r#type.as_ref())
            .and_then(|vt| vt.r#type.as_ref())
            .and_then(|t| match t {
                apple_proto::mil_spec::value_type::Type::TensorType(tt) => {
                    if tt.dimensions.is_empty() {
                        None
                    } else {
                        Some(tt.dimensions.clone())
                    }
                }
                _ => None,
            })
            .map(|dims| {
                dims.iter()
                    .filter_map(|d| match &d.dimension {
                        Some(apple_proto::mil_spec::dimension::Dimension::Constant(cd)) => {
                            Some(cd.size)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty());

        if let (Some(sx), Some(sy), Some(os)) = (&shape_x, &shape_y, &output_shape) {
            let x_rank = sx.len();
            let y_rank = sy.len();

            if x_rank >= 2 && y_rank >= 2 {
                let lhs_rows = sx[x_rank - 2];
                let lhs_cols = sx[x_rank - 1];
                let rhs_rows = sy[y_rank - 2];
                let rhs_cols = sy[y_rank - 1];

                // Validate inner dimensions
                if lhs_cols != rhs_rows && lhs_cols > 0 && rhs_rows > 0 {
                    return Err(ProtoValidationError {
                        kind: ProtoValidationKind::MatMulInnerDimMismatch,
                        message: format!(
                            "matmul inner dims incompatible: x={:?} y={:?} — \
                             lhs_cols ({}) != rhs_rows ({}). Core ML will reject this.",
                            sx, sy, lhs_cols, rhs_rows
                        ),
                        op_name: Some(op_name),
                    });
                }

                // Compute expected output: broadcast(batch_x, batch_y) + [lhs_rows, rhs_cols]
                let batch_x = &sx[..x_rank - 2];
                let batch_y = &sy[..y_rank - 2];
                let max_batch_rank = batch_x.len().max(batch_y.len());
                let mut expected = Vec::with_capacity(max_batch_rank + 2);
                let mut batch_compatible = true;
                for i in 0..max_batch_rank {
                    let da = if i < max_batch_rank - batch_x.len() {
                        1u64
                    } else {
                        batch_x[i - (max_batch_rank - batch_x.len())]
                    };
                    let db = if i < max_batch_rank - batch_y.len() {
                        1u64
                    } else {
                        batch_y[i - (max_batch_rank - batch_y.len())]
                    };
                    if da != db && da != 1 && db != 1 {
                        batch_compatible = false;
                        break;
                    }
                    expected.push(da.max(db));
                }

                if batch_compatible {
                    expected.push(lhs_rows);
                    expected.push(rhs_cols);

                    if expected != *os {
                        return Err(ProtoValidationError {
                            kind: ProtoValidationKind::MatMulOutputShapeMismatch,
                            message: format!(
                                "matmul output shape is {:?} but inferred from x={:?} y={:?} \
                                 is {:?} — Core ML will reject this. The matmul shape inference \
                                 must compute broadcast_batch + [lhs_rows, rhs_cols], not reuse x's shape.",
                                os, sx, sy, expected
                            ),
                            op_name: Some(op_name),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Sanitize SSA variable identifiers in a list of operations.
///
/// MIL (and Python) treat `.` as an attribute accessor, so names like
/// `lm_head.weight` are parsed as `lm_head` accessing attribute `weight`.
/// This pass replaces all SSA variable identifiers with sanitized versions
/// (dots → underscores, etc.) while preserving the original dotted names
/// in the `name = string(...)` attribute for debugging/metadata.
///
/// The pass works in two phases:
/// 1. **Collect**: scan all output names and build an `original → sanitized` map
/// 2. **Replace**: walk all operations and replace output names and input
///    name references with their sanitized equivalents
fn sanitize_mil_identifiers(
    operations: &mut [apple_proto::mil_spec::Operation],
    block_output_names: &mut [String],
    fn_input_names: &mut [apple_proto::mil_spec::NamedValueType],
) {
    // Phase 1: Collect all names that need sanitization
    let mut name_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    for op in operations.iter() {
        for output in &op.outputs {
            let sanitized = sanitize_mil_ident(&output.name);
            if sanitized != output.name {
                name_map.entry(output.name.clone()).or_insert(sanitized);
            }
        }
        // Also collect input name references that might be dotted
        for arg in op.inputs.values() {
            for binding in &arg.arguments {
                if let Some(apple_proto::mil_spec::argument::binding::Binding::Name(ref n)) =
                    binding.binding
                {
                    let sanitized = sanitize_mil_ident(n);
                    if sanitized != *n {
                        name_map.entry(n.clone()).or_insert(sanitized);
                    }
                }
            }
        }
    }

    // Also check block outputs and function inputs
    for name in block_output_names.iter() {
        let sanitized = sanitize_mil_ident(name);
        if sanitized != *name {
            name_map.entry(name.clone()).or_insert(sanitized);
        }
    }
    for input in fn_input_names.iter() {
        let sanitized = sanitize_mil_ident(&input.name);
        if sanitized != input.name {
            name_map.entry(input.name.clone()).or_insert(sanitized);
        }
    }

    // If no names need sanitization, skip the mutation phase
    if name_map.is_empty() {
        return;
    }

    eprintln!(
        "[INFO] Sanitizing {} MIL identifier(s) (dots → underscores) for Core ML compatibility",
        name_map.len()
    );

    // Phase 2: Replace all occurrences
    for op in operations.iter_mut() {
        // Replace output names (these become MIL variable identifiers)
        for output in &mut op.outputs {
            if let Some(sanitized) = name_map.get(&output.name) {
                output.name = sanitized.clone();
            }
        }
        // Replace input name references (these must match output names for SSA wire-up)
        for (_key, arg) in op.inputs.iter_mut() {
            for binding in &mut arg.arguments {
                if let Some(apple_proto::mil_spec::argument::binding::Binding::Name(ref mut n)) =
                    binding.binding
                {
                    if let Some(sanitized) = name_map.get(n) {
                        *n = sanitized.clone();
                    }
                }
            }
        }
        // NOTE: We do NOT sanitize the "name" attribute value — it's a string
        // constant, not an identifier, and should preserve the original dotted
        // name for debugging/metadata: name = string("lm_head.weight")
    }

    // Sanitize block output names
    for name in block_output_names.iter_mut() {
        if let Some(sanitized) = name_map.get(name) {
            *name = sanitized.clone();
        }
    }

    // Sanitize function input names
    for input in fn_input_names.iter_mut() {
        if let Some(sanitized) = name_map.get(&input.name) {
            input.name = sanitized.clone();
        }
    }
}

/// Convert a `CoreMlFunction` into an Apple-compatible `MILSpec.Function`.
fn function_to_apple_proto(
    func: &CoreMlFunction,
    weight_entries: &[WeightEntry],
    opset: &str,
) -> Result<apple_proto::mil_spec::Function, ProtoValidationError> {
    // Regular tensor inputs
    let mut fn_inputs: Vec<apple_proto::mil_spec::NamedValueType> = func
        .inputs
        .iter()
        .map(|td| {
            let dtype = coreml_dtype_to_apple_mil(&td.dtype);
            make_apple_named_value_type(&td.name, dtype, &td.shape)
        })
        .collect();

    // State variable declarations: Core ML requires state variables to be
    // declared as MIL function inputs with `ValueType.type = StateType`.
    // Without these declarations, `read_state` and `write_state` ops
    // reference unbound names and the runtime rejects the model.
    for state in &func.states {
        let dtype = coreml_dtype_to_apple_mil(&state.dtype);
        fn_inputs.push(make_apple_state_named_value_type(&state.name, dtype, &state.shape));
    }

    let operations: Vec<apple_proto::mil_spec::Operation> = func
        .operations
        .iter()
        .flat_map(|op| mir_op_to_apple_ops(op, weight_entries, &func.node_shapes))
        .collect();

    // Sanitize MIL variable identifiers: replace dots in names like
    // "lm_head.weight" → "lm_head_weight" so the MIL text parser doesn't
    // interpret the dot as an attribute accessor. The original dotted name
    // is preserved in the name=string(...) attribute for debugging.
    let mut operations = operations;
    let mut fn_inputs_mut = fn_inputs;
    let mut block_output_names: Vec<String> =
        func.outputs.iter().map(|td| td.name.clone()).collect();
    sanitize_mil_identifiers(&mut operations, &mut block_output_names, &mut fn_inputs_mut);
    let fn_inputs = fn_inputs_mut; // consume mutable back

    // Validate fill/fill_like dtype consistency before writing.
    // Core ML rejects fill ops where the value parameter dtype differs from the output dtype.
    validate_fill_dtype_consistency(&operations)?;

    // Validate elementwise broadcast output shapes before writing.
    // Core ML rejects mul/add/etc. ops where the declared output shape doesn't match
    // the broadcast result of the inputs.
    validate_elementwise_broadcast_shapes(&operations)?;

    // Validate matmul output shapes before writing.
    // Core ML rejects matmul ops where the declared output shape doesn't match
    // the inferred batched matmul result, or where inner dimensions are incompatible.
    validate_matmul_shapes(&operations)?;

    let block = apple_proto::mil_spec::Block {
        inputs: vec![],
        outputs: block_output_names,
        operations,
        attributes: HashMap::new(),
    };

    let mut block_specializations = HashMap::new();
    block_specializations.insert(opset.to_string(), block);

    Ok(apple_proto::mil_spec::Function {
        inputs: fn_inputs,
        opset: opset.to_string(),
        block_specializations,
        attributes: HashMap::new(),
    })
}

/// Convert a `CoreMlModel` to an Apple-compatible `apple_proto::Model`.
///
/// This produces a Model that matches Apple's actual wire format exactly,
/// enabling Core ML's runtime to decode the .mlpackage correctly.
///
/// Key differences from the legacy `convert_to_proto_model()`:
/// - Uses `MILSpec.Program` (field 502) instead of `MLProgram` (field 20)
/// - Uses `MILSpec.Function` with `block_specializations` instead of `MilFunction` with `block`
/// - Operations use generic `type` + `inputs` + `outputs` instead of per-op-type `oneof`
/// - Data types use Apple enum values (FLOAT16=10, FLOAT32=11, etc.)
/// - Weight references use `BlobFileValue` with `fileName="weight.bin"`
/// - Model description uses `FunctionDescription` with field 20
pub fn convert_to_apple_proto_model(
    model: &CoreMlModel,
    weight_entries: &[WeightEntry],
) -> Result<apple_proto::Model, ProtoValidationError> {
    // The MIL opset version is independent of the specification version.
    // Apple's reference models use opset "CoreML9" even with spec version 10.
    // The opset determines which MIL ops are available; spec version determines
    // the overall model format capabilities (e.g., multi-function, state support).
    let opset = "CoreML9".to_string();

    // Build function descriptions for ModelDescription
    let function_descriptions: Vec<apple_proto::FunctionDescription> = model
        .functions
        .iter()
        .map(|func| {
            let input_descs: Vec<apple_proto::FeatureDescription> = func
                .inputs
                .iter()
                .map(|td| make_apple_feature_desc(&td.name, &td.dtype, &td.shape))
                .collect();
            let output_descs: Vec<apple_proto::FeatureDescription> = func
                .outputs
                .iter()
                .map(|td| make_apple_feature_desc(&td.name, &td.dtype, &td.shape))
                .collect();
            // Populate state feature descriptions for models with state (e.g., KV-cache)
            let state_descs: Vec<apple_proto::FeatureDescription> = func
                .states
                .iter()
                .map(|td| make_apple_state_feature_desc(&td.name, &td.dtype, &td.shape))
                .collect();
            apple_proto::FunctionDescription {
                name: func.name.clone(),
                input: input_descs,
                output: output_descs,
                state: state_descs,
                predicted_feature_name: String::new(),
                predicted_probabilities_name: String::new(),
            }
        })
        .collect();

    // Build MILSpec.Program functions map
    let mut program_functions = HashMap::new();
    for func in &model.functions {
        program_functions
            .insert(func.name.clone(), function_to_apple_proto(func, weight_entries, &opset)?);
    }

    let program = apple_proto::mil_spec::Program {
        version: 1,
        functions: program_functions,
        doc_string: String::new(),
        attributes: HashMap::new(),
    };

    // Core ML's model classification rule is binary: if ModelDescription.functions
    // is non-empty, the model is "multi-function" and MUST NOT have top-level
    // input/output/state feature descriptions. All I/O lives inside
    // FunctionDescription entries instead. This applies regardless of how many
    // functions exist — even a single-function model with functions=[] populated
    // is classified as multi-function by Core ML.
    //
    // For single-function models, we use the single-function schema pattern:
    //   - description.input/output are populated from the function's I/O
    //   - description.functions is empty
    //   - description.defaultFunctionName is ""
    //   - mlProgram.functions uses "main" as the function name
    //
    // This ensures Core ML's document decoder finds top-level outputSchema and
    // doesn't throw missingMetadataField(named: "outputSchema").
    //
    // For multi-function models, we use the multi-function schema pattern:
    //   - description.input/output/state are empty
    //   - description.functions is populated
    //   - description.defaultFunctionName is set
    //   - mlProgram.functions use the original shard names
    let is_single_function = model.functions.len() == 1;

    let (
        model_input_descs,
        model_output_descs,
        model_state_descs,
        final_function_descriptions,
        final_default_fn_name,
        final_program,
    ) = if is_single_function {
        // Single-function: populate top-level I/O, leave functions empty,
        // rename MIL program function to "main"
        let func = &model.functions[0];
        let top_inputs: Vec<apple_proto::FeatureDescription> = func
            .inputs
            .iter()
            .map(|td| make_apple_feature_desc(&td.name, &td.dtype, &td.shape))
            .collect();
        let top_outputs: Vec<apple_proto::FeatureDescription> = func
            .outputs
            .iter()
            .map(|td| make_apple_feature_desc(&td.name, &td.dtype, &td.shape))
            .collect();
        let top_states: Vec<apple_proto::FeatureDescription> = func
            .states
            .iter()
            .map(|td| make_apple_state_feature_desc(&td.name, &td.dtype, &td.shape))
            .collect();

        // Rename the single function to "main" in the MIL Program
        let mut main_program_functions = HashMap::new();
        if let Some(mil_func) = program.functions.get(&func.name) {
            main_program_functions.insert("main".to_string(), mil_func.clone());
        }

        (
            top_inputs,
            top_outputs,
            top_states,
            vec![],        // empty functions list for single-function
            String::new(), // no defaultFunctionName for single-function
            apple_proto::mil_spec::Program {
                version: program.version,
                functions: main_program_functions,
                doc_string: program.doc_string,
                attributes: program.attributes,
            },
        )
    } else {
        // Multi-function: top-level I/O empty, functions populated
        (
            vec![],
            vec![],
            vec![],
            function_descriptions,
            model.default_function_name.clone(),
            program,
        )
    };

    Ok(apple_proto::Model {
        specification_version: model.spec_version.proto_value(),
        description: Some(apple_proto::ModelDescription {
            functions: final_function_descriptions,
            default_function_name: final_default_fn_name,
            metadata: None,
            input: model_input_descs,
            output: model_output_descs,
            state: model_state_descs,
            predicted_feature_name: String::new(),
            predicted_probabilities_name: String::new(),
            training_input: vec![],
        }),
        is_updatable: false,
        r#type: Some(apple_proto::model::Type::MlProgram(final_program)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_data_type_from_mir() {
        assert_eq!(
            CoreMlDataType::from_mir_dtype(&mir_compat::MilDtypeCompat::Fp16),
            CoreMlDataType::Float16
        );
        assert_eq!(
            CoreMlDataType::from_mir_dtype(&mir_compat::MilDtypeCompat::Fp32),
            CoreMlDataType::Float32
        );
    }

    #[test]
    fn test_coreml_data_type_element_size() {
        assert_eq!(CoreMlDataType::Float16.element_size(), 2);
        assert_eq!(CoreMlDataType::Float32.element_size(), 4);
        // T-71: Float64 is 8 bytes (double precision), not 4
        assert_eq!(CoreMlDataType::Float64.element_size(), 8);
        assert_eq!(CoreMlDataType::Int32.element_size(), 4);
        assert_eq!(CoreMlDataType::UInt8.element_size(), 1);
        assert_eq!(CoreMlDataType::Int8.element_size(), 1);
        assert_eq!(CoreMlDataType::Bool.element_size(), 1);
        assert_eq!(CoreMlDataType::Int4.element_size(), 1);
        assert_eq!(CoreMlDataType::UInt4.element_size(), 1);
        assert_eq!(CoreMlDataType::E4M3.element_size(), 1);
        assert_eq!(CoreMlDataType::E5M2.element_size(), 1);
        assert_eq!(CoreMlDataType::UInt16.element_size(), 2);
        assert_eq!(CoreMlDataType::Unknown.element_size(), 0);
    }

    #[test]
    fn test_spec_version_state_support() {
        assert!(!SpecVersion::V7.supports_state());
        assert!(SpecVersion::V8.supports_state());
    }

    #[test]
    fn test_compute_unit_from_mir_hint() {
        assert_eq!(
            CoreMlComputeUnit::from_mir_hint(&mir_compat::ComputeUnitHintCompat::CPUAndNE),
            CoreMlComputeUnit::CpuAndNe
        );
    }

    #[test]
    fn test_weight_entry_serialization() {
        let entry = WeightEntry {
            name: "weight_0".to_string(),
            offset: 0,
            size: 1024,
            shape: vec![32, 64],
            dtype: CoreMlDataType::Float16,
            data: vec![0u8; 1024],
        };
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: WeightEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "weight_0");
        assert_eq!(deserialized.shape, vec![32, 64]);
    }

    #[test]
    fn test_shared_weight_ref() {
        let entry = WeightEntry {
            name: "shared_weight".to_string(),
            offset: 0,
            size: 2048,
            shape: vec![128, 128],
            dtype: CoreMlDataType::Float16,
            data: vec![0u8; 2048],
        };
        let shared = SharedWeightRef {
            weight: entry,
            referencing_functions: vec!["embedding".to_string(), "decode_step".to_string()],
        };
        assert_eq!(shared.referencing_functions.len(), 2);
    }

    // ─── Proto conversion tests ─────────────────────────────────────────────

    #[test]
    fn test_data_type_to_proto() {
        assert_eq!(
            data_type_to_proto(&CoreMlDataType::Float16),
            proto::ArrayDataType::Float16 as i32
        );
        assert_eq!(
            data_type_to_proto(&CoreMlDataType::Float32),
            proto::ArrayDataType::Float32 as i32
        );
        assert_eq!(data_type_to_proto(&CoreMlDataType::Int32), proto::ArrayDataType::Int32 as i32);
    }

    #[test]
    fn test_compute_unit_to_proto() {
        assert_eq!(
            compute_unit_to_proto(&CoreMlComputeUnit::CpuAndNe),
            proto::ComputeUnit::CpuAndNe as i32
        );
        assert_eq!(compute_unit_to_proto(&CoreMlComputeUnit::All), proto::ComputeUnit::All as i32);
    }

    #[test]
    fn test_spec_version_to_proto() {
        assert_eq!(
            spec_version_to_proto(&SpecVersion::V7),
            proto::SpecificationVersion::SpecificationVersion7 as i32
        );
        assert_eq!(
            spec_version_to_proto(&SpecVersion::V8),
            proto::SpecificationVersion::SpecificationVersion8 as i32
        );
    }

    #[test]
    fn test_shape_to_proto() {
        let shape = shape_to_proto(&[32, 64]);
        assert_eq!(shape.dimensions.len(), 2);
        assert!(matches!(
            &shape.dimensions[0].dimension_value,
            Some(proto::dimension_value::DimensionValue::Fixed(32))
        ));
    }

    #[test]
    fn test_tensor_desc_to_proto() {
        let td = TensorDesc {
            name: "input".to_string(),
            shape: vec![1, 128],
            dtype: CoreMlDataType::Float16,
            is_state: false,
        };
        let fd = tensor_desc_to_proto(&td);
        assert_eq!(fd.name, "input");
        assert!(matches!(
            fd.feature_type,
            Some(proto::feature_description::FeatureType::Tensor(_))
        ));
    }

    #[test]
    fn test_weight_entry_to_proto_file_ref() {
        let entry = WeightEntry {
            name: "weight_0".to_string(),
            offset: 256,
            size: 1024,
            shape: vec![32, 64],
            dtype: CoreMlDataType::Float16,
            data: vec![0u8; 1024],
        };
        let wd = weight_entry_to_proto(&entry);
        assert!(matches!(wd.weight_data, Some(proto::weight_data::WeightData::FileRef(_))));
        if let Some(proto::weight_data::WeightData::FileRef(fr)) = wd.weight_data {
            assert_eq!(fr.offset, 256);
            assert_eq!(fr.size, 1024);
        }
    }

    #[test]
    fn test_mir_op_to_proto_linear() {
        let weight_entries = vec![WeightEntry {
            name: "w".to_string(),
            offset: 0,
            size: 128,
            shape: vec![32, 64],
            dtype: CoreMlDataType::Float16,
            data: vec![0u8; 128],
        }];
        let op = mir_compat::MirOpCompat::Linear {
            name: "output".to_string(),
            x: "input".to_string(),
            weight_name: "w".to_string(),
            bias_name: None,
        };
        let proto_op = mir_op_to_proto_op(&op, &weight_entries);
        assert_eq!(proto_op.name, "output");
        assert!(matches!(proto_op.operation, Some(proto::mil_operation::Operation::LinearOp(_))));
    }

    #[test]
    fn test_mir_op_to_proto_all_variants() {
        let weight_entries: Vec<WeightEntry> = vec![];

        // Test every MirOpCompat variant converts without panic
        let ops: Vec<mir_compat::MirOpCompat> = vec![
            mir_compat::MirOpCompat::Const {
                name: "c".to_string(),
                data: vec![0u8; 4],
                dtype: mir_compat::MilDtypeCompat::Fp16,
                shape: vec![2],
            },
            mir_compat::MirOpCompat::Linear {
                name: "l".to_string(),
                x: "x".to_string(),
                weight_name: "w".to_string(),
                bias_name: None,
            },
            mir_compat::MirOpCompat::MatMul {
                name: "mm".to_string(),
                x: "x".to_string(),
                y: "y".to_string(),
                transpose_y: false,
            },
            mir_compat::MirOpCompat::Add {
                name: "a".to_string(),
                x: "x".to_string(),
                y: "y".to_string(),
            },
            mir_compat::MirOpCompat::Mul {
                name: "m".to_string(),
                x: "x".to_string(),
                y: "y".to_string(),
            },
            mir_compat::MirOpCompat::Sub {
                name: "s".to_string(),
                x: "x".to_string(),
                y: "y".to_string(),
            },
            mir_compat::MirOpCompat::Abs { name: "ab".to_string(), x: "x".to_string() },
            mir_compat::MirOpCompat::Reshape {
                name: "r".to_string(),
                x: "x".to_string(),
                shape: vec![1, 2],
            },
            mir_compat::MirOpCompat::Transpose {
                name: "t".to_string(),
                x: "x".to_string(),
                perm: vec![1, 0],
            },
            mir_compat::MirOpCompat::SliceByIndex {
                name: "sbi".to_string(),
                x: "x".to_string(),
                begin: vec![0],
                end: vec![1],
                stride: vec![],
                begin_mask: vec![false],
                end_mask: vec![false],
                squeeze_mask: vec![false],
            },
            mir_compat::MirOpCompat::SliceUpdate {
                name: "su".to_string(),
                x: "x".to_string(),
                update: "u".to_string(),
                begin: vec![0],
                end: vec![1],
            },
            mir_compat::MirOpCompat::Concat {
                name: "cat".to_string(),
                values: vec!["a".to_string(), "b".to_string()],
                axis: 0,
            },
            mir_compat::MirOpCompat::Softmax {
                name: "sm".to_string(),
                x: "x".to_string(),
                axis: -1,
            },
            mir_compat::MirOpCompat::Gelu {
                name: "g".to_string(),
                x: "x".to_string(),
                mode: "EXACT".to_string(),
            },
            mir_compat::MirOpCompat::ScaledDotProductAttention {
                name: "sdpa".to_string(),
                query: "q".to_string(),
                key: "k".to_string(),
                value: "v".to_string(),
                attention_mask: None,
                scale: None,
            },
            mir_compat::MirOpCompat::ReadState {
                name: "rs".to_string(),
                state_id: "s1".to_string(),
                shape: vec![128],
                dtype: mir_compat::MilDtypeCompat::Fp16,
            },
            mir_compat::MirOpCompat::CoremlUpdateState {
                name: "us".to_string(),
                state_id: "s1".to_string(),
                value: "v".to_string(),
            },
            mir_compat::MirOpCompat::Gather {
                name: "ga".to_string(),
                x: "x".to_string(),
                indices: "idx".to_string(),
                axis: 0,
            },
            mir_compat::MirOpCompat::ReduceMean {
                name: "rm".to_string(),
                x: "x".to_string(),
                axes: vec![1],
                keep_dims: true,
            },
            mir_compat::MirOpCompat::Rsqrt { name: "rsqrt".to_string(), x: "x".to_string() },
            mir_compat::MirOpCompat::RealDiv {
                name: "rd".to_string(),
                x: "x".to_string(),
                y: "y".to_string(),
            },
            mir_compat::MirOpCompat::LayerNorm {
                name: "ln".to_string(),
                x: "x".to_string(),
                weight_name: "w".to_string(),
                bias_name: None,
                epsilon: 1e-5,
                axes: vec![-1],
            },
            mir_compat::MirOpCompat::Topk {
                name: "tk".to_string(),
                x: "x".to_string(),
                k: 10,
                axis: -1,
            },
            mir_compat::MirOpCompat::Cos { name: "cos".to_string(), x: "x".to_string() },
            mir_compat::MirOpCompat::Sin { name: "sin".to_string(), x: "x".to_string() },
            mir_compat::MirOpCompat::Cast {
                name: "cast".to_string(),
                x: "x".to_string(),
                dtype: mir_compat::MilDtypeCompat::Fp16,
            },
            mir_compat::MirOpCompat::Tile {
                name: "tile".to_string(),
                x: "x".to_string(),
                reps: vec![1, 2, 1, 1],
            },
            mir_compat::MirOpCompat::Fill {
                name: "fill".to_string(),
                shape: vec![1, 1, 2, 1, 1],
                value: 1.0,
                dtype: mir_compat::MilDtypeCompat::Fp16,
            },
            mir_compat::MirOpCompat::FillLike {
                name: "fill_like".to_string(),
                ref_tensor: "x".to_string(),
                value: 0.0,
                dtype: mir_compat::MilDtypeCompat::Fp16,
            },
            mir_compat::MirOpCompat::Neg { name: "neg".to_string(), x: "x".to_string() },
        ];

        for op in &ops {
            let _proto_op = mir_op_to_proto_op(op, &weight_entries);
            // Each variant should convert without panic
        }
    }

    #[test]
    fn test_convert_to_proto_model_roundtrip() {
        use prost::Message;

        let model = CoreMlModel {
            spec_version: SpecVersion::V10,
            description: ModelDescriptionCompat {
                inputs: vec![TensorDesc {
                    name: "x".to_string(),
                    shape: vec![1, 64],
                    dtype: CoreMlDataType::Float16,
                    is_state: false,
                }],
                outputs: vec![TensorDesc {
                    name: "output".to_string(),
                    shape: vec![1, 32],
                    dtype: CoreMlDataType::Float16,
                    is_state: false,
                }],
                states: vec![],
            },
            functions: vec![CoreMlFunction {
                name: "main".to_string(),
                inputs: vec![TensorDesc {
                    name: "x".to_string(),
                    shape: vec![1, 64],
                    dtype: CoreMlDataType::Float16,
                    is_state: false,
                }],
                outputs: vec![TensorDesc {
                    name: "output".to_string(),
                    shape: vec![1, 32],
                    dtype: CoreMlDataType::Float16,
                    is_state: false,
                }],
                states: vec![],
                operations: vec![
                    mir_compat::MirOpCompat::Const {
                        name: "weight".to_string(),
                        data: vec![0u8; 128],
                        dtype: mir_compat::MilDtypeCompat::Fp16,
                        shape: vec![32, 64],
                    },
                    mir_compat::MirOpCompat::Add {
                        name: "output".to_string(),
                        x: "x".to_string(),
                        y: "weight".to_string(),
                    },
                ],
                node_shapes: std::collections::HashMap::new(),
            }],
            default_function_name: "main".to_string(),
            weights: vec![WeightEntry {
                name: "weight".to_string(),
                offset: 0,
                size: 128,
                shape: vec![32, 64],
                dtype: CoreMlDataType::Float16,
                data: vec![0u8; 128],
            }],
            shared_weights: vec![],
            compute_unit: CoreMlComputeUnit::CpuAndNe,
            user_defined_metadata: std::collections::HashMap::new(),
        };

        let weight_entries = model.weights.clone();
        let proto_model = convert_to_proto_model(&model, &weight_entries);

        // Verify key fields — V9 maps to raw value 9 (not in legacy proto enum)
        assert_eq!(proto_model.specification_version, 10);
        assert!(proto_model.description.is_some());
        assert!(proto_model.ml_program.is_some());

        let ml_prog = proto_model.ml_program.as_ref().unwrap();
        assert!(ml_prog.functions.contains_key("main"));

        // Serialize to protobuf bytes
        let bytes = proto_model.encode_to_vec();
        assert!(!bytes.is_empty());

        // Parse back
        let parsed = proto::Model::decode(bytes.as_slice()).unwrap();
        // V9 → raw value 9 (SpecificationVersion9 not in legacy proto enum)
        assert_eq!(parsed.specification_version, 10);
        assert!(parsed.ml_program.is_some());
        assert!(parsed.ml_program.as_ref().unwrap().functions.contains_key("main"));
    }

    /// Verify that shape/perm/axis/reps immediate values are emitted as INT32
    /// (RepeatedInts), not INT64 (RepeatedLongInts). Core ML's ios19 ops
    /// (reshape, transpose, tile, fill, etc.) reject INT64 tensors for these
    /// parameters. For example, ios19.reshape expects shape to be one of
    /// { tensor<int32, [?]>, tensor<int16, [?]>, tensor<int8, [?]> } and
    /// rejects tensor<int64, [N]>.
    #[test]
    fn test_shape_params_use_int32_not_int64() {
        // ── Tile.reps should be INT32 (Ints), not INT64 (LongInts) ──
        let tile_op = mir_compat::MirOpCompat::Tile {
            name: "tile_out".to_string(),
            x: "x".to_string(),
            reps: vec![1, 2, 1, 1],
        };
        let ops = mir_op_to_apple_ops(&tile_op, &[], &std::collections::HashMap::new());
        let tile_mil = ops.iter().find(|op| op.r#type == "tile").expect("tile op");
        let reps_arg = tile_mil.inputs.get("reps").expect("reps input");

        let reps_binding = match &reps_arg.arguments.first().unwrap().binding {
            Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => v,
            _ => panic!("reps should be a Value binding"),
        };
        let reps_imm = match &reps_binding.value {
            Some(apple_proto::mil_spec::value::Value::ImmediateValue(iv)) => iv,
            _ => panic!("reps should be ImmediateValue"),
        };
        let reps_tensor = match &reps_imm.value {
            Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv)) => tv,
            _ => panic!("reps should be Tensor"),
        };
        match &reps_tensor.value {
            Some(apple_proto::mil_spec::tensor_value::Value::Ints(ints)) => {
                assert_eq!(
                    ints.values,
                    vec![1i32, 2, 1, 1],
                    "tile.reps should be stored as INT32 elements"
                );
            }
            Some(apple_proto::mil_spec::tensor_value::Value::LongInts(_)) => {
                panic!(
                    "tile.reps must NOT use INT64/LongInts — \
                       Core ML ios19 rejects INT64 for shape-like parameters"
                );
            }
            Some(apple_proto::mil_spec::tensor_value::Value::Bytes(_)) => {
                panic!("tile.reps must NOT use bytes storage");
            }
            other => {
                panic!("tile.reps unexpected variant: {:?}", other);
            }
        }

        // Verify INT32 DataType in the type field
        if let Some(vt) = &reps_binding.r#type {
            if let Some(apple_proto::mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
                assert_eq!(
                    tt.data_type,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    "tile.reps tensor type should be INT32"
                );
            }
        }

        // ── Fill is decomposed to const(0) + add(zero, val) ──
        let fill_op = mir_compat::MirOpCompat::Fill {
            name: "fill_out".to_string(),
            shape: vec![1, 1, 2, 1, 1],
            value: 1.0,
            dtype: mir_compat::MilDtypeCompat::Fp16,
        };
        let ops = mir_op_to_apple_ops(&fill_op, &[], &std::collections::HashMap::new());
        // Fill is now decomposed to const + add (ANE-legal), not raw "fill"
        let _const_op =
            ops.iter().find(|op| op.r#type == "const").expect("const op in Fill decomposition");
        let add_op =
            ops.iter().find(|op| op.r#type == "add").expect("add op in Fill decomposition");

        // Verify the add op's output has the correct name
        assert_eq!(add_op.outputs[0].name, "fill_out", "add output should be fill_out");

        // Verify the add op has x (zeros) and y (value) inputs
        assert!(add_op.inputs.contains_key("x"), "add should have x input");
        assert!(add_op.inputs.contains_key("y"), "add should have y input");

        // ── Reshape.shape should be INT32 ──
        let reshape_op = mir_compat::MirOpCompat::Reshape {
            name: "reshape_out".to_string(),
            x: "x".to_string(),
            shape: vec![1, 512, 16, 128],
        };
        let ops = mir_op_to_apple_ops(&reshape_op, &[], &std::collections::HashMap::new());
        let reshape_mil = ops.iter().find(|op| op.r#type == "reshape").expect("reshape op");
        let shape_arg = reshape_mil.inputs.get("shape").expect("shape input");

        let shape_binding = match &shape_arg.arguments.first().unwrap().binding {
            Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => v,
            _ => panic!("shape should be a Value binding"),
        };
        match &shape_binding.value {
            Some(apple_proto::mil_spec::value::Value::ImmediateValue(iv)) => match &iv.value {
                Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv)) => {
                    match &tv.value {
                        Some(apple_proto::mil_spec::tensor_value::Value::Ints(ints)) => {
                            assert_eq!(
                                ints.values,
                                vec![1i32, 512, 16, 128],
                                "reshape.shape should be stored as INT32 elements"
                            );
                        }
                        Some(apple_proto::mil_spec::tensor_value::Value::LongInts(_)) => {
                            panic!(
                                "reshape.shape must NOT use INT64/LongInts — \
                                   Core ML ios19.reshape rejects INT64: \
                                   'Expected {{ tensor<int32, [?]>, tensor<int16, [?]>, \
                                   tensor<int8, [?]> }}; got tensor<int64, [N]>'"
                            );
                        }
                        other => panic!("reshape.shape unexpected variant: {:?}", other),
                    }
                }
                other => panic!("reshape.shape should be Tensor: {:?}", other),
            },
            other => panic!("reshape.shape should be ImmediateValue: {:?}", other),
        }

        // Verify INT32 DataType in the reshape shape type field
        if let Some(vt) = &shape_binding.r#type {
            if let Some(apple_proto::mil_spec::value_type::Type::TensorType(tt)) = &vt.r#type {
                assert_eq!(
                    tt.data_type,
                    apple_proto::mil_spec::DataType::Int32 as i32,
                    "reshape.shape tensor type should be INT32, not INT64"
                );
            }
        }

        // ── Transpose.perm should be INT32 ──
        let transpose_op = mir_compat::MirOpCompat::Transpose {
            name: "trans_out".to_string(),
            x: "x".to_string(),
            perm: vec![0, 1, 3, 2],
        };
        let ops = mir_op_to_apple_ops(&transpose_op, &[], &std::collections::HashMap::new());
        let trans_mil = ops.iter().find(|op| op.r#type == "transpose").expect("transpose op");
        let perm_arg = trans_mil.inputs.get("perm").expect("perm input");

        let perm_binding = match &perm_arg.arguments.first().unwrap().binding {
            Some(apple_proto::mil_spec::argument::binding::Binding::Value(v)) => v,
            _ => panic!("perm should be a Value binding"),
        };
        let perm_imm = match &perm_binding.value {
            Some(apple_proto::mil_spec::value::Value::ImmediateValue(iv)) => iv,
            _ => panic!("perm should be ImmediateValue"),
        };
        let perm_tensor = match &perm_imm.value {
            Some(apple_proto::mil_spec::value::immediate_value::Value::Tensor(tv)) => tv,
            _ => panic!("perm should be Tensor"),
        };
        match &perm_tensor.value {
            Some(apple_proto::mil_spec::tensor_value::Value::Ints(ints)) => {
                assert_eq!(
                    ints.values,
                    vec![0i32, 1, 3, 2],
                    "transpose.perm should be stored as INT32 elements"
                );
            }
            Some(apple_proto::mil_spec::tensor_value::Value::LongInts(_)) => {
                panic!(
                    "transpose.perm must NOT use INT64/LongInts — \
                       Core ML ios19 rejects INT64 for perm parameters"
                );
            }
            Some(apple_proto::mil_spec::tensor_value::Value::Bytes(_)) => {
                panic!("transpose.perm must NOT use bytes storage");
            }
            other => {
                panic!("transpose.perm unexpected variant: {:?}", other);
            }
        }
    }

    // ─── SDPA attention_mask + scale proto emission tests (T-31) ────

    #[test]
    fn test_sdpa_proto_emission_with_mask_and_scale() {
        // SDPA with both attention_mask and scale should emit attn_mask and scale in proto
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_test".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: Some("causal_mask".to_string()),
            scale: Some(0.125),
        };
        let proto_op = mir_op_to_proto_op(&compat, &[]);
        assert_eq!(proto_op.name, "sdpa_test");
        match &proto_op.operation {
            Some(proto::mil_operation::Operation::ScaledDotProductAttentionOp(sdpa_op)) => {
                assert_eq!(sdpa_op.query.as_ref().unwrap().name, "q");
                assert_eq!(sdpa_op.key.as_ref().unwrap().name, "k");
                assert_eq!(sdpa_op.value.as_ref().unwrap().name, "v");
                assert_eq!(
                    sdpa_op.attn_mask.as_ref().unwrap().name,
                    "causal_mask",
                    "attn_mask should be set in proto when attention_mask is Some"
                );
                assert!(
                    sdpa_op.has_attn_mask,
                    "has_attn_mask should be true when attention_mask is Some"
                );
                assert!(!sdpa_op.causal, "causal should remain false by default");
                assert!(
                    (sdpa_op.scale - 0.125).abs() < f32::EPSILON,
                    "scale should be 0.125 in proto"
                );
            }
            other => panic!("Expected ScaledDotProductAttentionOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sdpa_proto_emission_without_mask_and_scale() {
        // SDPA without mask or scale should emit attn_mask=None, has_attn_mask=false, scale=0.0
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_plain".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: None,
            scale: None,
        };
        let proto_op = mir_op_to_proto_op(&compat, &[]);
        assert_eq!(proto_op.name, "sdpa_plain");
        match &proto_op.operation {
            Some(proto::mil_operation::Operation::ScaledDotProductAttentionOp(sdpa_op)) => {
                assert!(
                    sdpa_op.attn_mask.is_none(),
                    "attn_mask should be None in proto when attention_mask is None"
                );
                assert!(
                    !sdpa_op.has_attn_mask,
                    "has_attn_mask should be false when attention_mask is None"
                );
                assert_eq!(sdpa_op.scale, 0.0, "scale should be 0.0 in proto when scale is None");
            }
            other => panic!("Expected ScaledDotProductAttentionOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sdpa_proto_emission_mask_only() {
        // SDPA with mask but no scale
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_masked".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: Some("mask".to_string()),
            scale: None,
        };
        let proto_op = mir_op_to_proto_op(&compat, &[]);
        match &proto_op.operation {
            Some(proto::mil_operation::Operation::ScaledDotProductAttentionOp(sdpa_op)) => {
                assert_eq!(sdpa_op.attn_mask.as_ref().unwrap().name, "mask");
                assert!(sdpa_op.has_attn_mask);
                assert_eq!(sdpa_op.scale, 0.0, "scale should default to 0.0");
            }
            other => panic!("Expected ScaledDotProductAttentionOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sdpa_proto_emission_scale_only() {
        // SDPA with scale but no mask
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_scaled".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: None,
            scale: Some(0.0625),
        };
        let proto_op = mir_op_to_proto_op(&compat, &[]);
        match &proto_op.operation {
            Some(proto::mil_operation::Operation::ScaledDotProductAttentionOp(sdpa_op)) => {
                assert!(sdpa_op.attn_mask.is_none());
                assert!(!sdpa_op.has_attn_mask);
                assert!(
                    (sdpa_op.scale - 0.0625).abs() < f32::EPSILON,
                    "scale should be 0.0625 in proto"
                );
            }
            other => panic!("Expected ScaledDotProductAttentionOp, got {:?}", other),
        }
    }

    #[test]
    fn test_sdpa_apple_proto_emission_with_mask_and_scale() {
        // Apple-proto emission should include attn_mask input and scale input
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_apple".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: Some("mask".to_string()),
            scale: Some(0.125),
        };
        let shapes =
            std::collections::HashMap::from([("sdpa_apple".to_string(), vec![1usize, 8, 64, 64])]);
        let ops = mir_op_to_apple_ops(&compat, &[], &shapes);
        assert_eq!(ops.len(), 1);
        let op = &ops[0];
        assert_eq!(op.r#type, "scaled_dot_product_attention");
        // Should have 5 inputs: query, key, value, attn_mask, scale
        assert!(op.inputs.contains_key("query"), "should have query input");
        assert!(op.inputs.contains_key("key"), "should have key input");
        assert!(op.inputs.contains_key("value"), "should have value input");
        assert!(
            op.inputs.contains_key("attn_mask"),
            "should have attn_mask input when attention_mask is Some"
        );
        assert!(op.inputs.contains_key("scale"), "should have scale input when scale is Some");
    }

    #[test]
    fn test_sdpa_apple_proto_emission_without_mask_and_scale() {
        // Apple-proto emission without mask/scale should only have q, k, v inputs
        let compat = mir_compat::MirOpCompat::ScaledDotProductAttention {
            name: "sdpa_plain".to_string(),
            query: "q".to_string(),
            key: "k".to_string(),
            value: "v".to_string(),
            attention_mask: None,
            scale: None,
        };
        let shapes =
            std::collections::HashMap::from([("sdpa_plain".to_string(), vec![1usize, 8, 64, 64])]);
        let ops = mir_op_to_apple_ops(&compat, &[], &shapes);
        assert_eq!(ops.len(), 1);
        let op = &ops[0];
        assert!(op.inputs.contains_key("query"));
        assert!(op.inputs.contains_key("key"));
        assert!(op.inputs.contains_key("value"));
        assert!(
            !op.inputs.contains_key("attn_mask"),
            "should NOT have attn_mask input when attention_mask is None"
        );
        assert!(!op.inputs.contains_key("scale"), "should NOT have scale input when scale is None");
    }

    // ─── T-43: ProtoValidationError Tests ───────────────────────────

    #[test]
    fn test_proto_validation_error_display_with_op_name() {
        let err = ProtoValidationError {
            kind: ProtoValidationKind::FillDtypeMismatch,
            message: "dtype mismatch".to_string(),
            op_name: Some("fill_out".to_string()),
        };
        let display = format!("{}", err);
        assert!(display.contains("FillDtypeMismatch"), "display should contain kind");
        assert!(display.contains("fill_out"), "display should contain op name");
        assert!(display.contains("dtype mismatch"), "display should contain message");
    }

    #[test]
    fn test_proto_validation_error_display_without_op_name() {
        let err = ProtoValidationError {
            kind: ProtoValidationKind::MatMulInnerDimMismatch,
            message: "inner dims mismatch".to_string(),
            op_name: None,
        };
        let display = format!("{}", err);
        assert!(display.contains("MatMulInnerDimMismatch"), "display should contain kind");
        assert!(display.contains("inner dims mismatch"), "display should contain message");
        assert!(!display.contains("None"), "display should not contain 'None' for op_name");
    }

    #[test]
    fn test_proto_validation_kind_display() {
        assert_eq!(format!("{}", ProtoValidationKind::FillDtypeMismatch), "FillDtypeMismatch");
        assert_eq!(
            format!("{}", ProtoValidationKind::FillFp16FloatsStorage),
            "FillFp16FloatsStorage"
        );
        assert_eq!(
            format!("{}", ProtoValidationKind::FillFp32BytesStorage),
            "FillFp32BytesStorage"
        );
        assert_eq!(
            format!("{}", ProtoValidationKind::ElementwiseBroadcastShapeMismatch),
            "ElementwiseBroadcastShapeMismatch"
        );
        assert_eq!(
            format!("{}", ProtoValidationKind::MatMulInnerDimMismatch),
            "MatMulInnerDimMismatch"
        );
        assert_eq!(
            format!("{}", ProtoValidationKind::MatMulOutputShapeMismatch),
            "MatMulOutputShapeMismatch"
        );
    }

    #[test]
    fn test_proto_validation_error_is_std_error() {
        let err = ProtoValidationError {
            kind: ProtoValidationKind::FillDtypeMismatch,
            message: "test".to_string(),
            op_name: None,
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_validate_fill_dtype_consistency_ok() {
        // A well-formed fill op with matching dtypes should pass validation
        let fill_op = mir_compat::MirOpCompat::Fill {
            name: "fill_out".to_string(),
            value: 1.0,
            shape: vec![1, 512],
            dtype: mir_compat::MilDtypeCompat::Fp16,
        };
        let ops = mir_op_to_apple_ops(&fill_op, &[], &std::collections::HashMap::new());
        let result = validate_fill_dtype_consistency(&ops);
        assert!(result.is_ok(), "valid fill should pass validation");
    }

    #[test]
    fn test_validate_elementwise_broadcast_shapes_ok() {
        // A well-formed add op should pass validation
        let add_op = mir_compat::MirOpCompat::Add {
            name: "add_out".to_string(),
            x: "a".to_string(),
            y: "b".to_string(),
        };
        let ops = mir_op_to_apple_ops(&add_op, &[], &std::collections::HashMap::new());
        let result = validate_elementwise_broadcast_shapes(&ops);
        assert!(result.is_ok(), "valid add should pass validation");
    }

    #[test]
    fn test_validate_matmul_shapes_ok() {
        // A well-formed matmul op should pass validation
        let matmul_op = mir_compat::MirOpCompat::MatMul {
            name: "mm_out".to_string(),
            x: "a".to_string(),
            y: "b".to_string(),
            transpose_y: false,
        };
        let shapes = std::collections::HashMap::from([
            ("a".to_string(), vec![1usize, 128, 64]),
            ("b".to_string(), vec![1usize, 64, 32]),
            ("mm_out".to_string(), vec![1usize, 128, 32]),
        ]);
        let ops = mir_op_to_apple_ops(&matmul_op, &[], &shapes);
        let result = validate_matmul_shapes(&ops);
        assert!(result.is_ok(), "valid matmul should pass validation");
    }

    #[test]
    fn test_validate_fill_dtype_consistency_no_fill_ops() {
        // Non-fill ops should pass fill validation trivially
        let add_op = mir_compat::MirOpCompat::Add {
            name: "add_out".to_string(),
            x: "a".to_string(),
            y: "b".to_string(),
        };
        let ops = mir_op_to_apple_ops(&add_op, &[], &std::collections::HashMap::new());
        let result = validate_fill_dtype_consistency(&ops);
        assert!(result.is_ok(), "non-fill ops should pass fill validation");
    }

    #[test]
    fn test_validate_elementwise_no_elementwise_ops() {
        // Non-elementwise ops should pass elementwise validation trivially
        let fill_op = mir_compat::MirOpCompat::Fill {
            name: "fill_out".to_string(),
            value: 1.0,
            shape: vec![1, 512],
            dtype: mir_compat::MilDtypeCompat::Fp16,
        };
        let ops = mir_op_to_apple_ops(&fill_op, &[], &std::collections::HashMap::new());
        let result = validate_elementwise_broadcast_shapes(&ops);
        assert!(result.is_ok(), "non-elementwise ops should pass elementwise validation");
    }

    #[test]
    fn test_validate_matmul_no_matmul_ops() {
        // Non-matmul ops should pass matmul validation trivially
        let add_op = mir_compat::MirOpCompat::Add {
            name: "add_out".to_string(),
            x: "a".to_string(),
            y: "b".to_string(),
        };
        let ops = mir_op_to_apple_ops(&add_op, &[], &std::collections::HashMap::new());
        let result = validate_matmul_shapes(&ops);
        assert!(result.is_ok(), "non-matmul ops should pass matmul validation");
    }

    #[test]
    fn test_validate_empty_operations() {
        // Empty operations list should pass all validations
        let ops: Vec<apple_proto::mil_spec::Operation> = vec![];
        assert!(validate_fill_dtype_consistency(&ops).is_ok());
        assert!(validate_elementwise_broadcast_shapes(&ops).is_ok());
        assert!(validate_matmul_shapes(&ops).is_ok());
    }

    #[test]
    fn test_proto_validation_error_kind_equality() {
        // ProtoValidationKind should support equality comparison
        assert_eq!(ProtoValidationKind::FillDtypeMismatch, ProtoValidationKind::FillDtypeMismatch);
        assert_ne!(
            ProtoValidationKind::FillDtypeMismatch,
            ProtoValidationKind::FillFp16FloatsStorage
        );
    }

    #[test]
    fn test_proto_validation_error_clone() {
        let err = ProtoValidationError {
            kind: ProtoValidationKind::MatMulOutputShapeMismatch,
            message: "shape mismatch".to_string(),
            op_name: Some("mm_out".to_string()),
        };
        let cloned = err.clone();
        assert_eq!(cloned.kind, err.kind);
        assert_eq!(cloned.message, err.message);
        assert_eq!(cloned.op_name, err.op_name);
    }

    #[test]
    fn test_convert_to_apple_proto_model_valid_model() {
        // A minimal valid model should produce Ok
        let model = CoreMlModel {
            spec_version: crate::SpecVersion::V7,
            description: crate::ModelDescriptionCompat {
                inputs: vec![crate::TensorDesc {
                    name: "x".to_string(),
                    dtype: crate::CoreMlDataType::Float16,
                    shape: vec![1, 128],
                    is_state: false,
                }],
                outputs: vec![crate::TensorDesc {
                    name: "y".to_string(),
                    dtype: crate::CoreMlDataType::Float16,
                    shape: vec![1, 128],
                    is_state: false,
                }],
                states: vec![],
            },
            functions: vec![CoreMlFunction {
                name: "main".to_string(),
                inputs: vec![crate::TensorDesc {
                    name: "x".to_string(),
                    dtype: crate::CoreMlDataType::Float16,
                    shape: vec![1, 128],
                    is_state: false,
                }],
                outputs: vec![crate::TensorDesc {
                    name: "y".to_string(),
                    dtype: crate::CoreMlDataType::Float16,
                    shape: vec![1, 128],
                    is_state: false,
                }],
                states: vec![],
                operations: vec![mir_compat::MirOpCompat::Identity {
                    name: "y".to_string(),
                    x: "x".to_string(),
                    dtype: mir_compat::MilDtypeCompat::Fp16,
                }],
                node_shapes: std::collections::HashMap::new(),
            }],
            default_function_name: "main".to_string(),
            weights: vec![],
            shared_weights: vec![],
            compute_unit: crate::CoreMlComputeUnit::CpuAndNe,
            user_defined_metadata: HashMap::new(),
        };
        let result = convert_to_apple_proto_model(&model, &[]);
        assert!(result.is_ok(), "valid model should produce Ok: {:?}", result.err());
    }
}
