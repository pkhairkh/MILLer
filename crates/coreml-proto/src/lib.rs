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

/// Prost-generated Core ML protobuf types.
/// These are the real protobuf message types compiled from the .proto files.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/coreml.rs"));
}

use serde::{Deserialize, Serialize};

// ─── Core ML Types (hand-written for environments without proto compilation) ─

/// Data type for tensor elements.
/// Mirrors `ArrayDataType` from DataStructures.proto.
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
}

impl CoreMlDataType {
    /// Convert from MIR dtype to Core ML data type.
    pub fn from_mir_dtype(mir_dtype: &crate::mir_compat::MilDtypeCompat) -> Self {
        match mir_dtype {
            crate::mir_compat::MilDtypeCompat::Fp16 => CoreMlDataType::Float16,
            crate::mir_compat::MilDtypeCompat::Fp32 => CoreMlDataType::Float32,
            crate::mir_compat::MilDtypeCompat::Int32 => CoreMlDataType::Int32,
            crate::mir_compat::MilDtypeCompat::UInt8 => CoreMlDataType::UInt8,
        }
    }

    /// Size in bytes per element.
    pub fn element_size(&self) -> usize {
        match self {
            CoreMlDataType::Float32 | CoreMlDataType::Float64 => 4,
            CoreMlDataType::Float16 => 2,
            CoreMlDataType::Int32 => 4,
            CoreMlDataType::UInt8 | CoreMlDataType::Int8 | CoreMlDataType::Bool => 1,
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
}

impl SpecVersion {
    /// Protobuf specification version number.
    pub fn proto_value(&self) -> i32 {
        match self {
            SpecVersion::V7 => 7,
            SpecVersion::V8 => 8,
        }
    }

    /// Whether this version supports stateful models (mb.read_state / mb.coreml_update_state).
    pub fn supports_state(&self) -> bool {
        matches!(self, SpecVersion::V8)
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
/// the package structure. This is the Rust-side representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    /// Schema version (currently "1.0").
    #[serde(rename = "schemaVersion")]
    pub schema_version: String,
    /// Model identifier string.
    #[serde(rename = "modelId")]
    pub model_id: String,
    /// List of file entries in the package.
    pub files: Vec<PackageManifestEntry>,
    /// Metadata about the model.
    pub metadata: PackageManifestMetadata,
}

/// A single file entry in the package manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifestEntry {
    /// Path relative to the mlpackage root.
    pub path: String,
    /// File role (e.g., "model", "weights").
    pub role: String,
}

/// Metadata in the package manifest.
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
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
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
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub enum MilDtypeCompat {
        Fp16,
        Fp32,
        Int32,
        UInt8,
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
            shape: Vec<i64>,
        },
        Transpose {
            name: String,
            x: String,
            perm: Vec<i64>,
        },
        SliceByIndex {
            name: String,
            x: String,
            begin: Vec<i64>,
            end: Vec<i64>,
        },
        SliceUpdate {
            name: String,
            x: String,
            update: String,
            begin: Vec<i64>,
            end: Vec<i64>,
        },
        Concat {
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
        /// Catch-all for MIL ops that don't have specialized compat representations.
        /// The proto emission layer handles these by emitting the appropriate
        /// MIL builder call based on the op_kind string.
        Unsupported {
            op_kind: String,
            name: String,
            /// Serialized JSON of the op's parameters for flexible emission
            params_json: String,
        },
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
    }
}

/// Convert `MilDtypeCompat` → `proto::ArrayDataType` (prost enum as i32).
pub fn mil_dtype_to_proto(dt: &mir_compat::MilDtypeCompat) -> i32 {
    match dt {
        mir_compat::MilDtypeCompat::Fp16 => proto::ArrayDataType::Float16 as i32,
        mir_compat::MilDtypeCompat::Fp32 => proto::ArrayDataType::Float32 as i32,
        mir_compat::MilDtypeCompat::Int32 => proto::ArrayDataType::Int32 as i32,
        mir_compat::MilDtypeCompat::UInt8 => proto::ArrayDataType::Uint8 as i32,
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
        | mir_compat::MilDtypeCompat::UInt8 => {
            proto::weight_value::Value::FloatValue(data.to_vec())
        }
        mir_compat::MilDtypeCompat::Int32 => proto::weight_value::Value::IntValue(data.to_vec()),
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
        mir_compat::MirOpCompat::MatMul { name, x, y } => (
            name.clone(),
            proto::mil_operation::Operation::MatmulOp(proto::MilMatMulOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                y: Some(proto::OperandRef { name: y.clone() }),
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
                shape: shape.clone(),
            }),
        ),
        mir_compat::MirOpCompat::Transpose { name, x, perm } => (
            name.clone(),
            proto::mil_operation::Operation::TransposeOp(proto::MilTransposeOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                perm: perm.clone(),
            }),
        ),
        mir_compat::MirOpCompat::SliceByIndex { name, x, begin, end } => (
            name.clone(),
            proto::mil_operation::Operation::SliceByIndexOp(proto::MilSliceByIndexOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                begin: begin.clone(),
                end: end.clone(),
                stride: vec![],
                begin_mask: 0,
                end_mask: 0,
            }),
        ),
        mir_compat::MirOpCompat::SliceUpdate { name, x, update, begin, end } => (
            name.clone(),
            proto::mil_operation::Operation::SliceUpdateOp(proto::MilSliceUpdateOp {
                x: Some(proto::OperandRef { name: x.clone() }),
                update: Some(proto::OperandRef { name: update.clone() }),
                begin: begin.clone(),
                end: end.clone(),
            }),
        ),
        mir_compat::MirOpCompat::Concat { name, values, axis } => (
            name.clone(),
            proto::mil_operation::Operation::ConcatOp(proto::MilConcatOp {
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
        mir_compat::MirOpCompat::ScaledDotProductAttention { name, query, key, value } => (
            name.clone(),
            proto::mil_operation::Operation::ScaledDotProductAttentionOp(
                proto::MilScaledDotProductAttentionOp {
                    query: Some(proto::OperandRef { name: query.clone() }),
                    key: Some(proto::OperandRef { name: key.clone() }),
                    value: Some(proto::OperandRef { name: value.clone() }),
                    attn_mask: None,
                    has_attn_mask: false,
                    causal: false,
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
        mir_compat::MirOpCompat::Conv { name, x, weight, pad_type, groups } => {
            let weight_data = weight_entries
                .iter()
                .find(|w| w.name == *weight)
                .map(weight_entry_to_proto)
                .unwrap_or_else(|| proto::WeightData { weight_data: None });
            (
                name.clone(),
                proto::mil_operation::Operation::ConvOp(proto::MilConvOp {
                    x: Some(proto::OperandRef { name: x.clone() }),
                    weight: Some(weight_data),
                    pad_type: pad_type.clone(),
                    groups: *groups,
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
        // Unsupported ops are emitted as identity pass-through with a comment
        // marker in the function name. The op_kind and params are preserved
        // for downstream Python emission or manual inspection.
        mir_compat::MirOpCompat::Unsupported { op_kind, name, params_json: _ } => {
            // Emit as identity to preserve graph structure; downstream
            // Python bridge will handle the actual op emission.
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
        assert_eq!(CoreMlDataType::Int32.element_size(), 4);
        assert_eq!(CoreMlDataType::UInt8.element_size(), 1);
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
            spec_version: SpecVersion::V8,
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

        // Verify key fields
        assert_eq!(
            proto_model.specification_version,
            proto::SpecificationVersion::SpecificationVersion8 as i32
        );
        assert!(proto_model.description.is_some());
        assert!(proto_model.ml_program.is_some());

        let ml_prog = proto_model.ml_program.as_ref().unwrap();
        assert!(ml_prog.functions.contains_key("main"));

        // Serialize to protobuf bytes
        let bytes = proto_model.encode_to_vec();
        assert!(!bytes.is_empty());

        // Parse back
        let parsed = proto::Model::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            parsed.specification_version,
            proto::SpecificationVersion::SpecificationVersion8 as i32
        );
        assert!(parsed.ml_program.is_some());
        assert!(parsed.ml_program.as_ref().unwrap().functions.contains_key("main"));
    }
}
