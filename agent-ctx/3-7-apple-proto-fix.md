# Task 3-7: Fix Apple Protobuf Wire Format Incompatibility

## Summary

Replaced the custom protobuf definitions with Apple's actual wire format, fixing the "Cannot decode metadata" error when Core ML tries to open emitted `.mlpackage` files.

## Changes Made

### 1. New Apple-compatible proto files (`proto/coremlv2/`)

Created three new proto files that match Apple's actual format exactly (same field numbers, same package names):

- **`FeatureTypes.proto`** — `package CoreML.Specification`
  - `ArrayFeatureType` with correct enum values: FLOAT32=65568, FLOAT16=65552, INT32=131104
  - `FeatureType` with multiArrayType at field 5
  - `StateFeatureType`, `SizeRange`, etc.

- **`MIL.proto`** — `package CoreML.Specification.MILSpec`
  - `Program` with `map<string, Function> functions`
  - `Function` with `repeated NamedValueType inputs`, `string opset`, `map<string, Block> block_specializations`
  - `Block` with `repeated NamedValueType inputs`, `repeated string outputs`, `repeated Operation operations`
  - `Operation` with generic `string type`, `map<string, Argument> inputs`, `repeated NamedValueType outputs`
  - `Argument.Binding` with `string name` (SSA ref) or `Value value` (constant)
  - `Value` with `ImmediateValue` or `BlobFileValue` (fileName="weight.bin", offset)
  - `DataType` enum: FLOAT16=10, FLOAT32=11, INT32=23, etc.
  - `TensorType`, `Dimension`, `ValueType`, etc.

- **`Model.proto`** — `package CoreML.Specification`
  - `Model` with `int32 specificationVersion`, `ModelDescription description`, `MILSpec.Program mlProgram` at field **502**
  - `ModelDescription` with `repeated FunctionDescription functions` at field 20, `defaultFunctionName` at 21, `Metadata metadata` at 100
  - `FunctionDescription` with name, input, output, state
  - `Metadata` with shortDescription, versionString, author, license, `map<string, string> userDefined` at 100

### 2. Updated `build.rs`

Added compilation of the new `coremlv2/` proto files alongside the legacy `coreml/` files. The generated Rust types are placed in separate modules:
- `coreml.rs` — legacy format
- `core_ml.specification.rs` — Apple Model/FeatureTypes
- `core_ml.specification.mil_spec.rs` — Apple MIL

### 3. Updated `lib.rs` — New `apple_proto` module + conversion functions

Added `apple_proto` module with `mil_spec` submodule to expose the new generated types.

Added `convert_to_apple_proto_model()` function that produces an `apple_proto::Model` using Apple's actual wire format:
- Uses `MILSpec.Program` (field 502) instead of legacy `MLProgram` (field 20)
- Operations use generic `type` + `inputs` + `outputs` format
- Data types use Apple enum values (FLOAT16=10, FLOAT32=11)
- Weight references use `BlobFileValue` with `fileName="weight.bin"`
- Model description uses `FunctionDescription` at field 20

Added helper functions:
- `mil_dtype_to_apple()` — MirDtypeCompat → Apple MILSpec.DataType
- `coreml_dtype_to_apple_mil()` — CoreMlDataType → Apple MILSpec.DataType
- `coreml_dtype_to_apple_array()` — CoreMlDataType → Apple ArrayFeatureType.ArrayDataType
- `mir_op_to_apple_op()` — Full mapping of all 36 MirOpCompat variants to generic MILSpec.Operation format
- `function_to_apple_proto()` — CoreMlFunction → Apple MILSpec.Function with block_specializations

Legacy code preserved: `convert_to_proto_model()` and all legacy `mir_op_to_proto_op()` code still work.

### 4. Updated `mir_to_proto.rs`

Changed `model_to_protobuf_bytes()` to use `convert_to_apple_proto_model()` instead of the legacy `convert_to_proto_model()`. Updated all test assertions to use `apple_proto::Model` for deserialization.

### 5. Fixed `package.rs`

- `schemaVersion`: "1.0" → "1.0.0" (Core ML requires semver format)
- `user_defined` → `userDefined` (added `#[serde(rename = "userDefined")]`)

## Key Format Differences (Old → Apple)

| Aspect | MILLer Legacy | Apple Actual |
|--------|--------------|-------------|
| Model type field | 20 (MLProgram) | 502 (MILSpec.Program) |
| Operations | per-op-type `oneof` messages | generic `string type` + `map<string,Argument> inputs` |
| MILSpec.DataType | N/A (custom ArrayDataType) | FLOAT16=10, FLOAT32=11, INT32=23 |
| ArrayFeatureType | FLOAT32=1, FLOAT16=2 | FLOAT32=65568, FLOAT16=65552 |
| Function structure | `MilFunction { block }` | `Function { inputs, opset, block_specializations }` |
| Weight references | `FileReference { offset, size }` | `BlobFileValue { fileName, offset }` |
| ModelDescription | custom structure | `functions` at field 20, `metadata` at field 100 |
| Package name | `coreml` | `CoreML.Specification` / `CoreML.Specification.MILSpec` |

## Test Results

All 551 tests pass across the workspace:
- `ane-coreml-proto`: 15 tests (legacy format preserved)
- `ane-coreml-emit`: 31 tests (new Apple format tests added)
- `ane-bridge`: 27 tests
- All other crates: pass

New tests specifically validate:
- Apple protobuf serialization roundtrip
- Correct field numbers (specVersion=1, mlProgram=502, functions=20)
- BlobFileValue weight references with correct fileName and offset
- Generic Operation format with string type and named arguments
- Correct enum values (ArrayFeatureType and MILSpec.DataType)
