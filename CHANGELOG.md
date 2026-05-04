# Changelog

## Current Status — 2026-05-04

- **1561 tests passing**, 0 failures
- IR Cleanliness Score: 89%
- 0 clippy warnings, 0 errors
- **20 open tasks** (1 CRITICAL, 3 HIGH, 8 MEDIUM, 8 LOW) — see [TASKS.md](TASKS.md)
- **24 tasks resolved** across sprint cycles

Audit details: [docs/audit/tabula-rasa-v3.md](docs/audit/tabula-rasa-v3.md)
Violation report: [docs/audit/ane-violations.md](docs/audit/ane-violations.md)

---

## [sprint-ane-runtime-safety] — 2026-05-04

### Sprint: ANE Runtime Safety & Palettization Hardening (T-118, T-119, T-120, T-122)

Resolved 4 tasks (all MEDIUM) from the NECROSCOPY forensic audit
(ane-violations.md). All changes add runtime safety checks for Orion
constraints that previously had no compile-time or runtime enforcement,
preventing silent ANE failures.

#### Tasks Resolved

| Task | Description | Issues Fixed |
|------|-------------|--------------|
| T-118 | Add Palette Bits Validation with Version-Conditional Support | I-93 |
| T-119 | Add Minimum IOSurface Size Validation | I-94 |
| T-120 | Add Compilation Count Per Process Tracking | I-95 |
| T-122 | Add Weight Dict Initialization Check | I-97 |

#### Added

- T-118: `validate_palette_bits_for_family()` in `ane_layout.rs` — validates
  palette bit-widths against target hardware family. 3-bit and 6-bit
  palettization rejected on A11Legacy/A12/A13 (A14+ only per ANEC binary
  evidence). Uses existing `uses_a14minus_converters()` for family detection.
  Re-exported from `palettize_weights.rs`. 10 new tests.
- T-119: `MIN_IOSURFACE_BYTES` constant (~49 KB per Orion #4) and
  `validate_iosurface_sizes()` in `mir_to_proto.rs`. Computes output buffer
  sizes (shape_product × dtype_size) and warns when below minimum. Called
  from `convert_mir_to_proto_multifunction()`. 4 new tests.
- T-120: Global `COMPILATION_COUNT: AtomicU64` counter in `emitter.rs`.
  `emit_model()` increments counter, returns error at limit (119), warns
  at threshold (95). Public constants `COMPILATION_LIMIT`,
  `COMPILATION_WARNING_THRESHOLD`, function `compilation_count()`.
  Added `compilation_number: u64` to `ProtoEmitResult`. 3 new tests.
- T-122: Weight dictionary validation in `MlPackageWriter::write()` — warns
  when model has functions but zero weights (nil weight dict crashes ANEC
  per Orion #11). 2 new tests.

#### Changed

- `ane_layout.rs`: New `validate_palette_bits_for_family()` function alongside
  existing `validate_palette_bits()` for backward compatibility.
- `mir_to_proto.rs`: New `validate_iosurface_sizes()` function called before
  model return in `convert_mir_to_proto_multifunction()`.
- `emitter.rs`: `emit_model()` now tracks compilation count via global atomic.
  `ProtoEmitResult` gains `compilation_number` field. Added `log` dependency
  to `ane-coreml-emit`.
- `package.rs`: Weight dictionary validation added before weight.bin build.
- `lib.rs` (coreml-emit): New public exports for T-120 constants and function.
- `palettize_weights.rs`: Re-exports `validate_palette_bits_for_family`.

#### Tests

- T-118: 10 tests covering 3-bit/6-bit rejection on A11Legacy/A12/A13,
  acceptance on A14+, all-families coverage for 1/2/4/8-bit, None family
  basic check.
- T-119: 4 tests covering constant value, small/large buffer detection,
  buffer size computation correctness.
- T-120: 3 tests covering limit constants, count increment, ProtoEmitResult
  field existence.
- T-122: 2 tests covering empty weights with functions (warning) and empty
  weights without functions (no warning).

#### Issues Closed (4 issues)

I-93, I-94, I-95, I-97

---

## [sprint-pir-shard-hardening] — 2026-05-04

### Sprint: PIR/Shard Plan Hardening (T-109, T-104, T-114, T-115)

Resolved 4 tasks (all MEDIUM) from the NECROSCOPY forensic audit
(ane-violations.md). All changes harden the shard plan and PIR pipeline
by removing hardcoded defaults, adding error enforcement, and making
configuration explicit and independently controllable.

#### Tasks Resolved

| Task | Description | Issues Fixed |
|------|-------------|--------------|
| T-109 | Make StateTopologyPass Return Errors | I-84 |
| T-104 | Derive State Shape from ReadState Op | I-79, I-89 |
| T-114 | Fix PIR Tensor Spec Dtype Hardcoding | I-89 |
| T-115 | Make Opset Version and Deployment Target Configurable | I-90 |

#### Added

- T-109: `strict: bool` field on `StateTopologyPass` (default: true). Strict mode
  returns `Err` for ReadState without matching WriteState. WriteState without
  ReadState still logs info (valid for initial state writes). Added
  `new_lenient()` and `with_strict()` constructors. Module-level doc comment
  expanded with strict mode documentation.
- T-104: `read_state_map` in `convert_mir_to_proto_multifunction()` collects all
  ReadState ops before the state-building loop. `CoremlUpdateState` and
  `StateWrite` now derive shape/dtype from ReadState map. Returns explicit
  error when no ReadState exists and shape is empty (Core ML rejects empty-
  dimension state tensors).
- T-114: `derive_primary_dtype()` method on `ShardPlanPass` scans SIR graph for
  dtype-bearing ops (Const, Cast, Fill, FillLike, Quantize, Dequantize).
  Replaced all 9 hardcoded `"fp16"` values in `shard_plan.rs` with
  `primary_dtype.clone()`. Fallback to `"fp16"` with explicit `log::warn!`
  when no dtype-bearing op is found.
- T-115: `DEFAULT_MINIMUM_DEPLOYMENT_TARGET` constant in `lib.rs`, separate
  from `DEFAULT_OPSET_VERSION`. `minimum_deployment_target` field on
  `ShardPipelineSpec`, decoupled from `opset_version`. Builder method
  `with_deployment_target()` for independent configuration. Updated
  `shard_desc.rs` and `serialize.rs` to use the separate constant.

#### Changed

- `state_topology.rs`: Added `strict` field and conditional error returns.
  `ReadState` without matching `WriteState` now returns `Err` in strict mode
  (default). Non-strict mode preserves old warning-only behavior.
- `mir_to_proto.rs`: State declarations no longer default to empty shape and
  Float16 dtype. Shape and dtype are derived from ReadState ops. Missing
  ReadState for a state is now a hard error.
- `shard_plan.rs`: All TensorSpec dtype fields now use `primary_dtype.clone()`
  instead of hardcoded `"fp16"`. StateDeclaration and Handoff dtype fields
  similarly updated. Opset and deployment target use named constants instead
  of inline `"iOS18"` strings.
- `pir.rs`: `ShardPipelineSpec` gains `minimum_deployment_target` field and
  `with_deployment_target()` builder. `to_pir_graph()` uses the dedicated
  field instead of cloning `opset_version`.
- `lib.rs`: New `DEFAULT_MINIMUM_DEPLOYMENT_TARGET` constant with
  documentation explaining the decoupling rationale.
- `serialize.rs`: Uses `DEFAULT_MINIMUM_DEPLOYMENT_TARGET` constant.
- `shard_desc.rs`: Uses `DEFAULT_MINIMUM_DEPLOYMENT_TARGET` constant.

#### Tests

- T-109: Strict mode validation (ReadState without WriteState returns Err),
  non-strict mode validation (warning only), existing tests updated.
- T-114: `test_t114_derive_primary_dtype_fallback` — fallback to "fp16" when
  no dtype-bearing op in graph. `test_t114_derive_primary_dtype_from_const` —
  derives "fp32" from Const op. `test_t114_build_sharded_plan_fp32_dtype` —
  verifies fp32 propagation through PIR packages, inputs, outputs, and
  handoffs.
- T-115: `test_deployment_target_independent_of_opset` — verifies
  `with_deployment_target()` sets target independently. Default values
  verified for both `opset_version` and `minimum_deployment_target`.
- T-104: State shape derivation from ReadState verified. Error behavior for
  missing ReadState verified.

#### Issues Closed (4 issues)

I-79, I-84, I-89, I-90

---

## [sprint-dtype-validation-hardening] — 2026-03-05

### Sprint: Dtype Validation & Hard Error Hardening (T-97, T-101, T-102, T-103, T-111)

Resolved 5 tasks (2 HIGH, 3 MEDIUM) from the NECROSCOPY forensic audit
(ane-violations.md). All changes close dtype and weight validation gaps that
allowed invalid models to pass through to the ANE compiler, or replace silent
fallback defaults with hard errors.

#### Tasks Resolved

| Task | Description | Issues Fixed |
|------|-------------|--------------|
| T-97 | Add Dtype Cross-Validation and Rejection | I-72, I-98, I-99, I-100, I-102 |
| T-101 | Replace Fallback Shapes/Dtypes with Hard Errors | I-76, I-86 |
| T-102 | Fix F32 Weight Passthrough Without FP16 Conversion | I-77, I-87 |
| T-103 | Map Bool/Float64/Unknown Dtypes Correctly in Weights | I-78, I-88 |
| T-111 | Fix Interleave Validation When Channels Unknown | I-83 |

#### Added

- T-97: `CrossTypeViolation` and `AsymmetricQuantViolation` error variants in
  `dtype_constraints.rs`. `validate_cross_type_compatibility()` for BF16/F16
  cross-type checks, rejecting all 9 documented ANEC cross-type combinations.
  `is_fp32_compute_supported()` — returns false for A11Legacy/A12 families.
  `validate_anec_quantization_symmetry()` — rejects asymmetric quant on ANE.
  E5M2 removed from quantize validator accepted output dtypes. Comprehensive
  tests for all new functions.
- T-102: `convert_f32_to_fp16()` in `safetensors_resolver.rs`. F32 safetensors
  data now converts to FP16 using the same path as BF16→FP16 (via
  `half::f16::from_f32()`). Tests: byte size halving, value preservation,
  special values (NaN/Inf/subnormals), same-path-as-BF16 verification.
- T-103: `coreml_dtype_to_blob_dtype()` now returns `Result<u32>` instead of
  `u32`. Bool, Float64, and Unknown dtypes return explicit errors instead of
  silently mapping to Float32. `WeightBinBuilder::build()` returns
  `Result<WeightBinResult>`. All callers and tests updated.
- T-111: Interleave validation now enforces valid interleave factors, const→1,
  and int4/uint4→8 even when channels is None. Previously the entire validation
  was skipped when channels was unknown.

#### Changed

- `dtype_constraints.rs`: New error variants and validation functions for
  cross-type, FP32 architecture-conditional, and asymmetric quantization checks.
  E5M2 removed from quantize validator accepted output dtypes (V-051, V-111).
- `safetensors_resolver.rs`: F32 weights now convert to FP16 instead of passing
  through raw bytes. Uses same conversion path as existing BF16→FP16.
- `weights.rs`: `coreml_dtype_to_blob_dtype()` signature changed from `u32` to
  `Result<u32>`. Bool, Float64, Unknown return explicit errors. All callers
  updated.
- `mir_to_compat.rs`: Missing input/output MIR nodes now produce `bail!()`
  errors instead of falling back to shape `[1]`, dtype Fp16. No more silent
  wrong defaults.
- `role_mir.rs`: Now populates `input_shapes` from ShardSpec, enabling
  hard-error path in mir_to_compat.
- `placement_validate.rs`: Interleave validation no longer skipped when channels
  is None. Non-channel-dependent checks (valid factors, const→1, int4/uint4→8)
  always enforced.

#### Tests

- Cross-type compatibility validation (BF16/F16 mixed operand rejection)
- FP32 architecture-conditional check (A11Legacy/A12 rejection)
- Asymmetric quantization rejection on ANE
- E5M2 removed from accepted quantize output dtypes
- F32→FP16 conversion: byte size halving, value preservation, special values
- Bool/Float64/Unknown dtype explicit error return
- WeightBinBuilder Result return type propagation
- Missing I/O node hard error behavior
- Interleave validation with channels=None

#### Issues Closed (12 issues)

I-72, I-76, I-77, I-78, I-83, I-86, I-87, I-88, I-98, I-99, I-100, I-102

---

## [sprint-constraint-validation] — 2026-05-04

### Sprint: ANE Constraint Validation (T-92, T-93, T-94, T-99, T-100)

Resolved 5 HIGH-severity audit findings from the NECROSCOPY forensic audit
(ane-violations.md). All changes add compile-time constraint validation that
catches violations before models reach the ANE compiler.

#### Added
- T-92: Conv kernel power-of-2 validation; stencil (depthwise conv) constraints
  (5D, non-4D kernel, non-sum reduction, dilated, strided rejection)
- T-93: ANEC large kernel mode constraints (LARGE_KERNEL_THRESHOLD=16, W/H
  multiple of 8, stride 1-2, no depth>1, no grouped, no dilation)
- T-94: Deconvolution constraint validation (no dilation, SOx==2, no large
  kernel, no vector palettization, stride>2+depth>1 rejection)
- T-99: Conv 32K-channel limit (Orion #16) — `max_conv_channels` field added
  to AneHwLimits with `validate_conv_channels()` method
- T-100: Non-constant gather axis rejection per ANEC binary evidence

#### Changed
- `validate_conv_constraints()`: Now checks power-of-2 for kernel W/H/D before
  the existing range check. Non-power-of-2 sizes (3,5,6,7) are now rejected.
- `validate_gather_constraints()`: New `axis_is_constant` parameter (breaking
  API change). All callers updated.
- `AneHwLimits`: New `max_conv_channels: u64` field (32768 for all revisions).
- `ane_hw_limits_seed.json`: New `max_conv_channels` field for all 11 revisions.
- `LARGE_KERNEL_THRESHOLD` named constant replaces inline `16` literals.

#### Tests
- 156 new unit tests across op_constraints and ane_hw_limits modules
- Total: 708 tests pass (276 ane-ir + 432 ane-passes), 0 failures

---

## 2026-05-04 — Test Coverage & Code Quality Sprint

### Resolved (T-86, T-87, T-88, T-89, T-90)

| Task | Description | Key Change |
|------|-------------|------------|
| T-86 | Add Tests for Zero-Coverage Lab Modules | Added 33 tests across `device_meta.rs` (9), `run_dir.rs` (17), `host_inspect.rs` (7). Covers host_only/device_backed factory methods, all 11 chip→device class mappings, is_device_backed, JSON serialization roundtrips, layout constants, LabRunWriter construction/directory creation/write methods, directory validation, generate_run_id format, and host inspector logic. |
| T-87 | Add Tests for Zero-Coverage Report/Trace/Passes Modules | Added 39 tests across `json_report.rs` (9), `graph.rs` (15), `state_topology.rs` (5), `knowledge_query.rs` (10). Covers report generation, all 34 TracedOp variant JSON roundtrips, TensorShape methods, StateTopologyPass behavior, and NoKnowledge query methods. |
| T-88 | Code Quality Sweep — CQ-9, CQ-15, CQ-21, eprintln in StateTopology | Fixed 4 issues: (1) Added `log::warn!()` deprecation notice when max_seq_len defaults to 32768 in mir_to_compat.rs. (2) Ran `cargo fmt --all` across 16 files. (3) Replaced `.unwrap()` with `.expect("write to String cannot fail")` in session.rs. (4) Replaced `eprintln!` with `log::warn!`/`log::info!` in state_topology.rs. |
| T-89 | Expand Precision Hazard Op Pattern Coverage | Expanded `op_pattern_for_node()` from 14 to 47 specific pattern strings. Added coverage for normalization, linear/FC, convolution, elementwise, reduction, pooling, tensor transform, scatter/gather, attention, quantization, and constants. Added comprehensive test verifying 12 key pattern mappings. |
| T-90 | Fix Attention Reshape Placeholder Zero Warning | Added `log::warn!()` when `DecompositionContext` is None in `decompose_attention_block()`. This makes the placeholder-zero problem visible in logs rather than silently producing invalid shapes. |

### New Tests Added (74 tests)

**device_meta.rs (9 tests):**
- `test_device_metadata_host_only` — all fields of host_only() verified
- `test_device_metadata_device_backed_on_non_apple` — returns HostOnly on non-Apple
- `test_parse_device_class_known_chips` — all 11 chip→device class mappings
- `test_parse_device_class_unknown_chip` — returns None for unknown chips
- `test_is_device_backed_host_only` — returns false for host-only metadata
- `test_metadata_source_serialization` — MetadataSource roundtrip
- `test_run_type_serialization` — RunType::Cold and RunType::Warm roundtrip
- `test_execution_context_serialization` — ExecutionContext roundtrip
- `test_device_metadata_serialization` — DeviceMetadata::host_only() roundtrip

**run_dir.rs (17 tests):**
- `test_layout_constants` — all layout constants match expected strings
- `test_lab_run_writer_new` — construction stores output_dir
- `test_create_run_directory` — creates mlpackage/ and knowledge/ subdirs
- `test_write_run_record` — writes valid JSON LabRun
- `test_write_manifest` — writes JSON manifest
- `test_write_mir` — writes JSON MIR dump
- `test_write_knowledge_update` — writes to knowledge/update_task.json
- `test_write_inspection` — writes inspection JSON
- `test_write_timing` — writes timing JSON
- `test_write_fallback` — writes fallback JSON
- `test_write_baseline` — writes baseline JSON
- `test_write_drift` — writes drift JSON
- `test_validate_run_directory_missing_required` — missing run.json and manifest.json flagged
- `test_validate_run_directory_valid` — full structure validates clean
- `test_validate_run_directory_nonexistent` — "does not exist" issue returned
- `test_generate_run_id_format` — format starts with "run_"
- `test_generate_run_id_with_sha256_prefix` — sha256: prefix handled correctly

**host_inspect.rs (7 tests):**
- `test_host_inspector_new` — construction stores paths
- `test_inspect_nonexistent_path` — package_present=false for missing path
- `test_inspect_empty_directory` — package_present=true, manifest_readable=false
- `test_inspect_with_valid_manifest` — manifest_readable=true with valid JSON
- `test_inspect_with_invalid_manifest_json` — manifest_readable=false, warnings mention invalid JSON
- `test_inspect_with_empty_weights_dir` — warnings mention empty weights
- `test_structure_inspection_result_default_fields` — Python bridge fails gracefully on Linux

**json_report.rs (9 tests):**
- `test_json_reporter_new` — creates without error
- `test_json_report_default` — Default trait matches new()
- `test_generate_compilation_report` — report_type, version, data fields verified
- `test_generate_compilation_report_with_bridge_result` — bridge_result section has total_size_bytes
- `test_generate_compilation_report_with_error` — error field appears
- `test_generate_knowledge_report` — report_type="knowledge", observation_count
- `test_generate_diagnostics_report` — report_type="diagnostics"
- `test_write_to_file` — file roundtrip
- `test_json_report_serialization_roundtrip` — all fields preserved

**graph.rs (15 tests):**
- `test_tensor_shape_rank` — rank() for various shapes
- `test_tensor_shape_num_elements` — element counting
- `test_tensor_shape_ane_compatible` — rank <= 5 returns true
- `test_tensor_shape_default` — empty dims and dtype
- `test_traced_op_serialization` — all 34 variants JSON roundtrip with #[serde(tag="type")]
- `test_model_config_serialization` — roundtrip with rope_theta default
- `test_discovered_features_default` — all fields empty/zero/false
- `test_weight_info_serialization` — with/without quantized field
- `test_weight_name_map_entry_serialization` — module_path, weight, bias
- `test_state_declaration_serialization` — state_id, shape, dtype, layer_idx, is_key
- `test_traced_graph_minimal_deserialization` — minimal JSON deserializes
- `test_trace_metadata_serialization` — timestamp, duration, num_nodes
- `test_traced_node_serialization` — id, op, name, inputs, output_shape, is_parameter
- `test_quantized_weight_info_serialization` — scheme and bit_width
- `test_tensor_spec_serialization` — name and shape

**state_topology.rs (5 tests):**
- `test_state_topology_pass_new` — construction
- `test_state_topology_pass_default` — Default trait
- `test_run_stateless_graph` — no StateRead/StateWrite → no-op
- `test_run_graph_with_state_read_and_write` — matching read/write → Ok
- `test_run_graph_with_read_no_write` — read without write → Ok with warning

**knowledge_query.rs (10 tests):**
- `test_no_knowledge_query_legality` — returns None
- `test_no_knowledge_query_risk` — returns None
- `test_no_knowledge_query_precision_hazard` — returns None
- `test_no_knowledge_query_compute_plan` — returns None
- `test_legality_info_construction` — all fields verified
- `test_risk_info_construction` — all fields verified
- `test_precision_hazard_info_construction` — all fields including description
- `test_compute_plan_placement_info_construction` — all fields verified
- `test_legality_info_debug_format` — Debug contains field names
- `test_risk_info_debug_format` — Debug contains field names

**precision_policy.rs (1 test):**
- `test_expanded_op_patterns_cover_key_categories` — verifies 12 key pattern mappings for DecodeStep, Sampler, LayerNorm, BatchNorm, MatMul, Conv, Silu, Gelu, MaxPool, ReduceMean, ScaledDotProductAttention, Quantize, Gather

### Files Modified

- `crates/lab/src/device_meta.rs` — 9 new tests
- `crates/lab/src/run_dir.rs` — 17 new tests
- `crates/lab/src/host_inspect.rs` — 7 new tests
- `crates/report/src/json_report.rs` — 9 new tests
- `crates/report/Cargo.toml` — added tempfile dev-dependency
- `crates/trace/src/graph.rs` — 15 new tests
- `crates/passes/src/state_topology.rs` — 5 new tests + eprintln→log::warn!/log::info!
- `crates/passes/src/knowledge_query.rs` — 10 new tests
- `crates/passes/src/precision_policy.rs` — expanded op_pattern_for_node (14→47 patterns) + 1 new test
- `crates/bridge/src/mir_to_compat.rs` — added log::warn! for CQ-9 max_seq_len default
- `crates/lab/src/session.rs` — .unwrap()→.expect() for CQ-21
- `crates/passes/src/legality_rewrite.rs` — added log::warn! for B-12 attention reshape
- 16 files — cargo fmt --all applied (CQ-15)
- `TASKS.md` — T-86 through T-90 added and marked resolved
- `ISSUES.md` — I-61 through I-65 added and marked fixed
- `CHANGELOG.md` — sprint entry added

## 2026-05-04 — Emission Equivalence & Compat Expansion Sprint

### Resolved (T-61, T-66)

| Task | Description | Key Change |
|------|-------------|------------|
| T-61 | Add Cross-Validation Test for Python vs Rust Emission | Created 10 structural equivalence tests in `crates/coreml-emit/tests/cross_validation.rs`: linear projection topology, multi-function topology, spec version propagation, weight embedding, I/O descriptors, attention-like graph, pooling ops, op coverage matrix documentation, stateful decode step topology, normalization ops. Documented which ops are supported by each path with a cross-validated op coverage matrix. |
| T-66 | Add Remaining MirOpCompat Variants (partial) | Added 12 new `MirOpCompat` variants with full conversion, input_names, remap_inputs, rename_output, and tests: MaxPool, AvgPool, L2Pool (pooling), DepthToSpace, SpaceToDepth, PixelShuffle, PixelUnshuffle (spatial rearrangement), BatchNorm, InstanceNorm, L2Norm (normalization), Quantize, Dequantize (quantization). Updated `mir_op_to_compat()` and `mir_op_to_unsupported()` in `mir_to_compat.rs`. |

### New MirOpCompat Variants (12 variants)

**Pooling (T-66):**
- `MaxPool { name, x, kernel_sizes, strides, pad_type, pad_amounts }` — Core ML MIL op type: "max_pool"
- `AvgPool { name, x, kernel_sizes, strides, pad_type, pad_amounts, count_include_padding }` — Core ML MIL op type: "avg_pool"
- `L2Pool { name, x, kernel_sizes, strides, pad_type, pad_amounts }` — Core ML MIL op type: "l2_pool"

**Spatial Rearrangement (T-66):**
- `DepthToSpace { name, x, block_size }` — Core ML MIL op type: "depth_to_space"
- `SpaceToDepth { name, x, block_size }` — Core ML MIL op type: "space_to_depth"
- `PixelShuffle { name, x, upscale_factor }` — Core ML MIL op type: "pixel_shuffle"
- `PixelUnshuffle { name, x, downscale_factor }` — Core ML MIL op type: "pixel_unshuffle"

**Normalization (T-66):**
- `BatchNorm { name, x, mean, variance, gamma, beta, epsilon }` — Core ML MIL op type: "batch_norm"
- `InstanceNorm { name, x, gamma, beta, epsilon }` — Core ML MIL op type: "instance_norm"
- `L2Norm { name, x, epsilon, axes }` — Core ML MIL op type: "l2_norm"

**Quantization (T-66):**
- `Quantize { name, x, scale, zero_point, axis, output_dtype }` — Core ML MIL op type: "quantize"
- `Dequantize { name, x, scale, zero_point, axis, output_dtype }` — Core ML MIL op type: "dequantize"

### New Tests Added (16 tests)

**cross_validation.rs (10 tests — T-61):**
- `test_linear_projection_topology_equivalence` — const→const→linear topology matches Python bridge
- `test_multifunction_topology_equivalence` — embedding + decode_step with shared weights
- `test_spec_version_propagation_equivalence` — V7 and V10 preserved in protobuf
- `test_weight_embedding_equivalence` — const ops have BlobFileValue references
- `test_io_descriptor_equivalence` — input "x" and output "output" match Python bridge
- `test_attention_like_graph_topology` — const→linear→reshape→gelu chain
- `test_pooling_ops_mir_compat_to_apple_proto` — MaxPool emits via Rust proto-direct path
- `test_op_coverage_matrix_documentation` — living documentation of 75+ ops across both paths
- `test_stateful_decode_step_topology` — read_state→linear→slice_by_index→write_state→linear
- `test_normalization_ops_mir_compat_to_apple_proto` — BatchNorm emits via Rust proto-direct path

**mir_conversion.rs (6 tests — T-66):**
- `test_t66_input_names` — input_names() correctness for all 12 new variants
- `test_t66_output_name` — output_name() returns the name field
- `test_t66_remap_inputs` — remap_inputs() remaps String input fields correctly
- Updated `test_mir_op_to_compat_exhaustive_coverage` — added field-value assertions for 12 new variants
- Updated `test_mir_op_field_values_preserved` — expanded to cover new variant fields
- Updated `test_mir_op_dtype_conversion` — verified dtype mapping for Quantize/Dequantize

### Files Modified

- `crates/coreml-proto/src/lib.rs` — 12 new MirOpCompat variants + output_name/input_names/remap_inputs/rename_output updates
- `crates/bridge/src/mir_to_compat.rs` — 12 new conversion arms in mir_op_to_compat() + 12 unreachable markers in mir_op_to_unsupported()
- `crates/coreml-proto/tests/mir_conversion.rs` — 6 new/updated tests for T-66 variants
- `crates/coreml-emit/tests/cross_validation.rs` — 10 new cross-validation tests (T-61)
- `TASKS.md` — T-61 and T-66 marked resolved
- `ISSUES.md` — I-35 and I-40 marked fixed
- `CHANGELOG.md` — sprint entry added

---

## 2026-05-04 — Knowledge Seed Consistency & Constraint Integrity Sprint

### Resolved (T-86, T-87, T-89, T-88, T-91)

| Task | Description | Key Change |
|------|-------------|------------|
| T-86 | Align knowledge seed family mappings (V-001, V-002) | Fixed `ane_hw_limits_seed.json`: V6→A13 (was A14), V11→A17 (was A16). Added `test_hw_limits_seed_family_consistency()` to prevent future drift |
| T-87 | Resolve three-way knowledge seed contradictions (V-003, V-004, V-005) | Removed 6 comparison ops from `cpu_only_ops_seed.json` (they have ANEC converters on A14+). Marked logical_and/or/not as unsupported in `ane_op_family_matrix.json` (no dedicated ANEC converter). Changed `mb.gather` to `ane_legal: true` with `limited_index_range` constraint in `legality_seed.json` |
| T-89 | Fix Gelu mode contradictions (V-099, V-113) | Changed SIR builder from `"EXACT"` to `"TANH_APPROXIMATION"` in both `sir_build.rs:518` and `sir_build.rs:1415`. Updated test fixture in `staticize.rs`. ANEC only supports tanh approximation — EXACT mode has no converter |
| T-88 | Replace silent Fp16 dtype default with error (V-011) | `shard_desc.rs` now returns explicit error for unrecognized dtype strings instead of silently defaulting to Fp16. Added Int8 and UInt8 as recognized dtype strings |
| T-91 | Make zero-weight placeholders a hard error by default (V-007) | `mir_to_compat.rs` now errors by default when weights can't be resolved. Added `allow_missing_weights` parameter to `mir_graph_to_compat_with_arch()`. Added `mir_graph_to_compat_with_allow_missing()` convenience function. Error message directs users to `--allow-missing-weights` flag |

### New Tasks Derived from NECROSCOPY Audit (T-86 through T-131)

46 new tasks derived from the 138 violations in `docs/audit/ane-violations.md`. Tasks organized by severity:
- CRITICAL: T-86, T-87, T-89, T-90
- HIGH: T-88, T-91 through T-102
- MEDIUM: T-103 through T-123
- LOW: T-124 through T-131

### New Issues Derived from NECROSCOPY Audit (I-61 through I-98)

38 new issues derived from the forensic audit findings. All violations from ane-violations.md §III are now tracked as issues with detailed intent, mitigation, and Definition of Done.

### New Tests Added (5 tests)

- `test_hw_limits_seed_family_consistency` — verifies all revision→family mappings match between Rust code and knowledge seed JSON (T-86)
- `test_lower_shard_to_mir_default_dtype` — updated: unrecognized dtype now produces error instead of Fp16 default (T-88)
- `test_lower_shard_to_mir_int8_dtype` — verifies Int8 is a recognized dtype string (T-88)
- `test_lower_shard_to_mir_uint8_dtype` — verifies UInt8 is a recognized dtype string (T-88)
- `test_missing_weights_hard_error_by_default` — verifies missing weights produce hard error by default (T-91)
- `test_missing_weights_allowed_with_flag` — verifies allow_missing_weights=true permits zero-fill (T-91)

---

## 2026-05-04 — Test Coverage Sprint

### Resolved (T-58, T-59)

| Task | Description | Key Change |
|------|-------------|------------|
| T-58 | Add Tests for ir::payload, ir::shard_desc, ir::serialize | Added 52 tests: payload.rs (28 tests covering all 5 family payload types, from_spec/from_spec_with_override, wrong-op-type rejection, FamilyPayload JSON roundtrip, constants, descriptor serialization), shard_desc.rs (14 tests covering sharded_pipeline_shards structure/roles/dims, lower_shard_to_mir node/dtype verification, ShardedShardPayload construction/override/decode-step, build_sharded_pipeline_pir packages/handoffs/template), serialize.rs (10 tests covering SIR/AIR/MIR/PIR round-trip, corrupt/empty bytes errors, generic serialize/deserialize_graph, node preservation) |
| T-59 | Add Tests for lab::session, lab::harness, lab::fallback | Added 52 tests: session.rs (16 tests covering compute_task_hash determinism/uniqueness/format, build_artifact_manifest success/failure/limitations, build_knowledge_update success/failure/residuals, build_knowledge_update_with_drift computed/unavailable, ingest_knowledge_observations valid/empty/missing), harness.rs (24 tests covering LabRunBuilder all builder paths, LabRun JSON roundtrip/write_to_file, VerificationScope/EnvironmentSummary/CompileStepResult/InspectionStepResult/TimingResult/FallbackSuspicionResult/GeneratorProvenance serialization, schema version), fallback.rs (12 tests covering FallbackDetector default/custom threshold, detect_from_timing host-only/no-baseline/anomaly/normal, evidence kinds, assess_overall_level, FallbackLogEvidence serialization) |

### New Tests Added (104 tests)

**payload.rs (28 tests):**
- `test_linear_projection_payload_from_spec` — field verification for LinearProjection payload
- `test_linear_projection_payload_dtype_override` — fp32 override propagation
- `test_linear_projection_payload_wrong_op_type` — wrong TaskOp returns Err
- `test_lut_projection_payload_from_spec` — LUT-specific fields (vocab_size, embed_dim, num_groups, lut_bitwidth)
- `test_lut_projection_payload_dtype_override` — dtype override for LUT
- `test_lut_projection_payload_wrong_op_type` — wrong TaskOp returns Err
- `test_decode_step_payload_from_spec` — decode-step fields (embed_dim, num_heads, head_dim, kv_len, stateful=true)
- `test_decode_step_payload_dtype_override` — dtype override for decode-step
- `test_decode_step_payload_wrong_op_type` — wrong TaskOp returns Err
- `test_mlp_block_payload_from_spec` — MLP fields (input_dim, hidden_dim, output_dim, activation)
- `test_mlp_block_payload_dtype_override` — dtype override for MLP
- `test_mlp_block_payload_wrong_op_type` — wrong TaskOp returns Err
- `test_attention_payload_from_spec` — attention fields (embed_dim, num_heads, head_dim, seq_len)
- `test_attention_payload_dtype_override` — dtype override for attention
- `test_attention_payload_wrong_op_type` — wrong TaskOp returns Err
- `test_family_payload_from_spec_linear` — FamilyPayload from LinearProjection
- `test_family_payload_from_spec_lut` — FamilyPayload from LutProjection
- `test_family_payload_from_spec_decode_step` — FamilyPayload from DecodeStep
- `test_family_payload_from_spec_mlp_block` — FamilyPayload from MlpBlock
- `test_family_payload_from_spec_attention` — FamilyPayload from Attention
- `test_family_payload_dtype_override` — dtype override propagation
- `test_family_payload_to_json` — valid JSON output
- `test_family_payload_to_json_pretty` — valid pretty JSON output
- `test_family_payload_json_roundtrip` — serialize/deserialize preserves fields
- `test_bridge_version_constant` — BRIDGE_VERSION == 1
- `test_default_seed_constant` — DEFAULT_SEED == 42
- `test_function_descriptor_serialization` — FunctionDescriptor roundtrip
- `test_tensor_descriptor_serialization` — TensorDescriptor roundtrip

**shard_desc.rs (14 tests):**
- `test_sharded_pipeline_shards_structure` — 3 shards with correct roles/dims
- `test_sharded_pipeline_shards_wrong_op_type` — wrong TaskOp returns Err
- `test_shard_desc_serialization` — ShardDesc roundtrip
- `test_lower_shard_to_mir_structure` — 4 MIR nodes (weight, bias, matmul, add)
- `test_lower_shard_to_mir_dtypes` — fp16/fp32/int4/e4m3/e5m2 dtype mapping
- `test_lower_shard_to_mir_default_dtype` — unknown dtype defaults to fp16
- `test_sharded_shard_payload_from_shard` — all fields verified
- `test_sharded_shard_payload_dtype_override` — override propagation
- `test_sharded_shard_payload_decode_step` — decode-step command, 3 inputs, stateful=true
- `test_sharded_shard_payload_serialization` — roundtrip
- `test_build_sharded_pipeline_pir_structure` — 3 packages + shard template
- `test_build_sharded_pipeline_pir_wrong_op_type` — wrong TaskOp returns Err
- `test_build_sharded_pipeline_pir_handoffs` — entry→interior→exit handoff structure
- `test_build_sharded_pipeline_pir_serialization` — PirGraph roundtrip

**serialize.rs (10 tests):**
- `test_serialize_deserialize_sir_roundtrip` — SirGraph round-trip
- `test_serialize_deserialize_air_roundtrip` — AirGraph round-trip
- `test_serialize_deserialize_mir_roundtrip` — MirGraph round-trip
- `test_serialize_deserialize_pir_roundtrip` — PirGraph round-trip
- `test_deserialize_corrupt_bytes_returns_error` — invalid bytes → Err
- `test_deserialize_empty_bytes_returns_error` — empty bytes → Err
- `test_serialize_graph_generic` — generic serialize_graph function
- `test_deserialize_graph_generic` — generic deserialize_graph function
- `test_sir_roundtrip_preserves_nodes` — node count and op types preserved
- `test_mir_roundtrip_preserves_nodes` — node count and dtypes preserved

**session.rs (16 tests):**
- `test_compute_task_hash_deterministic` — same spec → same hash
- `test_compute_task_hash_different_specs` — different specs → different hashes
- `test_compute_task_hash_format` — "sha256:" prefix + 64 hex chars
- `test_compute_task_hash_uses_identity_string` — identity_string difference changes hash
- `test_build_artifact_manifest_success` — 1 package with correct fields
- `test_build_artifact_manifest_failure` — failed bridge → no packages
- `test_build_artifact_manifest_has_environment_limitations` — 3 limitations present
- `test_build_knowledge_update_success` — version 2, 2 observations, confidence values
- `test_build_knowledge_update_failure` — ane_legal=false, confidence=0.7
- `test_build_knowledge_update_has_residuals` — 3 residuals with expected content
- `test_build_knowledge_update_with_drift_computed` — version 3, drift metrics present
- `test_build_knowledge_update_with_drift_unavailable` — computation_status="unavailable"
- `test_build_knowledge_update_with_drift_version` — drift variant uses version 3
- `test_ingest_knowledge_observations_valid` — 2 observations ingested
- `test_ingest_knowledge_observations_empty_observations` — 0 ingested
- `test_ingest_knowledge_observations_missing_observations` — returns Err

**harness.rs (24 tests):**
- `test_lab_run_builder_minimal` — required fields only, defaults verified
- `test_lab_run_builder_all_fields` — all optional fields set
- `test_lab_run_builder_payload_hash` — payload_hash propagation
- `test_lab_run_builder_compile_result` — compile_result propagation
- `test_lab_run_builder_inspect_result` — inspect_result propagation
- `test_lab_run_builder_timing` — timing Some after set
- `test_lab_run_builder_fallback_suspicion` — fallback_suspicion Some after set
- `test_lab_run_builder_warnings` — multiple warnings accumulated
- `test_lab_run_builder_generator_provenance` — provenance propagation
- `test_lab_run_builder_adaptation_readiness` — readiness propagation
- `test_lab_run_completed_at_set` — build() sets completed_at
- `test_lab_run_to_json` — valid JSON output
- `test_lab_run_json_roundtrip` — serialize/deserialize preserves fields
- `test_lab_run_write_to_file` — file write and read back
- `test_verification_scope_serialization` — all 3 variants roundtrip
- `test_environment_summary_detect` — reasonable host values
- `test_environment_summary_bridge_version` — bridge_version set
- `test_compile_step_result_serialization` — roundtrip
- `test_inspection_step_result_serialization` — roundtrip
- `test_timing_result_serialization` — roundtrip
- `test_fallback_suspicion_result_serialization` — roundtrip
- `test_generator_provenance_serialization` — roundtrip
- `test_lab_run_schema_version` — schema version constant
- `test_lab_run_builder_chained` — builder method chaining

**fallback.rs (12 tests):**
- `test_fallback_detector_default_threshold` — latency_threshold_ratio = 3.0
- `test_fallback_detector_custom_threshold` — with_threshold_ratio(5.0)
- `test_detect_from_timing_host_only_returns_unavailable` — host-only → Unavailable
- `test_detect_from_timing_no_baseline_returns_unavailable` — no expected latency → Unavailable
- `test_detect_from_timing_latency_anomaly` — observed >> expected → LowConfidenceSuspicion
- `test_detect_from_timing_latency_normal` — observed ≈ expected → NoConclusion
- `test_detect_from_timing_no_compute_plan_evidence` — compute_plan_unavailable evidence
- `test_detect_from_timing_evidence_kinds` — correct evidence kind strings
- `test_detect_from_timing_suspicion_explanation_not_empty` — non-empty explanation
- `test_fallback_log_evidence_serialization` — roundtrip
- `test_assess_overall_level_latency_anomaly` — internal: anomaly → LowConfidenceSuspicion
- `test_assess_overall_level_no_anomaly` — internal: no anomaly → NoConclusion

---

## 2026-05-04 — Validation & Code Quality Sprint

### Resolved (T-64, T-60, T-81)

| Task | Description | Key Change |
|------|-------------|------------|
| T-64 | Centralize Palette Bit-Width Validation | Moved `validate_palette_bits()`, `VALID_PALETTE_BITS`, and `clamp_to_valid_palette_bits()` to `ane-ir::ane_layout`; updated 3 call sites (`palettize_weights.rs`, `lut_projection.rs`, `task_spec.rs`) to use centralized versions; fixed doc comments in `sir.rs` to list correct valid set {1,2,3,4,6,8} |
| T-60 | Fix Tile Decomposition Placeholder Zeros | Added `tile_input_dim()` method to `DecompositionContext` for concrete shape resolution; Tile decomposition now uses ctx dimensions when available, avoiding the batch=1 heuristic in `resolve_reshape_zeros()`; fixed final_shape to be at the original input rank (4D) instead of expanded rank (5D); logs warning when ctx is unavailable |
| T-81 | Fix `compat_input_dtype` String Matching | Removed `name.contains("input_ids")` heuristic that could misfire; now trusts the MIR node's declared `dtype` field directly via `mil_dtype_to_compat()`, since the MIR builder correctly assigns `MilDtype::Int32` to input_ids tensors |

### New Tests Added (9 tests)

- `test_validate_palette_bits_valid` — all valid ANE bit-widths accepted (T-64)
- `test_validate_palette_bits_invalid` — invalid bit-widths rejected (T-64)
- `test_clamp_to_valid_palette_bits` — clamping rounds down correctly (T-64)
- `test_tile_decomposition_with_ctx_uses_concrete_shapes` — ctx produces concrete reshape/final shapes (T-60)
- `test_tile_decomposition_without_ctx_uses_placeholders` — no-ctx falls back to 0 placeholders (T-60)
- `test_tile_input_dim_4d` — `DecompositionContext.tile_input_dim()` resolves 4D Tile dims (T-60)
- `test_tile_input_dim_non_4d` — non-4D ranks return None (T-60)
- `test_tile_input_dim_default_ctx` — zero ctx returns None for all dims (T-60)
- `test_compat_input_dtype_no_name_based_override` — name heuristics no longer override dtype (T-81)
- `test_compat_input_dtype_input_ids_with_fp16_returns_fp16` — declared dtype is respected (T-81)
- `test_compat_input_dtype_int32_passthrough` — Int32 dtype maps correctly regardless of name (T-81)

---

## 2026-05-04 — Bridge, FFI & Code Quality Sprint

### Resolved (T-75, T-76, T-77, T-78, T-82, T-83)

| Task | Description | Key Change |
|------|-------------|------------|
| T-75 | Fix FFI `coreml_model_destroy` Unsoundness | Documented allocation contract on `ModelHandleInner`; `coreml_model_load` MUST use `Box::new` so `destroy` can safely call `Box::from_raw`; added contract test |
| T-76 | Add Tests for coreml-ffi::api Module | Added 11 new tests: error type verification for all 5 `CoreMlApi` methods, JSON serialization roundtrips for result types, field-level validation |
| T-77 | Enforce PythonBridge Timeout | Replaced `Command::output()` with `spawn` + poll-based timeout loop; on timeout the child is killed and a timeout error is returned; no new dependencies |
| T-78 | Remove Dead-Code `compare_with_python_bridge` | Removed method, `ComparisonReport`, and `WeightBinComparison` types — all were dead code |
| T-82 | Remove Dead-Code `mir_node_to_compat` | Removed `#[allow(dead_code)]`, gated with `#[cfg(test)]`, added documentation explaining when to use it vs. shape-aware version |
| T-83 | Add BF16→FP16 Edge-Case Tests | Added 7 edge-case tests: NaN, infinity, negative zero, subnormals, max overflow, bulk conversion |

### New Tests Added (19 tests)

- `test_model_destroy_allocated_handle` — verifies Box-allocated handle can be destroyed without UB (T-75)
- `test_coreml_api_version_unavailable` — error type verification for `CoreMlApi::version()` (T-76)
- `test_coreml_api_compile_model_unavailable` — error type verification for `CoreMlApi::compile_model()` (T-76)
- `test_inspect_model_structure_unavailable` — error type verification for `inspect_model_structure()` (T-76)
- `test_inspect_compute_plan_unavailable` — error type verification for `inspect_compute_plan()` (T-76)
- `test_model_structure_result_serialization` — JSON roundtrip for `ModelStructureResult` (T-76)
- `test_compute_plan_result_serialization` — JSON roundtrip for `ComputePlanResult` (T-76)
- `test_model_structure_result_empty` — empty result structure validation (T-76)
- `test_compute_plan_result_unavailable` — unavailable result structure validation (T-76)
- `test_op_placement_all_compute_units` — CPU/GPU/ANE compute unit coverage (T-76)
- `test_function_structure_fields` — `FunctionStructure` field validation (T-76)
- `test_state_declaration_dtype_field` — `StateDeclaration` dtype field validation (T-76)
- `test_bf16_to_fp16_nan_preservation` — quiet + signaling NaN (T-83)
- `test_bf16_to_fp16_infinity_preservation` — +Inf and -Inf (T-83)
- `test_bf16_to_fp16_negative_zero` — signed zero preservation (T-83)
- `test_bf16_to_fp16_subnormal_handling` — subnormal/flush-to-zero behavior (T-83)
- `test_bf16_to_fp16_max_finite_value` — overflow to +Inf (T-83)
- `test_bf16_to_fp16_bulk_conversion` — full `convert_bf16_to_fp16` pipeline test (T-83)

---

## 2026-05-04 — Bridge Model Leakage & Code Quality Sprint

### Resolved (T-70, T-72, T-73, T-74, T-79, T-80, T-84, T-85)

| Task | Description | Key Change |
|------|-------------|------------|
| T-70 | Fix K/V Projection Alias Map Drop | Used `k_proj`/`v_proj` patterns to build separate K/V alias entries; Q/K/V aliases now point to their respective projection nodes |
| T-72 | Fix Palettize Qwen3 Name Heuristics | Added `run_palettize_weights_pass_with_arch()` using `ModelArchitecture` pattern methods instead of hardcoded name checks |
| T-73 | Fix `LM_HEAD_SHARD_SIZE` Hardcoding | Shard size derived from `vocab_size / TARGET_SHARD_COUNT` (8) instead of hardcoded 19000 |
| T-74 | Fix `resolve_shard` FP16-Only Byte Offsets | Element size derived from `data.len() / total_elements` instead of hardcoded 2; added byte-range overflow guard |
| T-79 | Log Warning When SafetensorsResolver Is Empty | Added `log::warn!()` in `from_traced_graph` when all resolution strategies fail |
| T-80 | Fix Fill Op `input_names()` Empty Vec | `input_names()` now returns `vec![format!("{}_shape", name)]` for Fill ops |
| T-84 | Replace `eprintln!` With `log::warn!` | Replaced in `ane_hw_limits.rs::AneHwLimits::a12()`; added `log` dependency to `ane-ir` |
| T-85 | Gate Deprecated `kv_cache_rewrite` | Module gated behind `deprecated-kv-cache-rewrite` feature flag in `ane-passes`; not compiled by default |

### New Tests Added (7 tests)

- `test_palettize_with_explicit_qwen3_architecture` — verifies architecture-aware palettization with Qwen3
- `test_palettize_with_generic_architecture` — verifies Generic architecture with GPT-2-like patterns
- `test_resolve_shard_weight` — updated for dynamic shard size derivation (T-73)
- `test_resolve_shard_weight_f32` — verifies F32 shard byte offsets (T-74)
- `test_resolve_shard_qwen3_vocab` — regression test for Qwen3-0.6B vocab size (T-73)
- Updated Fill `input_names()` test to verify shape input name (T-80)
- `kv_cache_rewrite` tests now gated behind `deprecated-kv-cache-rewrite` feature (T-85)

---

## 2026-05-04 — Placement & Classification Integrity Sprint

### Resolved (T-67, T-65, T-68, T-69, T-71)

| Task | Description | Key Change |
|------|-------------|------------|
| T-67 | Fix CPU_ONLY_OPS name mismatches (CRITICAL) | `"negative"`→`"neg"`, removed dead entries, added `"round"`, moved MILNeg to None |
| T-65 | Unify CPU-only classification | Added `is_cpu_only_unified()`, placement validator checks `default_engine()==None` first |
| T-68 | Fix `extract_whdc()` NCHW dimensional swap | Rank-4 NCHW: `(shape[3], shape[2], 1, shape[1])` instead of CDHW |
| T-69 | Wire pooling kernel size validation | `kernel_size` validated against max_pooling_kernel_dim=27 |
| T-71 | Fix Float64 element_size=4→8 | Split match arm: `Float32 => 4, Float64 => 8` |

### New Tests Added (15 tests)

- `test_cpu_only_covers_all_default_engine_none` — verifies all MirOp None-branch ops are in CPU_ONLY_OPS
- `test_t67_fixed_names_in_cpu_only` — verifies `"neg"` and `"round"` are CPU-only
- `test_t67_removed_names_not_in_cpu_only` — verifies removed dead-code entries are gone
- `test_extract_whdc_rank1` through `test_extract_whdc_regression_nchw_channels_vs_batch` — 9 tests for NCHW dimension extraction
- `test_pooling_kernel_size_within_limit`, `test_pooling_kernel_size_exceeds_limit`, `test_pooling_kernel_size_zero_rejected`, `test_pooling_kernel_size_large_rejected` — 4 tests for pooling kernel validation
- Expanded `test_coreml_data_type_element_size` to cover all 12 CoreMlDataType variants

---

## 2026-05-04 — Tabula Rasa v3 Audit Cycle

### Resolved (T-36 through T-57)

| Task | Description | Key Change |
|------|-------------|------------|
| T-36 | Parameterize model-specific constants | Added `ModelArchConfig` / `ModelArchitecture` |
| T-37 | SIR→AIR roundtrip tests | 14 roundtrip tests with Qwen3-0.6B dimensions |
| T-38 | ToProto trait for MirOp + MirOpCompat | Unified 167-variant mapping |
| T-39 | Constexpr* MirOpCompat variants | 7 variants for palettized weight emission |
| T-40 | V17 (M1) → A14 family mapping | M1 is A14-class, not A18 |
| T-41 | cargo fmt + clippy --fix | 52 files reformatted |
| T-42 | Chip comment errors | Corrected A11≠M1, A12≠M2, A14≠M3 |
| T-43 | Proto panic!() → Result | `ProtoValidationError` return type |
| T-44 | too_many_arguments refactor | `DecompositionEnv` + `DecodeWeights` structs |
| T-45 | Deprecated kv_cache_rewrite → pub(crate) | Prevents external access |
| T-46 | Shared shape_ops module | MILTile bug fix + 30+ bridge variants |
| T-47 | 4 ops with PE engine but no converter | Moved to None; added to CPU_ONLY_OPS |
| T-48 | Palettize pass is no-op | Added palette_bits field with validation |
| T-49 | ~30 missing CPU_ONLY_OPS | Added (with name mismatches, see I-41/I-42) |
| T-51 | ReduceMin non-FP guard | supports_reducemin_all_dtypes() |
| T-52 | A17 family + E4M3 fix | V11→A17 remapped |
| T-53 | HW tensor dim limits enforced | validate_tensor_dims() in placement pipeline |
| T-54 | panic→bail in legality_rewrite | Result-based error propagation |
| T-56 | ModelArchConfig::default() deprecation | qwen3_0_6b() factory |
| T-57 | Bridge Qwen3 architecture fallback | log::warn!() + deprecation warnings |
| T-58 | A13 broadcast FP16-only guard | Verified correct; A13 excluded from FP16-only |
| T-62 | Conv kernel_d/stride validation | Implemented per-family limits |
| T-63 | Zero-channels interleave bypass | Changed to if-let-Some pattern |

### Open Issues from v3 Audit (I-41 through I-60)

| Priority | Issues | Key Findings |
|----------|--------|--------------|
| CRITICAL | I-41, I-42 | CPU_ONLY_OPS name mismatches — MILNeg passes CPU-only gate |
| HIGH | I-43 through I-49 | extract_whdc swap, pooling discard, Float64 size, K/V alias drop, palettize heuristics, shard hardcodes, FP16 assumption |
| MEDIUM | I-50 through I-55 | FFI unsoundness, zero test coverage, timeout not enforced, dead code stub, empty resolver, Fill input_names |
| LOW | I-56 through I-60 | String matching, dead code, BF16 edge cases, eprintln, deprecated module |

Full details in [ISSUES.md](ISSUES.md).

---

## 2026-05-03 — Tabula Rasa v1/v2 Audit Cycle

Resolved 40 issues (I-01 through I-40): three-way source alignment, CPU-only hard gate, A13 family mapping, interleave/dtype/matmul/pad validators, reshape panic→Result, zero-dim validation, modulo-1 logic bug, SDPA compat fields, ArgMinMax A18 guard, 153 shape_inference tests, 62 staticize tests, MilDtype expansion.

Key infrastructure added: `AneFamily::A13`, `mil_op_name()`, `is_cpu_only()`, `PlacementContext`, `validate_matmul_constraints()`, `validate_pad_constraints()`, `resolve_reshape_zeros()`.
