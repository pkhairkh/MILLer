# MILLer — Prioritised Action List

> Ranked by **severity × impact**. Each task references findings in `docs/audit/ane-violations.md` and `ISSUES.md`.
> Estimates assume a single experienced Rust/Python developer.
> Generated from **ANE Violations Deep Forensic Audit** (ane-violations.md) on 2026-05-07.
> All prior tasks **T-01 through T-90** are **RESOLVED** — see archive summary below and `CHANGELOG.md` for details.
> New tasks numbered from **T-91**, derived from V-XXX violation IDs in ane-violations.md.


---

## Archive Summary — Resolved Tasks (T-01 through T-90)

All tasks from the v1 (T-01–T-46), v2 (T-47–T-66), and v3 (T-60–T-90) audits are **resolved**.
Full resolution details are in `CHANGELOG.md`.

| Range | Audit | Count | Status | Details |
|-------|-------|-------|--------|---------|
| T-01 – T-46 | v1 (tabula-rasa-v1) | 46 | ✅ All Resolved | See `CHANGELOG.md` |
| T-47 – T-57 | v2 (tabula-rasa-v2) | 11 | ✅ All Resolved | See `CHANGELOG.md` |
| T-58 – T-66 | v2/v3 overlap | 9 | ✅ All Resolved | See `CHANGELOG.md` |
| T-67 – T-90 | v3 (tabula-rasa-v3) | 24 | ✅ All Resolved | See `CHANGELOG.md` |

**Total resolved: 90 tasks across 3 audit cycles.**

### T-86 · ~~Align Knowledge Seed Family Mappings with Rust Code~~

~~**ISSUES ref**: I-61~~
~~**AUDIT ref**: V-001, V-002 (ane-violations.md §III)~~
~~**Severity**: CRITICAL~~
~~**Effort**: S (0.5 day)~~
**✅ RESOLVED** — Fixed `ane_hw_limits_seed.json`: V6→A13 (was A14), V11→A17 (was A16). Added `test_hw_limits_seed_family_consistency()` to prevent future drift.

---

### T-87 · ~~Resolve Three-Way Knowledge Seed Contradictions~~

~~**ISSUES ref**: I-62~~
~~**AUDIT ref**: V-003, V-004, V-005 (ane-violations.md §III)~~
~~**Severity**: CRITICAL~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — Removed 6 comparison ops from `cpu_only_ops_seed.json`. Marked logical_and/or/not as unsupported in `ane_op_family_matrix.json`. Changed `mb.gather` to `ane_legal: true` with `limited_index_range` constraint.

---

### T-88 · ~~Replace Silent Fp16 Dtype Default with Explicit Error~~

~~**ISSUES ref**: I-63~~
~~**AUDIT ref**: V-011 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: S (0.5 day)~~
**✅ RESOLVED** — `shard_desc.rs` now returns explicit error for unrecognized dtype strings. Added Int8 and UInt8 as recognized dtype strings.

---

### T-89 · ~~Fix Gelu Mode Contradictions — Standardize on TANH_APPROXIMATION~~

~~**ISSUES ref**: I-64~~
~~**AUDIT ref**: V-099, V-113 (ane-violations.md §III)~~
~~**Severity**: CRITICAL~~
~~**Effort**: S (0.5 day)~~
**✅ RESOLVED** — Changed SIR builder from `"EXACT"` to `"TANH_APPROXIMATION"` in sir_build.rs. Updated test fixture in staticize.rs.

---

### T-91 · ~~Make Zero-Weight Placeholders a Hard Error by Default~~

~~**ISSUES ref**: I-66~~
~~**AUDIT ref**: V-007 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — `mir_to_compat.rs` now errors by default when weights can't be resolved. Added `allow_missing_weights` parameter. Added `mir_graph_to_compat_with_allow_missing()` convenience function.

---

## 🔴 CRITICAL — Silent Miscompilation or Data Corruption

### T-90 · ~~Replace Concat Emissions with ANE-Legal Alternatives~~

~~**ISSUES ref**: I-65~~
~~**AUDIT ref**: V-098, V-130 (ane-violations.md §III)~~
~~**Severity**: CRITICAL~~
~~**Effort**: L (2 days)~~
**✅ RESOLVED** — Replaced all MILConcat emissions in SDPA decomposition (mil_lower.rs) and RoPE rotate_half (legality_rewrite.rs) with Stack+Reshape pattern. AttentionBlock and DecodeStep concats replaced with Stack. Added placement validator gate rejecting non-channel-axis MILConcat (Orion #1). 6 new tests.

---


### T-92 · ~~Add Conv/Pool Constraint Validation~~

- **ISSUES ref**: I-67
- **AUDIT ref**: V-009, V-132, V-128 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**✅ RESOLVED** — Added power-of-2 kernel validation for conv W/H/D, dilated stencil rejection, stencil constraints (5D rejection, non-4D kernel, non-sum reduction, dilated, strided). Added `validate_stencil_constraints()` function. Updated existing conv tests to use power-of-2 kernel sizes. New tests for all constraints.

---

### T-93 · ~~Add Large Kernel Mode Constraints~~

- **ISSUES ref**: I-68
- **AUDIT ref**: V-115 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**✅ RESOLVED** — Added `LARGE_KERNEL_THRESHOLD=16` named constant. Added `validate_large_kernel_constraints()` checking: W/H multiple of 8, stride 1-2 only, no depth>1, no grouped conv, no dilation. Wired into `validate_conv_constraints()`. Tests for each constraint.

---

### T-94 · ~~Add Deconvolution Constraint Validation~~

- **ISSUES ref**: I-69
- **AUDIT ref**: V-116, V-048 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (1 day)

**✅ RESOLVED** — Added `validate_deconv_constraints()` checking: no dilation, SOx==2, no large kernel, no vector palettization, stride>2 with depth>1 rejected. Updated placement_validate.rs ConvTranspose comment. Tests for each constraint.

---

### T-95 · ~~Add Multi-Surface Ordering and Uniformity Validation~~

~~**ISSUES ref**: I-70~~
~~**AUDIT ref**: V-105, V-117, V-118, V-119, V-120, V-121 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: M (1.5 days)~~
**✅ RESOLVED** — Added alphabetical sorting of multi-output (Orion #3) and multi-input (Orion #19) surfaces before emission. Added `validate_surface_uniformity()` for Orion #2/#18 output/input buffer size checks. Added `validate_flat_buffer_layout()` for Orion #20 [1,C,1,S] validation. 9 new tests.

---

### T-96 · ~~Fix MILLinear→MILMatMul Defeating Conv1x1AsLinear Optimization~~

~~**ISSUES ref**: I-71~~
~~**AUDIT ref**: V-114 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — Changed ANE legality rewrite to convert MILLinear → MILConv(1x1) instead of MILMatMul for ANE targets (Orion #17: conv 1x1 is 3x faster than matmul). CPU path retains MILLinear→MILMatMul. Bias field now logged with warning when dropped. Updated precision override test. 3 new tests.

---

### T-97 · ~~Add Dtype Cross-Validation and Rejection~~

- **ISSUES ref**: I-72, I-98, I-99, I-102
- **AUDIT ref**: V-125, V-126, V-051/V-111, V-134 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**✅ RESOLVED** — Added `CrossTypeViolation` and `AsymmetricQuantViolation` error variants in `dtype_constraints.rs`. Added `validate_cross_type_compatibility()` for BF16/F16 cross-type checks, rejecting all 9 documented ANEC cross-type combinations. Added `is_fp32_compute_supported()` — returns false for A11Legacy/A12 families where FP32 is rejected. Added `validate_anec_quantization_symmetry()` — rejects asymmetric quantization on ANE. Removed E5M2 from quantize validator accepted output dtypes (universally rejected by ANEC per V-051/V-111). Added comprehensive tests for all new functions.

---

### T-98 · ~~Add Quantized Conv Weight Attributes~~

- **ISSUES ref**: I-73, I-94
- **AUDIT ref**: V-110 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: L (2 days)

**✅ RESOLVED** — Added `kernel_scale: Option<f32>`, `kernel_zero_point: Option<i32>`, `kernel_palettized_lut: Option<String>` fields to MILConv struct in mir.rs. Added same fields to MirOpCompat::Conv and wired through conversion in coreml-proto/src/lib.rs and bridge/src/mir_to_compat.rs. Added `kernel_scale`, `kernel_zero_point`, `kernel_palettized_lut` fields to MilConvOp proto message in MIL.proto. Emit quantized conv attributes as named const nodes in Apple proto format (kernel_scale → Float32 const, kernel_zero_point → Int32 const, kernel_palettized_LUT → String const). Legacy proto emission also carries the fields with defaults (0.0, 0, ""). Added `populate_conv_quantization_fields()` function and `ConvQuantizationInfo` struct in palettize_weights.rs. 6 new tests (3 in palettize_weights.rs, 3 in mir.rs).

---

### T-99 · ~~Add Conv 32K-Channel Limit Validation~~

- **ISSUES ref**: I-74
- **AUDIT ref**: V-103 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `max_conv_channels: 32768` field to AneHwLimits struct. Added `validate_conv_channels()` method. Updated `ane_hw_limits_seed.json` with new field for all 11 revisions. Tests verify 32K boundary.

---

### T-100 · ~~Add Non-Constant Gather Axis Rejection~~

- **ISSUES ref**: I-75
- **AUDIT ref**: V-136 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Extended `validate_gather_constraints()` with `axis_is_constant: bool` parameter. Non-constant axis rejected with "Gather with non-constant axis is not supported on ANE". Updated all callers. Tests for both constant and dynamic axis.

---

### T-101 · ~~Replace Fallback Shapes/Dtypes with Hard Errors~~

- **ISSUES ref**: I-76, I-86
- **AUDIT ref**: V-023, V-025 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1 day)

**✅ RESOLVED** — Missing input nodes now produce `bail!()` errors instead of defaulting to shape `[1]`, dtype Fp16. Missing output nodes now produce `bail!()` errors instead of silent fallbacks. Updated `role_mir.rs` to populate `input_shapes` from ShardSpec. Fixed test graphs to provide `input_shapes`.

---

### T-102 · ~~Fix F32 Weight Passthrough Without FP16 Conversion~~

- **ISSUES ref**: I-77, I-87
- **AUDIT ref**: V-026 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `convert_f32_to_fp16()` in `safetensors_resolver.rs`. F32 safetensors data now converts to FP16 using the same path as BF16→FP16 (via `half::f16::from_f32()`). Added tests: byte size halving, value preservation, special values (NaN/Inf/subnormals), and same-path-as-BF16 verification.

---


### T-103 · ~~Map Bool/Float64/Unknown Dtypes Correctly in Weights~~

- **ISSUES ref**: I-78, I-88
- **AUDIT ref**: V-027 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Changed `coreml_dtype_to_blob_dtype()` to return `Result<u32>` instead of `u32`. Bool, Float64, and Unknown dtypes now return explicit errors instead of silently mapping to Float32. Changed `WeightBinBuilder::build()` to return `Result<WeightBinResult>`. Updated all callers and tests.

---

### T-104 · ~~Derive State Shape from ReadState Op~~

- **ISSUES ref**: I-79
- **AUDIT ref**: V-028 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — `mir_to_proto.rs` now derives state shape/dtype from matching ReadState ops. Returns error when no ReadState exists and shape is empty. Tests verify shape derivation and error behavior.

---

### T-105 · ~~Resolve Softmax/InstanceNorm Family Gating Contradiction~~

- **ISSUES ref**: I-80
- **AUDIT ref**: V-029, V-030, V-101 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**✅ RESOLVED** — Added architecture-conditional soft-warning at placement for Softmax/InstanceNorm on A11Legacy/A12/A13. Added `architecture_conditional` and `architecture_conditional_note` fields to ane_op_family_matrix.json entries. Tests verify warning behavior.

---

### T-106 · ~~Add Pooling Stride-3 Avg-Only Check and Other Pool Constraints~~

- **ISSUES ref**: I-81
- **AUDIT ref**: V-127 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added stride-3 avg-only pool validation, stride>3 rejected for all modes, large-stride Min/Max pool with padding rejected. Tests for each constraint.

---

### T-107 · ~~Implement or Remove StaticizePass~~

- **ISSUES ref**: I-82
- **AUDIT ref**: V-014 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**✅ RESOLVED** — StaticizePass removed from compile pipeline — was a phantom no-op pass. Struct preserved with #[deprecated] for backward compatibility. All pipeline references and step numbering updated.

---

### T-108 · ~~Expand Precision Policy Coverage~~

- **ISSUES ref**: I-83
- **AUDIT ref**: V-015 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: L (2 days)

**✅ RESOLVED** — Expanded `op_pattern_for_node()` from ~33% to 100% SirOp variant coverage. Added 53 new match arms covering all previously-uncategorized ops: Comparison (6), Logical (4), Activation (12), Trigonometric (9), Reduction (8), Tensor Transform (21), Image Resize (8), Scatter/Gather (4), Constexpr (7), Recurrent (3), Control Flow (8), Random (4), Topk/Classify (2), Rounding (3), Mathematical (4), Conditional (2). Replaced `_ => "Other"` with `Misc_{VariantName}` catch-all. Changed return type from `&str` to `String`. 2 new tests verify 50+ specific patterns and no bare "Other" mappings.

---

### T-109 · ~~Make StateTopologyPass Return Errors~~

- **ISSUES ref**: I-84
- **AUDIT ref**: V-016 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `strict: bool` field (default: true). Strict mode returns `Err` for ReadState without matching WriteState. WriteState without ReadState still logs info (valid for initial state). Added `new_lenient()` and `with_strict()` constructors. Tests for strict/non-strict modes.

---

### T-110 · ~~Derive FunctionEntry Shapes from Graph~~

- **ISSUES ref**: I-85
- **AUDIT ref**: V-017 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**✅ RESOLVED** — Added `derive_primary_shapes()` method that scans StateRead ops (KV cache shapes) to extract batch, seq, embed dimensions. Replaced all 7 hardcoded `vec![1, 1]` values in shard_plan.rs with derived shapes. Fallback to vec![1, 1] only with explicit warning when no shape info found. 4 new tests.

---

### T-111 · ~~Fix Interleave Validation When Channels Unknown~~

- **ISSUES ref**: I-83
- **AUDIT ref**: V-020 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — When channels is None, interleave validation still enforces: valid interleave factors, const→1, int4/uint4→8. Previously the entire validation was skipped when channels was unknown, allowing invalid dtype/interleave combinations to pass silently.

---

### T-112 · ~~Fix Knowledge Conflict Detection Symmetry~~

- **ISSUES ref**: I-84, I-85
- **AUDIT ref**: V-021, V-022 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**✅ RESOLVED** — `check_conflicts_for_entry()` now back-patches existing entries when a new entry conflicts with them, making conflict detection symmetric. `claims_agree()` now implements field-level comparison for all 9 knowledge types (was only 1/8 before). Added payload accessors: `payload_survival_rate()`, `payload_fallback_engine()`, `payload_num_partitions()`. 6 new tests.

---

### T-113 · ~~Make default_engine() Revision-Aware~~

- **ISSUES ref**: I-88
- **AUDIT ref**: V-010 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**✅ RESOLVED** — Added `default_engine_for_revision(Option<AneRevision>)` method that cross-references family capabilities. ReduceArgmax/ReduceArgmin return None on A18 (no LSE_7 converter). ReduceL2Norm returns None on A11Legacy/A12/A13. MILSquare returns None on A14Minus families. SDPA returns None pre-A16. Existing `default_engine()` preserved for backward compat. 7 new tests.

---

### T-114 · ~~Fix PIR Tensor Spec Dtype Hardcoding~~

- **ISSUES ref**: I-89
- **AUDIT ref**: V-038 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `derive_primary_dtype()` method to ShardPlanPass that scans SIR graph for dtype-bearing ops (Const, Cast, Fill, Quantize, Dequantize). Replaced all 9 hardcoded `"fp16"` values in shard_plan.rs with `primary_dtype.clone()`. Fallback to "fp16" with explicit warning when no dtype-bearing op found. 3 new tests verify derivation, fallback, and fp32 propagation.

---

### T-115 · ~~Make Opset Version and Deployment Target Configurable~~

- **ISSUES ref**: I-90
- **AUDIT ref**: V-037, V-046 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `DEFAULT_MINIMUM_DEPLOYMENT_TARGET` constant separate from `DEFAULT_OPSET_VERSION`. Decoupled `minimum_deployment_target` from `opset_version` in `ShardPipelineSpec`. Added `with_deployment_target()` builder method. Updated `shard_desc.rs` and `serialize.rs` to use separate constant. Tests verify independent setting and default values.

---

### T-116 · ~~Add ANEC Attribute Shape Validation~~

~~**ISSUES ref**: I-91~~
~~**AUDIT ref**: V-107 (ane-violations.md §III)~~
~~**Severity**: MEDIUM~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — Added `validate_anec_attribute_shapes()` in op_constraints.rs validating stride, pad_amounts, and dilation vector shapes for conv, pool, and deconv operations per ANEC schema. Wired into mir_to_compat conversion to validate before attributes are dropped. 10 new tests.

---

### T-117 · ~~Extend AneFamily to Cover All 8 Binary-Defined Families~~

~~**ISSUES ref**: I-92~~
~~**AUDIT ref**: V-109 (ane-violations.md §III)~~
~~**Severity**: MEDIUM~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — Added `minimum_family_value()` mapping each AneFamily to ANEC binary MinimumFamily discriminant (0-7). Added `from_minimum_family_value()` constructor. Added `family_level()` ordering method. Added `ALL_FAMILIES` constant. 6 new tests.

---

### T-118 · ~~Add Palette Bits Validation with Version-Conditional Support~~

- **ISSUES ref**: I-93
- **AUDIT ref**: V-033 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `validate_palette_bits_for_family()` in `ane_layout.rs` that checks version-conditional constraints: 3-bit and 6-bit palettization rejected on A11Legacy/A12/A13 (A14+ only). Uses existing `uses_a14minus_converters()` for family detection. Re-exported from `palettize_weights.rs`. 10 new tests verify all combinations.

---

### T-119 · ~~Add Minimum IOSurface Size Validation~~

- **ISSUES ref**: I-94
- **AUDIT ref**: V-104, V-122 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `MIN_IOSURFACE_BYTES` constant (~49 KB per Orion #4). Added `validate_iosurface_sizes()` in `mir_to_proto.rs` that computes output buffer sizes (shape_product × dtype_size) and warns when below minimum. Called from `convert_mir_to_proto_multifunction()`. 4 new tests verify constant, small/large buffer detection, and size computation.

---

### T-120 · ~~Add Compilation Count Per Process Tracking~~

- **ISSUES ref**: I-95
- **AUDIT ref**: V-106, V-123 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added global `COMPILATION_COUNT: AtomicU64` counter in `emitter.rs`. `emit_model()` increments counter and returns error at limit (119), warns at threshold (95). Added `COMPILATION_LIMIT`, `COMPILATION_WARNING_THRESHOLD`, and `compilation_count()` public API. Added `compilation_number: u64` to `ProtoEmitResult`. 3 new tests.

---

### T-121 · ~~Add Vector Palettization At-Cout Constraint~~

- **ISSUES ref**: I-96
- **AUDIT ref**: V-133 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `validate_vector_palettization_constraints()` in op_constraints.rs enforcing three ANEC constraints: vector palettization only at Cout dimension, zero point rejected for vector palettized kernel, palette size 256 rejected. Re-exported from palettize_weights.rs. 8 new tests.

---

### T-122 · ~~Add Weight Dict Initialization Check~~

- **ISSUES ref**: I-97
- **AUDIT ref**: V-124 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added weight dictionary validation in `MlPackageWriter::write()` that warns when a model has functions but zero weights (nil weight dict crashes ANEC per Orion #11). 2 new tests verify warning behavior for empty weights with functions and no-warning for no functions.

---

### T-123 · ~~Add Stencil Constraints~~

- **ISSUES ref**: I-98
- **AUDIT ref**: V-137 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `validate_stencil_constraints()` with 5 checks: 5D stencil rejection, non-4D kernel rejection, non-sum reduction mode rejection, dilated stencil rejection, strided stencil rejection. Tests for each constraint.

---
## 🟡 MEDIUM — Configuration / Documentation / Quality
### T-124 · ~~Add V26 Speculative Warning~~
- **AUDIT ref**: V-031, V-088
**✅ RESOLVED** — Added log::warn!() in AneHwLimits::future() (V26 path) that limits are speculative. Follows same pattern as A12 warning.

### T-125 · ~~Replace Hardcoded Dimension Defaults~~
- **AUDIT ref**: V-089
**✅ RESOLVED** — Replaced 8 unwrap_or(64/48/32) dimension defaults in role_mir.rs with ok_or_else() error returns. Missing shape specs now produce explicit errors.

### T-126 · ~~Add Canonicalization Cycle Limit Diagnostic~~
- **AUDIT ref**: V-090
**✅ RESOLVED** — Added log::warn!() when canonicalization substitution chain resolution hits 100-step limit.

### T-127 · ~~Document UInt16/Bool Limited Support~~
- **AUDIT ref**: V-091, V-092. **✅ RESOLVED** — Added `validate_uint16_constraints()` and `validate_bool_constraints()` in dtype_constraints.rs. UInt16 only valid as output of TopK/Sort/ReduceArgmax/ReduceArgmin. Bool only valid as mask input for Select/Where. Added `UInt16ConstraintViolation` and `BoolConstraintViolation` error variants. Module-level documentation added for V-091/V-092 constraints. 6 new tests.

### T-128 · ~~Expand CPU_ONLY_OPS_DETAILED~~
- **AUDIT ref**: V-093. **✅ RESOLVED** — Expanded `CPU_ONLY_OPS_DETAILED` from 38 to 154 entries (100% coverage of CPU_ONLY_OPS set). Added 116 new entries with reason codes: NoConverter (56), ControlFlow (18), Gradient (11), Logical (10), TrigInverse (1), ComplexNumber (1), Fft (2), Rnn (3), Cumulative (3), Random (3), Scatter (1), Sparse (3), ShapeQuery (1). Verified no duplicate entries. 3 new tests.

### T-129 · ~~Fix Knowledge Schema/Seed Format Mismatch~~
- **AUDIT ref**: V-071. **✅ RESOLVED** — Added 4 missing `KnowledgeType` variants (CpuOnlyOps, AneHwLimits, PalettizationConstraints, AneOpFamilyMatrix) to schema and Rust enum. Updated `claims_agree()` for new types. Fixed `precision_hazard_seed.json` field rename `op` → `op_pattern`. Added "Seed File Formats (Current)" section to knowledge_schema.md documenting actual seed formats and migration path. 7 new seed validation integration tests.

### T-130 · ~~Wire --seed Parameter or Remove~~
- **AUDIT ref**: V-087
**✅ RESOLVED** — Removed --seed from Compile and CompileFull CLI subcommands — was dead code. Retained for sharded/lab/generate commands where functional.

### T-131 · ~~Emit Matmul Transpose Flags as Named Const Nodes~~
- **AUDIT ref**: V-129

**✅ RESOLVED** — Replaced `make_value_arg(make_immediate_bool_value(...))` with named const node pattern for transpose_x and transpose_y in Apple proto emission. Per Orion #12, transpose flags must be emitted as named const nodes, not immediate bools. MatMul now emits 3 operations: `{name}_transpose_x_0` const, `{name}_transpose_y_0` const, and the matmul op referencing them by name. 1 new test in mir.rs.

---

## Task → Issue Cross-Reference

| Task | Issue(s) | Severity | Effort | Status |
|------|----------|----------|--------|--------|
| T-90 | I-65 | CRITICAL | L | ✅ Resolved |
| T-95 | I-70 | HIGH | M | ✅ Resolved |
| T-96 | I-71 | HIGH | M | ✅ Resolved |
| T-98 | I-73, I-94 | HIGH | L | ✅ Resolved |
| T-105 | I-80 | MEDIUM | M | ✅ Resolved |
| T-107 | I-82 | MEDIUM | M | ✅ Resolved |
| T-108 | I-83 | MEDIUM | L | ✅ Resolved |
| T-110 | I-85 | MEDIUM | M | ✅ Resolved |
| T-112 | I-84, I-85 | MEDIUM | M | ✅ Resolved |
| T-113 | I-88 | MEDIUM | M | ✅ Resolved |
| T-116 | I-91 | MEDIUM | M | ✅ Resolved |
| T-117 | I-92 | MEDIUM | M | ✅ Resolved |
| T-118 | I-93 | MEDIUM | S | ✅ Resolved |
| T-119 | I-94 | MEDIUM | S | ✅ Resolved |
| T-120 | I-95 | MEDIUM | S | ✅ Resolved |
| T-121 | I-96 | MEDIUM | S | ✅ Resolved |
| T-122 | I-97 | MEDIUM | S | ✅ Resolved |
| T-124 | — | LOW | S | ✅ Resolved |
| T-125 | — | LOW | S | ✅ Resolved |
| T-126 | — | LOW | S | ✅ Resolved |
| T-127 | — | LOW | S | ✅ Resolved |
| T-128 | — | LOW | S | ✅ Resolved |
| T-129 | — | LOW | M | ✅ Resolved |
| T-130 | — | LOW | S | ✅ Resolved |
| T-131 | — | LOW | S | ✅ Resolved |
| T-86 | I-61 | CRITICAL | S | ✅ Resolved |
| T-87 | I-62 | CRITICAL | M | ✅ Resolved |
| T-89 | I-64 | CRITICAL | S | ✅ Resolved |
| T-91 | I-66 | HIGH | M | ✅ Resolved |
| T-92 | I-67 | HIGH | M | ✅ Resolved |
| T-93 | I-68 | HIGH | M | ✅ Resolved |
| T-94 | I-69 | HIGH | S | ✅ Resolved |
| T-97 | I-72 | HIGH | M | ✅ Resolved |
| T-99 | I-74 | HIGH | S | ✅ Resolved |
| T-100 | I-75 | HIGH | S | ✅ Resolved |
| T-101 | I-76 | HIGH | M | ✅ Resolved |
| T-102 | I-77 | HIGH | S | ✅ Resolved |
| T-103 | I-78 | HIGH | S | ✅ Resolved |
| T-104 | I-79 | MEDIUM | S | ✅ Resolved |
| T-106 | I-81 | MEDIUM | S | ✅ Resolved |
| T-109 | I-84 | MEDIUM | S | ✅ Resolved |
| T-111 | I-86 | MEDIUM | S | ✅ Resolved |
| T-114 | I-89 | MEDIUM | S | ✅ Resolved |
| T-115 | I-90 | MEDIUM | S | ✅ Resolved |
| T-123 | I-98 | MEDIUM | S | ✅ Resolved |

---

## Task → Violation Cross-Reference

| Task | Violation(s) | Area |
|------|-------------|------|
| T-91 | V-001, V-002 | Knowledge seed family mapping |
| T-92 | V-003, V-004, V-005 | Knowledge seed three-way contradictions |
| T-93 | V-098, V-130 | MILConcat emission (Orion #1) |
| T-94 | V-113, V-099 | Gelu EXACT mode (Orion #10) |
| T-95 | V-007 | Zero-fill weight placeholders |
| T-96 | V-003, V-004, V-005, V-029, V-030, V-072 | Knowledge seed vs. binary evidence |
| T-97 | V-009, V-103, V-115, V-132 | Conv/pool constraint validation |
| T-98 | V-116, V-048 | Deconvolution constraint validation |
| T-99 | V-117, V-118, V-119, V-120, V-121 | Orion surface ordering/uniformity (#2,#3,#18,#19,#20) |
| T-100 | V-125, V-126 | BF16/F16 cross-type, FP32 arch-conditional |
| T-101 | V-012, V-025, V-035 | Model architecture leakage (Generic→Qwen3) |
| T-102 | V-051, V-111, V-050, V-134, V-128, V-127 | Dtype validation gaps |
| T-103 | V-014, V-016, V-080, V-081, V-082 | Stub-mimic / phantom capabilities |
| T-104 | V-023, V-085, V-062 | I/O node / shape fallback |
| T-105 | V-026, V-027 | F32→FP16 conversion, dtype blob mapping |
| T-106 | V-114 | MILLinear→MILMatMul regression (Orion #17) |
| T-107 | V-100, V-138, V-110 | Missing ANEC operation mappings |
| T-108 | V-019, V-136 | Gather contradiction resolution |
| T-109 | V-137, V-133, V-135 | Stencil constraints, vector palettization |
| T-110 | V-021, V-022, V-053, V-055, V-056, V-071 | Knowledge system integrity |
| T-111 | V-031, V-034, V-037, V-046, V-087 | Hardcoded constants / phantom capabilities |
| T-112 | V-017, V-038, V-045, V-047 | PIR/shard plan hardcoded values |
| T-113 | V-104, V-106, V-124 | Orion runtime constraints (#4,#5,#11) |
| T-114 | V-076, V-077, V-079, V-083 | Lab module stubs / hardcoded values |
| T-115 | V-108, V-109, V-131 | AneFamily/AneRevision enum gaps |
| T-116 | V-059, V-089, V-090, V-091, V-092, V-093, V-094, V-095, V-097, V-129 | Doc/code mismatches, minor quality |

---
### v3 Audit Sprint Tasks (T-58 through T-85)

All 28 tasks resolved across four sprints. See CHANGELOG.md for details.

### v2 Audit (T-47 through T-66)

Tasks T-47 through T-57, T-60, T-62, T-63, T-64, T-65 resolved. T-61, T-66 remain open (carried forward above).

## Summary Statistics

| Severity | Count | Total Effort |
|----------|-------|-------------|
| 🔴 CRITICAL | 0 | 0 |
| **Total** | **0** | **0** |

> **All tasks resolved.** No open tasks remain.
>
> **Resolved**: 45 tasks (T-86, T-87, T-89, T-90, T-91–T-99, T-100–T-108, T-109–T-131).
> Tasks T-01 through T-85 are all resolved — see archive summary above and `CHANGELOG.md`.
