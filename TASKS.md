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

### T-90 · Replace Concat Emissions with ANE-Legal Alternatives

- **ISSUES ref**: I-65
- **AUDIT ref**: V-098, V-130 (ane-violations.md §III)
- **Severity**: CRITICAL
- **Effort**: L (2 days)

**Intent**: MILLer emits `MILConcat` in two critical code paths — the SDPA (scaled dot-product attention) decomposition path and the RoPE (rotary position embedding) rotate_half path. The ANE compiler rejects concat operations per Orion constraint #1 — concat is only supported along the channel axis with a constant positive axis. The SDPA path emits concat along non-channel axes, and the RoPE path emits `concat(-x2, x1, axis=-1)` which uses a negative axis value. Both paths produce models that fail at ANEC compile time. The binary forensic evidence confirms extensive concat constraints: "Concat supports only 1 axis", "ANE Concat supports only const positive axis", "failed: only works when concat is applied on the channel axis".

**Mitigation / Implementation**:
1. **SDPA path** (`crates/passes/src/mil_lower.rs:2842-2858`): Replace concat emission with reshape+stack sequence or emit as fused SDPA for A16+ targets (where `ConvertScaledDotProductAttention` is reliable). For pre-A16 targets, decompose into ANE-legal reshape+transpose+stack operations.
2. **RoPE rotate_half** (`crates/passes/src/legality_rewrite.rs:3098,3622`): Replace `concat(-x2, x1, axis=-1)` with an equivalent reshape+transpose sequence: (a) Reshape the two halves to separate channel dimensions, (b) Stack along the channel axis using a reshape+transpose pattern, (c) Reshape back to the original output shape.
3. Add a test that constructs a MIR graph with the old concat patterns, runs the lowering/rewrite passes, and verifies that no `MILConcat` nodes remain in the output graph for ANE-targeted paths.
4. Add a linter check in the placement validator that warns if `MILConcat` appears in a graph targeting the ANE with a non-channel axis.
5. Document the concat constraints from Orion #1 in `ane-constraints-docs/` if not already present.

**Definition of Done**:
- [ ] No `MILConcat` nodes emitted in SDPA decomposition path for ANE targets
- [ ] No `MILConcat` nodes emitted in RoPE rotate_half path for ANE targets
- [ ] New test verifies zero concat nodes in ANE-targeted graph output
- [ ] Placement validator warns on ANE-targeted concat with non-channel axis
- [ ] `cargo test` passes with zero failures
- [ ] Existing SDPA and RoPE functionality preserved (output numerical equivalence)

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

### T-95 · Add Multi-Surface Ordering and Uniformity Validation

- **ISSUES ref**: I-70
- **AUDIT ref**: V-105, V-117, V-118, V-119, V-120, V-121 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**Intent**: The ANE requires multi-output and multi-input surfaces to be in alphabetical order (Orion #3, #19) and have uniform allocation sizes (Orion #2, #18). MILLer neither sorts surfaces alphabetically nor validates size uniformity. Incorrect surface ordering causes silent data corruption — correct values written to wrong output tensors. Non-uniform sizes cause 0x1d runtime errors with no compile-time indication. The ANE flat buffer layout packed `[1,C,1,S]` is also not validated, meaning data written in wrong layout produces silently incorrect inference.

**Mitigation / Implementation**:
1. In `crates/coreml-emit/src/mir_to_proto.rs`: Sort multi-output tensor names alphabetically before emission. Add comment: `// Orion #3: ANE reads outputs in alphabetical order`.
2. In `crates/coreml-emit/src/mir_to_proto.rs`: Sort multi-input tensor names alphabetically before emission. Add comment: `// Orion #19: ANE reads inputs in alphabetical order`.
3. In `crates/coreml-emit/src/mir_to_proto.rs`: Add `validate_surface_uniformity()` that checks all output buffers have the same byte size and all input buffers have the same byte size. Return error if non-uniform.
4. In `crates/coreml-emit/src/mir_to_proto.rs`: Add `validate_flat_buffer_layout()` that checks tensor data is written in packed `[1,C,1,S]` format.
5. Add tests for alphabetical ordering, uniformity validation, and layout validation.

**Definition of Done**:
- [ ] Multi-output surfaces sorted alphabetically before emission
- [ ] Multi-input surfaces sorted alphabetically before emission
- [ ] Surface size uniformity validated (both input and output)
- [ ] Flat buffer layout [1,C,1,S] validated
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

---

### T-96 · Fix MILLinear→MILMatMul Defeating Conv1x1AsLinear Optimization

- **ISSUES ref**: I-71
- **AUDIT ref**: V-114 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1 day)

**Intent**: The AIR-to-MIR pipeline correctly creates `Conv1x1AsLinear` in the legality rewrite pass (`legality_rewrite.rs:354-370`), but the MIL lowering pass then converts ALL `MILLinear` to `MILMatMul` (`mil_lower.rs:3268-3307`). Orion #17 documents that conv 1x1 is 3x faster than matmul on ANE. The comment says "The linear op may not have an ANE execution converter" but binary evidence shows `ConvertLayer` has 97 instances (far more than `ConvertMatMul`'s 8 family instantiations). The conversion to matmul loses the 3x performance benefit and may also lose palettization support.

**Mitigation / Implementation**:
1. In `crates/passes/src/mil_lower.rs`: Change the `MILLinear` lowering to preserve `Conv1x1AsLinear` as `MILConv(1x1)` instead of converting to `MILMatMul` for ANE targets.
2. Add a `target: AneTarget` parameter to the lowering pass so it can make target-aware decisions.
3. For CPU targets, keep the existing MILLinear→MILMatMul conversion (matmul may be more efficient on CPU).
4. Add a test that verifies a linear layer with kernel shape `[out, in, 1, 1]` produces `MILConv` (not `MILMatMul`) when targeting ANE.
5. Add a test that verifies the existing MILLinear→MILMatMul path still works for CPU targets.

**Definition of Done**:
- [ ] MILLinear with 1x1 kernel shape preserved as MILConv for ANE targets
- [ ] MILLinear→MILMatMul conversion retained for CPU targets
- [ ] Lowering pass accepts target parameter
- [ ] Tests verify both ANE and CPU paths
- [ ] `cargo test` passes with zero failures
- [ ] Performance: linear layers on ANE use Conv1x1 (3x faster per Orion #17)

---

### T-97 · ~~Add Dtype Cross-Validation and Rejection~~

- **ISSUES ref**: I-72, I-98, I-99, I-102
- **AUDIT ref**: V-125, V-126, V-051/V-111, V-134 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**✅ RESOLVED** — Added `CrossTypeViolation` and `AsymmetricQuantViolation` error variants in `dtype_constraints.rs`. Added `validate_cross_type_compatibility()` for BF16/F16 cross-type checks, rejecting all 9 documented ANEC cross-type combinations. Added `is_fp32_compute_supported()` — returns false for A11Legacy/A12 families where FP32 is rejected. Added `validate_anec_quantization_symmetry()` — rejects asymmetric quantization on ANE. Removed E5M2 from quantize validator accepted output dtypes (universally rejected by ANEC per V-051/V-111). Added comprehensive tests for all new functions.

---

### T-98 · Add Quantized Conv Weight Attributes

- **ISSUES ref**: I-73
- **AUDIT ref**: V-110 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: L (2 days)

**Intent**: ANEC's convolution schema includes `kernel_scale`, `kernel_zero_point`, `kernel_palettized_LUT`, and `kernel_mutable_palettized_LUT` attributes for quantized/palettized weights. MILLer's `MILConv` and `MilConvOp` proto don't carry any of these attributes, meaning quantized/palettized convolution emission is incomplete and will fail for any non-FP16 weight format. This is a blocker for full quantized model support on the ANE.

**Mitigation / Implementation**:
1. In `crates/ir/src/mir.rs`: Add `kernel_scale`, `kernel_zero_point`, `kernel_palettized_lut` fields to `MILConv` struct.
2. In `crates/coreml-proto/src/lib.rs`: Add corresponding `ToProto` emission for the new fields, mapping to the ANEC convolution proto attributes.
3. In `crates/coreml-proto/proto/coreml/MIL.proto`: Verify that the proto definition includes these fields; add if missing.
4. In `crates/passes/src/palettize_weights.rs`: When palettizing conv weights, populate `kernel_scale`, `kernel_zero_point`, and `kernel_palettized_lut` in the MILConv node.
5. Add tests for quantized conv emission with all new attributes.

**Definition of Done**:
- [ ] MILConv struct includes kernel_scale, kernel_zero_point, kernel_palettized_lut
- [ ] ToProto emission handles all new fields
- [ ] Palettize pass populates new fields for conv ops
- [ ] Tests for quantized conv emission
- [ ] `cargo test` passes with zero failures

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

### T-105 · Resolve Softmax/InstanceNorm Family Gating Contradiction

- **ISSUES ref**: I-80
- **AUDIT ref**: V-029, V-030, V-101 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: ConvertSoftmax and ConvertInstanceNorm are family-agnostic converters (no MinimumFamily trait in binary), but ANEC has architecture-conditional rejection strings. Neither the per-family matrix nor MILLer's constraint model captures this nuance — converters exist for all families but specific architecture variants may reject the operation at compile time.

**Definition of Done**: Documentation and constraint model updated to reflect "converter available for all families but architecture-conditional rejection possible". Add soft-warning at placement for Softmax/InstanceNorm on older architectures. Test verifies warning is emitted.

---

### T-106 · ~~Add Pooling Stride-3 Avg-Only Check and Other Pool Constraints~~

- **ISSUES ref**: I-81
- **AUDIT ref**: V-127 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added stride-3 avg-only pool validation, stride>3 rejected for all modes, large-stride Min/Max pool with padding rejected. Tests for each constraint.

---

### T-107 · Implement or Remove StaticizePass

- **ISSUES ref**: I-82
- **AUDIT ref**: V-014 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: `StaticizePass::run()` is a pure pass-through — `Ok(input)`. Its documentation claims it replaces symbolic dimensions, resolves variable-length sequences, and records decisions. None of this is implemented. The phantom pass wastes developer trust and obscures the actual pipeline.

**Definition of Done**: Either StaticizePass is implemented (resolves symbolic dims) or removed from the pipeline with documentation explaining its absence. If removed, all references to it in the pipeline are cleaned up.

---

### T-108 · Expand Precision Policy Coverage

- **ISSUES ref**: I-83
- **AUDIT ref**: V-015 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: L (2 days)

**Intent**: Only 14 of ~167 SIR op types query precision hazards. All others silently use default fp16 even if stored knowledge indicates a hazard. This means precision-sensitive ops like attention projections and normalization layers may get wrong dtype assignments.

**Definition of Done**: Top-30 most common op types covered by precision policy; remainder marked with explicit "coverage gap" log warnings. Test verifies coverage for top-30 ops.

---

### T-109 · ~~Make StateTopologyPass Return Errors~~

- **ISSUES ref**: I-84
- **AUDIT ref**: V-016 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `strict: bool` field (default: true). Strict mode returns `Err` for ReadState without matching WriteState. WriteState without ReadState still logs info (valid for initial state). Added `new_lenient()` and `with_strict()` constructors. Tests for strict/non-strict modes.

---

### T-110 · Derive FunctionEntry Shapes from Graph

- **ISSUES ref**: I-85
- **AUDIT ref**: V-017 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: FunctionEntry TensorSpec shapes are hardcoded as `vec![1,1]` throughout `shard_plan.rs`. Comments say "derived from graph" but derivation is not implemented. This produces wrong PIR shapes for any model with batch > 1 or sequence length > 1.

**Definition of Done**: Shapes derived from MIR graph by walking node dimensions. Fallback to vec![1,1] only with explicit warning. Test verifies correct shapes for known models.

---

### T-111 · ~~Fix Interleave Validation When Channels Unknown~~

- **ISSUES ref**: I-83
- **AUDIT ref**: V-020 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — When channels is None, interleave validation still enforces: valid interleave factors, const→1, int4/uint4→8. Previously the entire validation was skipped when channels was unknown, allowing invalid dtype/interleave combinations to pass silently.

---

### T-112 · Fix Knowledge Conflict Detection Symmetry

- **ISSUES ref**: I-87
- **AUDIT ref**: V-021, V-022 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: Knowledge conflict detection marks new entry as `ConflictedWith(existing)` but never back-patches the existing entry. Querying only existing entries misses mutual conflicts. Additionally, `claims_agree` defaults to `true` for 7/8 knowledge types, preventing contradiction detection for PrecisionHazard, SurvivalMatrixEntry, etc.

**Definition of Done**: Conflict detection is symmetric (both entries marked). `claims_agree` has field-level comparison for all 8 types. Tests verify symmetry and field-level comparison.

---

### T-113 · Make default_engine() Revision-Aware

- **ISSUES ref**: I-88
- **AUDIT ref**: V-010 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: `default_engine()` returns static engine assignment per op regardless of AneRevision. Ops assigned to PE may be placed on families that don't support them (e.g., ArgMinMax on A18 has no converter but default_engine() returns Some(PE)).

**Definition of Done**: `default_engine()` takes `Option<AneRevision>` parameter and cross-references family capabilities. Returns `None` for ops not supported on the target family. Test verifies revision-aware behavior for all ops.

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

### T-116 · Add ANEC Attribute Shape Validation

- **ISSUES ref**: I-91
- **AUDIT ref**: V-107 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: ANEC defines precise attribute shapes for all 98 operations (e.g., convolution: stride=shape{3}, padding=shape{6}). MILLer doesn't validate that emitted attributes match ANEC expectations. Wrong-shaped attributes fail at ANEC compile time with cryptic errors.

**Definition of Done**: Validation function checks attribute shapes per ANEC schema before emission. At minimum, validate conv, pool, and matmul attribute shapes. Tests for each op type.

---

### T-117 · Extend AneFamily to Cover All 8 Binary-Defined Families

- **ISSUES ref**: I-92
- **AUDIT ref**: V-109 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: The MinimumFamily enum in the ANEC binary has values 0-7 (8 families), but MILLer only models 6 (A11Legacy through A18). Families 6-7 may correspond to A16+ variants or future architectures. MILLer cannot express constraints for these families.

**Definition of Done**: AneFamily enum extended to 8 variants matching binary MinimumFamily. Revision-to-family mapping updated. Test verifies all 8 families are mapped.

---

### T-118 · Add Palette Bits Validation with Version-Conditional Support

- **ISSUES ref**: I-93
- **AUDIT ref**: V-033 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Palette bits documented as valid {1,2,3,4,6,8} but no validation enforces this. Binary shows version-conditional constraints: "3-bit palettization is only supported from version {1}", "6-bit palettization is only supported from version {1}". Out-of-range values accepted silently.

**Definition of Done**: `validate_palette_bits()` checks against valid set per version. 3-bit and 6-bit rejected on pre-threshold versions. Test validates version-conditional behavior.

---

### T-119 · Add Minimum IOSurface Size Validation

- **ISSUES ref**: I-94
- **AUDIT ref**: V-104, V-122 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: ANE requires a minimum IOSurface size of ~49 KB for eval buffers (Orion #4). Models with smaller output buffers fail at runtime with 0x1d error and no compile-time indication.

**Definition of Done**: Emission pipeline validates output buffer sizes meet ~49 KB minimum. Warning emitted for borderline cases. Test verifies enforcement.

---

### T-120 · Add Compilation Count Per Process Tracking

- **ISSUES ref**: I-95
- **AUDIT ref**: V-106, V-123 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: ANE has a ~119 compilation limit per process (Orion #5). Long-running processes that repeatedly compile models silently fail after hitting this limit. No counter or warning mechanism exists.

**Definition of Done**: Global compilation counter tracks count per process. Warning emitted at ~100 compilations. Error at ~119. Test verifies counter behavior.

---

### T-121 · Add Vector Palettization At-Cout Constraint

- **ISSUES ref**: I-96
- **AUDIT ref**: V-133 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Vector palettization is only supported at Cout dimension for ANE. No enforcement exists — vector palettization at other dimensions will fail at ANEC compile time. Additionally, "zero point is not supported for vector palettized kernel" and "Quantized kernel with palettize size=256 is not supported" are unenforced.

**Definition of Done**: Vector palettization rejected at non-Cout dimensions. Zero point rejected for vector palettized kernel. Size=256 rejected. Tests for each constraint.

---

### T-122 · Add Weight Dict Initialization Check

- **ISSUES ref**: I-97
- **AUDIT ref**: V-124 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Nil weight dict causes immediate crash at ANEC compile time (Orion #11). MILLer's emission path does not verify that the weight dictionary is properly initialized as `@{}` (empty NSDictionary, not nil).

**Definition of Done**: Emission pipeline validates weight dict is non-nil before ANEC compilation. Test verifies check.

---

### T-123 · ~~Add Stencil Constraints~~

- **ISSUES ref**: I-98
- **AUDIT ref**: V-137 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**✅ RESOLVED** — Added `validate_stencil_constraints()` with 5 checks: 5D stencil rejection, non-4D kernel rejection, non-sum reduction mode rejection, dilated stencil rejection, strided stencil rejection. Tests for each constraint.

---
## 🟡 MEDIUM — Configuration / Documentation / Quality
### T-124 · Add V26 Speculative Warning
- **AUDIT ref**: V-031, V-088. Add explicit warning in `for_revision(V26)` that limits are speculative. **DoD**: Warning logged when V26 used.

### T-125 · Replace Hardcoded Dimension Defaults
- **AUDIT ref**: V-089. Replace `unwrap_or` defaults (64, 48, 32) with explicit config or fail-closed. **DoD**: No silent dimension defaults.

### T-126 · Add Canonicalization Cycle Limit Diagnostic
- **AUDIT ref**: V-090. Log warning when 100-step limit hit. **DoD**: Diagnostic output on limit hit.

### T-127 · Document UInt16/Bool Limited Support
- **AUDIT ref**: V-091, V-092. Specify which ops/families support UInt16 and Bool. **DoD**: Constraint docs updated.

### T-128 · Expand CPU_ONLY_OPS_DETAILED
- **AUDIT ref**: V-093. Cover all 120+ CPU-only ops with reason codes. **DoD**: All CPU-only ops have documented reasons.

### T-129 · Fix Knowledge Schema/Seed Format Mismatch
- **AUDIT ref**: V-071. Align seed JSON format with knowledge_schema.md. **DoD**: Seeds validate against schema.

### T-130 · Wire --seed Parameter or Remove
- **AUDIT ref**: V-087. Either wire through compile pipeline or remove from CLI. **DoD**: --seed either works or is removed.

### T-131 · Emit Matmul Transpose Flags as Named Const Nodes
- **AUDIT ref**: V-129. Per Orion #12, named const nodes instead of immediate bools. **DoD**: Transpose flags emitted as named const nodes.

---

## Task → Issue Cross-Reference

| Task | Issue(s) | Severity | Effort | Status |
|------|----------|----------|--------|--------|
| T-90 | I-65 | CRITICAL | L | 🔴 Open |
| T-95 | I-70 | HIGH | M | 🟠 Open |
| T-96 | I-71 | HIGH | M | 🟠 Open |
| T-98 | I-73 | HIGH | L | 🟠 Open |
| T-105 | I-80 | MEDIUM | M | 🟡 Open |
| T-107 | I-82 | MEDIUM | M | 🟡 Open |
| T-108 | I-83 | MEDIUM | L | 🟡 Open |
| T-110 | I-85 | MEDIUM | M | 🟡 Open |
| T-112 | I-84, I-85 | MEDIUM | M | 🟡 Open |
| T-113 | I-88 | MEDIUM | M | 🟡 Open |
| T-116 | I-91 | MEDIUM | M | 🟡 Open |
| T-117 | I-92 | MEDIUM | M | 🟡 Open |
| T-118 | I-93 | MEDIUM | S | 🟡 Open |
| T-119 | I-94 | MEDIUM | S | 🟡 Open |
| T-120 | I-95 | MEDIUM | S | 🟡 Open |
| T-121 | I-96 | MEDIUM | S | 🟡 Open |
| T-122 | I-97 | MEDIUM | S | 🟡 Open |
| T-124 | — | LOW | S | 🔵 Open |
| T-125 | — | LOW | S | 🔵 Open |
| T-126 | — | LOW | S | 🔵 Open |
| T-127 | — | LOW | S | 🔵 Open |
| T-128 | — | LOW | S | 🔵 Open |
| T-129 | — | LOW | M | 🔵 Open |
| T-130 | — | LOW | S | 🔵 Open |
| T-131 | — | LOW | S | 🔵 Open |
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
| 🔴 CRITICAL | 1 | ~2 days |
| 🟠 HIGH | 3 | ~7.5 days |
| 🟡 MEDIUM | 12 | ~10.5 days |
| 🔵 LOW | 8 | ~4.5 days |
| **Total** | **24** | **~24.5 days** |

> **Priority guidance**: CRITICAL task T-90 must be resolved before any production compilation.
> HIGH tasks (T-95, T-96, T-98) should be addressed in the next 2–3 sprint cycles.
> MEDIUM/LOW tasks (T-105–T-131) are technical debt that should be chipped away at consistently.
>
> **Resolved**: 20 tasks (T-86, T-87, T-89, T-91–T-94, T-97, T-99–T-104, T-106, T-109, T-111, T-114, T-115, T-123).
> Tasks T-01 through T-85 are all resolved — see archive summary above and `CHANGELOG.md`.
