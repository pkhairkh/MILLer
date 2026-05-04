# MILLer — Prioritised Action List

> Ranked by **severity × impact**. Each task references findings in `docs/audit/ane-violations.md` and `ISSUES.md`.
> Estimates assume a single experienced Rust/Python developer.
> Generated from **ANE Violations Deep Forensic Audit** (ane-violations.md) on 2026-05-07.
> All prior tasks **T-01 through T-90** are **RESOLVED** — see archive summary below and `CHANGELOG.md` for details.
> New tasks numbered from **T-91**, derived from V-XXX violation IDs in ane-violations.md.
> Ranked by **impact × urgency**. Each task references findings in `docs/audit/ane-violations.md` (§III, §VI) and `docs/audit/tabula-rasa-v3.md` (§VII) and `ISSUES.md`.
> Estimates assume a single experienced Rust/Python developer.
> Generated from TABULA RASA v3 + NECROSCOPY forensic audits on 2026-05-04.
> Tasks T-01 through T-85 are all resolved — see CHANGELOG.md for details.

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

### T-89 · ~~Fix Gelu Mode Contradictions — Standardize on TANH_APPROXIMATION~~

~~**ISSUES ref**: I-64~~
~~**AUDIT ref**: V-099, V-113 (ane-violations.md §III)~~
~~**Severity**: CRITICAL~~
~~**Effort**: S (0.5 day)~~
**✅ RESOLVED** — Changed SIR builder from `"EXACT"` to `"TANH_APPROXIMATION"` in sir_build.rs. Updated test fixture in staticize.rs.

---

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

## 🔴 CRITICAL — Silent Miscompilation or Data Corruption

### T-91 · Fix Knowledge Seed Family Assignment Mismatches (V-001, V-002)

**ISSUES ref**: I-66
**AUDIT ref**: V-001, V-002 (ane-violations.md §III CRITICAL)
**Severity**: CRITICAL
**Effort**: S (0.5 day)

**Intent**: The `ane_hw_limits_seed.json` maps V6→family "A14" but Rust code (`ane_target.rs`) maps V6→A13. Similarly, V11→family "A16" in the JSON but V11→A17 in Rust. These mismatches grant A14-class capabilities to A13 hardware and miss A17 E4M3 support. Every model compiled for A13 or A17 hardware uses wrong constraint data, potentially placing ops on the ANE that the hardware cannot execute, causing silent runtime failures.

**Mitigation/Implementation**:
1. In `knowledge/ane_hw_limits_seed.json`, change V6's family field from "A14" to "A13" and V11's family field from "A16" to "A17".
2. Add a CI test that loads `ane_hw_limits_seed.json` and verifies every V→family mapping matches the Rust `ane_target.rs` `revision_to_family()` function. The test should enumerate all defined AneRevision variants, look up their family in both JSON and Rust, and assert equality.
3. Add inline comment in JSON documenting the source of truth (Rust code).

**Definition of Done**:
- [ ] `ane_hw_limits_seed.json` V6 family == "A13" matching Rust
- [ ] `ane_hw_limits_seed.json` V11 family == "A17" matching Rust
- [ ] New test `test_hw_limits_seed_family_consistency()` passes, verifying ALL V→family mappings
- [ ] `cargo test` passes with zero failures
- [ ] No other seed JSONs have V→family mismatches
### T-88 · ~~Replace Silent Fp16 Dtype Default with Explicit Error~~

~~**ISSUES ref**: I-63~~
~~**AUDIT ref**: V-011 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: S (0.5 day)~~
**✅ RESOLVED** — `shard_desc.rs` now returns explicit error for unrecognized dtype strings. Added Int8 and UInt8 as recognized dtype strings.

---

### T-91 · ~~Make Zero-Weight Placeholders a Hard Error by Default~~

~~**ISSUES ref**: I-66~~
~~**AUDIT ref**: V-007 (ane-violations.md §III)~~
~~**Severity**: HIGH~~
~~**Effort**: M (1 day)~~
**✅ RESOLVED** — `mir_to_compat.rs` now errors by default when weights can't be resolved. Added `allow_missing_weights` parameter. Added `mir_graph_to_compat_with_allow_missing()` convenience function.

---

### T-92 · Add Conv/Pool Constraint Validation

- **ISSUES ref**: I-67
- **AUDIT ref**: V-009, V-132, V-128 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**Intent**: MILLer defines several conv/pool constraint fields in `AneHwLimits` but never validates them in `validate_tensor_dims()`. Additionally, the conv kernel range check (1–7) allows kernel sizes 3, 5, 6, 7 which fail the ANEC power-of-2 requirement. Dilated pooling and dilated stencil are rejected by ANEC but MILLer has no dilation check for these operations. These unenforced constraints mean models with oversized kernels, non-power-of-2 kernels, or dilated pooling pass validation but fail at ANE runtime with cryptic errors.

**Mitigation / Implementation**:
1. In `crates/passes/src/op_constraints.rs`: Add `validate_conv_kernel_constraints()` function that checks: (a) kernel width and height are power of 2, (b) kernel depth is power of 2 for 3D convolutions, (c) kernel size is within revision-specific limits from `AneHwLimits`. Wire into `validate_conv_constraints()`.
2. In `crates/passes/src/op_constraints.rs`: Add dilation check to `validate_pooling_constraints()` — if dilation is present and > 1, return `bail!("Dilated pooling is not supported on ANE")`.
3. Add `validate_stencil_constraints()` for depthwise conv: reject 5D stencil, non-4D kernel, non-sum reduction mode, dilated stencil, and strided stencil.
4. Add tests for each new constraint: power-of-2 kernel validation, dilated pooling rejection, and stencil constraints.
5. Update `validate_tensor_dims()` to call the new validation functions for conv/pool ops.

**Definition of Done**:
- [ ] Conv kernel power-of-2 validation enforced
- [ ] Dilated pooling rejected with clear error message
- [ ] Dilated stencil rejected
- [ ] 5D stencil rejected
- [ ] Non-4D stencil kernel rejected
- [ ] Revision-specific conv kernel limits enforced
- [ ] Tests cover all new constraints
- [ ] `cargo test` passes with zero failures

---

### T-93 · Add Large Kernel Mode Constraints

- **ISSUES ref**: I-68
- **AUDIT ref**: V-115 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**Intent**: ANEC has a "large kernel" mode (activated when kernel width or height exceeds a threshold, likely 16 based on the existing dead-code check in `op_constraints.rs`) with 12+ additional constraints that MILLer doesn't enforce: kernel W/H must be multiple of 8, stride must be 1–2 only, zero padding only, no depth > 1, no palettized weights, input/output x and y strides must match, no grouped conv, no dynamic shape, no dilation. Without these validations, large-kernel convolutions pass MILLer's placement but fail at ANEC with opaque error messages.

**Mitigation / Implementation**:
1. In `crates/passes/src/op_constraints.rs`: Add `validate_large_kernel_constraints()` that checks all 12 constraints when kernel dimensions exceed the large-kernel threshold.
2. Define `LARGE_KERNEL_THRESHOLD: usize = 16` as a named constant (replacing the existing dead-code comparison at line 41).
3. For each constraint, return a specific error message matching the ANEC rejection: "Large kernel mode requires kernel W/H multiple of 8", "Large kernel mode requires stride 1 or 2", etc.
4. Wire into `validate_conv_constraints()`.
5. Add tests for each constraint: valid large kernels, invalid large kernels with each violation.

**Definition of Done**:
- [ ] `LARGE_KERNEL_THRESHOLD` defined as named constant
- [ ] All 12 large-kernel constraints validated
- [ ] Specific error messages matching ANEC rejection patterns
- [ ] Tests for each constraint (valid + invalid cases)
- [ ] `cargo test` passes with zero failures

---

### T-94 · Add Deconvolution Constraint Validation

- **ISSUES ref**: I-69
- **AUDIT ref**: V-116, V-048 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (1 day)

**Intent**: Deconvolution (ConvTranspose) has five ANEC-specific constraints that MILLer doesn't enforce: (1) no dilation, (2) SOx must equal 2, (3) no large kernel, (4) no vector palettization, (5) stride > 2 does not support kernel depth > 1. Currently, `ConvTranspose` always passes placement validation unconditionally (no kernel size, stride, or group checks). Models violating these constraints compile through MILLer but fail at ANEC.

**Mitigation / Implementation**:
1. In `crates/passes/src/op_constraints.rs`: Add `validate_deconv_constraints()` that checks all five constraints.
2. In `crates/passes/src/placement_validate.rs`: Wire deconv validation into the ConvTranspose placement check (currently unconditional pass).
3. Add tests for each constraint: valid deconv parameters, and invalid deconv with each violation.

**Definition of Done**:
- [ ] Deconvolution dilation rejected
- [ ] Deconv SOx != 2 rejected
- [ ] Deconv large kernel rejected
- [ ] Deconv vector palettization rejected
- [ ] Deconv stride > 2 with kernel depth > 1 rejected
- [ ] All checks wired into placement validator
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

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

### T-97 · Add Dtype Cross-Validation and Rejection

- **ISSUES ref**: I-72
- **AUDIT ref**: V-125, V-126, V-051/V-111, V-134 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1.5 days)

**Intent**: Four dtype-related validation gaps allow invalid models through to ANEC: (1) BF16/F16 cross-type operations are explicitly rejected by ANEC but `dtype_constraints.rs` has no cross-type validation — operations mixing BF16 and F16 operands will fail at ANEC compile time. (2) FP32 computation is rejected on some architectures but `is_dtype_ane_legal()` approves FP32 for all families without architecture check. (3) E5M2 is accepted by the quantize validator but universally rejected by ANEC ("E4M3 or E5M2 format not supported"). (4) Asymmetric quantization is not supported on ANEC but no check prevents it in the ANE path.

**Mitigation / Implementation**:
1. In `crates/passes/src/dtype_constraints.rs`: Add `validate_dtype_cross_type()` that rejects operations where any operand is BF16 and any other is F16, per the 9 ANEC cross-type rejection strings.
2. In `crates/passes/src/dtype_constraints.rs`: Make `is_dtype_ane_legal()` for FP32 architecture-conditional — return false for families where "Float32 not supported for architecture" applies. Add a new method `is_fp32_compute_supported(family: AneFamily) -> bool`.
3. In `crates/passes/src/dtype_constraints.rs`: Remove E5M2 from the quantize validator's accepted output dtypes. Add a comment: `// E5M2 is universally rejected by ANEC (V-051, V-111)`.
4. In `crates/passes/src/palettize_weights.rs`: Add asymmetric quantization rejection — if `quantization_type == "asymmetric"`, return error for ANE-targeted models.
5. Add tests for each new constraint.

**Definition of Done**:
- [ ] BF16/F16 cross-type operations rejected
- [ ] FP32 rejection architecture-conditional
- [ ] E5M2 removed from quantize validator accepted types
- [ ] Asymmetric quantization rejected for ANE path
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

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

### T-99 · Add Conv 32K-Channel Limit Validation

- **ISSUES ref**: I-74
- **AUDIT ref**: V-103 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**Intent**: `max_tensor_channels` in `AneHwLimits` is set to 65536 for newer revisions, but ANEC has a conv-specific channel limit of 32768 (Orion #16). The general channel validation uses `max_tensor_channels` (65536) which exceeds the actual conv-specific limit. Convolutions with channel counts between 32768 and 65536 pass validation but will fail at ANEC compile time.

**Mitigation / Implementation**:
1. In `crates/ir/src/ane_hw_limits.rs`: Add `max_conv_channels: usize` field to `AneHwLimits` with value 32768 for all revisions.
2. In `crates/passes/src/op_constraints.rs`: Add conv-specific channel check in `validate_conv_constraints()` using `max_conv_channels` instead of `max_tensor_channels`.
3. Add test: conv with 32769 channels should be rejected; conv with 32768 channels should pass.
4. Add comment: `// Orion #16: Conv-specific channel limit is 32K, lower than general max_tensor_channels`.

**Definition of Done**:
- [ ] `max_conv_channels` field added to AneHwLimits
- [ ] Conv validation uses max_conv_channels (32K) instead of max_tensor_channels
- [ ] Test validates 32K boundary
- [ ] `cargo test` passes with zero failures

---

### T-100 · Add Non-Constant Gather Axis Rejection

- **ISSUES ref**: I-75
- **AUDIT ref**: V-136 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**Intent**: ANEC rejects gather operations with non-constant axes ("gather with non-constant axis is not supported on ANEs"). MILLer emits dynamic-axis gather for embedding lookups in `mil_lower.rs` and the legality rewrite pass generates Gather for RoPE table lookups with potentially non-constant axes. These models will fail at ANEC compile time.

**Mitigation / Implementation**:
1. In `crates/passes/src/op_constraints.rs`: Add `validate_gather_constraints()` that checks if the gather axis is a compile-time constant. If not, reject with `"Gather with non-constant axis is not supported on ANE"`.
2. In `crates/passes/src/placement_validate.rs`: Wire gather constraint validation for MILGather ops.
3. Add tests: gather with constant axis should pass, gather with dynamic axis should be rejected.
4. Document that embedding gather operations must use constant-axis patterns.

**Definition of Done**:
- [ ] Non-constant gather axis rejected at constraint validation
- [ ] Constant-axis gather allowed
- [ ] Tests for both cases
- [ ] `cargo test` passes with zero failures

---

### T-101 · Replace Fallback Shapes/Dtypes with Hard Errors

- **ISSUES ref**: I-76
- **AUDIT ref**: V-023, V-025 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: M (1 day)

**Intent**: In `crates/bridge/src/mir_to_compat.rs`, when input/output nodes are missing from the MIR graph, the code falls back to shape `vec![1]` and dtype `Fp16`. These defaults are almost certainly wrong for any real model — a model with batch size > 1, sequence length > 1, or non-FP16 dtypes will get silently incorrect descriptors. Similarly, the default architecture silently defaults to Qwen3 weight name patterns (V-025), causing wrong input remapping and undefined references for non-Qwen3 models.

**Mitigation / Implementation**:
1. In `mir_to_compat.rs`: When an input/output node is not found in the MIR graph, return `bail!("Node '{}' not found in MIR graph. Cannot determine shape/dtype for compat layer.", name)` instead of using fallback defaults.
2. For the architecture default: if no architecture is specified and no Qwen3 patterns match, return an error requiring explicit architecture specification rather than defaulting to Qwen3.
3. Add tests: one verifying that missing nodes produce errors, and one verifying that explicit architecture specification works.

**Definition of Done**:
- [ ] Missing MIR nodes produce hard errors instead of fallback defaults
- [ ] Architecture must be explicitly specified when not matching Qwen3
- [ ] Tests verify error behavior
- [ ] `cargo test` passes with zero failures

---

### T-102 · Fix F32 Weight Passthrough Without FP16 Conversion

- **ISSUES ref**: I-77
- **AUDIT ref**: V-026 (ane-violations.md §III)
- **Severity**: HIGH
- **Effort**: S (0.5 day)

**Intent**: In `crates/bridge/src/safetensors_resolver.rs`, F32 weight data is passed through without conversion to FP16, even though BF16 gets converted. If the proto declares the weight as FP16 but the raw data is F32 (4 bytes per element), the weight file will contain double the expected bytes, causing buffer over-reads or misalignment at model load time.

**Mitigation / Implementation**:
1. In `safetensors_resolver.rs`: Add F32→FP16 conversion when the target dtype is FP16. Use the same conversion path as BF16→FP16 (via `half::f16::from_f32()`).
2. Add a test: load F32 weights, convert to FP16, verify the byte size is halved and values are approximately preserved.
3. Add a log message when F32→FP16 conversion occurs: `log::info!("Converting F32 weight '{}' to FP16 for ANE compatibility", name)`.

**Definition of Done**:
- [ ] F32 weights converted to FP16 when target is FP16
- [ ] Conversion uses same path as BF16→FP16
- [ ] Test verifies byte size halving and value preservation
- [ ] Log message on conversion
- [ ] `cargo test` passes with zero failures

---

### T-92 · Resolve Knowledge Seed Three-Way Contradictions (V-003, V-004, V-005)

**ISSUES ref**: I-67, I-68, I-69
**AUDIT ref**: V-003, V-004, V-005 (ane-violations.md §III CRITICAL)
**Severity**: CRITICAL
**Effort**: M (1 day)

**Intent**: Three knowledge seeds contradict each other and binary evidence. (1) Comparison ops (equal, not_equal, greater, etc.) are listed as CPU-only in `cpu_only_ops_seed.json` but have ConvertBinaryCompare ANEC converters on A14+. (2) Logical AND/OR/NOT are listed as "supported" A12+ in `ane_op_family_matrix.json` but have no ANEC converter — per-op doc confirms they never land on ANE. (3) `mb.gather` is declared ANE-illegal (ane_legal: false) in `legality_seed.json` but `anec.gather` exists with a ConvertGather converter. These contradictions cause the compiler to misclassify ops: either blocking ANE-legal ops from the ANE or allowing ANE-illegal ops onto the ANE.

**Mitigation/Implementation**:
1. In `cpu_only_ops_seed.json`, remove comparison ops (equal, not_equal, greater, less, greater_equal, less_equal) from the CPU-only list. Add them to `ane_op_family_matrix.json` with A14+ scope and appropriate confidence level.
2. In `ane_op_family_matrix.json`, change logical_and/or/not from "supported" to "cpu_only" for all families. Document that `anec.equal_zero` covers NOT but there are no dedicated logical_and/or converters.
3. In `legality_seed.json`, change `mb.gather` from `ane_legal: false` to `ane_legal: true` with a new constraint tag `limited_index_range` and appropriate notes about constant-axis requirement.
4. Add a test that cross-validates: every op listed as CPU-only in cpu_only_ops_seed is NOT listed as ANE-supported in ane_op_family_matrix, and vice versa. Every op listed in legality_seed as ane_legal:false must NOT have a converter in the ANEC catalog.

**Definition of Done**:
- [ ] Comparison ops removed from cpu_only_ops_seed.json, added to ane_op_family_matrix.json with A14+ scope
- [ ] Logical AND/OR/NOT marked as cpu_only in ane_op_family_matrix.json
- [ ] Gather changed to ane_legal:true with limited_index_range constraint in legality_seed.json
- [ ] New cross-validation test `test_knowledge_seed_consistency()` passes
- [ ] `cargo test` passes with zero failures
### T-103 · Map Bool/Float64/Unknown Dtypes Correctly in Weights

- **ISSUES ref**: I-78
- **AUDIT ref**: V-027 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: `crates/coreml-emit/src/weights.rs:116-119` silently maps Bool, Float64, and Unknown data types to Float32 blob format. This is data-corrupting for Bool tensors (1 bit vs 32 bit representation), incorrect for Float64 (8 bytes vs 4 bytes), and dangerous for Unknown (should be rejected entirely).

**Definition of Done**: Bool mapped to dedicated Bool blob type or rejected; Float64 mapped to 8-byte Float64 blob; Unknown dtype rejected early. Tests for each case.

---

### T-104 · Derive State Shape from ReadState Op

- **ISSUES ref**: I-79
- **AUDIT ref**: V-028 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: State declarations default to empty shape + Fp16 when only a write op is present. Core ML rejects protos with empty-dimension state tensors.

**Definition of Done**: State shape derived from corresponding ReadState op; error if no ReadState and no explicit shape. Test verifies non-empty state shapes.

---

### T-105 · Resolve Softmax/InstanceNorm Family Gating Contradiction

- **ISSUES ref**: I-80
- **AUDIT ref**: V-029, V-030, V-101 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: ConvertSoftmax and ConvertInstanceNorm are family-agnostic converters (no MinimumFamily trait in binary), but ANEC has architecture-conditional rejection strings. Neither the per-family matrix nor MILLer's constraint model captures this nuance — converters exist for all families but specific architecture variants may reject the operation at compile time.

**Definition of Done**: Documentation and constraint model updated to reflect "converter available for all families but architecture-conditional rejection possible". Add soft-warning at placement for Softmax/InstanceNorm on older architectures. Test verifies warning is emitted.

---

### T-106 · Add Pooling Stride-3 Avg-Only Check and Other Pool Constraints

- **ISSUES ref**: I-81
- **AUDIT ref**: V-127 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Pool stride 3 is only supported for Avg mode ("Pool with strides of 3 is only supported with Avg mode"). Additionally, "Large stride Min/Max pool with padding is not supported". Neither constraint is enforced.

**Definition of Done**: Stride-3 MaxPool rejected; large-stride Min/Max pool with padding rejected. Tests for each case.

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

### T-109 · Make StateTopologyPass Return Errors

- **ISSUES ref**: I-84
- **AUDIT ref**: V-016 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: StateTopologyPass claims to "verify" and "ensure" state patterns but only logs eprintln warnings — never returns Err. Invalid state naming conventions and capacity violations pass silently.

**Definition of Done**: Invalid state patterns return `Err` instead of eprintln. Tests verify error returns. Pass documentation updated to reflect enforcement behavior.

---

### T-110 · Derive FunctionEntry Shapes from Graph

- **ISSUES ref**: I-85
- **AUDIT ref**: V-017 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: M (1 day)

**Intent**: FunctionEntry TensorSpec shapes are hardcoded as `vec![1,1]` throughout `shard_plan.rs`. Comments say "derived from graph" but derivation is not implemented. This produces wrong PIR shapes for any model with batch > 1 or sequence length > 1.

**Definition of Done**: Shapes derived from MIR graph by walking node dimensions. Fallback to vec![1,1] only with explicit warning. Test verifies correct shapes for known models.

---

### T-111 · Fix Interleave Validation When Channels Unknown

- **ISSUES ref**: I-86
- **AUDIT ref**: V-020 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Interleave constraints are skipped entirely when channels is None, including non-channel-dependent checks (const→1, int4→8). This means Int4/UInt4 dtypes pass validation without the required interleave==8 check.

**Definition of Done**: Non-channel-dependent interleave checks (const→1, int4→8) enforced even when channels is None. Test verifies enforcement.

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

### T-114 · Fix PIR Tensor Spec Dtype Hardcoding

- **ISSUES ref**: I-89
- **AUDIT ref**: V-038 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: PIR tensor specs are hardcoded to dtype "fp16" ignoring actual task spec dtype. Wrong for fp32/int4/int8 tasks.

**Definition of Done**: PIR tensor spec dtype derived from task spec. Test verifies correct dtype propagation.

---

### T-115 · Make Opset Version and Deployment Target Configurable

- **ISSUES ref**: I-90
- **AUDIT ref**: V-037, V-046 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: `DEFAULT_OPSET_VERSION = "iOS18"` and `minimum_deployment_target` are hardcoded. Models will fail on older iOS at load time. Wrong for A11/A12 (iOS 16-era hardware).

**Definition of Done**: Both values configurable from CLI or task spec. Defaults remain iOS18 for backward compatibility. Test verifies CLI override.

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

### T-123 · Add Stencil Constraints

- **ISSUES ref**: I-98
- **AUDIT ref**: V-137 (ane-violations.md §III)
- **Severity**: MEDIUM
- **Effort**: S (0.5 day)

**Intent**: Five stencil (depthwise conv) constraints not enforced: (1) 5D stencil rejected, (2) non-4D kernel rejected, (3) non-sum reduction mode rejected, (4) dilated stencil rejected, (5) strided stencil rejected.

**Definition of Done**: All five stencil constraints validated in op_constraints.rs. Tests for each.

---

### T-93 · Replace MILConcat Emission with ANE-Legal Alternatives (V-098, V-130)

**ISSUES ref**: I-71
**AUDIT ref**: V-098, V-130 (ane-violations.md §III CRITICAL, Orion #1)
**Severity**: CRITICAL
**Effort**: L (3 days)

**Intent**: MILLer emits MILConcat (mb.concat) in three critical paths: (1) SDPA decomposition in `mil_lower.rs:2842-2858`, (2) RoPE rotate_half in `legality_rewrite.rs:3098,3622`, and (3) embedding gather in `mil_emitter.py:432`. However, Orion #1 documents that the concat MIL op is rejected by the ANE compiler. Binary forensic evidence confirms extensive concat constraints: channel-axis-only, const-positive-axis required, no interleaved on some dimensions, no symbolic shape propagation. All models using SDPA decomposition will fail ANE compilation. This is the single most impactful violation because it affects every attention-based model compiled through the Rust pipeline.

**Mitigation/Implementation**:
1. **SDPA path** (`mil_lower.rs:2842-2858`): Replace concat with reshape+stack or emit as fused SDPA for A16+ targets. For pre-A16 targets, decompose Q/K/V projections without concat, using separate reshape+transpose sequences.
2. **RoPE rotate_half** (`legality_rewrite.rs:3098,3622`): Replace `concat(-x2, x1, axis=-1)` with equivalent reshape+transpose sequence. The mathematical operation is: split input into two halves along the last dimension, negate the second half, and interleave. This can be expressed as: reshape → transpose → reshape without concat.
3. **Embedding gather** (`mil_emitter.py:432`): Replace concat with reshape+stack or use SliceByIndex instead.
4. Add tests verifying that no MILConcat nodes survive in the MIR graph after legality rewriting for any standard model topology (attention, RoPE, embedding).

**Definition of Done**:
- [ ] No MILConcat emission in SDPA decomposition path
- [ ] No MILConcat emission in RoPE rotate_half path
- [ ] No MILConcat emission in embedding gather path
- [ ] New test `test_no_concat_in_ane_path()` passes for attention, RoPE, and embedding topologies
- [ ] Existing `cargo test` passes with zero failures
- [ ] Python bridge emission also uses concat-free alternatives

---

### T-94 · Fix Gelu EXACT Mode — ANEC Only Supports Tanh Approximation (V-113, V-099)

**ISSUES ref**: I-72
**AUDIT ref**: V-113, V-099 (ane-violations.md §III CRITICAL, Orion #10)
**Severity**: CRITICAL
**Effort**: M (1.5 days)

**Intent**: The SIR builder hardcodes Gelu mode="EXACT" in `sir_build.rs:518,1415`, but ANEC's ConvertElementwiseUnary(Gelu) only supports tanh approximation. The SIR→AIR→MIR pipeline preserves whatever mode the SIR builder sets, so models compiled through the Rust pipeline emit mb.gelu(mode="EXACT") which is not supported by ANEC. Meanwhile, the Python emitter and role_mir.rs use "TANH_APPROXIMATION". The Rust and Python paths produce incompatible gelu modes. Models using gelu through the Rust path will fail ANE compilation.

**Mitigation/Implementation**:
1. In `sir_build.rs:518,1415`, change `"EXACT"` to `"TANH_APPROXIMATION"` for all gelu emission.
2. In `staticize.rs:527` test fixture, change `"exact"` to `"tanh_approximation"` for consistency.
3. Add a validation in `mir_to_compat.rs` that rejects gelu with mode != "TANH_APPROXIMATION" for ANE targets, with a clear error message.
4. Add a test that verifies all gelu nodes in a compiled MIR graph have mode="TANH_APPROXIMATION".
5. Document in SIR reference that ANEC only supports tanh-approximation gelu.

**Definition of Done**:
- [ ] `sir_build.rs` uses "TANH_APPROXIMATION" for all gelu emission
- [ ] `staticize.rs` test fixtures use "tanh_approximation"
- [ ] Validation rejects non-TANH_APPROXIMATION gelu for ANE targets
- [ ] Test verifies gelu mode consistency through SIR→AIR→MIR pipeline
- [ ] `cargo test` passes with zero failures

---

### T-95 · Replace Zero-Fill Weight Placeholders with Hard Errors (V-007)

**ISSUES ref**: I-70
**AUDIT ref**: V-007 (ane-violations.md §III CRITICAL)
**Severity**: CRITICAL
**Effort**: M (1 day)

**Intent**: When a weight cannot be resolved, MILLer silently produces zero-filled placeholder data. The model compiles and loads successfully but produces completely incorrect inference results. The only indication is a `log::warn!()` message that users routinely miss. This is the most dangerous class of failure — silently wrong results with no error signal. The T-79 fix only added a warning when ALL resolution strategies fail, but partial failures (some weights resolved, others zero-filled) still pass silently.

**Mitigation/Implementation**:
1. In `safetensors_resolver.rs` and `mir_to_compat.rs`, change the zero-fill fallback to return `Err(...)` instead of silently producing zero-filled weights.
2. Add a `--allow-missing-weights` CLI flag (default: false) that restores the old behavior for intentional zero-fill scenarios (testing, development).
3. When the flag is not set and any weight resolves to zero-fill, fail compilation with a clear error listing which weights could not be resolved.
4. Add test verifying that compilation with missing weights fails by default but succeeds with `--allow-missing-weights`.

**Definition of Done**:
- [ ] Compilation fails with hard error when any weight resolves to zero-fill
- [ ] `--allow-missing-weights` flag restores old behavior when explicitly set
- [ ] Error message lists all unresolvable weight names
- [ ] Test verifies default-fail and flag-allow paths
- [ ] `cargo test` passes with zero failures

---

## 🟠 HIGH — Missing Enforcement / Model Leakage / Untested Paths

### T-96 · Fix Knowledge Seed Contradictions and Align with Binary Evidence (V-003, V-004, V-005, V-029, V-030, V-072)

**ISSUES ref**: I-67, I-68, I-69, I-90, I-130
**AUDIT ref**: V-003, V-004, V-005, V-029, V-030, V-072
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: Multiple knowledge seeds contain contradictions with binary forensic evidence. Softmax and InstanceNorm are listed as "supported for all families" but binary shows architecture-conditional rejection. SDPA is marked "unreliable" for A12-A15 but binary shows ConvertScaledDotProductAttention is family-agnostic with MinimumFamily=A11Legacy. These contradictions cause the constraint model to either over-permit or over-restrict op placement.

**Mitigation/Implementation**:
1. In `ane_op_family_matrix.json`, add `architecture_conditional: true` flag for Softmax and InstanceNorm entries. Add notes about specific architecture rejections from binary evidence.
2. For SDPA, change "unreliable" to "supported" for A13+ with `architecture_conditional: true` and add note about family-agnostic converter with MinimumFamily=A11Legacy.
3. Add a `device_class_to_family` mapping document in `knowledge/` that translates device class names (M2, M3) to AneFamily values.
4. Add test validating that all op entries in ane_op_family_matrix have consistent support flags across related families.

**Definition of Done**:
- [ ] Softmax and InstanceNorm entries have architecture_conditional flag
- [ ] SDPA support scope updated to reflect binary evidence
- [ ] device_class_to_family mapping document exists
- [ ] Cross-consistency test for ane_op_family_matrix passes
- [ ] `cargo test` passes with zero failures

---

### T-97 · Implement Conv/Pool Constraint Validation (V-009, V-132, V-115, V-103)

**ISSUES ref**: I-73, I-91, I-95, I-101
**AUDIT ref**: V-009, V-103, V-115, V-132
**Severity**: HIGH
**Effort**: L (3 days)

**Intent**: MILLer defines 7 conv/pool/PE-specific constraint fields in AneHwLimits but never validates them. Conv kernel dimensions must be power of 2 (ANEC rejects 3,5,6,7). Conv has a 32K-channel limit (Orion #16) but max_tensor_channels allows 65536. Large kernel mode has 12+ constraints not enforced (W/H multiple of 8, stride 1-2, zero padding only, no depth>1, no palettized weights, matching strides, no grouped conv, no dynamic shape, no dilation). These missing validations allow ops to pass placement but fail at ANE emission time.

**Mitigation/Implementation**:
1. In `validate_tensor_dims()`, wire the 7 unused constraint fields (pe_max_pooling_kh, pe_max_pooling_kw, etc.) into actual validation logic.
2. In `op_constraints.rs`, add conv kernel power-of-2 validation: reject kernel sizes that are not powers of 2. Add `is_power_of_two()` helper.
3. Add conv-specific 32K-channel limit in `op_constraints.rs`, distinct from general max_tensor_channels.
4. Add large kernel mode detection (kernel W/H > threshold) and validate: W/H multiple of 8, stride 1-2 only, zero padding only, no depth>1, no palettized weights.
5. Add tests for each new constraint with valid and invalid cases.

**Definition of Done**:
- [ ] All 7 AneHwLimits constraint fields validated in validate_tensor_dims()
- [ ] Conv kernel power-of-2 validation rejects sizes 3,5,6,7
- [ ] Conv 32K-channel limit enforced
- [ ] Large kernel mode constraints enforced (at least 6 of 12+)
- [ ] Tests for each new validation path with valid + invalid cases
- [ ] `cargo test` passes with zero failures

---

### T-98 · Implement Deconvolution Constraint Validation (V-116, V-048)

**ISSUES ref**: I-96
**AUDIT ref**: V-116, V-048
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: ConvTranspose (deconvolution) currently passes placement validation unconditionally — no kernel size, stride, or group checks. Binary forensic evidence shows deconv has extensive constraints: SOx must equal 2, no large kernel, no vector palettization, no dilation, stride>2 does not support kernel depth>1. ANEC will reject deconvolutions violating these constraints.

**Mitigation/Implementation**:
1. In `placement_validate.rs`, add deconv-specific validation for ConvTranspose ops.
2. Add constraint checks: (a) SOx==2 validation, (b) reject large kernel, (c) reject vector palettization, (d) reject dilation, (e) reject kernel depth>1 when stride>2.
3. In `op_constraints.rs`, add `validate_deconv_constraints()` function with these checks.
4. Add tests for each deconv constraint with valid and invalid cases.

**Definition of Done**:
- [ ] ConvTranspose no longer passes placement validation unconditionally
- [ ] SOx==2 validation for deconv
- [ ] Large kernel rejection for deconv
- [ ] Vector palettization rejection for deconv
- [ ] Dilation rejection for deconv
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

---

### T-99 · Enforce Orion Surface Ordering and Uniformity Constraints (V-117, V-118, V-119, V-120, V-121)

**ISSUES ref**: I-92, I-97
**AUDIT ref**: V-117, V-118, V-119, V-120, V-121 (Orion #2, #3, #18, #19, #20)
**Severity**: HIGH
**Effort**: L (2 days)

**Intent**: Five Orion constraints for surface handling are completely unenforced: (1) multi-output buffers must have uniform sizes (Orion #2), (2) multi-output surfaces must be alphabetically ordered (Orion #3), (3) multi-input surfaces must be alphabetically ordered (Orion #19), (4) multi-input surfaces must have uniform alloc sizes (Orion #18), (5) ANE reads flat buffer as packed [1,C,1,S] (Orion #20). Violations of #2/#3/#18/#19 cause silent data corruption — correct values written to wrong output/input tensors. Violation of #20 causes silently incorrect inference.

**Mitigation/Implementation**:
1. In `coreml-emit/src/mir_to_proto.rs` or emission pipeline, add alphabetical sorting of multi-output and multi-input surfaces before writing.
2. Add uniform size validation for all output buffers and input surfaces before emission. Fail with clear error if non-uniform.
3. Add buffer layout validation ensuring data follows packed [1,C,1,S] format.
4. Add tests: (a) alphabetical ordering test, (b) uniformity rejection test, (c) layout validation test.

**Definition of Done**:
- [ ] Multi-output surfaces sorted alphabetically before emission
- [ ] Multi-input surfaces sorted alphabetically before emission
- [ ] Non-uniform output buffer sizes cause compilation error
- [ ] Non-uniform input surface sizes cause compilation error
- [ ] Buffer layout validated as packed [1,C,1,S]
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

---

### T-100 · Add BF16/F16 Cross-Type and Architecture-Conditional FP32 Validation (V-125, V-126)

**ISSUES ref**: I-98, I-99
**AUDIT ref**: V-125, V-126
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: ANEC explicitly rejects BF16/F16 cross-type operations (9 constraint strings from binary forensic) but MILLer has no cross-type validation. FP32 computation is rejected on some architectures ("Float32 not supported for architecture") but `is_dtype_ane_legal()` approves FP32 for all families. MILLer will approve operations that ANEC rejects, causing compilation failures at runtime.

**Mitigation/Implementation**:
1. In `dtype_constraints.rs`, add `validate_cross_type_compatibility()` that checks input/output dtype pairs. Reject: BF16+F16, F16+BF16, complex+integer, float+integer, different-integer-type operands.
2. In `is_dtype_ane_legal()`, add architecture-conditional check for FP32: reject on families where binary evidence shows "Float32 not supported for architecture". At minimum, warn when FP32 is approved without architecture verification.
3. Add tests for cross-type rejection and FP32 architecture-conditional behavior.

**Definition of Done**:
- [ ] BF16/F16 cross-type operations rejected at validation time
- [ ] All 9 binary-documented cross-type rejections enforced
- [ ] FP32 architecture-conditional check added to is_dtype_ane_legal()
- [ ] Tests for cross-type and FP32 checks
- [ ] `cargo test` passes with zero failures

---

### T-101 · Fix Model Architecture Leakage — Generic→Qwen3 Fallback (V-012, V-025, V-035)

**ISSUES ref**: I-76, I-107
**AUDIT ref**: V-012, V-025, V-035
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: When model architecture is not recognized, MILLer silently falls back to Qwen3 weight patterns in three places: (1) `common.rs:297-308` Generic model architecture, (2) `mir_to_compat.rs:458-468` default architecture patterns, (3) `ModelArchConfig::default()` still callable producing Qwen3-0.6B defaults. Non-Qwen3/LLaMA models will have silently broken weight resolution, wrong input remapping, and undefined references.

**Mitigation/Implementation**:
1. Remove Generic→Qwen3 fallback in `common.rs`. Return an error when model architecture is not recognized, requiring explicit architecture specification.
2. Remove default architecture fallback in `mir_to_compat.rs`. Fail with clear error listing supported architectures.
3. Remove `Default` impl for `ModelArchConfig` or make it return an error. Add `ModelArchConfig::unspecified()` for placeholder cases with explicit warning.
4. Add test that compilation fails when architecture is unspecified.

**Definition of Done**:
- [ ] Generic model architecture no longer falls back to Qwen3
- [ ] Default architecture in mir_to_compat produces error, not silent fallback
- [ ] ModelArchConfig::default() removed or returns error
- [ ] ModelArchConfig::unspecified() available for explicit placeholder
- [ ] Test verifies compilation fails without architecture specification
- [ ] `cargo test` passes with zero failures

---

### T-102 · Fix Dtype Validation Gaps — E5M2, Int4 Interleave, Asymmetric Quant, Dilated Pool (V-051, V-050, V-134, V-128, V-127)

**ISSUES ref**: I-100, I-117, I-118, I-102, I-148
**AUDIT ref**: V-051, V-111, V-050, V-134, V-128, V-127
**Severity**: HIGH
**Effort**: M (1.5 days)

**Intent**: Multiple dtype-related validation gaps: (1) Quantize validator accepts E5M2 as output dtype but ANEC universally rejects it ("E4M3 or E5M2 format not supported"). (2) Int4/UInt4 interleave==8 constraint is deferred to caller with no enforcement. (3) Asymmetric quantization is not supported on ANEC ("Asym quantization is not supported") but no check prevents it. (4) Dilated pooling is rejected by ANEC but MILLer has no dilation check. (5) Pooling stride 3 only supported for Avg mode but not enforced.

**Mitigation/Implementation**:
1. In `dtype_constraints.rs` quantize validator, reject E5M2 as output dtype. Binary evidence confirms it is universally unsupported.
2. Add interleave==8 validation directly in `is_dtype_ane_legal()` for Int4/UInt4 instead of deferring to caller.
3. Add asymmetric quantization rejection check in `palettize_weights.rs` or `op_constraints.rs` for ANE path.
4. Add dilation check in `validate_pooling_constraints()` — reject dilated pooling.
5. Add stride-3 Avg-only check — reject MaxPool/L2Pool with stride 3.
6. Add tests for each new validation.

**Definition of Done**:
- [ ] E5M2 rejected as quantize output dtype
- [ ] Int4/UInt4 interleave==8 enforced in is_dtype_ane_legal()
- [ ] Asymmetric quantization rejected for ANE path
- [ ] Dilated pooling rejected
- [ ] MaxPool/L2Pool stride-3 rejected
- [ ] Tests for each validation
- [ ] `cargo test` passes with zero failures

---

### T-103 · Replace Stub-Mimic and Phantom Capabilities with Honest Implementations (V-014, V-016, V-080, V-081, V-082)

**ISSUES ref**: I-78, I-79, I-137
**AUDIT ref**: V-014, V-016, V-080, V-081, V-082
**Severity**: HIGH
**Effort**: M (1.5 days)

**Intent**: Multiple functions present themselves as real logic but are stubs: (1) StaticizePass::run() is pure pass-through, documentation claims it replaces symbolic dims. (2) StateTopologyPass only logs warnings, never returns Err despite claiming to "verify" and "ensure". (3) CoreMlApi::version() returns "unknown", compile_model() returns Err after passing is_available(). (4) FfiModel::load() returns Ok with handle:None. (5) coreml_model_info() writes zeroed info with Ok status. These phantom capabilities waste developer trust and mask real gaps.

**Mitigation/Implementation**:
1. For StaticizePass: either implement (resolve symbolic dimensions) or remove from pipeline and add clear documentation that static dimension resolution is not yet implemented.
2. For StateTopologyPass: change return type to Result, return Err for invalid state patterns. Document advisory vs. enforcement mode.
3. For CoreMlApi stubs: document clearly that these are stubs. Return specific error types (NotAvailableOnPlatform) instead of generic errors.
4. For FfiModel::load(): return Err when handle would be None instead of Ok.
5. For coreml_model_info(): return error status instead of Ok with zeroed data.

**Definition of Done**:
- [ ] StaticizePass either implemented or removed from pipeline with documentation
- [ ] StateTopologyPass returns Err for invalid patterns
- [ ] CoreMlApi stubs return specific NotAvailableOnPlatform errors
- [ ] FfiModel::load() returns Err instead of Ok(None)
- [ ] coreml_model_info() returns error instead of zeroed Ok
- [ ] Tests updated to match new behavior
- [ ] `cargo test` passes with zero failures

---

### T-104 · Add I/O Node and Shape Fallback Hard Errors (V-023, V-085, V-062)

**ISSUES ref**: I-86, I-125, I-140
**AUDIT ref**: V-023, V-085, V-062
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: Multiple code paths silently fall back to wrong defaults: (1) Input/output nodes missing from MIR graph default to shape vec![1] and dtype Fp16 — almost certainly wrong. (2) SIR builder missing input shape falls back to (1,32) silently. (3) Shape inference catch-all returns empty shape for unrecognized MirOp variants. These silent fallbacks produce wrong models without any error indication.

**Mitigation/Implementation**:
1. In `mir_to_compat.rs`, replace fallback shapes/dtypes with hard errors when nodes are missing from MIR graph.
2. In `sir_build.rs`, add `log::warn!()` when input shape fallback is used. Consider making shape a required parameter.
3. In `shape_inference.rs`, replace empty shape catch-all with `log::warn!()` identifying the unrecognized variant. At minimum, document which ops lack shape inference.
4. Add tests verifying that missing nodes cause compilation failure.

**Definition of Done**:
- [ ] Missing I/O nodes cause hard errors instead of wrong defaults
- [ ] SIR input shape fallback emits warning
- [ ] Shape inference catch-all emits warning for unrecognized variants
- [ ] Tests verify failure on missing nodes
- [ ] `cargo test` passes with zero failures

---

### T-105 · Implement F32→FP16 Weight Conversion and Fix Dtype Blob Mapping (V-026, V-027)

**ISSUES ref**: I-87, I-88
**AUDIT ref**: V-026, V-027
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: Two weight data format issues: (1) F32 weight data is passed through without FP16 conversion in safetensors_resolver — BF16 gets converted but F32 does not, producing corrupted data when F32 bytes are written but declared as FP16 in proto. (2) Bool, Float64, and Unknown data types are silently mapped to Float32 blob format — data-corrupting for Bool tensors and incorrect for Float64.

**Mitigation/Implementation**:
1. In `safetensors_resolver.rs`, add F32→FP16 conversion when target dtype is FP16. Ensure data format matches proto declaration.
2. In `weights.rs`, map Bool to dedicated Bool blob type or reject at validation time. Map Float64 to 8-byte blob (after T-71 fix). Reject Unknown dtype early with clear error.
3. Add tests for F32→FP16 conversion correctness and Bool/Float64/Unknown dtype handling.

**Definition of Done**:
- [ ] F32 weights converted to FP16 when target dtype is FP16
- [ ] Bool dtype either has dedicated blob format or is rejected
- [ ] Float64 uses 8-byte blob format
- [ ] Unknown dtype rejected with clear error
- [ ] Tests for each conversion path
- [ ] `cargo test` passes with zero failures

---

### T-106 · Fix MILLinear→MILMatMul Performance Regression — Preserve Conv1x1 (V-114)

**ISSUES ref**: I-104
**AUDIT ref**: V-114 (Orion #17)
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: Orion #17 documents that conv 1x1 is 3x faster than matmul on ANE. The pipeline correctly creates Conv1x1AsLinear in AIR (legality_rewrite.rs:354-370), but the ANE legality pass then converts ALL MILLinear to MILMatMul (mil_lower.rs:3268-3307). This defeats the Conv1x1AsLinear optimization, causing a 3x performance regression for all linear projections. Binary shows ConvertLayer has 97 instances and ConvertMatMul has 8 family instantiations, meaning both are available.

**Mitigation/Implementation**:
1. In `mil_lower.rs:3268-3307`, replace MILLinear→MILMatMul conversion with MILLinear→MILConv(1x1) for ANE targets, preserving the Conv1x1AsLinear optimization.
2. Keep MILLinear→MILMatMul as fallback only when Conv1x1 constraints cannot be satisfied.
3. Add performance regression test that verifies linear ops emit Conv1x1 when possible.
4. Document the decision with reference to Orion #17.

**Definition of Done**:
- [ ] MILLinear emits as Conv1x1 for ANE targets instead of MatMul
- [ ] MILLinear→MILMatMul only used when Conv1x1 constraints not satisfiable
- [ ] Performance test verifies Conv1x1 emission for linear ops
- [ ] Documentation references Orion #17
- [ ] `cargo test` passes with zero failures

---

### T-107 · Add Missing ANEC Operation Mappings to MirOpCompat (V-100, V-138, V-110)

**ISSUES ref**: I-153, I-94
**AUDIT ref**: V-100, V-138, V-110
**Severity**: HIGH
**Effort**: L (3 days)

**Intent**: ANEC defines 98+ operations but MILLer's MirOpCompat only models ~30 variants. 70+ ANEC operations including 27 ConvertElementwiseUnary variants (Elu, LeakyRelu, Sqr, Rsqrt, Sign, Ceil, Floor, Exp2, Log2, Trunc, NRelu, Dirac, Degamma, HighPrecisionSigmoid, RoundNearest, Square, ClampedRelu) have no MILLer equivalents. Additionally, ANEC convolution has kernel_scale, kernel_zero_point, kernel_palettized_LUT attributes for quantized/palettized weights that are not modeled. These missing mappings cause ops to fail emission or be incorrectly lowered.

**Mitigation/Implementation**:
1. Add MirOpCompat variants for the 27 ConvertElementwiseUnary operations that have dedicated converters: ClampedRelu, Elu, LeakyRelu, Sqr, Rsqrt, Sign, Ceil, Floor, Exp2, Log2, Trunc, NRelu, Dirac, Degamma, HighPrecisionSigmoid, RoundNearest, Square. Each needs: conversion logic, input_names, remap_inputs, rename_output methods, and tests.
2. Add quantized weight attributes to MILConv: kernel_scale, kernel_zero_point, kernel_palettized_LUT. Add these to MilConvOp proto as well.
3. Add RingBufferReader/Writer and State variants for KV-cache models.
4. Add tests for each new variant.

**Definition of Done**:
- [ ] 27+ new MirOpCompat variants with full conversion paths
- [ ] MILConv has kernel_scale, kernel_zero_point, kernel_palettized_LUT fields
- [ ] Tests for each new variant
- [ ] RingBufferReader/Writer and State variants added
- [ ] `cargo test` passes with zero failures

---

### T-108 · Implement Gather Contradiction Resolution (V-019, V-136)

**ISSUES ref**: I-82
**AUDIT ref**: V-019, V-136
**Severity**: HIGH
**Effort**: M (1 day)

**Intent**: Gather is listed in CPU_ONLY_OPS but mil_lower actively emits MILGather for embedding lookup and legality_rewrite generates Gather for RoPE table lookups. Binary evidence confirms gather with non-constant axis is rejected ("gather with non-constant axis is not supported on ANEs"). This contradiction means: (a) Gather is classified as CPU-only, forcing all gather ops off the ANE, or (b) Gather is emitted for embedding lookups that might have constant axes. The resolution must handle both the classification and the emission.

**Mitigation/Implementation**:
1. Remove Gather from CPU_ONLY_OPS. Instead, add a const-axis check: Gather with constant axis is ANE-legal, dynamic axis is not.
2. In `legality_rewrite.rs`, replace Gather emission for RoPE with SliceByIndex or ensure axis is const.
3. In `mil_lower.rs`, ensure embedding Gather uses const axis (embeddings use constant index tensors).
4. Add `validate_gather_axis_constness()` in op_constraints.rs.
5. Add tests for constant-axis and dynamic-axis Gather.

**Definition of Done**:
- [ ] Gather removed from CPU_ONLY_OPS
- [ ] Const-axis Gather classified as ANE-legal
- [ ] Dynamic-axis Gather rejected at validation time
- [ ] RoPE Gather replaced with const-axis or SliceByIndex
- [ ] Embedding Gather uses const axis
- [ ] Tests for both Gather paths
- [ ] `cargo test` passes with zero failures

---

### T-109 · Implement Stencil Constraints and Vector Palettization at-Cout (V-137, V-133, V-135)

**ISSUES ref**: I-150, I-103, I-151
**AUDIT ref**: V-137, V-133, V-135
**Severity**: HIGH
**Effort**: M (1.5 days)

**Intent**: Multiple ANEC constraint categories not enforced: (1) Stencil (depthwise conv) has 5 constraints: 5D stencil rejected, non-4D kernel rejected, non-sum reduction rejected, dilated stencil rejected, strided stencil rejected. (2) Vector palettization only supported at Cout for ANE. (3) Width wrap axis not supported on some architectures. All violations cause ANEC compilation failures that could be caught earlier.

**Mitigation/Implementation**:
1. Add stencil constraint validation in `op_constraints.rs`: reject 5D stencil, non-4D kernel, non-sum reduction, dilated stencil, strided stencil.
2. Add vector palettization at-Cout constraint in `palettize_weights.rs` or `op_constraints.rs`.
3. Add width wrap axis architecture check in `op_constraints.rs`.
4. Add tests for each constraint.

**Definition of Done**:
- [ ] 5D stencil rejected
- [ ] Non-4D kernel stencil rejected
- [ ] Non-sum reduction stencil rejected
- [ ] Dilated stencil rejected
- [ ] Strided stencil rejected
- [ ] Vector palettization at non-Cout rejected
- [ ] Width wrap axis architecture check added
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

---

## 🟡 MEDIUM — Technical Debt / Drift / Code Quality

### T-110 · Fix Knowledge System Integrity Issues (V-021, V-022, V-053, V-055, V-056, V-071)

**ISSUES ref**: I-84, I-85, I-120, I-121, I-122
**AUDIT ref**: V-021, V-022, V-053, V-055, V-056, V-071
**Severity**: MEDIUM
**Effort**: M (1.5 days)

**Intent**: Multiple knowledge system integrity issues: (1) claims_agree defaults to true for 7/8 knowledge types — no contradiction detection for PrecisionHazard, SurvivalMatrixEntry, etc. (2) Conflict detection marks new entry as ConflictedWith(existing) but never back-patches existing entry. (3) Documentation says "never start above 0.5" but CompileFailure starts at 0.7. (4) Doc says pattern-level transfer scales "0.5–0.8" but code uses hardcoded 0.65. (5) ComputePlanObservation doc says "confidence always 0.9" but code accepts any value. (6) Seed JSONs don't follow the knowledge_schema.md format — missing unit wrapper, provenance, conflict_status fields.

**Mitigation/Implementation**:
1. Implement claims_agree for all 8 knowledge types with field-level comparison.
2. Make conflict detection symmetric — back-patch existing entries when new entry conflicts.
3. Fix documentation to match actual code behavior (confidence values, transfer scaling).
4. Either align seed JSONs with schema or update schema to match actual format.
5. Add tests for knowledge consistency and conflict detection.

**Definition of Done**:
- [ ] claims_agree implemented for all 8 knowledge types
- [ ] Conflict detection is symmetric
- [ ] Documentation matches code behavior
- [ ] Seed JSONs either follow schema or schema updated
- [ ] Tests for knowledge consistency
- [ ] `cargo test` passes with zero failures

---

### T-111 · Fix Hardcoded Constants and Phantom Capabilities (V-031, V-034, V-037, V-046, V-087)

**ISSUES ref**: I-105, I-106, I-109, I-141
**AUDIT ref**: V-031, V-034, V-037, V-046, V-087
**Severity**: MEDIUM
**Effort**: M (1 day)

**Intent**: Multiple hardcoded constants and phantom capabilities: (1) V26 limits are fabricated (inherits A18 + num_nes=16) with no warning. (2) KvCacheLayout::Paged is constructible but unimplemented. (3) DEFAULT_OPSET_VERSION and minimum_deployment_target are hardcoded to "iOS18". (4) --seed CLI parameter is accepted but discarded.

**Mitigation/Implementation**:
1. Add explicit "speculative — not based on any hardware" warning in for_revision(V26) return.
2. Gate KvCacheLayout::Paged behind feature flag or add serde validation rejecting it.
3. Make opset version and deployment target configurable from CLI or task spec.
4. Either wire --seed through compile pipeline or remove from CLI.
5. Add tests for each fix.

**Definition of Done**:
- [ ] V26 returns speculative warning
- [ ] KvCacheLayout::Paged gated behind feature flag or rejected on deserialization
- [ ] Opset version and deployment target configurable
- [ ] --seed either wired or removed
- [ ] Tests for each change
- [ ] `cargo test` passes with zero failures

---

### T-112 · Fix PIR/Shard Plan Hardcoded Values (V-017, V-038, V-045, V-047)

**ISSUES ref**: I-80, I-110, I-114, I-115
**AUDIT ref**: V-017, V-038, V-045, V-047
**Severity**: MEDIUM
**Effort**: M (1 day)

**Intent**: Multiple hardcoded values in shard plan and PIR: (1) FunctionEntry TensorSpec shapes hardcoded as vec![1,1]. (2) PIR tensor specs hardcoded to dtype "fp16". (3) PIR context_length always 0. (4) KV cache default shape fallback vec![2,1,1,1,1] is arbitrary.

**Mitigation/Implementation**:
1. Derive FunctionEntry shapes from MIR graph instead of hardcoding. Walk the graph to extract actual batch/seq dimensions.
2. Use actual dtype from task spec for PIR tensor specs.
3. Derive context_length from graph or task spec.
4. Add configuration for KV cache default shape.
5. Add tests verifying derived values match expected dimensions.

**Definition of Done**:
- [ ] FunctionEntry shapes derived from MIR graph
- [ ] PIR tensor specs use actual dtype from task spec
- [ ] context_length derived from graph or task spec
- [ ] KV cache default shape configurable
- [ ] Tests for derived values
- [ ] `cargo test` passes with zero failures

---

### T-113 · Add Orion Runtime Constraint Validations (V-104, V-106, V-124)

**ISSUES ref**: I-142, I-143, I-147
**AUDIT ref**: V-104, V-106, V-124 (Orion #4, #5, #11)
**Severity**: MEDIUM
**Effort**: M (1 day)

**Intent**: Three Orion runtime constraints not enforced: (1) Minimum IOSurface size (~49 KB) for eval — smaller buffers cause 0x1d runtime error. (2) ~119 compilation limit per process — exceeding causes silent crash. (3) Weight dict must be @{} not nil — nil weight dict crashes ANEC at compile time.

**Mitigation/Implementation**:
1. Add minimum IOSurface size validation (~49 KB) in emission pipeline.
2. Add compilation count tracking per process. Warn at ~100, error at ~119. Provide process restart advisory.
3. Add weight dict initialization check ensuring dict is properly initialized (not nil/empty when weights exist).
4. Add tests for each constraint.

**Definition of Done**:
- [ ] Minimum IOSurface size validated
- [ ] Compilation count tracked with warnings
- [ ] Weight dict initialization guaranteed
- [ ] Tests for each constraint
- [ ] `cargo test` passes with zero failures

---

### T-114 · Fix Lab Module Stub Functions and Hardcoded Values (V-076, V-077, V-079, V-083)

**ISSUES ref**: I-133, I-134, I-136, I-138
**AUDIT ref**: V-076, V-077, V-079, V-083
**Severity**: MEDIUM
**Effort**: S (0.5 day)

**Intent**: Lab module issues: (1) device_backed() always returns HostOnly even on macOS. (2) coremltools_available hardcoded to false. (3) LabRunBuilder allows timing on HostOnlyInspection despite doc saying MUST be None. (4) FFI validation rejects packages without weight.bin but some models don't require it.

**Mitigation/Implementation**:
1. Document that device_backed() returns HostOnly on all current platforms. Add TODO for macOS implementation.
2. Add Python bridge detection logic to set coremltools_available dynamically.
3. Add runtime enforcement: reject timing on HostOnlyInspection runs.
4. Make weight.bin optional in validation — only require when model declares external weights.

**Definition of Done**:
- [ ] device_backed() documented as stub
- [ ] coremltools_available dynamically detected
- [ ] Timing rejected on HostOnlyInspection runs
- [ ] weight.bin optional in validation
- [ ] Tests for each fix
- [ ] `cargo test` passes with zero failures

---

### T-115 · Extend AneFamily and AneRevision Enums (V-109, V-108, V-131)

**ISSUES ref**: I-144, I-145, I-149
**AUDIT ref**: V-108, V-109, V-131
**Severity**: MEDIUM
**Effort**: M (1 day)

**Intent**: Binary evidence shows: (1) AneRevision only defines 11 revisions but binary has 14 hardware versions with dedicated code paths. (2) AneFamily only models 6 families but binary MinimumFamily enum has 8 values (0-7). (3) Hardware sub-variants (NE/PE/DMA engines) are not modeled. MILLer will misattribute ops on future hardware and cannot express engine-specific constraints.

**Mitigation/Implementation**:
1. Add missing AneRevision variants for pre-A11 hardware (V0-V3) with minimal capabilities.
2. Extend AneFamily enum to cover all 8 binary-defined families. Add AneFamily::A17Pro and AneFamily::A18Pro (or equivalent) for families 6-7.
3. Document hardware sub-variant structure (NE/PE/DMA) in comments. Future work: add sub-variant constraints.
4. Add tests for new family and revision variants.

**Definition of Done**:
- [ ] Pre-A11 AneRevision variants added
- [ ] AneFamily extended to 8 values matching binary
- [ ] Sub-variant structure documented
- [ ] Tests for new variants
- [ ] `cargo test` passes with zero failures

---

## 🔵 LOW — Minor Quality / Style / Documentation

### T-116 · Fix Remaining Doc/Code Mismatches and Minor Quality Issues (V-059, V-089, V-090, V-091, V-092, V-093, V-094, V-095, V-097, V-129)

**ISSUES ref**: I-123, I-154, I-155, I-156, I-157, I-158, I-159, I-152, I-160
**AUDIT ref**: V-059, V-089, V-090, V-091, V-092, V-093, V-094, V-095, V-097, V-129
**Severity**: LOW
**Effort**: M (1 day)

**Intent**: Collection of minor quality issues: EmptyWeightResolver doc says "returns Some" but returns None. Hardcoded fallback dimensions. Canonicalization cycle limit without diagnostic. UInt16/Bool "limited support" undocumented. CPU_ONLY_OPS_DETAILED incomplete. Knowledge consistency all-or-nothing. UUID not globally unique. LayerNorm epsilon FP16 truncation. MatMul transpose flags as immediate bools.

**Mitigation/Implementation**:
1. Fix EmptyWeightResolver doc to match implementation.
2. Add diagnostic when canonicalization cycle limit is hit.
3. Add constraint documentation for UInt16 and Bool "limited support".
4. Expand CPU_ONLY_OPS_DETAILED with reason codes.
5. Replace binary knowledge_consistent with graded score.
6. Use model-specific salt in UUID generation.
7. Document LayerNorm epsilon FP16 truncation or compute in FP32.
8. Add TODO for MatMul transpose flags as named const nodes.

**Definition of Done**:
- [ ] EmptyWeightResolver doc matches implementation
- [ ] Canonicalization cycle limit diagnostic added
- [ ] UInt16/Bool limited support documented
- [ ] CPU_ONLY_OPS_DETAILED expanded
- [ ] Knowledge consistency uses graded score
- [ ] UUID uses model-specific salt
- [ ] LayerNorm epsilon documented
- [ ] MatMul transpose flags TODO documented
- [ ] `cargo test` passes with zero failures
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
| T-91 | I-66 | CRITICAL | S | 🔴 Open |
| T-92 | I-67, I-68, I-69 | CRITICAL | M | 🔴 Open |
| T-93 | I-71 | CRITICAL | L | 🔴 Open |
| T-94 | I-72 | CRITICAL | M | 🔴 Open |
| T-95 | I-70 | CRITICAL | M | 🔴 Open |
| T-96 | I-67, I-68, I-69, I-90, I-130 | HIGH | M | 🟠 Open |
| T-97 | I-73, I-91, I-95, I-101 | HIGH | L | 🟠 Open |
| T-98 | I-96 | HIGH | M | 🟠 Open |
| T-99 | I-92, I-97 | HIGH | L | 🟠 Open |
| T-100 | I-98, I-99 | HIGH | M | 🟠 Open |
| T-101 | I-76, I-107 | HIGH | M | 🟠 Open |
| T-102 | I-100, I-117, I-118, I-102, I-148 | HIGH | M | 🟠 Open |
| T-103 | I-78, I-79, I-137 | HIGH | M | 🟠 Open |
| T-104 | I-86, I-125, I-140 | HIGH | M | 🟠 Open |
| T-105 | I-87, I-88 | HIGH | M | 🟠 Open |
| T-106 | I-104 | HIGH | M | 🟠 Open |
| T-107 | I-153, I-94 | HIGH | L | 🟠 Open |
| T-108 | I-82 | HIGH | M | 🟠 Open |
| T-109 | I-150, I-103, I-151 | HIGH | M | 🟠 Open |
| T-110 | I-84, I-85, I-120, I-121, I-122 | MEDIUM | M | 🟡 Open |
| T-111 | I-105, I-106, I-109, I-141 | MEDIUM | M | 🟡 Open |
| T-112 | I-80, I-110, I-114, I-115 | MEDIUM | M | 🟡 Open |
| T-113 | I-142, I-143, I-147 | MEDIUM | M | 🟡 Open |
| T-114 | I-133, I-134, I-136, I-138 | MEDIUM | S | 🟡 Open |
| T-115 | I-144, I-145, I-149 | MEDIUM | M | 🟡 Open |
| T-116 | I-123, I-154, I-155, I-156, I-157, I-158, I-159, I-152, I-160 | LOW | M | 🔵 Open |
| T-86 | I-61 | CRITICAL | S | ✅ |
| T-87 | I-62 | CRITICAL | M | ✅ |
| T-89 | I-64 | CRITICAL | S | ✅ |
| T-90 | I-65 | CRITICAL | L | ⬜ |
| T-88 | I-63 | HIGH | S | ✅ |
| T-91 | I-66 | HIGH | M | ✅ |
| T-92 | I-67 | HIGH | M | ⬜ |
| T-93 | I-68 | HIGH | M | ⬜ |
| T-94 | I-69 | HIGH | S | ⬜ |
| T-95 | I-70 | HIGH | M | ⬜ |
| T-96 | I-71 | HIGH | M | ⬜ |
| T-97 | I-72 | HIGH | M | ⬜ |
| T-98 | I-73 | HIGH | L | ⬜ |
| T-99 | I-74 | HIGH | S | ⬜ |
| T-100 | I-75 | HIGH | S | ⬜ |
| T-101 | I-76 | HIGH | M | ⬜ |
| T-102 | I-77 | HIGH | S | ⬜ |
| T-61 | I-35 | MEDIUM | M | ⬜ |
| T-66 | I-40 | MEDIUM | M | ⬜ |
| T-103 | I-78 | MEDIUM | S | ⬜ |
| T-104 | I-79 | MEDIUM | S | ⬜ |
| T-105 | I-80 | MEDIUM | M | ⬜ |
| T-106 | I-81 | MEDIUM | S | ⬜ |
| T-107 | I-82 | MEDIUM | M | ⬜ |
| T-108 | I-83 | MEDIUM | L | ⬜ |
| T-109 | I-84 | MEDIUM | S | ⬜ |
| T-110 | I-85 | MEDIUM | M | ⬜ |
| T-111 | I-86 | MEDIUM | S | ⬜ |
| T-112 | I-87 | MEDIUM | M | ⬜ |
| T-113 | I-88 | MEDIUM | M | ⬜ |
| T-114 | I-89 | MEDIUM | S | ⬜ |
| T-115 | I-90 | MEDIUM | S | ⬜ |
| T-116 | I-91 | MEDIUM | M | ⬜ |
| T-117 | I-92 | MEDIUM | M | ⬜ |
| T-118 | I-93 | MEDIUM | S | ⬜ |
| T-119 | I-94 | MEDIUM | S | ⬜ |
| T-120 | I-95 | MEDIUM | S | ⬜ |
| T-121 | I-96 | MEDIUM | S | ⬜ |
| T-122 | I-97 | MEDIUM | S | ⬜ |
| T-123 | I-98 | MEDIUM | S | ⬜ |
| T-124 | — | LOW | S | ⬜ |
| T-125 | — | LOW | S | ⬜ |
| T-126 | — | LOW | S | ⬜ |
| T-127 | — | LOW | S | ⬜ |
| T-128 | — | LOW | S | ⬜ |
| T-129 | — | LOW | M | ⬜ |
| T-130 | — | LOW | S | ⬜ |
| T-131 | — | LOW | S | ⬜ |

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
| 🔴 CRITICAL | 5 | ~7 days |
| 🟠 HIGH | 14 | ~19.5 days |
| 🟡 MEDIUM | 6 | ~6.5 days |
| 🔵 LOW | 1 | ~1 day |
| **Total** | **26** | **~34 days** |

> **Priority guidance**: CRITICAL tasks (T-91–T-95) must be resolved before any production compilation.
> HIGH tasks (T-96–T-109) should be addressed in the next 2–3 sprint cycles.
> MEDIUM/LOW tasks (T-110–T-116) are technical debt that should be chipped away at consistently.
Tasks T-01 through T-46 all resolved. See CHANGELOG.md for details.
