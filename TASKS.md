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
