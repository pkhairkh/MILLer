# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-03 (TABULA RASA full audit refresh)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `AUDIT.md` (generated 2026-05-03)*

---

## P0 — CRITICAL (Silent Emission Failures / Wrong Constraints)

### I-01 · Three Sources of Truth Diverged (Engine / CPU-Only / Compat)

**Status:** ⬜ Open
**Files:** `crates/ir/src/mir.rs`, `crates/passes/src/cpu_only_ops.rs`, `crates/coreml-proto/src/lib.rs`
**AUDIT ref:** §II-A, §IV (B-1)
**Severity:** CRITICAL — 55 ops assigned ANE engines but have no compat converter; will silently fail at emission
**Effort:** L (3-5 days)
**Task:** T-22

The three sources of truth for op legality have diverged:
1. `MirOp::default_engine()` assigns ANE engines (NE/PE/TE) to ops
2. `CPU_ONLY_OPS` set declares which ops are CPU-exiled
3. `MirOpCompat` enum determines what can actually be emitted

55 ops have ANE engine assignments but map to `MirOpCompat::Unsupported`, meaning they pass placement validation as ANE-legal but will fail silently at emission time. Critical gaps include: ConvTranspose, all pooling ops, BatchNorm, InstanceNorm, Quantize/Dequantize, all resize/upsample ops, and numerous elementwise/reduction variants.

**Fix:** Audit every MirOp variant. For each, ensure one of:
- (a) ANE-legal: has compat converter + correct engine assignment
- (b) CPU-only: `default_engine() → None` + entry in `CPU_ONLY_OPS`
- (c) Transitional: compat converter exists but is incomplete → add compat variant

---

### I-02 · CPU-Only List Not Checked by Placement Validator

**Status:** ⬜ Open
**Files:** `crates/passes/src/placement_validate.rs`, `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-B, §IV (B-2d)
**Severity:** CRITICAL — CPU-only ops can be routed to ANE
**Effort:** S (0.5 day)
**Task:** T-23

`validate_placement()` uses only `op.default_engine().is_none()` to determine CPU-only status. The `cpu_only_ops::is_cpu_only()` function is never consulted. This means ops like `band_part`, `logical_and/or/not` (which are on the CPU-only list but have `default_engine() → Some(PE)`) pass ANE placement validation.

Additionally, `default_engine()` and `CPU_ONLY_OPS` are independently maintained and have diverged for at least 4 ops.

**Fix:** Add `cpu_only_ops::is_cpu_only()` check in `validate_placement()`. Also reconcile `default_engine()` with `CPU_ONLY_OPS`.

---

### I-03 · V6 (A12 Silicon) Mapped to A14 Family

**Status:** ⬜ Open
**Files:** `crates/ir/src/ane_target.rs:68`
**AUDIT ref:** §III (CQ-13), §IV (B-3)
**Severity:** HIGH — A12 hardware gets wrong broadcast/SDPA/LayerNorm gates
**Effort:** M (1 day)
**Task:** T-24

`AneRevision::V6` (Apple A12 Bionic) maps to `AneFamily::A14` in `ane_target.rs`. This causes A12 silicon to get A14 family constraints:
- Non-FP16 broadcast allowed (should be FP16-only)
- SDPA gate uses A14 rules (should be blocked)
- LayerNorm gate uses A14 rules (should be limited)

The `broadcast_fp16_only()` method correctly returns true for `A11Legacy | A12`, but V6 never maps to A12 family, so A12 silicon never hits this code path.

**Fix:** Either add an `A13` family variant for V6-V7, or map V6 to `A12` family (since A12 and A13 share FP16-only broadcast).

---

## P1 — HIGH (Missing Enforcement / Runtime Risks)

### I-04 · Interleave + Dtype Validators Dead Code (Not Wired)

**Status:** ⬜ Open
**Files:** `crates/ir/src/ane_layout.rs`, `crates/passes/src/dtype_constraints.rs`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-12)
**Severity:** HIGH — Implemented constraint checks are never enforced
**Effort:** S (0.5 day)
**Task:** T-25

`validate_interleave_constraints()` and `validate_channellast_constraints()` in `ane_layout.rs` are never called from any non-test code. Similarly, `is_dtype_ane_legal()`, `is_broadcast_dtype_legal()`, `is_blockwise_scale_supported()`, and `is_asymmetric_quantization_supported()` in `dtype_constraints.rs` are never called from `placement_validate.rs` or any hot-path module.

**Fix:** Wire these functions into `placement_validate.rs`.

---

### I-05 · Missing `validate_matmul_constraints()`

**Status:** ⬜ Open
**Files:** `crates/passes/src/op_constraints.rs`
**AUDIT ref:** §II-C, §IV (B-9)
**Severity:** HIGH — Illegal MatMul configurations pass validation
**Effort:** S (0.5 day)
**Task:** T-26

No MatMul-specific constraint validator exists. The ANE requires depth=1 for both inputs (interleaved format), but this is never checked. MatMul is the most performance-critical ANE op.

**Fix:** Add `validate_matmul_constraints()` enforcing depth=1 for both inputs, cout%ox==0, and other MatMul-specific rules from the constraint docs.

---

### I-06 · Missing `validate_pad_constraints()`

**Status:** ⬜ Open
**Files:** `crates/passes/src/op_constraints.rs`
**AUDIT ref:** §II-C, §IV (B-8)
**Severity:** HIGH — Replication/symmetric/negative/batch/channel/depth padding not rejected
**Effort:** M (1 day)
**Task:** T-27

No padding constraint validator exists. The ANE rejects:
- Replication padding
- Symmetric padding
- Negative padding mode
- Batch axis padding
- Channel axis padding
- Depth axis padding (for ANEC padding op)
- Mixed padding modes across axes

**Fix:** Add `validate_pad_constraints()` mirroring constraint document §4.13.

---

### I-07 · Reshape `.unwrap()` Panic in MIR Lowering

**Status:** ⬜ Open
**Files:** `crates/passes/src/mil_lower.rs:220-221`
**AUDIT ref:** §III (CQ-3), §IV (B-5)
**Severity:** HIGH — Compiler panics on edge-case reshape
**Effort:** S (0.5 day)
**Task:** T-28

`resolved.iter().position(|&d| d == 0).unwrap()` and `resolved.iter().rposition(|&d| d == 0).unwrap()` will panic if `resolved` has no zeros. This can happen when shape inference fails to produce a zero-dim placeholder for reshape inference targets.

**Fix:** Return `Result` with proper error type, or use `ok_or(MilLowerError::...)` with `?`.

---

### I-08 · Zero-Dimension Shapes Survive to Core ML Emission

**Status:** ⬜ Open
**Files:** `crates/passes/src/legality_rewrite.rs:464-465,2194,2282`, `crates/bridge/src/mir_to_compat.rs:2591-2599`
**AUDIT ref:** §II-E, §IV (B-6)
**Severity:** HIGH — Produces invalid Core ML models with literal zero dimensions
**Effort:** S (0.5 day)
**Task:** T-29

Tile decomposition and attention reshape use `0` placeholders for dimensions that should be inferred. When shape inference fails (e.g., no `DecompositionContext`), zeros propagate to emitted protobuf. Core ML treats 0 as a literal zero dimension, not "infer from input." A test in `mir_to_compat.rs` even asserts that zeros survive, codifying this bug.

**Fix:** Add a validation pass before emission that rejects any MIR shape containing 0 dims. Replace the test assertion with a check that zero shapes are caught.

---

### I-09 · `% 1 == 0` Always-True Logic Bug

**Status:** ⬜ Open
**Files:** `crates/bridge/src/mir_to_compat.rs:1265,1271`
**AUDIT ref:** §III (CQ-1), §IV (B-10)
**Severity:** HIGH — Likely placeholder code masking a real divisor bug
**Effort:** S (0.5 day)
**Task:** T-30

`remaining % 1 == 0` is always true (modulo 1 is always 0). The corresponding `/ 1` divisions on lines 1266/1272 are identity operations. Line 1287 uses `product_so_far` as the divisor, suggesting lines 1265/1271 should also use `product_so_far`.

**Fix:** Replace `% 1` with `% product_so_far` and remove `/ 1`.

---

### I-10 · SDPA Compat Missing `attention_mask` and `scale`

**Status:** ⬜ Open
**Files:** `crates/coreml-proto/src/lib.rs`
**AUDIT ref:** §III (CQ-23), §IV (B-11)
**Severity:** HIGH — RoPE-attention models cannot emit SDPA with masks via proto-direct
**Effort:** M (1 day)
**Task:** T-31

`MirOp::MILScaledDotProductAttention` carries `attention_mask: Option<MirNodeId>` and `scale: Option<f32>`, but `MirOpCompat::ScaledDotProductAttention` only has `query, key, value`. The proto-direct path cannot emit SDPA with attention masks or custom scale factors.

**Fix:** Add `attention_mask` and `scale` fields to `MirOpCompat::ScaledDotProductAttention` and wire through proto emission.

---

### I-11 · ArgMinMax Missing A18 Guard

**Status:** ⬜ Open
**Files:** `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-D, §IV (B-7)
**Severity:** MEDIUM — ArgMinMax silently fails on A18/M4 hardware
**Effort:** S (0.5 day)
**Task:** T-32

LSE_7 (A18/M4) has no ArgMinMax converter. No family-version guard blocks it. `MILReduceArgmax/Argmin` have `default_engine() → Some(PE)`, so they pass placement validation on any family including A18.

**Fix:** Add A18-specific check for ArgMinMax in `placement_validate.rs`.

---

## P2 — MEDIUM (Technical Debt / Test Gaps)

### I-12 · Zero Tests for `bridge::shape_inference.rs`

**Status:** ⬜ Open
**Files:** `crates/bridge/src/shape_inference.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM — 500+ lines of shape inference with zero test coverage
**Effort:** M (2 days)
**Task:** T-33

Shape bugs here silently produce wrong Core ML models. Covers broadcast, concat, slice-by-index, topk, where, layer-norm, pad, expand_dims, squeeze, and more.

**Fix:** Test `compat_output_shape` for every MirOp variant with concrete shapes.

---

### I-13 · Zero Tests for `passes::staticize.rs`

**Status:** ⬜ Open
**Files:** `crates/passes/src/staticize.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM — No tests at all for a compilation pass
**Effort:** S (0.5 day)
**Task:** T-34

Even the current pass-through implementation needs a smoke test. Any future change introducing real staticization logic will be completely untested.

**Fix:** Add basic smoke test verifying pass-through behavior.

---

### I-14 · `MilDtype` Missing Int4, UInt4, E4M3, E5M2

**Status:** ⬜ Open
**Files:** `crates/ir/src/common.rs:15-24`
**AUDIT ref:** §III (CQ-15)
**Severity:** MEDIUM — Cannot enforce Int4-per-cout-dequant or float8 rules
**Effort:** M (1 day)
**Task:** T-35

The `MilDtype` enum only has: Fp16, Fp32, Int32, UInt8, Bool, Fp64, Int8, Int16. Missing: Int4, UInt4, E4M3, E5M2, UInt16. Without these, the constraint "Int4 Per-Cout Dequant is not supported" and "E4M3 is not supported on this architecture" cannot be enforced.

**Fix:** Add missing dtype variants and enforce constraints in `dtype_constraints.rs`.

---

### I-15 · Model-Specific Constants in Compilation Logic

**Status:** ⬜ Open
**Files:** `crates/passes/src/role_mir.rs:640-642,1103`, `crates/passes/src/legality_rewrite.rs:1077,1876,2325`, `crates/bridge/src/mir_to_compat.rs:510`, `crates/bridge/src/shape_inference.rs:54-58`
**AUDIT ref:** §III (CQ-16, CQ-17, CQ-18, CQ-19)
**Severity:** MEDIUM — Will cause silent miscompilation for non-Qwen3 models
**Effort:** M (2 days)
**Task:** T-36

Hardcoded model-specific values:
- `vocab_size = 32000` (LLaMA-2 default, wrong for Qwen3's 151936)
- `head_dim = 128` fallback (should error rather than silently use wrong value)
- Qwen3 weight name patterns in `build_input_alias_map()`
- `vec![1, 512]` input shape fallback (Qwen3-0.6B assumption)

**Fix:** Read these from TaskSpec/ModelConfig. Error on missing values rather than using wrong defaults.

---

### I-16 · No SIR→AIR Roundtrip Test with Real Shapes

**Status:** ⬜ Open
**Files:** `crates/passes/src/legality_rewrite.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM
**Effort:** M (1 day)
**Task:** T-37

19 unit tests cover individual decompositions but there's no end-to-end test using `DecompositionContext::for_decode_step_full()` with realistic Qwen3-0.6B dimensions.

**Fix:** Add SIR→AIR roundtrip test with full DecompositionContext.

---

### I-17 · Unify MirOp + MirOpCompat via `ToProto` Trait

**Status:** ⬜ Open (continues T-17)
**Files:** `crates/coreml-proto/src/lib.rs`, `crates/bridge/src/mir_to_compat.rs`
**AUDIT ref:** §III (CQ-20, CQ-21)
**Severity:** MEDIUM — 167 vs ~50 variants, ~1150 lines boilerplate
**Effort:** L (3-5 days)
**Task:** T-38

MirOp has 167 variants; MirOpCompat has ~50 + Unsupported. The conversion, rename, remap, and input-name extraction functions each require per-variant match arms totaling ~1150 lines that must be updated in lockstep.

**Fix:** Implement `ToProto` trait on `MirOp` to replace `MirOpCompat`. Use derive macro or visitor pattern for boilerplate.

---

### I-18 · Proto-Direct Path Cannot Emit Palettized Weights

**Status:** ⬜ Open
**Files:** `crates/coreml-proto/src/lib.rs`
**AUDIT ref:** §III (CQ-24)
**Severity:** MEDIUM — Major functional gap vs Python bridge
**Effort:** M (2 days)
**Task:** T-39

`MirOpCompat` has no variants for ConstexprLutToDense, ConstexprBlockwiseShiftScale, ConstexprCast, ConstexprLutToSparse, ConstexprSparseToDense, ConstexprSparseBlockwiseShiftScale. The Python bridge can emit palettized weights; the proto-direct path cannot.

**Fix:** Add Constexpr* variants to MirOpCompat and implement proto emission.

---

### I-19 · V17 (M1) Incorrectly Mapped to A18 Family

**Status:** ⬜ Open
**Files:** `crates/ir/src/ane_target.rs:71-73`
**AUDIT ref:** §III (CQ-14), §IV (B-4)
**Severity:** MEDIUM — M1 gets A18's SDPA/LayerNorm gates (wrong)
**Effort:** S (0.5 day)
**Task:** T-40

V17 is Apple M1, which uses A14-class ANE. The code maps V17 to A18 family, giving M1 hardware A18's SDPA and LayerNorm support (which it doesn't have).

**Fix:** Map V17→A14 family.

---

## P3 — LOW (Code Quality)

### I-20 · Formatting + Clippy Cleanup

**Status:** ⬜ Open
**Files:** All crates
**AUDIT ref:** §III (CQ-9, CQ-10)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-41

49 files need formatting, ~57 clippy warnings are auto-fixable.

**Fix:** `cargo fmt && cargo clippy --fix`

---

## Pre-Audit Issues (Archived)

The following issues from the previous tracker have been resolved and archived to `CHANGELOG.md`:

| Old ID | Description | Status |
|--------|-------------|--------|
| ISSUE-001 | Mask path uses CPU-only ops | ✅ FIXED |
| ISSUE-002 | Fill/FillLike survive to emission | ✅ FIXED |
| ISSUE-003 | Equal/LessEqual on ANE path | ✅ FIXED |
| ISSUE-004 | Qwen3-specific alias map | 🟡 Partially Fixed → subsumed by I-15 |
| ISSUE-005 | output_dim_for_weight parses HF names | 🟡 Partially Fixed → subsumed by I-15 |
| ISSUE-006 | Hardcoded model constants in CLI | ✅ FIXED |
| ISSUE-007 | KV mask CPU-only path | ✅ FIXED |
| ISSUE-008 | Three mask implementations | ✅ FIXED |
| ISSUE-009 | DecompositionContext leaks model config | 🟡 Partially Fixed → subsumed by I-15 |
| ISSUE-010 | JSON alias resolution fragile | ⬜ Open (low priority) |
| ISSUE-011 | Where→Select double-rewrite | ✅ FIXED |
| ISSUE-012 | Shared node dedup post-hoc | ⬜ Open (low priority) |
| ISSUE-013 | for_qwen3_0_6b factory | ✅ FIXED |
| ISSUE-014 | Hardcoded shape in role_mir | ✅ FIXED |
| ISSUE-015 | kv_cache_rewrite dead code | ✅ FIXED (deprecated) |
| ISSUE-016 | RMSNorm fp16 overflow | ✅ FIXED |
| ISSUE-017 | QK norm not implemented | ✅ FIXED |
| ISSUE-018 | No model architecture detection | 🟡 Partially Fixed → subsumed by I-15 |
| ISSUE-019 | bool→fp16 cast on ANE | ✅ FIXED |
| ISSUE-020 | Reverse ring-buffer KV cache | ⬜ Open (medium priority) |
| ISSUE-021 | default_engine() misclassifies ops | ✅ FIXED (partially — I-01 continues this) |
| ISSUE-022 | Scalar constant resolution | ✅ FIXED |

---

## Summary Statistics

| Priority | Total | Open | In Progress | Fixed |
|----------|-------|------|-------------|-------|
| P0 | 3 | 3 | 0 | 0 |
| P1 | 8 | 8 | 0 | 0 |
| P2 | 8 | 8 | 0 | 0 |
| P3 | 1 | 1 | 0 | 0 |
| **Total** | **20** | **20** | **0** | **0** |

*Pre-audit issues: 14 fixed, 4 partially fixed (subsumed), 4 open (archived to CHANGELOG.md)*
