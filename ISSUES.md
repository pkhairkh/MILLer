# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-03 (TABULA RASA full audit refresh)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `AUDIT.md` (generated 2026-05-03)*

---

## P0 — CRITICAL (Silent Emission Failures / Wrong Constraints)

### I-01 · Three Sources of Truth Diverged (Engine / CPU-Only / Compat)

**Status:** ✅ FIXED (T-22)
**Files:** `crates/ir/src/mir.rs`, `crates/passes/src/cpu_only_ops.rs`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-A, §IV (B-1)
**Severity:** CRITICAL → RESOLVED
**Resolution:** Performed full three-way alignment of `MirOp::default_engine()`, `CPU_ONLY_OPS`, and `MirOpCompat` coverage. Moved 28 MirOp variants from PE/NE to None (CPU-only):
- Trig inverse/hyperbolic: Acos, Asin, Atan, Atanh, Tan, Cosh, Sinh
- Logical: LogicalAnd, LogicalOr, LogicalXor, LogicalNot
- Activation variants: Relu6, SigmoidHard, ThresholdedRelu, ClampedRelu, LinearActivation, Prelu, Softsign, ScaledTanh, Softplus, SoftplusParametric
- Other elementwise: Threshold, Inverse, Mod, Clip
- Miscellaneous: BandPart, ReverseSequence, Einsum

Added 10 entries to CPU_ONLY_OPS: relu6, sigmoid_hard, thresholded_relu, clamped_relu, linear_activation, scaled_tanh, softplus_parametric, threshold, inverse, einsum. Added `MirOp::mil_op_name()` method for cross-referencing.

ANE-legal ops that still lack MirOpCompat variants (BatchNorm, MaxPool, AvgPool, Quantize, Dequantize, all resize ops, etc.) remain in PE/NE — they have real ANEC converters per the per-op support matrix and need MirOpCompat variants (tracked by T-38/T-39).

---

### I-02 · CPU-Only List Not Checked by Placement Validator

**Status:** ✅ FIXED (T-22/T-23)
**Files:** `crates/passes/src/placement_validate.rs`, `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-B, §IV (B-2d)
**Severity:** CRITICAL → RESOLVED
**Resolution:** Added `cpu_only_ops::is_cpu_only(op.mil_op_name())` as a hard gate in `validate_placement()` before the `default_engine().is_none()` check. Also reconciled `default_engine()` with `CPU_ONLY_OPS` by moving all CPU-only ops from PE/NE to None (see I-01 resolution). The validator now provides defense-in-depth against future drift.

---

### I-03 · V6 (A13 Silicon) Mapped to A14 Family

**Status:** ✅ FIXED (T-24)
**Files:** `crates/ir/src/ane_target.rs`, `crates/trace/src/versioned.rs`, `crates/ir/src/strategy.rs`, `crates/cli/src/main.rs`, `crates/passes/src/dtype_constraints.rs`, `crates/ir/src/ane_hw_limits.rs`
**AUDIT ref:** §III (CQ-13), §IV (B-3)
**Severity:** HIGH → RESOLVED
**Resolution:** Added `AneFamily::A13` variant with distinct constraint profile. A13 has full-dtype broadcast (unlike A12's FP16-only) but retains A14Minus elementwise/reduction converters and FP-only ReduceMin (unlike A14's A14Plus). Mapped V6→A13 instead of V6→A14. Added `uses_a14minus_converters()` and `supports_reducemin_all_dtypes()` helper methods. Updated ReduceMin gate in versioned.rs to use `supports_reducemin_all_dtypes()` (catches A11/A12/A13). Updated all exhaustive match arms, family levels, CLI parsing, strategy KvCache benefit. Fixed chip comments and Mac-to-family mapping.

---

## P1 — HIGH (Missing Enforcement / Runtime Risks)

### I-04 · Interleave + Dtype Validators Dead Code (Not Wired)

**Status:** ✅ FIXED (T-25)
**Files:** `crates/ir/src/ane_layout.rs`, `crates/passes/src/dtype_constraints.rs`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-12)
**Severity:** HIGH → RESOLVED
**Resolution:** Wired all six constraint validators into `placement_validate.rs` via new `PlacementContext` struct and `validate_placement_with_context()` function. `is_dtype_ane_legal()` runs as a universal dtype gate for all ops (rejects Int32/Fp64). `is_broadcast_dtype_legal()` enforces FP16-only broadcast on A11/A12 for binary elementwise ops. `validate_interleave_constraints()` enforces valid interleave factors {1,2,3,4,8}, const→interleave-1, int4→interleave-8, and channel-divisibility. `validate_channellast_constraints()` enforces ChannelLast only for depthwise convolutions with interleave=1. `is_blockwise_scale_supported()` hard-rejects `ConstexprBlockwiseShiftScale` and `ConstexprSparseBlockwiseShiftScale`. `is_asymmetric_quantization_supported()` hard-rejects asymmetric quantization. All validators are opt-in via `PlacementContext` fields — backward-compatible `validate_placement()` continues to work with empty context.

---

### I-05 · Missing `validate_matmul_constraints()`

**Status:** ✅ FIXED (T-26)
**Files:** `crates/passes/src/op_constraints.rs`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-9)
**Severity:** HIGH → RESOLVED
**Resolution:** Added `validate_matmul_constraints()` to `op_constraints.rs` enforcing four ANE MatMul hard constraints: (1) depth=1 — both inputs must have rank ≤ 4 (rank-5 forces depth>1 in ANE NCDHW layout); (2) minimum rank 2 — both inputs must be at least 2D matrices; (3) inner dimensions must match — contraction dimension K of input A equals K of input B, handling `transpose_y` correctly; (4) output channels even — M dimension must be even for ANE tiling (cout % ox == 0 prerequisite). Wired into `placement_validate.rs` with dedicated `MILMatMul` match arm that calls the validator with both input shapes and the `transpose_y` flag. Added 27 new tests (14 unit + 13 integration).

---

### I-06 · Missing `validate_pad_constraints()`

**Status:** ✅ FIXED (T-27)
**Files:** `crates/passes/src/op_constraints.rs`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-8)
**Severity:** HIGH → RESOLVED
**Resolution:** Added `validate_pad_constraints()` to `op_constraints.rs` enforcing six ANE Pad hard constraints: replication/symmetric mode rejection, no negative padding, no batch-axis padding (rank ≥ 4), no channel-axis padding (rank-aware: axis 1 for rank ≥ 4, axis 0 for rank < 4), no depth-axis padding (rank-5 only), and pad_amounts length validation. Wired into `placement_validate.rs` with dedicated `MILPad` match arm. Added 25 unit tests + 10 integration tests.

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
| P0 | 3 | 0 | 0 | 3 |
| P1 | 8 | 5 | 0 | 3 |
| P2 | 8 | 8 | 0 | 0 |
| P3 | 1 | 1 | 0 | 0 |
| **Total** | **20** | **14** | **0** | **6** |

*P0 issues I-01, I-02, and I-03 all resolved. P1 issues I-04, I-05, and I-06 resolved by T-25, T-26, and T-27.*
