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

**Status:** ✅ FIXED (T-28)
**Files:** `crates/passes/src/mil_lower.rs:220-221`
**AUDIT ref:** §III (CQ-3), §IV (B-5)
**Severity:** HIGH → RESOLVED
**Resolution:** Extracted reshape zero-resolution logic into `resolve_reshape_zeros(input_shape, target_shape) -> Result<Vec<usize>>` with safe zero-position collection using `Iterator::filter().collect()` instead of `.unwrap()` on `.position()`/`.rposition()`. Changed `infer_shape` return type from `Vec<usize>` to `Result<Vec<usize>>` so reshape failures propagate as compilation errors via `?` operator instead of panicking. Added final validation that rejects shapes with unresolved zero placeholders (bails with diagnostic context including both input and target shapes). Updated caller in `run_with_weight_shapes` to use `?` for error propagation. Added 17 new unit tests.

---

### I-08 · Zero-Dimension Shapes Survive to Core ML Emission

**Status:** ✅ FIXED (T-29)
**Files:** `crates/bridge/src/mir_to_compat.rs`, `crates/coreml-emit/src/mir_to_proto.rs`
**AUDIT ref:** §II-E, §IV (B-6)
**Severity:** HIGH → RESOLVED
**Resolution:** Two-pronged fix: (1) Changed `eprintln!` warnings in `mir_op_to_compat_with_shapes()` and `mir_op_to_compat()` to `anyhow::bail!()` hard gates — reshape ops with unresolved zero dimensions now produce compilation errors with diagnostic context (zero positions, raw shape, node_shape, input_shape) instead of silently emitting zeros. (2) Added defense-in-depth zero-dim validation gate in `convert_mir_to_proto_multifunction()` that scans all `MirOpCompat::Reshape` and `MirOpCompat::Fill` ops for zero dimensions in shape vectors, bailing before emission. Fixed test that asserted zeros survive to expect an error instead. Added 7 new emission-layer tests.

---

### I-09 · `% 1 == 0` Always-True Logic Bug

**Status:** ✅ FIXED (T-30)
**Files:** `crates/bridge/src/mir_to_compat.rs`
**AUDIT ref:** §III (CQ-1), §IV (B-10)
**Severity:** HIGH → RESOLVED
**Resolution:** Fixed `% 1 == 0` always-true bug in `resolve_reshape_shape()`. The 2-zero case used `remaining % 1 == 0` which is trivially always true, making the else branch dead code; the corresponding `/ 1` divisions were identity operations. Unified the 2-zero and 3+-zero cases into a single match arm using `product_so_far` consistently. Also fixed a latent bug where failed positional resolution (target rank > input rank) left the `resolved` array partially modified, corrupting the subsequent `non_zero_product` calculation. Added 6 new tests.

---

### I-10 · SDPA Compat Missing `attention_mask` and `scale`

**Status:** ✅ FIXED (T-31)
**Files:** `crates/coreml-proto/src/lib.rs`, `crates/bridge/src/mir_to_compat.rs`, `crates/coreml-proto/proto/coreml/MIL.proto`
**AUDIT ref:** §III (CQ-23), §IV (B-11)
**Severity:** HIGH → RESOLVED
**Resolution:** Added `attention_mask: Option<String>` and `scale: Option<f32>` to `MirOpCompat::ScaledDotProductAttention`. Wired through all conversion paths (From impl, mir_op_to_compat, compat_input_names, remap_compat_inputs, rename_compat_output) and both proto emission paths (MIL proto with `attn_mask`/`has_attn_mask`/`scale` fields, Apple proto with `attn_mask` name-arg input and `scale` immediate float32 input). Added `float scale = 7` field to proto definition. 14 new tests.

---

### I-11 · ArgMinMax Missing A18 Guard

**Status:** ✅ FIXED (T-32)
**Files:** `crates/ir/src/ane_target.rs`, `crates/passes/src/placement_validate.rs`, `crates/trace/src/versioned.rs`
**AUDIT ref:** §II-D, §IV (B-7)
**Severity:** MEDIUM → RESOLVED
**Resolution:** Added `supports_argminmax()` method to `AneFamily` — returns `true` for all families except A18 (which has no LSE_7 converter for `ConvertReductionArg`). Added dedicated match arm in `placement_validate.rs` that hard-rejects `MILReduceArgmax/Argmin` on A18 with diagnostic message. Fixed SIR-level classification in `versioned.rs` — changed from unconditional `CpuOnly` to family-gated support (A11Legacy through A16 supported on PE, A18 rejected). Added A18 warning about ArgMinMax CPU fallback. 17 new tests.

---

## P2 — MEDIUM (Technical Debt / Test Gaps)

### I-12 · Zero Tests for `bridge::shape_inference.rs`

**Status:** ✅ FIXED (T-33)
**Files:** `crates/bridge/src/shape_inference.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM → RESOLVED
**Effort:** M (2 days)
**Task:** T-33

Shape bugs here silently produce wrong Core ML models. Covers broadcast, concat, slice-by-index, topk, where, layer-norm, pad, expand_dims, squeeze, and more.

**Resolution:** Added 153 comprehensive tests covering all three public functions and two private helpers. Tests cover every MirOp variant handled by `compat_output_shape` with concrete shapes, edge cases (unknown inputs, out-of-range axes, incompatible broadcast), and the `broadcast_shape_compat` and `reduce_shape` helpers directly. Two bugs discovered and fixed: (1) `MILTopk` negative axis — `saturating_add(*axis as usize)` wraps for negative isize producing usize::MAX instead of the correct positive index; replaced with `(rank + axis) as usize`; (2) `MILExpandDims` multi-axis — `ax + i` position adjustment produced wrong insertion positions; replaced with direct `ax` since sorted-order iteration handles position shifts automatically.

---

### I-13 · Zero Tests for `passes::staticize.rs`

**Status:** ✅ FIXED (T-34)
**Files:** `crates/passes/src/staticize.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM → RESOLVED
**Effort:** S (0.5 day)
**Task:** T-34

Even the current pass-through implementation needs a smoke test. Any future change introducing real staticization logic will be completely untested.

**Resolution:** Added 62 comprehensive tests covering the full SirOp variant surface and key invariants: empty graphs, single-node graphs, all composite ops (LinearProjection, RMSNorm, RoPETransform, DecodeStep, SDPA, AttentionBlock, Sampler), state ops (StateRead, StateWrite), 20 unary elementwise ops, 13 binary elementwise ops, 7 reduction ops, tensor transforms (Reshape, Transpose, Concat, Split, ExpandDims, Squeeze, Tile, Pad), Gather, MatMul, Cast, Select, Where, Softmax, Conv, SliceByIndex, Quantize/Dequantize, all Constexpr variants, normalization ops (LayerNorm, BatchNorm), a realistic multi-node decode pipeline, metadata preservation (all TaskOrigin variants, QualityContract, precision_override), graph I/O preservation, Result type consistency, idempotency (single and triple pass), a 50-node stress test, pooling ops, recurrent ops (RNN, GRU, LSTM), control flow (Cond, WhileLoop), random ops, ConvTranspose, space/depth rearrangement ops, parametric activations, and Einsum. Uses `assert_graphs_identical()` helper for deep structural comparison since `SirOp` does not derive `PartialEq`.

---

### I-14 · `MilDtype` Missing Int4, UInt4, E4M3, E5M2

**Status:** ✅ FIXED (T-35)
**Files:** `crates/ir/src/common.rs`, `crates/passes/src/dtype_constraints.rs`, `crates/ir/src/ane_target.rs`, `crates/coreml-proto/src/lib.rs`, `crates/bridge/src/mir_to_compat.rs`, `crates/coreml-emit/src/weights.rs`, `crates/trace/src/sir_build.rs`, `crates/trace/src/bin/verify_pipeline.rs`, `crates/ir/src/shard_desc.rs`, `crates/ir/src/linear_slice.rs`, `crates/passes/src/mil_lower.rs`, `crates/ir/tests/pipeline.rs`
**AUDIT ref:** §III (CQ-15)
**Severity:** MEDIUM → RESOLVED
**Effort:** M (1 day)
**Task:** T-35

The `MilDtype` enum only has: Fp16, Fp32, Int32, UInt8, Bool, Fp64, Int8, Int16. Missing: Int4, UInt4, E4M3, E5M2, UInt16. Without these, the constraint "Int4 Per-Cout Dequant is not supported" and "E4M3 is not supported on this architecture" cannot be enforced.

**Resolution:** Added 5 new dtype variants (Int4, UInt4, E4M3, E5M2, UInt16) to both `MilDtype` and `MilDtypeCompat`. Added `AneFamily::supports_e4m3()` for version gating. Enforced all ANE constraints: Int4/UInt4 require interleave=8, E4M3 only on A18, E5M2 rejected on all families. Updated quantization/dequantization constraints per ANE canon. 25 new tests.

---

### I-15 · Model-Specific Constants in Compilation Logic

**Status:** ✅ FIXED (T-36)
**Files:** `crates/ir/src/common.rs`, `crates/passes/src/role_mir.rs`, `crates/passes/src/legality_rewrite.rs`, `crates/bridge/src/mir_to_compat.rs`, `crates/bridge/src/shape_inference.rs`
**AUDIT ref:** §III (CQ-16, CQ-17, CQ-18, CQ-19)
**Severity:** MEDIUM → RESOLVED
**Effort:** M (2 days)
**Task:** T-36

Hardcoded model-specific values:
- `vocab_size = 32000` (LLaMA-2 default, wrong for Qwen3's 151936)
- `head_dim = 128` fallback (should error rather than silently use wrong value)
- Qwen3 weight name patterns in `build_input_alias_map()`
- `vec![1, 512]` input shape fallback (Qwen3-0.6B assumption)

**Resolution:** Added `ModelArchConfig` and `ModelArchitecture` to `ane-ir/src/common.rs`, centralizing all model-specific constants. (1) `role_mir.rs`: Replaced hardcoded `vocab_size=32000` and `embed_dim=128` with `arch_config()` fields; added `with_arch_config()` builder method. (2) `legality_rewrite.rs`: Replaced 4 silent `head_dim=128` fallbacks with strong diagnostic warnings. (3) `mir_to_compat.rs`: Replaced hardcoded Qwen3 patterns in `build_input_alias_map()` with `ModelArchitecture` pattern methods; added `mir_graph_to_compat_with_arch()`. (4) `shape_inference.rs`: Replaced `vec![1, 512]` with `max_seq_len` parameter; added convenience wrappers. Default config is Qwen3-0.6B. 8 new tests.

---

### I-16 · No SIR→AIR Roundtrip Test with Real Shapes

**Status:** ✅ FIXED (T-37)
**Files:** `crates/passes/src/legality_rewrite.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM → RESOLVED
**Effort:** M (1 day)
**Task:** T-37

19 unit tests cover individual decompositions but there's no end-to-end test using `DecompositionContext::for_decode_step_full()` with realistic Qwen3-0.6B dimensions.

**Fix:** Added 14 comprehensive SIR→AIR roundtrip tests using `for_decode_step_full()` and `for_attention_full()` with realistic Qwen3-0.6B dimensions. Added `collect_air_op_refs()` and `validate_air_graph_structural_invariants()` helpers for deep structural invariant validation (no duplicate AirNodeIds, reference integrity, output reachability).

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

**Status:** ✅ FIXED (T-40)
**Files:** `crates/ir/src/ane_target.rs`, `crates/ir/src/ane_hw_limits.rs`, `crates/trace/src/versioned.rs`, `knowledge/ane_hw_limits_seed.json`
**AUDIT ref:** §III (CQ-14), §IV (B-4)
**Severity:** MEDIUM → RESOLVED
**Effort:** S (0.5 day)
**Task:** T-40

V17 is Apple M1, which uses A14-class ANE. The code mapped V17 to A18 family, giving M1 hardware A18's SDPA and LayerNorm support (which it doesn't have).

**Resolution:** Mapped V17 → A14 family. M1 has A14-class constraint profile: no SDPA, no LayerNorm, A14Plus elementwise/reduction converters, full-dtype broadcast, ArgMinMax supported (LSE_3). Created dedicated `m1()` hardware limits function for Mac-specific NE count (6) and tensor width (262144). Fixed `a18()` to use `AneRevision::V19`. Changed `family_to_default_revision(A18)` from V17 to V19. Updated seed data. 15 new tests.

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
| P1 | 8 | 0 | 0 | 8 |
| P2 | 8 | 2 | 0 | 6 |
| P3 | 1 | 1 | 0 | 0 |
| **Total** | **20** | **3** | **0** | **17** |

*P0 issues I-01, I-02, and I-03 all resolved. P1 issues I-04 through I-11 all resolved by T-25 through T-32. P2 issues I-12, I-13, I-14, I-15, I-16, I-19 resolved by T-33, T-34, T-35, T-36, T-37, T-40.*
