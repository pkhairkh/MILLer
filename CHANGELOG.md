# CHANGELOG.md — MILLer Compiler

## 2026-05-04

### T-41 · Run `cargo fmt` and `cargo clippy --fix` ✅
- **ISSUES ref**: I-20
- **AUDIT ref**: §III (CQ-9, CQ-10)
- **Severity**: LOW → RESOLVED
- **Effort**: S (0.5 day)
- **Resolution**: Ran `cargo fmt --all` across the entire workspace, reformatting 52 files to comply with Rust style guidelines. Ran `cargo clippy --fix --allow-dirty --allow-staged` to address auto-fixable clippy warnings. After the fix, the only remaining clippy warnings are 11 `too_many_arguments` violations (8/7 through 16/7 parameters) which require manual refactoring and are tracked by T-44. All 1099 tests pass.

### T-39 · Add ConstexprLutToDense to MirOpCompat ✅
- **ISSUES ref**: I-18
- **AUDIT ref**: §III (CQ-24)
- **Severity**: MEDIUM → RESOLVED
- **Effort**: M (2 days)
- **Resolution**: Added 7 Constexpr* variants to MirOpCompat, closing the major functional gap where the proto-direct path could not emit palettized/quantized/compressed weights. Previously all 7 constexpr MirOp variants were mapped to `MirOpCompat::Unsupported` with all field data lost (empty `"{}"` params JSON). Changes across 3 crates: (1) `coreml-proto/src/lib.rs`: Added 7 new MirOpCompat variants (ConstexprAffineDequantize, ConstexprBlockwiseShiftScale, ConstexprLutToDense, ConstexprSparseToDense, ConstexprCast, ConstexprLutToSparse, ConstexprSparseBlockwiseShiftScale) with full field preservation. Updated `From<MirOp>` with proper type conversions. Added both MIL proto and Apple proto emission paths using `constexpr_*` op_type with weight-name inputs and scalar/vector attributes. (2) `bridge/src/mir_to_compat.rs`: Added 7 explicit match arms in `mir_op_to_compat()`. Marked constexpr arms in `mir_op_to_unsupported()` as `unreachable!()`. Added 7 arms each in `compat_input_names()`, `remap_compat_inputs()`, and `rename_compat_output()`. (3) `coreml-emit/src/mir_to_proto.rs`: Added 7 constexpr variants to `op_output_names` macro. All 1099 tests pass.

### T-40 · Fix V17 (M1) → A18 Mapping ✅
- **ISSUES ref**: I-19
- **AUDIT ref**: §III (CQ-14), §IV (B-4)
- **Severity**: MEDIUM → RESOLVED
- **Effort**: S (0.5 day)
- **Resolution**: Mapped V17 (M1) to A14 family instead of A18. M1 uses A14-class ANE with identical constraint profile: no SDPA, no LayerNorm, A14Plus elementwise/reduction converters, full-dtype broadcast, ArgMinMax supported (LSE_3 converter). Previously, V17 was grouped with V19/V20/V26 under A18 family, which incorrectly gave M1 hardware A18's SDPA and LayerNorm support — ops that the M1 ANE cannot actually execute. Changes: (1) `ane_target.rs`: Separated V17 from the A18 match arm into its own `AneRevision::V17 => AneFamily::A14` arm with detailed comment explaining why M1 is A14-class; (2) `ane_hw_limits.rs`: Created dedicated `m1()` hardware limits function (6 NEs, 262144 max tensor width — Mac-specific hardware but A14-class constraint profile), fixed `a18()` to use `AneRevision::V19` instead of `V17`, updated `for_revision(V17)` to call `Self::m1()`; (3) `versioned.rs`: Changed `family_to_default_revision(A18)` from `V17` to `V19`; (4) `ane_hw_limits_seed.json`: Updated V17's family from "A18" to "A14". Added 15 new tests: 11 in `ane_target.rs` (V17→A14, no SDPA, no LayerNorm, full-dtype broadcast, A14Plus converters, ArgMinMax supported, no E4M3, ReduceMin all dtypes, V19≠V17 for A18, V7=V17 same family, A18 vs M1 constraint diff) and 4 in `versioned.rs` (A18 default rev is V19, A14 default rev is V7, M1 no SDPA/LayerNorm via VersionedCompiler). All 1099 tests pass.

### T-37 · Add SIR→AIR Roundtrip Test ✅
- **ISSUES ref**: I-16
- **AUDIT ref**: §V
- **Severity**: MEDIUM → RESOLVED
- **Effort**: M (1 day)
- **Resolution**: Added 14 comprehensive SIR→AIR roundtrip tests using `DecompositionContext::for_decode_step_full()` with realistic Qwen3-0.6B dimensions (embed_dim=1024, num_heads=16, head_dim=128, kv_heads=8, intermediate_size=2048, vocab_size=151936). Tests cover: (1) Full DecodeStep roundtrip with RoPE+QK-norm+GQA, verifying Conv1x1AsLinear output_dim for Q/K/O projections (2048/1024/1024), per-head MatMul (NOT SDPA), StateReadFixed/StateWriteFixed for KV cache, SliceByIndex for head extraction, and no Tile/Split ops; (2) AttentionBlock roundtrip with for_attention_full(), verifying GQA reshape shapes (Q=[1,512,16,128], K/V=[1,512,8,128], attn_flat=[1,512,2048]); (3) Multi-layer decode roundtrip verifying shared node deduplication (shared_attn_scale appears exactly once across layers); (4) Full transformer layer roundtrip (LinearProjection→RMSNorm(axes=3)→AttentionBlock→Reshape→Add) verifying 4D reshape shapes for QK-norm; (5) Non-GQA model roundtrip (kv_heads==num_heads, LLaMA-2-like); (6) output_dim_for_weight validation for all Qwen3-0.6B projection types; (7) Conv1x1AsLinear output_dim consistency check across 7 linear projections; (8) Metadata propagation (TaskOrigin::RealModel, precision_override) through SIR→AIR; (9) Tile decomposition SSA validity; (10) Select/Where decomposition SSA validity; (11) Empty graph roundtrip; (12) Passthrough ops roundtrip; (13) RMSNorm+RoPE+DecodeStep combined roundtrip; (14) GQA fan_out computation. Added `collect_air_op_refs()` and `validate_air_graph_structural_invariants()` helpers for deep structural invariant validation (no duplicate AirNodeIds, reference integrity, output reachability). All 1083 tests pass.

### T-36 · Parameterize Model-Specific Constants ✅
- **ISSUES ref**: I-15
- **AUDIT ref**: §III (CQ-16, CQ-17, CQ-18, CQ-19)
- **Severity**: MEDIUM → RESOLVED
- **Effort**: M (2 days)
- **Resolution**: Added `ModelArchConfig` and `ModelArchitecture` to `ane-ir/src/common.rs`, centralizing all model-specific constants that were previously hardcoded throughout the codebase. (1) `role_mir.rs`: Replaced hardcoded `vocab_size=32000` and `embed_dim=128` with `arch_config().vocab_size` and `arch_config().embed_dim`; added `with_arch_config()` builder method to `RoleMirBuilder`. (2) `legality_rewrite.rs`: Replaced 4 silent `head_dim=128` fallbacks in `decompose_attention`, `decompose_decode_step`, `apply_rope_decode`, and RoPE decompose with strong `[ERROR]`/`[WARN]` diagnostic messages that explicitly flag wrong-scale risk for models with head_dim != 128. (3) `mir_to_compat.rs`: Replaced hardcoded Qwen3 weight name patterns in `build_input_alias_map()` with `ModelArchitecture` pattern methods (`q_proj_pattern()`, `k_proj_pattern()`, etc.); added `mir_graph_to_compat_with_arch()` for architecture-aware MIR-to-compat conversion. (4) `shape_inference.rs`: Replaced hardcoded `vec![1, 512]` fallback with `max_seq_len` parameter in `compat_input_shape()` and `compat_output_shape()`; added `compat_input_shape_default()` and `compat_output_shape_default()` convenience wrappers using Qwen3-0.6B defaults. Default config is Qwen3-0.6B (vocab_size=151936, embed_dim=1024, head_dim=128, kv_heads=8, intermediate_size=2048, max_seq_len=32768). Added `from_model_config()` factory for CLI integration. Added 8 new tests for `ModelArchConfig` and `ModelArchitecture`. All 1069 tests pass.

## Pre-Audit Resolved Tasks (Archived from TASKS.md)

These tasks (T-01 through T-21) were completed before the TABULA RASA audit on 2026-05-03.

---

### T-01 · Fix FFI Validation Path Bug ✅
- **ISSUES ref**: C-11
- **Resolution**: Changed path from `Model/` to `Data/` in `capi.rs`; added test.

### T-02 · Add CI/CD Pipeline ✅
- **ISSUES ref**: C-01
- **Resolution**: Added `.github/workflows/ci.yml` with check, fmt, clippy, test jobs.

### T-03 · Extract Shared Op Definitions Across SIR / AIR / MIR ✅
- **ISSUES ref**: C-04, C-06, S-07, S-08, S-09
- **Resolution**: Created `common.rs` with `MilDtype`, `ComputeUnitHint`, `IrNodeId`, `IrGraph`; genericised `serialize.rs`.

### T-04 · Break Up `linear_slice.rs` God Module ✅
- **ISSUES ref**: C-05, W-29, W-30, S-10, S-11
- **Resolution**: Extracted to `payload.rs`, `shard_desc.rs`; added `#[deprecated]`; replaced hardcoded opset/seed.

### T-05 · Move Lab Orchestration Out of CLI ✅
- **ISSUES ref**: C-12, W-23
- **Resolution**: Created `crates/lab/src/session.rs`; CLI delegates.

### T-06 · Deduplicate `mil_emitter.py` ✅
- **ISSUES ref**: C-13, W-20, W-27
- **Resolution**: Created `python/program_builder.py`; reduced from 2,933 to 1,947 lines (−33.6%); fixed decode_step routing; extracted shared role→op mapping.

### T-07 · Derive KV Cache Dimensions from Shard Spec ✅
- **ISSUES ref**: C-14
- **Resolution**: Added `num_heads`, `head_dim`, `context_length` to `AttentionComputation`; shapes derived from spec.

### T-08 · Centralise Workspace Dependencies ✅
- **ISSUES ref**: C-03
- **Resolution**: Added `safetensors`, `prost`, `prost-types`, `prost-build`, `half` to workspace deps; replaced all hardcoded versions.

### T-09 · Add Integration Tests ✅
- **ISSUES ref**: C-02
- **Resolution**: Added `crates/ir/tests/pipeline.rs`, `crates/knowledge/tests/round_trip.rs`, `crates/cli/tests/cli.rs`.

### T-10 · Fix Knowledge Store Duplications ✅
- **ISSUES ref**: C-07, C-08, C-09, W-04, W-05
- **Resolution**: Created `util.rs` with shared functions; replaced `expect()` with `Result`; reconciled confidence values; added typed accessors.

### T-11 · Add Tests for Core IR Types ✅ (partial)
- **ISSUES ref**: W-01, W-02
- **Resolution**: Added exhaustive MirOp tests; CLI integration tests. Artifacts and report crate tests still pending.

### T-12 · Slim Down the Bridge Crate ✅
- **ISSUES ref**: C-10, W-07, W-08, S-04
- **Resolution**: Extracted shape inference to `shape_inference.rs`; replaced `f32_to_f16` with `half` crate; documented Qwen3-specific code and boilerplate.

### T-13 · Deduplicate Python Module Constants and Helpers ✅
- **ISSUES ref**: W-17, W-18, W-19, W-26
- **Resolution**: Created `python/common.py` with `COMPUTE_MAP`, `_error_result()`, `_ensure_coremltools()`; `ImportError` now propagates.

### T-14 · Separate Bridge Concerns in `bridge.py` ✅
- **ISSUES ref**: W-21, W-22, W-25
- **Resolution**: `handle_host_inspect` delegates to `model_structure.py`; `handle_profile` delegates to `profiler.generate_inputs()`; merged `convert_stateful_milprogram` into `convert_milprogram`.

### T-15 · Make Emission Deterministic ✅
- **ISSUES ref**: W-10
- **Resolution**: Replaced `Uuid::new_v4()` with `Uuid::new_v5()` using deterministic namespace.

### T-16 · Remove or Deprecate Legacy Proto Format ✅
- **ISSUES ref**: W-11
- **Resolution**: Added `#[deprecated]` to legacy `proto` module with removal deadline.

### T-17 · Unify `MirOpCompat` with `MirOp` ✅ (lightweight)
- **ISSUES ref**: W-12, W-16
- **Resolution**: Added `From<MirOp> for MirOpCompat` conversion with exhaustive variant test. Full `ToProto` trait deferred to I-17.

### T-18 · Add `#[should_panic]` Tests ✅
- **Resolution**: Added `crates/ir/tests/should_panic.rs` and `crates/knowledge/tests/should_panic.rs`.

### T-19 · Fix A12 Hardware Limits ✅ (documented)
- **ISSUES ref**: S-12
- **Resolution**: Added doc comment noting estimated values; added runtime warning when A12 limits are used.

### T-20 · Update README Test Count ✅
- **ISSUES ref**: S-15
- **Resolution**: Updated from 440 to 637.

### T-21 · Resolve Remaining 12 Code Review Findings ✅
- **ISSUES ref**: W-03, W-06, W-09, W-13, W-14, S-01, S-02, S-03, S-05, S-06, S-13, S-14
- **Resolution**: All 12 findings addressed (see original TASKS.md for details).

---

## Pre-Audit Resolved Issues (Archived from ISSUES.md)

| Issue | Description | Resolution |
|-------|-------------|------------|
| ISSUE-001 | Mask path uses CPU-only ops | Replaced with precomputed tables + Gather |
| ISSUE-002 | Fill/FillLike survive to emission | Replaced with Const scalar + Add broadcasting |
| ISSUE-003 | Equal/LessEqual on ANE path | Eliminated by ISSUE-001 fix |
| ISSUE-006 | Hardcoded model constants in CLI | Now read from ModelConfig |
| ISSUE-007 | KV mask CPU-only path | Eliminated by ISSUE-001 fix |
| ISSUE-008 | Three mask implementations | Unified in legality_rewrite |
| ISSUE-011 | Where→Select double-rewrite | Rewrite removed; engine classification fixed |
| ISSUE-013 | for_qwen3_0_6b factory | Replaced with from_model_config() |
| ISSUE-014 | Hardcoded shape in role_mir | Derived from spec output_specs |
| ISSUE-015 | kv_cache_rewrite dead code | Deprecated |
| ISSUE-016 | RMSNorm fp16 overflow | Dynamic max-abs stabilization added |
| ISSUE-017 | QK norm not implemented | Full SLaNC implementation added |
| ISSUE-019 | bool→fp16 cast on ANE | Eliminated by ISSUE-001 fix |
| ISSUE-021 | default_engine() misclassifies ops | 8 ops moved to CPU-only |
| ISSUE-022 | Scalar constant resolution | scalar:// protocol added |

---

## TABULA RASA Audit — 2026-05-03

Full-spectrum diagnostic sweep performed. See `AUDIT.md` for complete findings.

**Key metrics:**
- 660 tests passing, 0 failures
- 3 clippy deny errors, 85 warnings
- 55 ops with engine/compat mismatch
- 9 missing constraint validators
- 2 modules with zero test coverage
- IR Cleanliness Score: 78%

**New issues filed:** I-01 through I-20 (see ISSUES.md)
**New tasks created:** T-22 through T-45 (see TASKS.md)

---

## Post-Audit Resolved Tasks

### T-22 · Align Three Sources of Truth (Engine / CPU-Only / Compat) ✅
- **Commit**: 063f225
- **ISSUES ref**: I-01
- **Resolution**: Performed full three-way alignment of `MirOp::default_engine()`, `CPU_ONLY_OPS`, and `MirOpCompat` coverage. Moved 28 MirOp variants from PE/NE to None (CPU-only). Added 10 entries to CPU_ONLY_OPS. Added `MirOp::mil_op_name()` method.

### T-23 · Wire `is_cpu_only()` into Placement Validation ✅
- **Commit**: 063f225 (bundled with T-22)
- **ISSUES ref**: I-02
- **Resolution**: Added `cpu_only_ops::is_cpu_only(op.mil_op_name())` check as a hard gate in `placement_validate.rs`.

### T-24 · Fix V6 (A13 Silicon) → A14 Family Mapping ✅
- **Commit**: 063f225
- **ISSUES ref**: I-03
- **Resolution**: Added `AneFamily::A13` variant with distinct constraint profile. Mapped `AneRevision::V6` to `A13` family (was incorrectly grouped with V7 under A14).

### T-25 · Wire Interleave + Dtype Validators into Pipeline ✅
- **Commit**: 5d2f85f
- **ISSUES ref**: I-04
- **Resolution**: Added `PlacementContext` struct and `validate_placement_with_context()` function. Wired six constraint validators as hard gates.

### T-26 · Add `validate_matmul_constraints()` ✅
- **Commit**: (bundled with T-25)
- **ISSUES ref**: I-05
- **Resolution**: Added `validate_matmul_constraints()` enforcing depth=1, minimum rank 2, inner dimensions match, and output channels even. 27 new tests.

### T-27 · Add `validate_pad_constraints()` ✅
- **ISSUES ref**: I-06
- **Resolution**: Added `validate_pad_constraints()` to `op_constraints.rs` enforcing six ANE Pad hard constraints: mode gate (reject replication/symmetric), no negative padding, no batch-axis padding (rank ≥ 4), no channel-axis padding (rank-aware axis mapping), no depth-axis padding (rank-5), and pad_amounts length validation. Wired into `placement_validate.rs` with dedicated `MILPad` match arm. 25 unit tests + 10 integration tests. 762 total tests passing.

### T-28 · Fix Reshape `.unwrap()` Panic ✅
- **ISSUES ref**: I-07
- **Resolution**: Extracted reshape zero-resolution logic into `resolve_reshape_zeros(input_shape, target_shape) -> Result<Vec<usize>>` with safe zero-position collection (replaces `.unwrap()` on `.position()`/`.rposition()`). Changed `infer_shape` return type from `Vec<usize>` to `Result<Vec<usize>>` so reshape failures propagate as compilation errors via `?` instead of panicking. Added final validation rejecting shapes with unresolved zeros. 17 new unit tests. 780+ total tests passing.

### T-29 · Add Zero-Dimension Validation Before Emission ✅
- **ISSUES ref**: I-08
- **Resolution**: Two-pronged approach: (1) Changed `eprintln!` warnings in `mir_to_compat.rs` to `anyhow::bail!()` hard gates — reshape ops with unresolved zero dimensions now produce compilation errors instead of silently emitting invalid shapes. Both `mir_op_to_compat_with_shapes()` and `mir_op_to_compat()` reject zeros with diagnostic messages including zero positions, raw shape, node_shape, and input_shape. (2) Added defense-in-depth zero-dim validation gate in `mir_to_proto.rs` `convert_mir_to_proto_multifunction()` that scans all `MirOpCompat::Reshape` and `MirOpCompat::Fill` ops for zero dimensions in shape vectors. Fixed `test_reshape_zero_placeholders_resolved_from_node_shape` Case 4 to assert error instead of asserting zeros survive. Added 7 new emission-layer tests. 785 total tests pass.

### T-30 · Fix `% 1 == 0` Logic Bug ✅
- **ISSUES ref**: I-09
- **Resolution**: Fixed `% 1 == 0` always-true logic bug in `resolve_reshape_shape()` in `mir_to_compat.rs`. The 2-zero case used `remaining % 1 == 0` which is trivially always true (modulo 1 is always 0), making the else branch dead code. The corresponding `/ 1` divisions were identity operations. Unified the 2-zero and 3+-zero cases into a single `2 | _` match arm using `product_so_far` consistently: all zeros except the last are set to 1, the last zero is set to `remaining / product_so_far`. Fixed `product_so_far *= 1` to `product_so_far *= resolved[pos]` for self-documenting correctness. Also fixed a latent bug where failed positional resolution (target rank > input rank) left the `resolved` array partially modified, corrupting the subsequent `non_zero_product` calculation — now resets to `target_shape` on positional failure. Added 6 new tests. 790 total tests pass.

### T-31 · Add `attention_mask` and `scale` to SDPA Compat ✅
- **ISSUES ref**: I-10
- **Resolution**: Added `attention_mask: Option<String>` and `scale: Option<f32>` fields to `MirOpCompat::ScaledDotProductAttention`, which previously only had `query`, `key`, and `value`. These fields existed in `MirOp::MILScaledDotProductAttention` (and in SIR/AIR counterparts) but were silently discarded (`..` / `_` patterns) during MirOp→MirOpCompat conversion. Updated seven code locations: (1) `MirOpCompat` enum definition in `coreml-proto/src/lib.rs`; (2) `From<MirOp>` impl — wired `attention_mask.map(|id| nid(id))` and `scale` instead of discarding with `..`; (3) `mir_to_compat.rs` `mir_op_to_compat()` — wired `attention_mask.as_ref().map(|id| id.0.clone())` and `scale: *scale` instead of `attention_mask: _, scale: _`; (4) `compat_input_names()` — includes `attention_mask` in input names when present (4 inputs: q, k, v, mask vs 3 without mask); (5) `remap_compat_inputs()` — remaps `attention_mask` name through alias map; (6) `rename_compat_output()` — preserves `attention_mask` and `scale` through rename; (7) proto emission — wires `attn_mask`, `has_attn_mask`, and `scale` in `MilScaledDotProductAttentionOp` proto message (added `float scale = 7` field to proto definition). Apple-proto emission emits `attn_mask` as a name-arg input and `scale` as an immediate float32 input. Added 14 new tests: 8 in `mir_to_compat.rs` (attention_mask preservation, scale preservation, both together, none, input_names with/without mask, remap, rename) and 6 in `coreml-proto/src/lib.rs` (proto emission with/without mask/scale, mask-only, scale-only, Apple-proto with/without). 804 total tests pass.

### T-32 · Add A18 Guard for ArgMinMax ✅
- **ISSUES ref**: I-11
- **Resolution**: Added `supports_argminmax()` method to `AneFamily` — returns `true` for all families except A18 (which has no LSE_7 converter for `ConvertReductionArg`). Added dedicated match arm in `placement_validate.rs` that hard-rejects `MILReduceArgmax/Argmin` on A18 with diagnostic message citing the missing LSE_7 converter. Fixed SIR-level classification in `versioned.rs` — changed `ReduceArgmax/Argmin` from unconditional `CpuOnly` ("has no ANEC converter; arg reduction is CPU-only") to family-gated: `AneSupported(PE)` on families that support argminmax (A11Legacy through A16), `CpuOnly` on A18. Added A18 version warning about ArgMinMax CPU fallback. Updated family capability table in `ane_target.rs` header doc to include ArgMinMax column. 17 new tests (11 placement validator tests covering rejection on A18, allowed on A16/A14/A12/A11Legacy, message content verification, interaction with dtype gate; 6 SIR-level versioned tests covering supported on A16/A14/A11Legacy and CpuOnly on A18). 821 total tests pass.

### T-33 · Add Tests for `bridge::shape_inference.rs` ✅
- **ISSUES ref**: I-12
- **AUDIT ref**: §V
- **Resolution**: Added 153 comprehensive tests for the `shape_inference.rs` module, which previously had zero test coverage (500+ lines of shape inference code). Tests cover all three public functions (`compat_input_dtype`, `compat_input_shape`, `compat_output_shape`) and two private helper functions (`broadcast_shape_compat`, `reduce_shape`). Coverage includes: (1) `compat_input_dtype` — input_ids special-case returning Int32, passthrough for other dtypes; (2) `compat_input_shape` — non-empty shape returns as-is, input_ids fallback `[1,512]`, generic fallback `[1]`; (3) `compat_output_shape` early-return paths — non-empty shape bypass, input_ids name heuristic; (4) 22 unary shape-propagating ops (Silu, Abs, Relu, Sigmoid, Tanh, Gelu, Exp, Cos, Sin, Cast, Rsqrt, Neg, Sqrt, LogicalNot, Ceil, Floor, Round, Sign, Log, LeakyRelu, Clip) with known and unknown inputs; (5) 17 binary ops with numpy-style broadcast (Add, Mul, Sub, Maximum, Minimum, RealDiv, Pow, FloorDiv, Mod, Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual, LogicalAnd, LogicalOr) — same-shape, different-rank broadcast, scalar broadcast, mixed-ones broadcast, partial-known fallbacks, incompatible fallback; (6) Softmax, Linear; (7) MatMul — 2D, batched, batched broadcast, partial-known, degenerate 1D fallback; (8) Reshape, Transpose (2D and 4D), Tile; (9) Fill, FillLike; (10) Gather (embedding, axis-last, no-indices fallback); (11) Reduction ops (ReduceMean, ReduceMax, ReduceMin, ReduceProd) — keep_dims/no-keep, single/multiple/all axes, out-of-range, unknown input; (12) ExpandDims (single axis, multiple axes, axis at end); (13) Squeeze (single/multiple axes, out-of-range); (14) Pad (symmetric, uneven, zero padding); (15) Concat (multi-input, single-input, axis out-of-range); (16) Where (all-same, broadcast condition, partial-known fallbacks); (17) LayerNorm, Topk (positive/negative axis), SDPA (with/without mask/scale); (18) ReadState, CoremlUpdateState, StateWrite; (19) Conv, Select, Split; (20) SliceByIndex (simple, begin/end masks, squeeze mask, negative end); (21) Identity (normal, placeholder, unknown); (22) Stack (axis 0, axis 2); (23) Const (node_shapes lookup, scalar:// pattern, priority, unknown); (24) Catch-all unknown op; (25) `broadcast_shape_compat` directly (same-shape, different-rank, scalar, mixed-ones, incompatible, empty); (26) `reduce_shape` directly (keep/no-keep, multiple axes, all axes, out-of-range). **Bug fixes discovered by tests**: (1) Fixed `MILTopk` negative axis handling — `saturating_add(*axis as usize)` wraps for negative isize values producing usize::MAX; replaced with `(rank + axis) as usize` which correctly computes the positive axis index; (2) Fixed `MILExpandDims` multi-axis position adjustment — `ax + i` produced wrong insertion positions when multiple axes are specified; replaced with direct `ax` since iterating in sorted order already accounts for position shifts. 974 total tests pass.

### T-34 · Add Tests for `passes::staticize.rs` ✅
- **ISSUES ref**: I-13
- **AUDIT ref**: §V
- **Resolution**: Added 62 comprehensive tests for the `StaticizePass` module, which previously had zero test coverage. Tests cover: (1) empty/minimal graphs (empty graph, default/new equivalence); (2) single-node graphs (Const, Fill, Identity placeholder); (3) LinearProjection (core vertical slice, with bias); (4) RMSNorm, RoPETransform; (5) DecodeStep (minimal, with QK-norm); (6) SDPA (without mask, with mask+scale) and AttentionBlock; (7) StateRead/StateWrite (KV cache operations); (8) 20 unary elementwise ops (Silu, Relu, Sigmoid, Tanh, Gelu, Exp, Neg, Abs, Sqrt, Rsqrt, Cos, Sin, Ceil, Floor, Round, Sign, Log, LogicalNot, Softplus, Softsign); (9) 13 binary elementwise ops (Add, Mul, Sub, Maximum, Minimum, RealDiv, Pow, Equal, NotEqual, Greater, GreaterEqual, Less, LessEqual); (10) 7 reduction ops (ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceArgmax, ReduceArgmin); (11) tensor transform ops (Reshape, Transpose, Concat, Split, ExpandDims, Squeeze, Tile, Pad); (12) Gather; (13) MatMul; (14) Cast, Select, Where; (15) Softmax; (16) Conv; (17) SliceByIndex; (18) Quantize/Dequantize and Constexpr ops (AffineDequantize, LutToDense, SparseToDense, ConstexprCast, ConstexprBlockwiseShiftScale); (19) Sampler; (20) LayerNorm, BatchNorm; (21) realistic multi-node decode pipeline; (22) metadata preservation (all fields, all TaskOrigin variants); (23) graph I/O preservation (multiple inputs, multiple outputs, empty inputs); (24) Result type consistency (always Ok, unwraps cleanly); (25) idempotency (single pass, multi-node triple pass); (26) 50-node stress test; (27) Topk; (28) pooling ops (MaxPool, AvgPool); (29) recurrent ops (RNN, GRU, LSTM); (30) control flow (Cond, WhileLoop); (31) random ops; (32) ConvTranspose; (33) ReshapeLike, Flatten2d, Reverse; (34) DepthToSpace/SpaceToDepth/PixelShuffle/PixelUnshuffle; (35) Cumsum, FillLike, Range1d; (36) parametric activations (LeakyRelu, ScaledTanh, ThresholdedRelu); (37) Einsum. Uses `assert_graphs_identical()` helper for deep structural comparison since `SirOp` does not derive `PartialEq`. 1036 total tests pass.

### T-35 · Expand `MilDtype` with Int4, UInt4, E4M3, E5M2, UInt16 ✅
- **ISSUES ref**: I-14
- **AUDIT ref**: §III (CQ-15)
- **Resolution**: Added 5 new dtype variants to `MilDtype` (Int4, UInt4, E4M3, E5M2, UInt16) and 5 corresponding variants to `MilDtypeCompat`. Updated all 18 exhaustive match sites across 8 crates: `common.rs`, `dtype_constraints.rs`, `ane_target.rs`, `coreml-proto/src/lib.rs` (MilDtypeCompat, CoreMlDataType, data_type_to_proto, mil_dtype_to_proto, weight_data_inline, mil_dtype_to_apple, coreml_dtype_to_apple_mil, compat_dtype_to_apple_mil, coreml_dtype_to_apple_array, Cast dtype_str), `mir_to_compat.rs` (mil_dtype_to_compat, compat_dtype_element_size, convert_dtype), `weights.rs` (BlobDataType, coreml_dtype_to_blob_dtype), `verify_pipeline.rs`, `pipeline.rs`, `sir_build.rs`, `shard_desc.rs`, `linear_slice.rs`, `mil_lower.rs`. Added `AneFamily::supports_e4m3()` method (true only on A18, matching per-op support matrix). Enforced ANE constraints in `dtype_constraints.rs`: (1) Int4/UInt4 legal but require interleave=8 via `validate_int4_interleave()`, `validate_uint4_interleave()`, `dtype_requires_interleave_8()`; (2) E4M3 version-gated — rejected on A11-A16 with `VersionGatedDtype` error, allowed on A18; (3) E5M2 rejected on all families with "E4M3 or E5M2 format not supported" error message matching actual ANE runtime error; (4) UInt16 legal on all families; (5) Updated quantization constraints — output can be Int8/UInt8/E4M3/E5M2 per ANE canon ("Quant layer must have int8, uint8, e4m3 or e5m2 output format"); (6) Updated dequantization constraints — input can be Int8/UInt8/Int4/E4M3 per ANE canon ("Dequant layer must have int8, uint8, int4 or e4m3 input format"), but Int4 per-cout dequant rejected; (7) Added `is_int4_per_cout_dequant_supported()` → false, `is_e4m3_zero_point_supported()` → false, `DequantScaleType` enum with `PerTensor` and `PerOutputChannel` variants; (8) Added `Int4ConstraintViolation` and `Float8ConstraintViolation` error variants to `DtypeConstraintError`. Updated `CoreMlDataType` with 5 new variants and proper proto value mappings (Int4=25, UInt4=35, E4M3=40, E5M2=41, UInt16=32) matching Apple's MIL proto v2 DataType enum. Updated `BlobDataType` with UInt16=7, Int4=8, UInt4=11, Float8E4M3FN=16, Float8E5M2=17 matching Apple's BlobDataType.hpp. Updated Cast op dtype strings for all new variants. Updated `parse_dtype` in sir_build.rs with all new string formats including aliases (fp64/float64/double, int16/short, e4m3/float8_e4m3/float8e4m3fn, e5m2/float8_e5m2/float8e5m2). Updated string→MilDtype matches in shard_desc.rs, linear_slice.rs, mil_lower.rs with all new dtype names. 25 new tests covering: Int4/UInt4 legal on all families, E4M3 rejected on pre-A17, E4M3 legal on A18, E5M2 rejected on all families, UInt16 legal on all families, Int4/UInt4 interleave must be 8, Int4 per-cout dequant not supported, E4M3 zero point not supported, quantize E4M3/E5M2 output, quantize Int4/UInt4 rejected, dequantize Int4/E4M3 input, dequantize Int4 per-cout rejected, dequantize Int4 per-tensor ok, dequantize E5M2/UInt4 rejected, dtype_requires_interleave_8, E5M2 error message content, E4M3 version-gated error, Int4 interleave error message. All 1061 tests pass.
