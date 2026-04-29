# ISSUES

Current high-signal issues after the Sprint 60 code quality, ANE constraints grounding, and per-op constraint implementation.

## Verified Snapshot

- `cargo test --workspace --quiet` passes on this host with **501 passing** and **1 ignored**.
- `python3 -m pytest python/tests/ -q` passes with **68 Python tests**.
- `cargo clippy --all-targets --all-features -- -D warnings` passes cleanly.
- `cargo fmt --check --all` passes cleanly.
- All eight task families are real `TaskFamilyTrait` implementations and are reachable from `ane-cli generate-tasks --family`.
- `HandoffKind::StateWriteRead` is active on the decode-step Interior → Exit shard boundary.
- `compile-sharded --proto-direct` and `compile-full-sharded --proto-direct` do reach `RoleMirBuilder` for sharded decode-step emission.
- All five declared `ElementWiseOp` variants (Add, Mul, Abs, Maximum, Minimum) now have AIR→MIR lowering paths (Sprint 55).
- AIR decompositions for AttentionBlock and DecodeStep now carry real task dimensions via `DecompositionContext` when available (Sprint 56).
- The current MIR surface exposes **37** `MirOp` variants. A simple exact-name diff against `MIL_OPS.md` still leaves **133** documented MIL op names uncovered; that raw count includes a few alias-like overlaps such as `Const` vs `MILConst` and `select` vs `MILWhere`.
- **Sprint 58 resolved**: `MilDtypeRepr` unified into `MilDtype`; `ComputeUnits` unified into `ComputeUnitHint`; `ane-ir` uses `anyhow::Result`; naming inconsistencies fixed (`MILAtanh`, `MILSinh`); stub passes implemented/removed; CI pipeline configured; Python tests added.
- **Sprint 59 resolved**: `AneFamily`/`AneRevision`/`AneTarget` types with chip-to-family mapping; `AneEngine` with per-MirOp engine assignment; `AnePlacementValidator` for hard constraint checks; per-family op legality matrix seeded; SDPA constraint validation in AIR→MIR lowering; palettization constraints seeded.
- **Sprint 60 resolved**: Per-op constraint validation for conv/linear/gather/pooling/ArgMinMax; dtype legality rules per ANE family; `CPU_ONLY_OPS` hard gate in `LegalityRewritePass`; `AneInterleave`/`AneLayout` types; `AneHwLimits` per-revision hardware limits.
- **SIR Trace Bugfix Sprint resolved**: Seven critical bugs in `sir_build.rs` fixed — (1) separate Q/K/V projections instead of single merged QKV, (2) no phantom split node references, (3) residual connections emitted as `SirOp::Add`, (4) RMSNorm epsilon validation with fallback chain, (5) causal mask references in SDPA, (6) SwiGLU auto-detection when both gate_proj and up_proj exist, (7) non-silent `resolve_input` warnings. Model registry references removed from all documentation — decomposition is fully config-driven. Additionally: QK-norm support (`has_qk_norm` flag for Qwen3-like architectures), `head_dim` from config override (Qwen3 sets `head_dim=128` independently of `hidden_size/num_heads`), and explicit residual inputs on `AttentionBlock`/`MlpBlock` nodes in the fallback structural graph.

## Current Priorities

### 1. Unify shard emission around one source of truth

The project still has two parallel role-specific shard emission descriptions:

- Rust-side `RoleMirBuilder` in `crates/passes/src/role_mir.rs`
- Python-side shard-role logic in `python/mil_emitter.py`

This is the highest-value cleanup because it affects compiler truth directly. Right now:

- `--proto-direct` for sharded decode-step uses `RoleMirBuilder`
- the default Python bridge path still uses its own independent role-specific logic

Proceed by making one representation authoritative and letting the other path consume it as a backend adapter, not as a second semantic source.

### 2. Make the active proto-direct shard path semantically honest

The active proto-direct shard CLI path is structurally real but still loses two important pieces of truth:

- `emit_role_shard_proto_direct()` and `emit_mir_graph_proto_direct()` currently use `EmptyWeightResolver`
- `RoleMirBuilder` defaults every node to `CPUAndNE` instead of consuming the shard's actual `compute_units`

So the Rust-only shard path currently preserves role-specific structure, but not real compile-time weights or real per-shard compute-unit intent.

Sprint 57 update: real shard compute-unit choices now thread into `RoleMirBuilder` via `compute_units_to_hint()`. The remaining gap is real compile-time weights into proto-direct emission (EmptyWeightResolver still used in `emit_role_shard_proto_direct()`).

Proceed by threading real compile-time weights into proto-direct emission.

### 3. Carry real shapes all the way into MIR and proto-direct artifacts

The AIR decomposition surface is no longer missing major ops, and placeholder shapes are now resolved when `DecompositionContext` is available (Sprint 56):

- `decompose_attention_block()` in `crates/passes/src/legality_rewrite.rs` now uses real dimensions from `DecompositionContext` for SliceByIndex bounds (e.g., Q=[0:batch, 0:seq, 0:embed]) and Reshape target shapes (e.g., [batch, seq, heads, head_dim])
- `decompose_decode_step()` now uses real dimensions for SliceByIndex bounds, Reshape target shapes, and StateReadFixed shapes
- When `DecompositionContext` is `None` (e.g., in tests or non-task compilation), placeholder zeros are still used for backward compatibility
- The CLI constructs `DecompositionContext` from the task spec for Attention, DecodeStep, and ShardedDecodeStep tasks

Sprint 57 update: `MirNode.shape` is now populated during AIR→MIR lowering via `infer_shape()`. `RoleMirBuilder` nodes also derive shapes from `spec.output_specs` when available.

**Remaining gaps:**

- some `RoleMirBuilder` stateful nodes (e.g., AttentionComputation read_state) still use hardcoded shape vectors
- proto-direct model construction still fills function input/output shapes with `vec![]` in some paths

Proceed by threading shape information from AIR through MIR lowering and into proto-direct model I/O metadata, so downstream consumers do not have to infer basic tensor structure from names alone.

### 4. Turn profiling into frontier exploration

The repo now has the right family surface for sweeps:

- `ShapeHostileFamily`
- `OpRemapFamily`
- `ShardSurvivalFamily`

But the CLI still treats profiling as single-package timing:

- `profile` in `crates/cli/src/main.rs` / `python/bridge.py` times one emitted package
- there is no built-in sweep/search/report path for failure boundaries, placement cliffs, or formulation comparisons

Proceed by adding a first-class frontier-search workflow that drives `compile`, `verify`, `profile`, and optionally `compute_plan_harvest` across generated families.

### 5. Integrate offline placement proof into verification on non-Apple hosts

The repo already has two placement-evidence mechanisms:

- online/macOS: `MLComputePlan` via `python/compute_plan.py`
- offline/host-side: `ComputePlanVerifier` in `crates/knowledge/src/compute_plan_verify.rs`

But `python/verify.py` only uses the first one and otherwise reports placement unavailable.

Sprint 57 update: Python verification now falls back to `predict_placement_from_ops()` when MLComputePlan is unavailable, using the same known op→device mappings as the Rust ComputePlanVerifier. `PlacementResult` includes `verification_method` and `prediction_confidence` fields to distinguish observed vs. predicted placement.

Proceed by validating the predicted placement against real MLComputePlan data when Apple hardware is available, to calibrate the prediction confidence scores.

### 6. Close the remaining declared lowering hole for `StaticLUTProjection`

One active declared lowering gap remains in compiler code:

- `AirOp::StaticLUTProjection` still errors with `StaticLUTProjection lowering not yet implemented`

This is now documented as a scope boundary in `AirOp::StaticLUTProjection`, but it is still an honest open gap:

- either add proper SIR→AIR→MIR/compiler ownership for LUT
- or remove this AIR variant if LUT will stay a dedicated Python/task-family path

Sprint 57 update: `StaticLUTProjection` now lowers to `MILGather` as a de-scoped approximation. The op is not used by any active SIR/task path; LUT projection has a dedicated Python emission path. This is the "de-scope" direction: the AIR variant remains for serializability and compat layer coverage, but the lowering is an approximation, not a faithful grouped-LUT implementation.

No further action needed unless LUT semantics need to be brought into the Rust compiler path.

### 7. Close op-surface reachability gaps inside the current compiler

The repo now has a second class of op problem besides fully missing ops: ops that are declared and even serialized, but are not reachable from the active SIR/task path.

Examples:

- MIR-only / effectively unreachable today: `MILSub`, `MILConv`, `MILStateWrite`, `MILCast`
- AIR+MIR, but no active SIR producer or task path: `SliceUpdate`, `Where`, `Exp`, `Sigmoid`, `Tanh`

These ops mostly exist in tests, compat conversion, or proto serialization, which is useful scaffolding, but still leaves a truth gap between “represented” and “compiler-reachable”.

Proceed by making each of these ops either:

- reachable from a real SIR/task path, or
- explicitly de-scoped from the current compiler surface and removed from coverage claims

### 8. Add an explicit MIL op coverage backlog from `MIL_OPS.md`

The current MIR surface covers 37 operations. The broader documented MIL space is much larger.

High-value missing groups include:

- activation/elementwise: `relu6`, `sigmoid_hard`, `thresholded_relu`, `clamped_relu`, `leaky_relu`, `linear_activation`, `prelu`, `softsign`, `silu`, `scaled_tanh`, `elu`, `softplus`, `softplus_parametric`, `clip`
- comparison/masking: `select`, `equal`, `greater`, `greater_equal`, `less`, `less_equal`, `not_equal`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`
- reductions/norm/pooling: `reduce_max`, `reduce_min`, `reduce_prod`, `reduce_sum_square`, `reduce_l2_norm`, `reduce_l1_norm`, `reduce_log_sum_exp`, `reduce_log_sum`, `batch_norm`, `instance_norm`, `l2_norm`, `local_response_norm`, `max_pool`, `avg_pool`, `l2_pool`
- unary math/transcendentals: `sqrt`, `inverse`, `ceil`, `floor`, `round`, `log`, `sign`, `exp2`, `atan`, `erf`, `acos`, `asin`, `cosh`, `sinh`, `mod`, `pow`, `atanh`, `tan`
- tensor/view/shape ops: `expand_dims`, `squeeze`, `reverse`, `reverse_sequence`, `slice_by_size`, `sliding_windows`, `reshape_like`, `pad`, `tile`, `stack`, `flatten2d`, `shape`, `range_1d`, `fill`, `fill_like`, `identity`
- resize/space/image ops: `resize`, `resize_nearest_neighbor`, `resize_bilinear`, `upsample_nearest_neighbor`, `upsample_bilinear`, `crop`, `crop_resize`, `affine`, `resample`, `depth_to_space`, `space_to_depth`, `pixel_shuffle`, `pixel_unshuffle`, `batch_to_space`, `space_to_batch`
- gather/scatter/index ops: `gather_along_axis`, `gather_nd`, `scatter`, `scatter_along_axis`, `scatter_nd`, `argsort`, `reduce_argmax`, `reduce_argmin`, `band_part`, `cumsum`, `one_hot`, `non_zero`, `non_maximum_suppression`
- quantization/constexpr: `quantize`, `dequantize`, `constexpr_affine_dequantize`, `constexpr_blockwise_shift_scale`, `constexpr_lut_to_dense`, `constexpr_sparse_to_dense`, `constexpr_cast`, `constexpr_lut_to_sparse`, `constexpr_sparse_blockwise_shift_scale`
- recurrent/control/random/container: `rnn`, `gru`, `lstm`, `cond`, `while_loop`, `make_list`, `list_length`, `list_write`, `list_read`, `list_gather`, `list_scatter`, `random_bernoulli`, `random_normal`, `random_uniform`, `random_categorical`, `classify`

Proceed by tracking this as a grouped backlog, not as an unbounded “more ops later” bucket.

### 9. Add a generic mechanism for equivalent-formulation choice

The compiler can now represent alternatives such as:

- `MILSliceUpdate`
- `MILWhere`

But it still does not choose among semantically equivalent formulations using evidence. The current path hard-codes one formulation and stops there.

Proceed on this only after shard-emission unification, proto-direct realism, and stronger frontier/profiling evidence, because a formulation-choice mechanism needs trustworthy evidence and fewer duplicate semantic paths.

### 10. Expose the multi-function path through the Rust CLI/artifact flow

Multi-function support is real:

- Python bridge has `emit_multifunction`, `emit_multifunction_shared_weights`, and `validate_multifunction`
- proto-direct has `emit_proto_direct_multifunction`

But the Rust CLI still has no first-class multi-function compile/package/report command.

Proceed by adding one narrow CLI path that emits and validates the existing embedding + decode-step package, then records function-level provenance in manifests and reports.

### 11. Consolidate stale truth-facing documentation

The top of `TASKS.md` and this file are now aligned with code reality, but `STATUS.md` still contains stale feature-matrix and residual statements such as:

- multi-function reported as seam-only in older sections
- stateful models reported as schema-only in older sections
- older sharded/runtime residuals that conflict with the current code

Proceed by collapsing or rewriting the stale sections so the project has one truth-facing status layer instead of a current summary plus contradictory historical fragments.

---

## ane-constraints-docs Audit Findings (2026-04-28)

The following issues were identified during a systematic comparison of the `ane-constraints-docs/` directory (the "points of truth") against the current implementation in `ane-knowledge` and `ane-trace`.

### Issue #27: Missing ANE Bandwidth Constraints In ane-knowledge

**Severity**: High
**Component**: ane-knowledge
**Discovered**: 2026-04-28

The `ane-constraints-docs` specifications define bandwidth constraints for ANE data paths (neural engine to DRAM, neural engine to SRAM, inter-neural-core bandwidth) that are not represented in the `ane-knowledge` crate. Without these constraints, the compiler cannot make informed decisions about tile scheduling, buffer allocation, and data movement optimization.

**Impact**: Tile assignment may exceed available bandwidth; no bandwidth-aware pass ordering or fusion decisions; performance predictions in ane-report may be inaccurate.

**Proposed Fix**: Add bandwidth constraint structures to `ane-knowledge`; define per-family bandwidth profiles (A11–A18); integrate bandwidth checks into the AIR to MIR lowering pass.

### Issue #28: Incomplete ANE Memory Layout Specifications

**Severity**: High
**Component**: ane-knowledge, ane-passes
**Discovered**: 2026-04-28

The `ane-constraints-docs` detail specific memory layout requirements for ANE tensors (planar format, channel alignment, stride requirements) that are only partially implemented in `ane-knowledge`. The documented per-tile memory layout constraints and the interaction between layout and kernel fusion are not captured.

**Impact**: Generated models may have suboptimal memory layouts; potential for misaligned tensors causing runtime errors on ANE; fusion decisions may violate layout compatibility requirements.

**Proposed Fix**: Implement full memory layout constraint structures in `ane-knowledge`; add layout verification pass in `ane-passes`; ensure `ane-trace` SIR construction produces layout-compatible graphs.

### Issue #29: Outdated ANE Kernel Fusion Rules

**Severity**: High
**Component**: ane-knowledge, ane-passes
**Discovered**: 2026-04-28

The kernel fusion rules in `ane-knowledge` do not match the current `ane-constraints-docs` specifications. The documented rules include version-specific fusion limits (3 ops for ANE1/A14, 5 for A15, 8 for A16, 12 for A18) and fusion compatibility matrices that are not fully represented. Additionally, the docs specify that certain op combinations (e.g., conv+bn+relu) have special fusion patterns that the current implementation does not handle.

**Impact**: Suboptimal fusion decisions leading to reduced ANE utilization; fusion passes may create invalid op combinations for specific ANE versions; performance regression compared to coremltools output.

**Proposed Fix**: Update fusion rules in `ane-knowledge` to match `ane-constraints-docs`; implement version-aware fusion compatibility matrix; add fusion verification pass; add special fusion pattern matching for common sequences.

### Issue #30: Missing ANE Precision Mode Constraints Per Version

**Severity**: Medium
**Component**: ane-knowledge, ane-trace
**Discovered**: 2026-04-28

The `ane-constraints-docs` specify precision mode constraints that differ across ANE versions. A11/A12/A14/A15 only support FP16, while A16+ supports mixed-precision (FP16/FP32). The current `ane-knowledge` crate does not fully encode these version-specific precision constraints, and the `VersionedCompiler` in `ane-trace` has a basic implementation that needs alignment with the documented constraints.

**Impact**: Traced models may use precision modes not supported by the target ANE version; no precision-aware lowering or casting in the compilation pipeline; potential for numerical accuracy issues when FP32 ops are forced to FP16.

**Proposed Fix**: Add precision constraint profiles to `ane-knowledge` per ANE version; enhance `VersionedCompiler` to apply precision constraints during SIR construction; add precision-aware casting pass in `ane-passes` for cross-version compatibility.

### Issue #31: ane-constraints-docs Has Constraint Categories Not Yet in ane-knowledge

**Severity**: Medium
**Component**: ane-knowledge
**Discovered**: 2026-04-28

The `ane-constraints-docs` directory contains several constraint categories that have no corresponding representation in the `ane-knowledge` crate:
1. **DMA transfer constraints** — limitations on data movement between ANE and main memory
2. **Neural core scheduling constraints** — rules for distributing work across multiple neural cores
3. **Power state constraints** — ANE behavior under different power states (performance vs. efficiency)
4. **Thermal constraints** — ANE throughput reduction under thermal pressure

**Impact**: Compiler cannot make DMA-aware scheduling decisions; no multi-core utilization optimization; no power/thermal-aware compilation mode.

**Proposed Fix**: Add constraint category structures for each missing area; implement DMA, scheduling, power, and thermal constraint specifications; add query interfaces for passes; consider a "power-aware" compilation mode for ane-cli.
