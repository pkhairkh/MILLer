# IR Reference

## Canonical v0 Task

**Name:** `linear_proj_slice`
**Family:** `LinearProjection`
**File:** `benchmarks/synthetic/linear_projection_slice.toml`

### Shapes

| Tensor | Shape | Dtype |
|--------|-------|-------|
| Input `x` | [1, 64] | fp16 |
| Weight `W` | [64, 32] | fp16 |
| Bias `b` | [32] | fp16 |
| Output `z` | [1, 32] | fp16 |

### Operation

```
z = linear(x, W, b)
```

Where `linear(x, W, b)` is the canonical `mb.linear` Core ML op, mathematically
equivalent to `x @ W + b`. The MIL emission path uses `mb.linear` (not separate
matmul + add ops). The Rust baseline computes `x @ W + b` as the FP32 reference.

### Artifact Path Semantics

```
artifacts/compile/{run_id}/
  manifest.json          — ArtifactManifest with truth fields
  mir.json               — MIR graph dump
  mlpackage/             — Emitted .mlpackage directory
    {task_name}.mlpackage/
  knowledge/
    update_{task_name}.json  — Knowledge update artifact
```

## IR Stack (Active Path)

The current vertical slice uses a **direct lowering path** that bypasses the full pass pipeline:

```
TOML task spec → SyntheticTaskSpec → SIR graph → MIR graph → Bridge payload → Python MIL

MIL emission uses canonical Core ML ops: mb.linear (FC projections), mb.gelu
(GELU activation), mb.scaled_dot_product_attention (attention, iOS 18+),
mb.reduce_mean (normalization), mb.rsqrt (normalization), mb.real_div (division),
mb.layer_norm (normalization), mb.topk (sampling), mb.gather (indexing),
mb.cos / mb.sin (RoPE positional encoding).
```

Specifically:
1. `task_spec::load_synthetic_task()` parses TOML into `SyntheticTaskSpec`
2. `linear_slice::sir_from_linear_projection()` builds SIR from spec
3. `linear_slice::lower_linear_projection_to_mir()` lowers directly to MIR
4. `LinearProjectionPayload::from_spec()` builds the bridge JSON payload

### Full IR Stack (compile-full CLI path)

The full IR stack is now wired into the `compile-full` CLI subcommand. The actual
pass pipeline ordering and signatures are:

```
SIR → (CanonicalizePass: SIR→SIR) → (StaticizePass: SIR→SIR)
  → (PrecisionPolicyPass: SIR→SIR) → (StateTopologyPass: SIR→SIR)
  → (LegalityRewritePass: SIR→AIR) → (RiskAnnotatePass: AIR→AIR)
  → (ShardPlanPass: &SIR→ShardPlan+PIR) → (MilLowerPass: &AIR+&ShardPlan→Vec<MIR>)
```

Note the key data-flow dependencies:
- CanonicalizePass, StaticizePass, PrecisionPolicyPass, and StateTopologyPass all
  transform SIR in place (pass-through for the current linear projection slice).
- LegalityRewritePass consumes the SIR and produces an AIR graph.
- RiskAnnotatePass annotates the AIR graph with risk scores.
- ShardPlanPass takes a reference to the original SIR to produce a ShardPlan and PIR.
- MilLowerPass takes references to both the AIR graph and the ShardPlan to produce
  one MIR graph per shard.

The active fast-path `compile` command still uses direct SIR→MIR lowering.
When the pass pipeline produces significantly different output from the direct
path, the `compile-full` command enables comparison and validation.

### Deviations from SPEC

- **AIR is skipped** in the fast-path `compile` command: SIR→MIR direct in `linear_slice.rs`
- **Pass pipeline is wired** into the `compile-full` CLI subcommand (8 passes)
- **Shard planning** produces a single-shard result in `compile-full`; multi-shard plans are produced by `compile-full-sharded` which runs the full pass pipeline per shard (Sprint 37)
- **Shard planning + MIR consistency** (S37.3): the compile-full-sharded path now runs ShardPlanPass and MilLowerPass for each shard, so MIR compute_unit_hint matches the per-shard compute unit assignment from the multi-shard plan
- **Risk-based knowledge in multi-shard planning** (S37.4): `build_sharded_plan_from_spec_with_risk_knowledge()` applies both template and risk-based knowledge at the plan-construction level, fixing the previous gap where compile-full-sharded ignored accumulated risk observations
- **PIR** is produced by the `compile-full` pass pipeline

See [SPEC.md](../SPEC.md) section 5 for the full IR specification.

## Structural Verification (Sprint 34)

Host-side inspection now includes structural verification via MLModelStructure,
replacing the previous approach of checking only file existence and Manifest.json
readability.

### Inspection Methods

| Method | Platform | What It Provides |
|--------|----------|-----------------|
| `mlmodel_structure` | macOS + Core ML runtime | Op inventory, function signatures, state declarations, I/O specs |
| `fallback_file_check` | Any | File inventory, weight file count, heuristic op name scanning |
| `none` | Any | No structural inspection performed |

### MIR-vs-Structure Comparison

The `mir_compare` module provides a canonical mapping from MIR op type names
to Core ML MIL op names. This mapping is the single source of truth for the
MIR→MIL correspondence:

- Rust side: `crates/lab/src/mir_compare.rs::mir_to_mil_name()`
- Python side: `python/model_structure.py::compare_mir_vs_structure()`

Both implement the same multiset comparison: for each MIR op, check whether
the corresponding MIL op name appears in the emitted structure. The result
includes:
- `op_fidelity_score` (0-1): fraction of MIR ops matched in the structure
- `missing_ops`: MIR ops not found in the emitted structure
- `extra_ops`: structure ops not expected by the MIR

### When MLModelStructure Is Unavailable

On non-Apple platforms (Linux), MLModelStructure requires the Core ML runtime
which is not available. The `model_structure` bridge command:
1. Attempts `MLModelStructure.load_from_path()`
2. If unavailable, reports the reason (e.g., "Apple Core ML runtime not available")
3. Falls back to `fallback_file_structure()` which provides file-based heuristics
4. The `inspection_method` field in the result explicitly identifies which method was used

This means host inspection on Linux is still weaker than on macOS, but the
weakness is now explicitly labeled rather than hidden.

## SIR→AIR Decomposition (Sprint 36)

All declared SIR ops now have active SIR→AIR decomposition paths in
`LegalityRewritePass`. Previously, `SirOp::AttentionBlock`, `SirOp::DecodeStep`,
`SirOp::RMSNorm`, `SirOp::RoPETransform`, and `SirOp::Sampler` would produce
an error in the legality rewrite pass. They now decompose into sequences of
lower-level AIR ops:

| SIR Op | AIR Decomposition |
|--------|-------------------|
| LinearProjection | Conv1x1AsLinear (canonical mb.linear) |
| AttentionBlock | Conv1x1AsLinear + SliceByIndex + Reshape + Transpose + ScaledDotProductAttention + Conv1x1AsLinear (+ optional QK-norm: LayerNorm/RMSNorm on Q and K before SDPA) |
| DecodeStep | Conv1x1AsLinear + SliceByIndex + StateReadFixed + Reshape + ScaledDotProductAttention + Conv1x1AsLinear + StateWriteFixed |
| RMSNorm | ReduceMean + Rsqrt + ElementWise::Mul + ElementWise::Mul |
| RoPETransform | Cos + Sin + ElementWise::Mul + ElementWise::Mul + ElementWise::Add |
| Sampler | Topk + Softmax + Gather |
| StateRead | StateReadFixed |
| StateWrite | StateWriteFixed |

### Critique Bug 1 Fix (Sprint 36)

`SirOp::LinearProjection` now lowers to `AirOp::Conv1x1AsLinear` (not `AirOp::MatMul`).
This closes the inconsistency where the Python emitter used `mb.linear` (Sprint 31)
but the SIR→AIR path still produced `MatMul`. The full pipeline is now consistent:
SIR → Conv1x1AsLinear → MILLinear → mb.linear.

## AIR→MIR Lowering Coverage (Sprint 36)

All previously "declared but no lowering" MIR ops now have active AIR→MIR lowering
paths in `MilLowerPass`:

| AIR Op | MIR Op |
|--------|--------|
| ScaledDotProductAttention | MIRScaledDotProductAttention |
| SliceByIndex | MILSliceByIndex |
| Gelu | MILGelu |
| Relu | MILCast (approximation — no MILRelu yet) |
| StateReadFixed | MILReadState |
| StateWriteFixed | MILCoremlUpdateState |
| Split | MILSplit |
| Concat | MILConcat |

The previous gap where 7 MIR ops were declared in the enum but had no AIR→MIR
lowering path is now fully closed.
