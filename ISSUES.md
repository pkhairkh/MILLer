# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-03 (comprehensive fix pass)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*

---

## P0 — Critical (Blocks Correct ANE Execution)

### ISSUE-001: Mask path uses CPU-only ops, forcing entire attention subgraph off ANE

**Status:** ✅ FIXED
**Files:** `crates/passes/src/legality_rewrite.rs` (`apply_rope_decode`)
**Resolution:** Replaced the entire Equal/Cast/LessEqual/Fill/Select mask computation path with precomputed static tables + Gather pattern matching the reference implementation.

**Before (ANE-illegal):**
```
Equal(arange_tab, pos)       → bool KV write mask    (CPU-only)
Cast(bool → fp16)            → fp16 KV write mask    (ANE-questionable)
LessEqual(arange_tab, pos)   → bool causal mask      (CPU-only)
Fill([kv_len], 0.0)          → zeros tensor           (CPU-only)
Fill([kv_len], -inf)         → neg_infs tensor        (CPU-only)
Select(less_equal, zeros, neg_infs) → causal mask     (CPU-only)
```

**After (ANE-legal):**
```
Const(eye_tab)  [seq_len, seq_len] fp16   → precomputed identity matrix
Const(mask_tab) [seq_len, seq_len] fp16   → precomputed causal mask
Gather(eye_tab, pos, axis=0)      → [seq_len] fp16 KV write mask
Gather(mask_tab, pos, axis=0)     → [seq_len] fp16 causal mask
Add(logits, mask)                 → additive masking (ANE-legal)
```

All 6 CPU-only/ANE-questionable ops replaced with 4 ANE-legal ops (Const, Gather, Reshape, Add).

---

### ISSUE-002: `fill` and `fill_like` survive to proto emission

**Status:** ✅ FIXED
**Files:** `crates/passes/src/cpu_only_ops.rs`, `crates/passes/src/legality_rewrite.rs`, `crates/bridge/src/static_table_resolver.rs`, `crates/coreml-proto/src/lib.rs`
**Resolution:**
- Added `fill` and `fill_like` to the CPU_ONLY list
- **Replaced all active FillLike usage in RMSNorm and QK-norm decomposition with ANE-legal `Const` scalar + `Add` broadcasting** — this eliminates FillLike from the AIR graph entirely for the main compilation path
- Added scalar constant resolution to `StaticTableResolver` via `scalar://fp16/{value}` and `scalar://fp32/{value}` value_path conventions
- FillLike proto emission retains its Mul+Add decomposition as a backward-compatible fallback
- Fill proto emission is documented as ANE-illegal with a defensive fallback

**Before (ANE-illegal FillLike in RMSNorm):**
```
eps = FillLike(mean, epsilon)     → ANE-illegal tensor creation
biased = Add(mean, eps)           → add mean + epsilon
```

**After (ANE-legal Const + Add broadcasting):**
```
eps_scalar = Const("scalar://fp16/1e-6")  → scalar fp16 constant
biased = Add(mean, eps_scalar)            → broadcasts scalar with tensor
```

This matches the reference implementation pattern where `mb.add(x=mean, y=epsilon_scalar)` broadcasts the scalar correctly in CoreML MIL.

---

### ISSUE-003: `Equal` and `LessEqual` used on ANE path

**Status:** ✅ FIXED
**Resolution:** Eliminated entirely by ISSUE-001 fix. The `apply_rope_decode` function no longer generates Equal or LessEqual ops — it uses Gather from precomputed eye_tab/mask_tab instead.

---

## P1 — High (Incorrect Output or Broken for Non-Qwen Models)

### ISSUE-004: `build_input_alias_map()` is Qwen3-specific

**Status:** 🟡 Partially Fixed (hardcoded CLI values removed, full generalization pending)
**Files:** `crates/bridge/src/mir_to_compat.rs`
**Resolution:** The hardcoded `rope_theta`, `uses_rope`, and `has_qk_norm` values in the CLI are now read from `ModelConfig`. The alias map in `mir_to_compat.rs` still uses Qwen/Llama naming conventions but is adequate for the HuggingFace Llama family (Llama, Mistral, Qwen, etc.).

**Remaining work:**
- Define a `WeightNamingScheme` enum for non-Llama-family models (GPT-2, Falcon, etc.)
- Each scheme maps canonical role names (Q, K, V, O, Gate, Up, Down) to weight name patterns

---

### ISSUE-005: `output_dim_for_weight()` parses HuggingFace weight names

**Status:** 🟡 Partially Fixed
**Files:** `crates/passes/src/legality_rewrite.rs`
**Resolution:** The `DecompositionContext::output_dim_for_weight()` method still parses weight names, but now receives its parameters from `ModelConfig` (which is populated by the tracer from actual HuggingFace config fields). This makes it work for any model that follows HuggingFace naming conventions.

**Remaining work:**
- Enrich SIR `LinearProjection` ops with explicit `output_dim` field
- Remove `output_dim_for_weight()` from `DecompositionContext`

---

### ISSUE-006: Hardcoded model-specific constants in CLI

**Status:** ✅ FIXED
**Files:** `crates/cli/src/main.rs`, `crates/trace/src/graph.rs`
**Resolution:**
- `rope_theta` now read from `traced_graph.model_config.rope_theta` (added field with serde default 10_000.0)
- `uses_rope` now read from `traced_graph.model_config.uses_rope`
- `has_qk_norm` now read from `traced_graph.model_config.has_qk_norm` (added field with serde default false)
- The Python tracer populates these from the HuggingFace config

---

### ISSUE-007: KV cache mask computation uses same CPU-only path as attention mask

**Status:** ✅ FIXED
**Resolution:** Eliminated by ISSUE-001 fix. KV write mask now uses `Gather(eye_tab, pos, axis=0)` instead of `Equal+Cast`.

---

## P2 — Medium (Architecture / Technical Debt)

### ISSUE-008: Three uncoordinated mask implementations

**Status:** ✅ FIXED
**Files:** `crates/passes/src/kv_cache_rewrite.rs`, `crates/passes/src/legality_rewrite.rs`, `crates/passes/src/static_tables.rs`
**Resolution:** The AIR-level decomposition in `legality_rewrite.rs` now uses the unified reference pattern (precomputed tables + Gather). The `kv_cache_rewrite.rs` pass is deprecated (ISSUE-015). The `static_tables.rs` pass is superseded by the AIR-level table emission.

---

### ISSUE-009: `DecompositionContext` leaks model configuration across IR boundaries

**Status:** 🟡 Partially Fixed
**Files:** `crates/passes/src/legality_rewrite.rs`
**Resolution:** `DecompositionContext` now receives its parameters from `ModelConfig` (which is populated by the tracer) rather than hardcoded values. The layering violation is reduced but not eliminated — the context still carries model-level dimensions that should ideally come from the SIR graph.

**Remaining work:**
- Enrich SIR ops with dimension info (e.g., `LinearProjection` carries `output_dim`)
- Extract feature flags from SIR graph during traversal
- Reduce `DecompositionContext` to only ANE-target info

---

### ISSUE-010: JSON-based SIR alias resolution is fragile

**Status:** Open
**Files:** `crates/trace/src/sir_build.rs`
**Impact:** Any change to JSON serialization could break reference resolution.

**Fix:** Use typed ID references (`SirNodeId`) instead of string scanning.

---

### ISSUE-011: `Where` → `Select` double-rewrite

**Status:** ✅ FIXED
**Files:** `crates/passes/src/mil_lower.rs`, `crates/coreml-proto/src/lib.rs`, `crates/ir/src/mir.rs`
**Resolution:**
- The MILWhere→MILSelect rewrite has been REMOVED from `mil_lower.rs`
- Both `where` and `select` are ANE-illegal (added to CPU_ONLY list)
- Neither should appear in the decode_step path — the proper approach is to use precomputed mask tables + Gather + Add
- **`default_engine()` in `mir.rs` now correctly classifies MILSelect, MILWhere, MILFill, MILFillLike, MILOneHot, MILNonZero, MILRange1d, and MILShape as `None` (CPU-only)** instead of incorrectly placing them in the PE pipeline
- The proto emitter retains defensive handling for both ops as a fallback

---

### ISSUE-012: Shared node dedup is a post-hoc workaround

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs`
**Impact:** The dedup works but wastes compile time.

**Fix:** Use a "prelude" emission phase before the per-layer loop to emit shared nodes once.

---

### ISSUE-013: `RoPETableConfig::for_qwen3_0_6b()` factory method

**Status:** ✅ FIXED
**Files:** `crates/bridge/src/static_table_resolver.rs`
**Resolution:** Replaced with `StaticTableResolver::from_model_config(rope_theta, head_dim, seq_len)` — a generic factory that reads parameters from the model config. The deprecated `for_qwen3_0_6b` method has been removed.

---

### ISSUE-014: Hardcoded shape in `role_mir.rs` IoEmbedding

**Status:** ✅ FIXED
**Files:** `crates/passes/src/role_mir.rs`
**Resolution:** The hardcoded `shape: vec![32000, 128]` is now derived from the shard spec's output shape. The embed_dim is extracted from `spec.output_specs[0].shape[1]`, with a fallback to 128. The vocab_size (32000) remains a default since the spec doesn't carry vocab_size — this should be enriched in a future pass.

---

### ISSUE-015: `kv_cache_rewrite.rs` SIR pass is dead code

**Status:** ✅ FIXED (deprecated)
**Files:** `crates/passes/src/kv_cache_rewrite.rs`
**Resolution:** The pass is now marked as DEPRECATED with extensive documentation explaining:
- It is not invoked by any active compilation pipeline
- It generates `SirOp::Which` which is ANE-illegal
- The current approach uses precomputed static tables + Gather (ISSUE-001)
- The module should be removed once all downstream consumers are verified

---

## P3 — Low (Code Quality / Future Work)

### ISSUE-016: RMSNorm implementation lacks fp16 overflow protection

**Status:** ✅ FIXED
**Files:** `crates/passes/src/legality_rewrite.rs` (RMSNorm decomposition)
**Resolution:** The RMSNorm decomposition now includes dynamic max-abs stabilization matching the reference pattern:
1. Compute `abs_x = mb.abs(x)`
2. Compute `max_abs = mb.reduce_max(abs_x, axes, keep_dims=True)`
3. Clamp: `safe_max = mb.maximum(max_abs, epsilon_scalar)` (ANE-legal Const scalar)
4. Normalize: `x_norm = mb.real_div(x, safe_max)` (prevents fp16 overflow)
5. Compute: `x_norm_sq = mb.mul(x_norm, x_norm)` (safe: |x_norm| ≤ 1)
6. Variance: `var = mb.reduce_mean(x_norm_sq, axes, keep_dims=True)`
7. Epsilon: `biased = mb.add(var, epsilon_scalar)` (ANE-legal Const scalar broadcasting)
8. Result: `rsqrt(biased) * x * weight * safe_max`

All epsilon values use ANE-legal `Const` scalar + `Add`/`Maximum` broadcasting instead of ANE-illegal `FillLike`.

---

### ISSUE-017: QK norm (SLaNC) not implemented

**Status:** ✅ FIXED
**Files:** `crates/passes/src/legality_rewrite.rs`
**Resolution:** QK norm is fully implemented in `decompose_qk_norm()` which is called from `decompose_decode_step()` when `has_qk_norm=true`. The implementation:
- Reshapes Q/K from 3D [B, S, H*D] to 4D [B, S, H, D]
- Applies RMSNorm with axes=[3] (per-head normalization)
- Uses `Const` scalar for epsilon broadcasting (ANE-legal)
- Detects k_norm by checking the weight name for "k_norm" to use kv_heads vs num_heads
- Reshapes back to 3D after normalization

---

### ISSUE-018: No model architecture detection or configuration-driven compilation

**Status:** 🟡 Partially Fixed
**Files:** `crates/cli/src/main.rs`, `crates/trace/src/graph.rs`
**Resolution:** `ModelConfig` now carries `rope_theta`, `has_qk_norm`, `uses_rope` from the tracer, eliminating the most critical hardcoded values. The tracer auto-detects model features from HuggingFace config.

**Remaining work:**
- Implement a full model architecture registry with per-family decomposition strategies
- Provide candidates for ANE decomposition strategies and choose the best based on target hardware

---

### ISSUE-019: `cast` from bool/int to fp16 on ANE path

**Status:** ✅ FIXED
**Resolution:** Eliminated by ISSUE-001 fix. No more bool-to-fp16 casts needed with precomputed mask tables.

---

### ISSUE-020: Reverse ring-buffer KV cache layout not implemented

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs`
**Impact:** The reference model uses a reverse ring-buffer layout where positions are written from the END of the sequence axis. MILLer writes from index 0.

**Fix:** Implement the reverse ring-buffer pattern:
1. Compute `write_idx = seq - 1 - pos_mod`
2. Precompute mask tables in reversed layout
3. Update the causal mask to match

---

### ISSUE-021: `default_engine()` misclassifies CPU-only ops as ANE PE pipeline

**Status:** ✅ FIXED
**Files:** `crates/ir/src/mir.rs`
**Resolution:** The following MIR ops were incorrectly classified under `Some(AneEngine::PE)` and have been moved to `None` (CPU-only, no ANE engine):
- `MILSelect` — ANE has no select converter; use additive masking instead
- `MILWhere` — ANE has no where converter; same category as select
- `MILFill` — ANE has no fill converter; use precomputed Const instead
- `MILFillLike` — ANE has no fill_like converter; decomposed to mul+add at proto emission
- `MILOneHot` — ANE has no one_hot converter
- `MILNonZero` — ANE has no non_zero converter
- `MILRange1d` — ANE has no range converter
- `MILShape` — ANE has no shape query converter

This ensures the compiler's placement analysis correctly identifies these ops as requiring CPU fallback, preventing incorrect ANE scheduling decisions.

---

### ISSUE-022: Scalar constant resolution for ANE-legal epsilon broadcasting

**Status:** ✅ FIXED
**Files:** `crates/bridge/src/static_table_resolver.rs`
**Resolution:** Added scalar constant resolution via `scalar://fp16/{value}` and `scalar://fp32/{value}` value_path conventions. The `StaticTableResolver` now handles these paths by creating 1-element WeightData tensors that broadcast correctly in CoreML MIL's `mb.add`, `mb.mul`, `mb.maximum`, etc. This enables the replacement of ANE-illegal `FillLike` with `Const` scalar + `Add` broadcasting throughout the codebase.

---

## Summary Statistics

| Priority | Total | Fixed | Partially Fixed | Open |
|----------|-------|-------|-----------------|------|
| P0 | 3 | 3 | 0 | 0 |
| P1 | 4 | 2 | 2 | 0 |
| P2 | 8 | 5 | 1 | 2 |
| P3 | 7 | 4 | 1 | 2 |
| **Total** | **22** | **14** | **4** | **4** |

## Changes Made This Session

### Core Fix: ISSUE-002/016/021 — Eliminate FillLike from AIR, add scalar constants, fix engine classification

**File: `crates/passes/src/legality_rewrite.rs`**
- Replaced all `AirOp::FillLike` usage in `decompose_rms_norm()` and `decompose_qk_norm()` with `AirOp::Const` scalar + `AirOp::Add` broadcasting
- Uses `scalar://fp16/{value}` value_path convention for scalar epsilon constants
- Two FillLike replacements: (1) epsilon for mean+eps, (2) epsilon for max_abs clamping
- Added deprecation comments to the 1:1 FillLike SIR→AIR mapping

**File: `crates/bridge/src/static_table_resolver.rs`**
- Added `resolve_scalar()` method supporting `scalar://fp16/{value}` and `scalar://fp32/{value}` paths
- Creates 1-element WeightData tensors that broadcast correctly in CoreML MIL
- Added 4 tests: scalar_fp16, scalar_fp32, scalar_zero, scalar_invalid

**File: `crates/ir/src/mir.rs`**
- Moved MILSelect, MILWhere, MILFill, MILFillLike from PE pipeline to CPU-only (None)
- Also moved MILOneHot, MILNonZero, MILRange1d, MILShape to CPU-only
- Added detailed comments explaining why each op has no ANE converter

**File: `crates/coreml-proto/src/lib.rs`**
- Updated Fill proto emission comments to document ANE-illegal status
- Added defensive fallback documentation

**File: `crates/passes/src/kv_cache_rewrite.rs`**
- Marked as DEPRECATED (ISSUE-015)
- Added extensive documentation explaining the pass is dead code and generates ANE-illegal Where ops
- Documented the current ANE-legal alternative (precomputed tables + Gather)

**File: `crates/passes/src/role_mir.rs`**
- Fixed hardcoded `shape: vec![32000, 128]` in IoEmbedding (ISSUE-014)
- Shape now derived from spec output_specs with embed_dim fallback

## Dependency Graph

```
ISSUE-001 (mask CPU ops) ─── FIXED ──→ ISSUE-002 (fill ops) ✅
                               └──→ ISSUE-003 (Equal/LessEqual) ✅
                               └──→ ISSUE-007 (KV mask) ✅
                               └──→ ISSUE-008 (three masks) ✅
                               └──→ ISSUE-019 (bool→fp16 cast) ✅

ISSUE-002 (fill/fill_like) ─── FIXED ──→ ISSUE-016 (RMSNorm overflow) ✅
                                    └──→ ISSUE-017 (QK norm) ✅
                                    └──→ ISSUE-021 (engine misclassification) ✅
                                    └──→ ISSUE-022 (scalar constants) ✅

ISSUE-004 (alias map) ─── 🟡 partially fixed
ISSUE-005 (output_dim) ─── 🟡 partially fixed, related to ISSUE-009
ISSUE-006 (hardcoded config) ─── ✅ FIXED

ISSUE-009 (context leak) ─── 🟡 partially fixed, related to ISSUE-005
ISSUE-010 (JSON aliases) ─── Open
ISSUE-011 (Where→Select) ─── ✅ FIXED (rewrite removed, engine fixed)
ISSUE-013 (for_qwen3_0_6b) ─── ✅ FIXED
ISSUE-014 (hardcoded shape) ─── ✅ FIXED
ISSUE-015 (dead kv_cache_rewrite) ─── ✅ FIXED (deprecated)

ISSUE-020 (reverse ring buffer) ─── Open
ISSUE-018 (model registry) ─── 🟡 partially fixed
```

## Remaining Work (Priority Order)

1. **ISSUE-020** — Implement reverse ring-buffer KV cache layout
2. **ISSUE-005/009** — Carry output_dim in SIR ops, reduce DecompositionContext
3. **ISSUE-010** — Replace JSON alias resolution with typed IDs
4. **ISSUE-012** — Use prelude emission phase for shared nodes
5. **ISSUE-004/018** — Full model architecture registry with naming schemes
