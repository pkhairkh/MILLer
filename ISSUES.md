# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-03*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*

---

## P0 — Critical (Blocks Correct ANE Execution)

### ISSUE-001: Mask path uses CPU-only ops, forcing entire attention subgraph off ANE

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 2334–2379)
**Impact:** Entire attention subgraph falls off ANE to CPU. This is the single most impactful ANE-legality bug.

**Current implementation:**
```
Equal(arange_tab, pos)       → bool KV write mask    (CPU-only op)
Cast(bool → fp16)            → fp16 KV write mask    (int→fp16 cast, ANE-questionable)
LessEqual(arange_tab, pos)   → bool causal mask      (CPU-only op)
Fill([kv_len], 0.0)          → zeros tensor           (ANE-questionable, not const-folded)
Fill([kv_len], -inf)         → neg_infs tensor        (ANE-questionable, not const-folded)
Select(less_equal, zeros, neg_infs) → causal mask     (correct but inputs are CPU-only)
```

**Reference implementation** (from `pkhairkh/qwen3-coreml-palettized`):
```python
# Precomputed at BUILD TIME:
mask_tab = np.full((seq, seq), np.float16(-np.inf))   # (4096, 4096) fp16
for idx in range(seq):
    mask_tab[idx, seq - (idx + 1):] = np.float16(0.0) # Reversed layout
eye_tab = np.eye(seq, dtype=np.float16)                # (4096, 4096) fp16

# At RUNTIME (all ANE-legal ops):
kv_mask_row = mb.gather(x=eye_tab, indices=write_idx, axis=0)    # gather: ANE-legal
mask_row = mb.gather(x=mask_tab, indices=pos_mod, axis=0)        # gather: ANE-legal
mask = mb.reshape(x=mask_row, shape=[1, 1, 1, seq])              # reshape: ANE-legal
mask_write = mb.reshape(x=kv_mask_row, shape=[1, 1, seq, 1])    # reshape: ANE-legal
mask_keep = mb.sub(x=1.0, y=mask_write)                          # sub: ANE-legal
logits = mb.add(x=logits, y=mask)                                # add: ANE-legal ← THE masking op
```

**Key differences:**
| Aspect | Current | Reference |
|--------|---------|-----------|
| Mask storage | Runtime computation via Equal/LessEqual/Fill | Precomputed static fp16 tables |
| Ops used | Equal, LessEqual, Fill, Select, Cast | Gather, Const, Reshape, Sub, Add |
| ANE legality | 4+ CPU-only ops | All ANE-legal |
| Memory | Small runtime | Larger static table (seq² × fp16) |
| KV mask | Same Equal+Cast pattern | Gather from eye_tab (identity matrix) |

**Fix:** Replace the entire mask computation path with the reference pattern:
1. Precompute `mask_tab` and `eye_tab` as static constant tensors at compile time
2. Use `Gather` to select the correct row by position
3. Use additive `Add(logits, mask)` for masking instead of `Select(condition, zeros, neg_infs)`

---

### ISSUE-002: `fill` and `fill_like` survive to proto emission

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 2356–2368), `crates/coreml-proto/src/lib.rs` (line 3643)
**Impact:** `fill`/`fill_like` are ANE-problematic. When `shape` is const and `value` is const, they should be const-folded (eliminated entirely). When shapes are dynamic, they force CPU fallback.

**Current state:**
- The previous "fix" (commit 230c5e8) incorrectly stopped replacing `MILFill` with `MILConst`, claiming MILFill is "ANE-supported (PE engine)". This is wrong.
- The proto emitter itself documents: `fill_like is ANE-illegal` (line 3643)
- The reference model never uses `fill` or `fill_like` — all mask constants are precomputed

**Fix:**
1. Replace all `Fill` ops with precomputed constant tensors (matching the reference pattern)
2. If dynamic fills are ever needed, decompose to `mul(zeros_like(x), 0) + value` at the AIR level, not at proto emission
3. Remove the `Fill`/`FillLike` proto emission paths once they are no longer needed

---

### ISSUE-003: `Equal` and `LessEqual` used on ANE path

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 2334, 2351)
**Impact:** Both `"equal"` and `"less_equal"` are listed in `CPU_ONLY_OPS` (`crates/passes/src/cpu_only_ops.rs`, lines 144, 148). Using them in the attention mask path guarantees CPU fallback.

**Fix:** Eliminated entirely by ISSUE-001 fix (precomputed mask tables + Gather).

---

## P1 — High (Incorrect Output or Broken for Non-Qwen Models)

### ISSUE-004: `build_input_alias_map()` is Qwen3-specific

**Status:** Open
**Files:** `crates/bridge/src/mir_to_compat.rs` (lines 467–535)
**Impact:** Non-Qwen3 models produce broken alias maps. Any model with different naming conventions (GPT-2's `c_attn`, Falcon's `query_key_value`, Mistral's different convention) will fail silently.

**Current state:** The function's doc comment admits: *"Qwen3-specific: This function hardcodes aliases that match the Qwen3 transformer architecture"*. It contains string matches like:
- `.contains(".self_attn.q_proj.weight")` — Qwen/Llama naming only
- `name == "mlp_silu"` — Qwen3's SiLU gate naming
- `name == "attn_qk"`, `"attn_softmax"`, `"attn_sv"` — synthetic names from Qwen3 decomposition

**Fix:** Generalize to a config-driven alias system:
1. Define a `WeightNamingScheme` enum (HuggingFaceLlama, HuggingFaceGPT2, etc.)
2. Each scheme maps canonical role names (Q, K, V, O, Gate, Up, Down) to weight name patterns
3. The SIR builder records the naming scheme used, and downstream passes use it

---

### ISSUE-005: `output_dim_for_weight()` parses HuggingFace weight names

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 205–234)
**Impact:** Unknown projections silently get `dim=0`, producing wrong shapes. This is a bridge-layer concern leaking into the legality rewrite pass.

**Current state:** The `DecompositionContext::output_dim_for_weight()` method parses weight name strings to infer output dimensions. Returns 0 for any unrecognized weight name.

**Fix:**
1. Pass output dimensions explicitly through the SIR op or task spec
2. The SIR-level `LinearProjection` op should carry its output dimension
3. Remove `output_dim_for_weight()` from `DecompositionContext`

---

### ISSUE-006: Hardcoded model-specific constants in CLI

**Status:** Open
**Files:** `crates/cli/src/main.rs` (lines 4184, 4420–4421)
**Impact:** Only Qwen3 produces correct output. Other models will have wrong RoPE frequencies, missing QK norm, etc.

**Current state:**
```rust
let rope_theta = 1_000_000.0; // Qwen3 default; TODO: read from model config
true,  // uses_rope: Qwen3 uses RoPE in decode
true,  // has_qk_norm: Qwen3 uses QK norm
```

**Fix:** Read all model configuration from `TracedGraph.model_config`:
- `rope_theta` from `config.rope_theta` (already in most HF configs)
- `uses_rope` from `config.rope_scaling` or model architecture detection
- `has_qk_norm` from checking for `q_norm`/`k_norm` weights in the model

---

### ISSUE-007: KV cache mask computation uses same CPU-only path as attention mask

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 1636–1646)
**Impact:** The KV cache write mask (`Equal(arange, pos) → Cast(bool, fp16)`) forces CPU fallback even for the KV cache update path, which is separate from the attention mask.

**Current state:** Same `Equal+Cast` pattern as ISSUE-001 but for the KV write mask. Reference uses `Gather(eye_tab, write_idx)` instead.

**Fix:** Eliminated by ISSUE-001 fix (use precomputed `eye_tab` + Gather for KV write mask).

---

## P2 — Medium (Architecture / Technical Debt)

### ISSUE-008: Three uncoordinated mask implementations

**Status:** Open
**Files:** `crates/passes/src/kv_cache_rewrite.rs`, `crates/passes/src/legality_rewrite.rs`, `crates/trace/src/static_tables.rs`
**Impact:** Maintenance burden, inconsistent ANE behavior, three different patterns for the same operation.

**Current implementations:**
1. `kv_cache_rewrite.rs`: `Where(valid_mask, new, cached)` — SIR level, uses `Where`
2. `legality_rewrite.rs decompose_decode_step`: `Equal+Cast+Fill+Select` — AIR level, CPU-only
3. `static_tables.rs`: Precomputed `mask_tab` constant — SIR level, but not used by decode step

**Fix:** Unify on the reference pattern (precomputed tables + Gather). Remove `kv_cache_rewrite.rs` SIR pass (dead code) and the `static_tables.rs` approach in favor of the precomputed-constant + Gather approach in the AIR decomposition.

---

### ISSUE-009: `DecompositionContext` leaks model configuration across IR boundaries

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 59–83)
**Impact:** The SIR→AIR decomposition pass requires external dimension hints that should come from the SIR graph itself. This violates the IR layering principle.

**Current state:** `DecompositionContext` carries `vocab_size`, `intermediate_size`, `uses_rope`, `has_qk_norm`, `uses_gqa` — model configuration that belongs at the SIR level.

**Fix:**
1. Enrich SIR ops with all needed dimension info (e.g., `LinearProjection` should carry output_dim)
2. Extract model feature flags (uses_rope, has_qk_norm, uses_gqa) from the SIR graph during traversal
3. Remove `DecompositionContext` or reduce it to only ANE-target information (seq_len, batch_size)

---

### ISSUE-010: JSON-based SIR alias resolution is fragile

**Status:** Open
**Files:** `crates/trace/src/sir_build.rs` (lines 257–272)
**Impact:** Any change to JSON serialization or a string value that starts with `"sir_"` could break reference resolution. Also slow (serialize → scan → string substitution).

**Current state:** The SIR builder resolves cross-node references by:
1. Serializing each `SirOp` to JSON
2. Scanning for `"sir_*"` string references
3. Replacing dangling aliases with actual IDs via string substitution

**Fix:** Use typed ID references (e.g., `SirNodeId`) instead of string scanning. Each SIR op should store its dependencies as `SirNodeId` values, not as string patterns that need post-hoc resolution.

---

### ISSUE-011: `Where` → `Select` double-rewrite

**Status:** Open
**Files:** `crates/passes/src/mil_lower.rs` (line 3413), `crates/coreml-proto/src/lib.rs` (line 3501)
**Impact:** The `MILWhere` → `MILSelect` rewrite in `mil_lower.rs` makes the `Where` case in proto emission unreachable. The proto-level rewrite is defensive but adds dead code.

**Fix:** Once the `Where` case is confirmed unreachable in practice, remove it from proto emission (or keep with a clear `unreachable!()` comment). The real fix is to ensure all `Where` ops are rewritten to `Select` at the MIR level.

---

### ISSUE-012: Shared node dedup is a post-hoc workaround

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (lines 727–730)
**Impact:** The dedup masks a real bug: shared AIR nodes are emitted once per layer instead of once globally. The dedup works but wastes compile time generating N copies and discarding N-1.

**Current state:**
```rust
let mut seen_ids: HashSet<String> = HashSet::new();
air_nodes.retain(|node| seen_ids.insert(node.id.0.clone()));
```

**Fix:** Track shared nodes properly in the `sir_to_air` map so they are emitted once and referenced by ID from then on. Use a "prelude" emission phase before the per-layer loop.

---

### ISSUE-013: `RoPETableConfig::for_qwen3_0_6b()` factory method

**Status:** Open
**Files:** `crates/bridge/src/static_table_resolver.rs` (line 108)
**Impact:** Hardcodes Qwen3-0.6B dimensions (theta=1_000_000, head_dim=128).

**Fix:** Replace with a generic factory that reads rope parameters from `ModelConfig`:
```rust
RoPETableConfig::from_model_config(config: &ModelConfig, seq_len: usize)
```

---

### ISSUE-014: Hardcoded shape in `role_mir.rs` IoEmbedding

**Status:** Open
**Files:** `crates/passes/src/role_mir.rs` (line 631)
**Impact:** `shape: vec![32000, 128]` is only correct for Qwen3-0.6B.

**Fix:** Derive embedding shape from task spec (vocab_size × embed_dim from ModelConfig).

---

### ISSUE-015: `kv_cache_rewrite.rs` SIR pass is dead code

**Status:** Open
**Files:** `crates/passes/src/kv_cache_rewrite.rs`
**Impact:** The decode step decomposition in `legality_rewrite.rs` implements its own inline blend pattern, making this pass unreachable for the main compilation path. It also uses a different pattern (`Where` vs `Equal+Cast`).

**Fix:** Remove the dead code path, or integrate it properly with the AIR-level decomposition if a SIR-level KV cache rewrite is desired.

---

## P3 — Low (Code Quality / Future Work)

### ISSUE-016: RMSNorm implementation lacks fp16 overflow protection

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (RMSNorm decomposition)
**Impact:** The reference model uses `rms_norm_dynamic_safe` with dynamic max-abs stabilization to prevent fp16 overflow when activations > 256. MILLer's RMSNorm decomposition may produce NaN for large activations on ANE.

**Reference pattern:**
```python
abs_x = mb.abs(x=x)
maxval = mb.reduce_max(x=abs_x, axes=[-1], keep_dims=True)
max_clp = mb.clip(x=maxval, alpha=2**-14, beta=inf)
z = mb.real_div(x=x, y=max_clp)        # Divide by max first
sq = mb.mul(x=z, y=z)                   # Then square (no overflow)
var = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
eps_eff = mb.real_div(x=eps, y=max_clp) # Double-division avoids forming max²
eps_eff = mb.real_div(x=eps_eff, y=max_clp)
inv_std = mb.rsqrt(x=mb.add(x=var, y=eps_eff))
```

**Fix:** Add dynamic max-abs scaling to the RMSNorm decomposition, matching the reference pattern.

---

### ISSUE-017: QK norm (SLaNC) not implemented

**Status:** Open
**Files:** N/A (missing feature)
**Impact:** Qwen3 uses QK normalization with per-layer learned scales. Without it, the model produces incorrect outputs. The current code sets `has_qk_norm=true` in the context but doesn't implement QK norm in the decomposition.

**Reference pattern:** `rms_norm_slanc_qk` in `rms_norm.py` — adds SLaNC pre-scaling with per-layer learned scales before the RMS computation.

**Fix:** Implement QK norm as a separate pass or as part of the attention decomposition.

---

### ISSUE-018: No model architecture detection or configuration-driven compilation

**Status:** Open
**Files:** `crates/cli/src/main.rs`
**Impact:** The compiler cannot adapt to different model architectures. All configuration is hardcoded for Qwen3.

**Fix:** Implement a model architecture registry that:
1. Detects the model type from HF config (`architectures` field)
2. Selects the appropriate naming scheme, RoPE configuration, norm type, etc.
3. Provides candidates for ANE decomposition strategies and chooses the best based on target hardware

---

### ISSUE-019: `cast` from bool/int to fp16 on ANE path

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs` (line 2341)
**Impact:** Casting from bool or int types to fp16 on the ANE path is conditionally ANE-legal — it may work but often forces CPU fallback.

**Fix:** Eliminated by ISSUE-001 fix (no more bool-to-fp16 casts needed with precomputed mask tables).

---

### ISSUE-020: Reverse ring-buffer KV cache layout not implemented

**Status:** Open
**Files:** `crates/passes/src/legality_rewrite.rs`
**Impact:** The reference model uses a reverse ring-buffer layout where positions are written from the END of the sequence axis (`write_idx = seq-1 - pos`). This means active context always lives in a contiguous suffix, which is more cache-friendly and matches the precomputed mask table layout.

**Current state:** MILLer writes positions starting from index 0, which requires a different mask table layout.

**Fix:** Implement the reverse ring-buffer pattern matching the reference:
1. Compute `write_idx = seq - 1 - pos_mod`
2. Precompute mask tables in reversed layout
3. Update the causal mask to match

---

## Summary Statistics

| Priority | Count | Description |
|----------|-------|-------------|
| P0 | 3 | Blocks correct ANE execution |
| P1 | 4 | Broken for non-Qwen models or incorrect output |
| P2 | 8 | Architecture violations, technical debt |
| P3 | 5 | Code quality, missing features |
| **Total** | **20** | |

## Dependency Graph

```
ISSUE-001 (mask CPU ops) ─── fixes ──→ ISSUE-002 (fill ops)
                               └──→ ISSUE-003 (Equal/LessEqual)
                               └──→ ISSUE-007 (KV mask)
                               └──→ ISSUE-019 (bool→fp16 cast)

ISSUE-004 (alias map) ─── independent
ISSUE-005 (output_dim) ─── related to ISSUE-009
ISSUE-006 (hardcoded config) ─── related to ISSUE-018

ISSUE-008 (three masks) ─── superseded by ISSUE-001 fix
ISSUE-009 (context leak) ─── related to ISSUE-005
ISSUE-010 (JSON aliases) ─── independent

ISSUE-016 (RMSNorm overflow) ─── independent
ISSUE-017 (QK norm) ─── independent
ISSUE-020 (reverse ring buffer) ─── related to ISSUE-001
```

## Recommended Fix Order

1. **ISSUE-001** — Replace mask path with precomputed tables + Gather + Add (fixes 001, 002, 003, 007, 019, 020)
2. **ISSUE-004** + **ISSUE-005** + **ISSUE-006** — Remove Qwen3 hardcoding (fixes 004, 005, 006, 013, 014)
3. **ISSUE-008** + **ISSUE-015** — Clean up dead/uncoordinated mask code
4. **ISSUE-009** — Fix `DecompositionContext` layering violation
5. **ISSUE-016** + **ISSUE-017** — Implement proper RMSNorm and QK norm
6. **ISSUE-010** — Replace JSON alias resolution with typed IDs
7. **ISSUE-018** — Implement model architecture detection
