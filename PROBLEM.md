# Why the current `model.mlpackage` still does not compile/open

Date: 2026-05-03

Scope: Analysis of remaining compilation blockers after the comprehensive fix pass.

## Current Status

The `concat.interleave` parameter blocker (documented in the previous version of this file) has been **FIXED** — the proto emitter now explicitly serializes `interleave=false` for all concat operations.

## Current Hard Blocker: Static Table Resolution

The `apply_rope_decode` function now emits `Const` nodes with `value_path` references to `eye_tab` and `mask_tab` static tables. These tables must be resolved by the `StaticTableResolver` during proto emission.

**Required tables per rope reference:**
- `static_tables/{ref}/cos_tab` — [1, 1, seq_len, head_dim] fp16
- `static_tables/{ref}/sin_tab` — [1, 1, seq_len, head_dim] fp16
- `static_tables/{ref}/eye_tab` — [seq_len, seq_len] fp16
- `static_tables/{ref}/mask_tab` — [seq_len, seq_len] fp16
- `static_tables/{ref}/arange_tab` — [seq_len] int32

The `StaticTableResolver` computes these tables when `ensure_tables_computed()` is called. The CLI must ensure this is called for every unique rope reference before proto emission.

## Previously Fixed Blockers

| Blocker | Status |
|---------|--------|
| `concat.interleave` missing | ✅ Fixed — interleave=false now explicitly serialized |
| `fill_like` ANE-illegal | ✅ Fixed — replaced with Gather from precomputed eye_tab |
| `select` ANE-illegal | ✅ Fixed — replaced with Gather from precomputed mask_tab + Add |
| `Equal`/`LessEqual` CPU-only | ✅ Fixed — eliminated by precomputed tables + Gather |
| `slice_by_index` mask dtype | ✅ Fixed — masks are bool[4] |
| Zero-dimension outputs | ✅ Fixed |

## ANE Operation Audit (Post-Fix)

The decode_step path now uses only these operation types:
- `const` — precomputed static tables and weight references
- `gather` — RoPE cos/sin row lookup, eye_tab/mask_tab row lookup
- `linear`/`matmul` — Q/K/V projections, attention scores, output projection
- `reshape` — 4D head layout, mask broadcasting
- `transpose` — head dimension transposition
- `mul`/`add`/`sub` — elementwise operations, RoPE rotation, residual connections
- `slice_by_index` — per-head extraction from Q/K/V tensors
- `softmax` — attention weight normalization
- `silu` — MLP activation
- `neg` — RoPE rotate-half negation
- `concat` — RoPE rotate-half concatenation, context assembly
- `read_state`/`coreml_update_state` — KV cache read/write

All of these are ANE-legal operations. No CPU-only ops remain in the attention path.
