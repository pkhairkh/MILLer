# Changelog

## Current Status — 2026-05-04

- **1270 tests passing**, 0 failures
- IR Cleanliness Score: 89%
- 0 clippy warnings, 0 errors
- **8 open issues** (0 CRITICAL, 2 HIGH, 4 MEDIUM, 2 LOW) — see [ISSUES.md](ISSUES.md)
- **6 open tasks** (T-58, T-59, T-60, T-61, T-64, T-66) — see [TASKS.md](TASKS.md)

Audit details: [docs/audit/tabula-rasa-v3.md](docs/audit/tabula-rasa-v3.md)
Violation report: [docs/audit/ane-violations.md](docs/audit/ane-violations.md)

---

## 2026-05-04 — Bridge Model Leakage & Code Quality Sprint

### Resolved (T-70, T-72, T-73, T-74, T-79, T-80, T-84, T-85)

| Task | Description | Key Change |
|------|-------------|------------|
| T-70 | Fix K/V Projection Alias Map Drop | Used `k_proj`/`v_proj` patterns to build separate K/V alias entries; Q/K/V aliases now point to their respective projection nodes |
| T-72 | Fix Palettize Qwen3 Name Heuristics | Added `run_palettize_weights_pass_with_arch()` using `ModelArchitecture` pattern methods instead of hardcoded name checks |
| T-73 | Fix `LM_HEAD_SHARD_SIZE` Hardcoding | Shard size derived from `vocab_size / TARGET_SHARD_COUNT` (8) instead of hardcoded 19000 |
| T-74 | Fix `resolve_shard` FP16-Only Byte Offsets | Element size derived from `data.len() / total_elements` instead of hardcoded 2; added byte-range overflow guard |
| T-79 | Log Warning When SafetensorsResolver Is Empty | Added `log::warn!()` in `from_traced_graph` when all resolution strategies fail |
| T-80 | Fix Fill Op `input_names()` Empty Vec | `input_names()` now returns `vec![format!("{}_shape", name)]` for Fill ops |
| T-84 | Replace `eprintln!` With `log::warn!` | Replaced in `ane_hw_limits.rs::AneHwLimits::a12()`; added `log` dependency to `ane-ir` |
| T-85 | Gate Deprecated `kv_cache_rewrite` | Module gated behind `deprecated-kv-cache-rewrite` feature flag in `ane-passes`; not compiled by default |

### New Tests Added (7 tests)

- `test_palettize_with_explicit_qwen3_architecture` — verifies architecture-aware palettization with Qwen3
- `test_palettize_with_generic_architecture` — verifies Generic architecture with GPT-2-like patterns
- `test_resolve_shard_weight` — updated for dynamic shard size derivation (T-73)
- `test_resolve_shard_weight_f32` — verifies F32 shard byte offsets (T-74)
- `test_resolve_shard_qwen3_vocab` — regression test for Qwen3-0.6B vocab size (T-73)
- Updated Fill `input_names()` test to verify shape input name (T-80)
- `kv_cache_rewrite` tests now gated behind `deprecated-kv-cache-rewrite` feature (T-85)

---

## 2026-05-04 — Placement & Classification Integrity Sprint

### Resolved (T-67, T-65, T-68, T-69, T-71)

| Task | Description | Key Change |
|------|-------------|------------|
| T-67 | Fix CPU_ONLY_OPS name mismatches (CRITICAL) | `"negative"`→`"neg"`, removed dead entries, added `"round"`, moved MILNeg to None |
| T-65 | Unify CPU-only classification | Added `is_cpu_only_unified()`, placement validator checks `default_engine()==None` first |
| T-68 | Fix `extract_whdc()` NCHW dimensional swap | Rank-4 NCHW: `(shape[3], shape[2], 1, shape[1])` instead of CDHW |
| T-69 | Wire pooling kernel size validation | `kernel_size` validated against max_pooling_kernel_dim=27 |
| T-71 | Fix Float64 element_size=4→8 | Split match arm: `Float32 => 4, Float64 => 8` |

### New Tests Added (15 tests)

- `test_cpu_only_covers_all_default_engine_none` — verifies all MirOp None-branch ops are in CPU_ONLY_OPS
- `test_t67_fixed_names_in_cpu_only` — verifies `"neg"` and `"round"` are CPU-only
- `test_t67_removed_names_not_in_cpu_only` — verifies removed dead-code entries are gone
- `test_extract_whdc_rank1` through `test_extract_whdc_regression_nchw_channels_vs_batch` — 9 tests for NCHW dimension extraction
- `test_pooling_kernel_size_within_limit`, `test_pooling_kernel_size_exceeds_limit`, `test_pooling_kernel_size_zero_rejected`, `test_pooling_kernel_size_large_rejected` — 4 tests for pooling kernel validation
- Expanded `test_coreml_data_type_element_size` to cover all 12 CoreMlDataType variants

---

## 2026-05-04 — Tabula Rasa v3 Audit Cycle

### Resolved (T-36 through T-57)

| Task | Description | Key Change |
|------|-------------|------------|
| T-36 | Parameterize model-specific constants | Added `ModelArchConfig` / `ModelArchitecture` |
| T-37 | SIR→AIR roundtrip tests | 14 roundtrip tests with Qwen3-0.6B dimensions |
| T-38 | ToProto trait for MirOp + MirOpCompat | Unified 167-variant mapping |
| T-39 | Constexpr* MirOpCompat variants | 7 variants for palettized weight emission |
| T-40 | V17 (M1) → A14 family mapping | M1 is A14-class, not A18 |
| T-41 | cargo fmt + clippy --fix | 52 files reformatted |
| T-42 | Chip comment errors | Corrected A11≠M1, A12≠M2, A14≠M3 |
| T-43 | Proto panic!() → Result | `ProtoValidationError` return type |
| T-44 | too_many_arguments refactor | `DecompositionEnv` + `DecodeWeights` structs |
| T-45 | Deprecated kv_cache_rewrite → pub(crate) | Prevents external access |
| T-46 | Shared shape_ops module | MILTile bug fix + 30+ bridge variants |
| T-47 | 4 ops with PE engine but no converter | Moved to None; added to CPU_ONLY_OPS |
| T-48 | Palettize pass is no-op | Added palette_bits field with validation |
| T-49 | ~30 missing CPU_ONLY_OPS | Added (with name mismatches, see I-41/I-42) |
| T-51 | ReduceMin non-FP guard | supports_reducemin_all_dtypes() |
| T-52 | A17 family + E4M3 fix | V11→A17 remapped |
| T-53 | HW tensor dim limits enforced | validate_tensor_dims() in placement pipeline |
| T-54 | panic→bail in legality_rewrite | Result-based error propagation |
| T-56 | ModelArchConfig::default() deprecation | qwen3_0_6b() factory |
| T-57 | Bridge Qwen3 architecture fallback | log::warn!() + deprecation warnings |
| T-58 | A13 broadcast FP16-only guard | Verified correct; A13 excluded from FP16-only |
| T-62 | Conv kernel_d/stride validation | Implemented per-family limits |
| T-63 | Zero-channels interleave bypass | Changed to if-let-Some pattern |

### Open Issues from v3 Audit (I-41 through I-60)

| Priority | Issues | Key Findings |
|----------|--------|--------------|
| CRITICAL | I-41, I-42 | CPU_ONLY_OPS name mismatches — MILNeg passes CPU-only gate |
| HIGH | I-43 through I-49 | extract_whdc swap, pooling discard, Float64 size, K/V alias drop, palettize heuristics, shard hardcodes, FP16 assumption |
| MEDIUM | I-50 through I-55 | FFI unsoundness, zero test coverage, timeout not enforced, dead code stub, empty resolver, Fill input_names |
| LOW | I-56 through I-60 | String matching, dead code, BF16 edge cases, eprintln, deprecated module |

Full details in [ISSUES.md](ISSUES.md).

---

## 2026-05-03 — Tabula Rasa v1/v2 Audit Cycle

Resolved 40 issues (I-01 through I-40): three-way source alignment, CPU-only hard gate, A13 family mapping, interleave/dtype/matmul/pad validators, reshape panic→Result, zero-dim validation, modulo-1 logic bug, SDPA compat fields, ArgMinMax A18 guard, 153 shape_inference tests, 62 staticize tests, MilDtype expansion.

Key infrastructure added: `AneFamily::A13`, `mil_op_name()`, `is_cpu_only()`, `PlacementContext`, `validate_matmul_constraints()`, `validate_pad_constraints()`, `resolve_reshape_zeros()`.
