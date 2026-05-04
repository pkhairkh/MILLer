# Changelog

## Current Status — 2026-05-04

- **1297 tests passing**, 0 failures
- IR Cleanliness Score: 89%
- 0 clippy warnings, 0 errors
- **4 open issues** (0 CRITICAL, 2 HIGH, 1 MEDIUM, 1 LOW) — see [ISSUES.md](ISSUES.md)
- **4 open tasks** (T-58, T-59, T-61, T-66) — see [TASKS.md](TASKS.md)

Audit details: [docs/audit/tabula-rasa-v3.md](docs/audit/tabula-rasa-v3.md)
Violation report: [docs/audit/ane-violations.md](docs/audit/ane-violations.md)

---

## 2026-05-04 — Validation & Code Quality Sprint

### Resolved (T-64, T-60, T-81)

| Task | Description | Key Change |
|------|-------------|------------|
| T-64 | Centralize Palette Bit-Width Validation | Moved `validate_palette_bits()`, `VALID_PALETTE_BITS`, and `clamp_to_valid_palette_bits()` to `ane-ir::ane_layout`; updated 3 call sites (`palettize_weights.rs`, `lut_projection.rs`, `task_spec.rs`) to use centralized versions; fixed doc comments in `sir.rs` to list correct valid set {1,2,3,4,6,8} |
| T-60 | Fix Tile Decomposition Placeholder Zeros | Added `tile_input_dim()` method to `DecompositionContext` for concrete shape resolution; Tile decomposition now uses ctx dimensions when available, avoiding the batch=1 heuristic in `resolve_reshape_zeros()`; fixed final_shape to be at the original input rank (4D) instead of expanded rank (5D); logs warning when ctx is unavailable |
| T-81 | Fix `compat_input_dtype` String Matching | Removed `name.contains("input_ids")` heuristic that could misfire; now trusts the MIR node's declared `dtype` field directly via `mil_dtype_to_compat()`, since the MIR builder correctly assigns `MilDtype::Int32` to input_ids tensors |

### New Tests Added (9 tests)

- `test_validate_palette_bits_valid` — all valid ANE bit-widths accepted (T-64)
- `test_validate_palette_bits_invalid` — invalid bit-widths rejected (T-64)
- `test_clamp_to_valid_palette_bits` — clamping rounds down correctly (T-64)
- `test_tile_decomposition_with_ctx_uses_concrete_shapes` — ctx produces concrete reshape/final shapes (T-60)
- `test_tile_decomposition_without_ctx_uses_placeholders` — no-ctx falls back to 0 placeholders (T-60)
- `test_tile_input_dim_4d` — `DecompositionContext.tile_input_dim()` resolves 4D Tile dims (T-60)
- `test_tile_input_dim_non_4d` — non-4D ranks return None (T-60)
- `test_tile_input_dim_default_ctx` — zero ctx returns None for all dims (T-60)
- `test_compat_input_dtype_no_name_based_override` — name heuristics no longer override dtype (T-81)
- `test_compat_input_dtype_input_ids_with_fp16_returns_fp16` — declared dtype is respected (T-81)
- `test_compat_input_dtype_int32_passthrough` — Int32 dtype maps correctly regardless of name (T-81)

---

## 2026-05-04 — Bridge, FFI & Code Quality Sprint

### Resolved (T-75, T-76, T-77, T-78, T-82, T-83)

| Task | Description | Key Change |
|------|-------------|------------|
| T-75 | Fix FFI `coreml_model_destroy` Unsoundness | Documented allocation contract on `ModelHandleInner`; `coreml_model_load` MUST use `Box::new` so `destroy` can safely call `Box::from_raw`; added contract test |
| T-76 | Add Tests for coreml-ffi::api Module | Added 11 new tests: error type verification for all 5 `CoreMlApi` methods, JSON serialization roundtrips for result types, field-level validation |
| T-77 | Enforce PythonBridge Timeout | Replaced `Command::output()` with `spawn` + poll-based timeout loop; on timeout the child is killed and a timeout error is returned; no new dependencies |
| T-78 | Remove Dead-Code `compare_with_python_bridge` | Removed method, `ComparisonReport`, and `WeightBinComparison` types — all were dead code |
| T-82 | Remove Dead-Code `mir_node_to_compat` | Removed `#[allow(dead_code)]`, gated with `#[cfg(test)]`, added documentation explaining when to use it vs. shape-aware version |
| T-83 | Add BF16→FP16 Edge-Case Tests | Added 7 edge-case tests: NaN, infinity, negative zero, subnormals, max overflow, bulk conversion |

### New Tests Added (19 tests)

- `test_model_destroy_allocated_handle` — verifies Box-allocated handle can be destroyed without UB (T-75)
- `test_coreml_api_version_unavailable` — error type verification for `CoreMlApi::version()` (T-76)
- `test_coreml_api_compile_model_unavailable` — error type verification for `CoreMlApi::compile_model()` (T-76)
- `test_inspect_model_structure_unavailable` — error type verification for `inspect_model_structure()` (T-76)
- `test_inspect_compute_plan_unavailable` — error type verification for `inspect_compute_plan()` (T-76)
- `test_model_structure_result_serialization` — JSON roundtrip for `ModelStructureResult` (T-76)
- `test_compute_plan_result_serialization` — JSON roundtrip for `ComputePlanResult` (T-76)
- `test_model_structure_result_empty` — empty result structure validation (T-76)
- `test_compute_plan_result_unavailable` — unavailable result structure validation (T-76)
- `test_op_placement_all_compute_units` — CPU/GPU/ANE compute unit coverage (T-76)
- `test_function_structure_fields` — `FunctionStructure` field validation (T-76)
- `test_state_declaration_dtype_field` — `StateDeclaration` dtype field validation (T-76)
- `test_bf16_to_fp16_nan_preservation` — quiet + signaling NaN (T-83)
- `test_bf16_to_fp16_infinity_preservation` — +Inf and -Inf (T-83)
- `test_bf16_to_fp16_negative_zero` — signed zero preservation (T-83)
- `test_bf16_to_fp16_subnormal_handling` — subnormal/flush-to-zero behavior (T-83)
- `test_bf16_to_fp16_max_finite_value` — overflow to +Inf (T-83)
- `test_bf16_to_fp16_bulk_conversion` — full `convert_bf16_to_fp16` pipeline test (T-83)

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
