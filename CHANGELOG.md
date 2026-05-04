# Changelog

## Current Status — 2026-05-04

- **1252 tests passing**, 0 failures
- IR Cleanliness Score: 87%
- 2 minor clippy warnings (clone_on_copy, last_on_doubled_ended), 0 errors
- **27 open issues** (2 CRITICAL, 9 HIGH, 11 MEDIUM, 5 LOW) — see [ISSUES.md](ISSUES.md)
- **19 open tasks** (T-67 through T-85) — see [TASKS.md](TASKS.md)

Audit details: [docs/audit/tabula-rasa-v3.md](docs/audit/tabula-rasa-v3.md)
Violation report: [docs/audit/ane-violations.md](docs/audit/ane-violations.md)

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
