# ♉ AUDIT.md — TABULA RASA Full-Spectrum Diagnostic (v2)

**Date:** 2026-05-04
**Scope:** Full repository sweep — 12 crates, ~566 tests, 9 constraint documents
**Method:** Automated lint + deep source walk + canon cross-reference + drift analysis
**Prior audit:** 2026-05-03 (all 20 issues I-01 through I-20 resolved)
**Verification:** Source-code spot-check of all CRITICAL/HIGH findings; I-24 retracted as false positive after cross-referencing per-family support matrix and Apple error messages

---

## I. EXECUTIVE JUDGEMENT

The MILLer compiler lattice has matured substantially since the first TABULA RASA audit. All 20 previously identified issues (I-01 through I-20) have been resolved: the three sources of truth (engine assignment, CPU-only list, compat coverage) were aligned, interleave and dtype validators were wired into the pipeline, reshape panics were converted to Result types, and model-specific constants were centralized into `ModelArchConfig`. The codebase now compiles with zero clippy warnings, passes all 566+ tests, and enforces a six-gate placement validation chain covering dtype, interleave, layout, blockwise-scale, asymmetric-quant, and per-op constraints. The `ToProto` trait unified MirOp-to-proto mapping across 167 variants, eliminating ~750 lines of boilerplate, and the `Constexpr*` MirOpCompat variants closed the palettized-weight emission gap.

However, this second audit reveals a new stratum of issues that were invisible to the first pass. Four MirOp variants (`MILSliceUpdate`, `MILReverse`, `MILSlidingWindows`, `MILArgsort`) are assigned `Some(AneEngine::PE)` in `default_engine()` but have no ANEC converter — they map to `MirOpCompat::Unsupported` in the proto emission layer. These ops will pass placement validation as ANE-legal but silently fail at emission time, forcing a CPU fallback with synchronization stalls. This is the same class of bug as the original I-01, but the previous fix only addressed the subset of ops that were *also* on the `CPU_ONLY_OPS` list; ops that were neither CPU-only nor compat-supported fell through the gap. Additionally, the `CPU_ONLY_OPS` HashSet is missing approximately 30 ops from the canonical CPU-only list (including `for`, `call`, `condition`, `yield`, `return`, `negative`, `reciprocal`, `is_finite`, `is_nan`, `one_hot`, and others), meaning these ops lack defense-in-depth protection against accidental ANE placement.

The most dangerous new finding is that the `palettize_weights` pass is a **functional no-op**: it computes `bits` values (which can produce invalid 5-bit and 7-bit widths) but then discards them with `_ = (weight, bits)`. No palette annotation is actually emitted, meaning palettization decisions are silently ignored during compilation. Combined with the still-present Qwen3 default leakage in `ModelArchConfig::default()`, `mir_to_compat.rs` architecture fallback, and `shape_inference.rs` max_seq_len default, the compiler remains correct for its primary Qwen3 target but will produce subtly wrong results for any other model architecture. The drift between the Python bridge and Rust proto-direct emission paths is also untracked: both exist independently with no cross-validation test ensuring they produce structurally equivalent MIL graphs for the same input.

---

## II. ANE-CONSTRAINT VIOLATIONS

### II-A. Ops With ANE Engine but No ANEC Converter (4 ops)

These ops pass `default_engine().is_some()` and are NOT in `CPU_ONLY_OPS`, but map to `MirOpCompat::Unsupported` at emission time. They will pass placement validation as ANE-legal but silently fail during proto emission.

| # | MirOp Variant | Assigned Engine | Compat Status | File:Line | Severity |
|---|---|---|---|---|---|
| 1 | `MILSliceUpdate` | `Some(PE)` | Unsupported | `mir.rs:1180` | **CRITICAL** |
| 2 | `MILReverse` | `Some(PE)` | Unsupported | `mir.rs:1182` | **CRITICAL** |
| 3 | `MILSlidingWindows` | `Some(PE)` | Unsupported | `mir.rs:1181` | **CRITICAL** |
| 4 | `MILArgsort` | `Some(PE)` | Unsupported | `mir.rs:1189` | **CRITICAL** |

### II-B. Missing Ops from CPU-ONLY List (~30 ops)

The `CPU_ONLY_OPS` HashSet in `cpu_only_ops.rs` is missing entries from the canonical ANE CPU-only list. These ops lack defense-in-depth protection: if any future code path accidentally assigns them an ANE engine, the validator will not catch it.

| Category | Missing Ops | Severity |
|---|---|---|
| Control flow | `for`, `call`, `condition`, `yield`, `return` | HIGH |
| Shape query | `shape`, `rank`, `size`, `dimension_size` | HIGH |
| Type check | `is_finite`, `is_infinite`, `is_nan` | MEDIUM |
| Elementwise | `negative`, `reciprocal`, `reverse_square_root`, `rint`, `signbit` | MEDIUM |
| Transform | `strided_slice_update`, `dynamic_shape_cast`, `reinterpret_cast`, `col_to_im` | MEDIUM |
| Sparse/buffer | `sparse_tensor_storage`, `materialize_sparse_tensor`, `buffer_tensor` | MEDIUM |
| Other | `one_hot`, `dequantize_lut`, `extract`, `from_elements`, `func`, `get_coordinates`, `local_convolution`, `lp_norm`, `prune`, `pruning_metric`, `pruning_structure`, `variable_from_tensor`, `assign_variable`, `placeholder`, `device_hint`, `nf`, `unrealized_fold` | MEDIUM |

### II-C. Family-Version Guard Gaps

| Guard | Canon Requirement | Current Implementation | File | Severity |
|---|---|---|---|---|
| ~~Broadcast FP16-only for A13~~ | ~~A13 also FP16-only~~ | **VERIFIED CORRECT** — A13 excluded from FP16-only (per-family matrix ✅; Apple error msg says "A11/A12" only; constraint-doc §3.2 A13 text has internal error) | `ane_target.rs:43-45` | ~~HIGH~~ **RETRACTED** |
| ReduceMin non-FP | A14+ only | `supports_reducemin_all_dtypes()` exists but NOT enforced in placement validator for MILReduceMin | `placement_validate.rs` | **HIGH** |
| E4M3 | A17+ (LSE_6) | A18 only — V11 (A17 Pro) maps to A16 family, denied E4M3 | `ane_target.rs:89-91` | **HIGH** |
| Square converter | A13Minus vs A14Plus | No per-family dtype validation | `placement_validate.rs` | MEDIUM |

### II-D. Constraint Validators Not Fully Wired

| Constraint | Canon Reference | Status | File | Severity |
|---|---|---|---|---|
| Tensor dimension limits (HW limits) | §3.4 `hal_params` | `validate_tensor_dims()` exists but NOT called from `validate_placement_with_context()` | `ane_hw_limits.rs:148-193` | **HIGH** |
| Conv kernel_d / stride | §4.1 | `validate_conv_constraints()` discards params with `let _ = (kernel_d, stride)` | `op_constraints.rs:37` | MEDIUM |
| Zero-channels bypass | §6.3 | `channels.unwrap_or(0)` trivially passes interleave divisibility check | `placement_validate.rs:244` | MEDIUM |
| Packed10 format | §5 | No MilDtype variant or explicit rejection exists | `ane_layout.rs` | LOW |

### II-E. Palettization Pass Is a No-Op

| Finding | File | Description | Severity |
|---|---|---|---|
| A-12 | `palettize_weights.rs:88-100` | Pass computes `bits` but discards it with `_ = (weight, bits)`. No annotation is emitted. The pass is effectively dead code. | **CRITICAL** |
| A-13 | Multiple | Palette bit-width validation {1,2,3,4,6,8} exists in `lut_projection.rs:151` and `task_spec.rs:937` but NOT in `palettize_weights.rs`. `attention_bits + 2` can produce 7 (invalid). | **HIGH** |

---

## III. CODE-QUALITY FINDINGS

| # | Smell | Location | Suggestion | Severity |
|---|---|---|---|---|
| CQ-1 | 4 `panic!()` in production emission code | `mir_to_proto.rs:865,877,994,1004` | Replace with `anyhow::bail!()` | **HIGH** |
| CQ-2 | 3 `panic!()` in MIL lowering | `mil_lower.rs:943,1412,3320` | Replace with `bail!()` returning LoweringError | **HIGH** |
| CQ-3 | ~20 `.unwrap()` in weight file I/O | `weights.rs:530-652` | Replace with `?` operator for Result propagation | **HIGH** |
| CQ-4 | `panic!()` in legality passthrough | `legality_rewrite.rs:3903,4271` | Replace with `bail!()` for Select/Where/Tile passthrough | MEDIUM |
| CQ-5 | `eprintln!` in library function | `ane_hw_limits.rs:77-80` | Use `log::warn!()` instead | LOW |
| CQ-6 | Deprecated module still compiled | `kv_cache_rewrite` (pub(crate)) | Gate behind feature flag or remove entirely | MEDIUM |
| CQ-7 | `ModelArchConfig::default()` hardcodes Qwen3-0.6B | `common.rs:248-258` | Remove `Default` impl or rename to `qwen3_0_6b()` | **HIGH** |
| CQ-8 | Bridge defaults to Qwen3 architecture | `mir_to_compat.rs:455` | Return error when no architecture provided | **HIGH** |
| CQ-9 | Shape inference defaults to 32768 max_seq_len | `shape_inference.rs:72,566` | Make max_seq_len a required parameter | **HIGH** |
| CQ-10 | Hardcoded 896/768 in test fixtures | `staticize.rs:167`, `sir_build.rs:1753,2204` | Use named constants or ModelArchConfig | MEDIUM |
| CQ-11 | Model-specific precision hazard in code | `precision_policy.rs:247` | Move to knowledge seed file | MEDIUM |
| CQ-12 | Unused variables in test compilation | `mir_engine_test.rs:962`, `coreml-proto/src/lib.rs:7088` | Prefix with `_` or remove | LOW |
| CQ-13 | Unused imports in test files | `pipeline.rs:7-12`, `should_panic.rs:6`, `report_test.rs:6` | Remove unused imports | LOW |
| CQ-14 | Deprecated test still runs | `kv_cache_rewrite.rs:225-310` | Move to ignored or remove | LOW |
| CQ-15 | `cargo fmt` drift (3 files) | `bridge/mir_to_compat.rs`, `ir/mir.rs`, `coreml-proto/src/lib.rs` | Run `cargo fmt --all` | LOW |

---

## IV. BUG REPORT

| # | Symptom | Trigger | Fix Direction | Severity |
|---|---|---|---|---|
| B-1 | `MILSliceUpdate` silently fails at emission | Any model using slice_update op (e.g., in-place KV cache update) | Move to `None` in `default_engine()`; add to `CPU_ONLY_OPS` | **CRITICAL** |
| B-2 | `MILReverse` silently fails at emission | Any model using reverse op | Move to `None` in `default_engine()`; add to `CPU_ONLY_OPS` | **CRITICAL** |
| B-3 | `MILSlidingWindows` silently fails at emission | Any model using sliding_windows op | Move to `None` in `default_engine()`; add to `CPU_ONLY_OPS` | **CRITICAL** |
| B-4 | `MILArgsort` silently fails at emission | Any model using argsort (sort on ANE) | Move to `None` in `default_engine()`; add to `CPU_ONLY_OPS` | **CRITICAL** |
| B-5 | Palettization decisions silently ignored | Any model with LUT/palettized weights | Wire `bits` into weight annotation; add bit-width validation | **CRITICAL** |
| ~~B-6~~ | ~~FP32 broadcast allowed on A13~~ | ~~A13 hardware with non-FP16 broadcast inputs~~ | **RETRACTED** — Per-family support matrix shows A13 broadcast = ✅; Apple error message specifically says "A11/A12"; constraint-doc A13 section text erroneously claims "same broadcast constraints" — code is correct | ~~HIGH~~ **RETRACTED** |
| B-7 | ReduceMin Int8 allowed on A11-A13 | Non-FP ReduceMin on pre-A14 hardware | Add MILReduceMin guard in placement validator | **HIGH** |
| B-8 | E4M3 denied on A17 Pro hardware | V11→A16 family doesn't support E4M3 | Add A17 family or revision-level override | **HIGH** |
| B-9 | Tile reshape zeros resolved incorrectly | Tile with multiple zero placeholders and ctx=None | Use ctx dimensions when available; require ctx for Tile | **HIGH** |
| B-10 | HW tensor dimension limits not enforced | Large tensors pass placement but fail at ANE runtime | Wire `validate_tensor_dims()` into placement validator | **HIGH** |
| B-11 | Sampler Gather forces CPU fallback | Sampler decomposition produces Gather (CPU-only) | Replace Gather with SliceByIndex or mark Sampler CPU-only | MEDIUM |
| B-12 | Attention reshape placeholder zero | Batch dim > 1 with zero placeholder | Use ctx.batch_size when available | MEDIUM |

---

## V. TEST COVERAGE MAP

| Module | # pub fn | # test fn | Est. Coverage | Priority |
|---|---|---|---|---|
| **ir::payload** | 16 | 0 | 0% | 🔴 Critical |
| **ir::shard_desc** | 6 | 0 | 0% | 🔴 Critical |
| **ir::serialize** | 8 | 0 | 0% | 🔴 Critical |
| **lab::session** | 7 | 0 | 0% | 🔴 Critical |
| **lab::harness** | 14 | 0 | 0% | 🔴 Critical |
| **lab::fallback** | 3 | 0 | 0% | 🔴 Critical |
| **passes::state_topology** | 2 | 0 | 0% | 🔴 Critical |
| **passes::knowledge_query** | 1 | 0 | 0% | 🔴 Critical |
| **report::json_report** | 5 | 0 | 0% | 🟠 High |
| **trace::graph** | 3 | 0 | 0% | 🟠 High |
| **lab::host_inspect** | 2 | 0 | 0% | 🟠 High |
| **lab::device_meta** | 4 | 0 | 0% | 🟠 High |
| **lab::run_dir** | 13 | 0 | 0% | 🟠 High |
| **ir::strategy** | 16 | 6 | 38% | 🟠 High |
| **ir::pir** | 12 | 3 | 25% | 🟠 High |
| **bridge::subprocess** | 3 | 1 | 15% | 🟠 High |
| **bridge::safetensors_resolver** | 10 | 8 | 55% | 🟠 High |
| **coreml-emit::emitter** | 5 | 2 | 20% | 🟠 High |
| **passes::canonicalize** | 2 | 3 | 50% | 🟠 High |
| **passes::static_tables** | 1 | 2 | 50% | 🟠 High |
| **passes::role_mir** | 7 | 8 | 40% | 🟠 High |
| **knowledge::snapshot** | 4 | 3 | 60% | 🟡 Medium |
| **passes::palettize_weights** | 1 | 2 | 50% | 🟡 Medium |
| **passes::shard_plan** | 7 | 21 | 75% | 🟡 Medium |
| **bridge::proto_direct** | 6 | 11 | 70% | 🟡 Medium |
| **ir::common** | 14 | 8 | 57% | 🟡 Medium |
| **knowledge::store** | 17 | 9 | 70% | 🟢 Low |
| **passes::legality_rewrite** | 8 | 33 | 85% | 🟢 Low |
| **passes::mil_lower** | 3 | 58 | 100%+ | 🟢 Low |
| **passes::placement_validate** | 3 | 71 | 100%+ | 🟢 Low |
| **bridge::shape_inference** | 4 | 173 | 100%+ | 🟢 Low |

---

## VI. IR CLEANLINESS SCORE

```
┌─────────────────────────────────────────────┐
│  IR CLEANLINESS SCORE                       │
│                                             │
│  SIR ──██████████████████████░░░  90%       │
│  AIR ──█████████████████████░░░░  87%       │
│  MIR ──████████████████████░░░░░  84%       │
│  PIR ──██████████████████████░░░  92%       │
│                                             │
│  OVERALL: ██████████████████████░░  89%       │
│                                             │
│  Deductions from 100%:                      │
│  - 4 ops with PE engine but no ANEC: -4%    │
│  - Palettize pass is no-op: -3%             │
│  - 30 missing CPU_ONLY_OPS entries: -2%     │
│  - panic!/unwrap in production paths: -2%   │
│                                             │
│  Improvements since v1 audit:               │
│  + 20 issues resolved (I-01 through I-20)   │
│  + 1 false positive retracted (I-24)        │
│  + ToProto trait unified mapping             │
│  + Constexpr* compat variants added          │
│  + 6-gate placement validation chain         │
│  + Zero clippy warnings, all tests pass      │
└─────────────────────────────────────────────┘
```

---

## VII. RECOMMENDED SPRINT BACKLOG

Sorted by **impact x urgency** (highest first):

| Rank | Task | Impact | Urgency | Effort | Issue Ref |
|---|---|---|---|---|---|
| 1 | **Fix 4 ops with PE engine but no ANEC converter**: Move MILSliceUpdate, MILReverse, MILSlidingWindows, MILArgsort to `None` in `default_engine()` and add to `CPU_ONLY_OPS` | CRITICAL | NOW | S (0.5d) | I-21 |
| 2 | **Fix palettize_weights no-op**: Wire `bits` value into weight annotation; add {1,2,3,4,6,8} bit-width validation | CRITICAL | NOW | M (1d) | I-22 |
| 3 | **Add ~30 missing ops to CPU_ONLY_OPS**: `for`, `call`, `condition`, `yield`, `return`, `shape`, `rank`, `size`, `dimension_size`, `is_finite`, `is_infinite`, `is_nan`, `negative`, `reciprocal`, `reverse_square_root`, `rint`, `signbit`, `strided_slice_update`, `one_hot`, etc. | HIGH | NOW | S (0.5d) | I-23 |
| ~~4~~ | ~~**Fix broadcast FP16-only to include A13**~~ | ~~HIGH~~ | **RETRACTED** | — | ~~I-24~~ |
| 5 | **Add ReduceMin non-FP dtype guard in placement validator**: Check `target_family.supports_reducemin_all_dtypes()` when dtype is non-FP | HIGH | NEXT | S (0.5d) | I-25 |
| 6 | **Fix E4M3 support for A17 Pro (V11)**: Add A17 family or revision-level override so V11 gets E4M3 capability | HIGH | NEXT | M (1d) | I-26 |
| 7 | **Wire `validate_tensor_dims()` into placement pipeline**: Call `AneHwLimits::for_revision().validate_tensor_dims()` from `validate_placement_with_context()` | HIGH | NEXT | S (0.5d) | I-27 |
| 8 | **Replace `panic!()` in emission and lowering code**: 7 panic!() calls in mir_to_proto.rs and mil_lower.rs | HIGH | NEXT | S (0.5d) | I-28 |
| 9 | **Replace `.unwrap()` in weights.rs**: ~20 unwrap calls in binary format parsing/writing | HIGH | NEXT | M (1d) | I-29 |
| 10 | **Remove Qwen3 `Default` impl for ModelArchConfig**: Force explicit config; rename to `qwen3_0_6b()` factory | HIGH | NEXT | S (0.5d) | I-30 |
| 11 | **Fix Qwen3 architecture fallback in mir_to_compat.rs**: Return error instead of defaulting to Qwen3 | HIGH | NEXT | S (0.5d) | I-31 |
| 12 | **Add tests for ir::payload, ir::shard_desc, ir::serialize**: Three 0%-coverage modules with 30 pub fn total | HIGH | NEXT | L (3d) | I-32 |
| 13 | **Add tests for lab::session, lab::harness, lab::fallback**: Three 0%-coverage critical modules | HIGH | NEXT | L (2d) | I-33 |
| 14 | **Fix Tile decomposition placeholder zeros**: Use ctx dimensions when available; document ctx requirement | MEDIUM | NEXT | S (0.5d) | I-34 |
| 15 | **Add cross-validation test for Python vs Rust emission**: Structural equivalence test for same MIR input | MEDIUM | LATER | M (1d) | I-35 |
| 16 | **Fix Conv constraint discarded params**: Implement kernel_d and stride validation | MEDIUM | LATER | S (0.5d) | I-36 |
| 17 | **Fix zero-channels interleave bypass**: Return `AneConditional` when channels unknown | MEDIUM | LATER | S (0.5d) | I-37 |
| 18 | **Centralize palette bit-width validation**: Single `validate_palette_bits()` in ane_layout | MEDIUM | LATER | S (0.5d) | I-38 |
| 19 | **Unify CPU-only classification**: Derive CPU_ONLY_OPS from `default_engine() == None` | MEDIUM | LATER | M (1d) | I-39 |
| 20 | **Add ReduceMin/ArgMinMax/etc. to MirOpCompat**: Close remaining compat coverage gaps for ops with real ANEC converters | MEDIUM | LATER | M (2d) | I-40 |
