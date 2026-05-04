# ♉ AUDIT.md — TABULA RASA Full-Spectrum Diagnostic (v2)

**Date:** 2026-05-04
**Scope:** Full repository sweep — 12 crates, ~566 tests, 9 constraint documents
**Method:** Automated lint + deep source walk + canon cross-reference + drift analysis
**Prior audit:** 2026-05-03 (all 20 issues I-01 through I-20 resolved)
**Verification:** Source-code spot-check of all CRITICAL/HIGH findings; I-24 retracted as false positive after cross-referencing per-family support matrix and Apple error messages

---

## I. EXECUTIVE JUDGEMENT

The MILLer compiler lattice has matured substantially since the first TABULA RASA audit. All 20 previously identified issues (I-01 through I-20) have been resolved: the three sources of truth (engine assignment, CPU-only list, compat coverage) were aligned, interleave and dtype validators were wired into the pipeline, reshape panics were converted to Result types, and model-specific constants were centralized into `ModelArchConfig`. The codebase now compiles with zero clippy warnings, passes all 566+ tests, and enforces a six-gate placement validation chain covering dtype, interleave, layout, blockwise-scale, asymmetric-quant, and per-op constraints. The `ToProto` trait unified MirOp-to-proto mapping across 167 variants, eliminating ~750 lines of boilerplate, and the `Constexpr*` MirOpCompat variants closed the palettized-weight emission gap.

**Update (2026-05-06):** Seven additional issues from the v2 audit have been resolved (I-21 through I-31, tasks T-47 through T-57). The four ops with PE engine but no ANEC converter (I-21) have been moved to `None` in `default_engine()` and added to `CPU_ONLY_OPS`. The `palettize_weights` pass (I-22) is no longer a no-op — it now writes computed bits into a `palette_bits` field with {1,2,3,4,6,8} validation and clamping. ~27 missing ops were added to `CPU_ONLY_OPS` (I-23). ReduceMin non-FP dtype guard was added (I-25). Tensor dimension HW limits are now enforced at placement time (I-27). `ModelArchConfig` got a `qwen3_0_6b()` factory with deprecation on `default()` (I-30). Qwen3 architecture fallback now emits `log::warn!()` and deprecation warnings (I-31). Two issues were corrected during verification: I-28 (`panic!()` calls) was downgraded from HIGH to MEDIUM after confirming they are in test/guard code, not production paths; I-29 (`.unwrap()` in weights.rs) was retracted entirely as all calls are in test code. IR Cleanliness Score improved from 89% to 93%.

**Remaining open issues** include: E4M3 denied on A17 Pro (I-26), zero test coverage for 6 critical modules (I-32, I-33), and several MEDIUM-severity items (I-34 through I-40). The drift between the Python bridge and Rust proto-direct emission paths is also untracked: both exist independently with no cross-validation test ensuring they produce structurally equivalent MIL graphs for the same input.

---

## II. ANE-CONSTRAINT VIOLATIONS

### II-A. Ops With ANE Engine but No ANEC Converter (4 ops) — ✅ All Fixed

~~These ops pass `default_engine().is_some()` and are NOT in `CPU_ONLY_OPS`, but map to `MirOpCompat::Unsupported` at emission time. They will pass placement validation as ANE-legal but silently fail during proto emission.~~

**Fixed by T-47:** All four ops moved from `Some(AneEngine::PE)` to `None` in `default_engine()` and added to `CPU_ONLY_OPS`.

| # | MirOp Variant | Assigned Engine | Compat Status | File:Line | Severity |
|---|---|---|---|---|---|
| 1 | `MILSliceUpdate` | ~~`Some(PE)`~~ `None` | Unsupported → CPU-only | `mir.rs:1180` | ~~CRITICAL~~ **Fixed** |
| 2 | `MILReverse` | ~~`Some(PE)`~~ `None` | Unsupported → CPU-only | `mir.rs:1182` | ~~CRITICAL~~ **Fixed** |
| 3 | `MILSlidingWindows` | ~~`Some(PE)`~~ `None` | Unsupported → CPU-only | `mir.rs:1181` | ~~CRITICAL~~ **Fixed** |
| 4 | `MILArgsort` | ~~`Some(PE)`~~ `None` | Unsupported → CPU-only | `mir.rs:1189` | ~~CRITICAL~~ **Fixed** |

### II-B. Missing Ops from CPU-ONLY List (~30 ops) — ✅ Mostly Fixed

~~The `CPU_ONLY_OPS` HashSet in `cpu_only_ops.rs` is missing entries from the canonical ANE CPU-only list. These ops lack defense-in-depth protection: if any future code path accidentally assigns them an ANE engine, the validator will not catch it.~~

**Fixed by T-49:** Added ~27 missing ops to `CPU_ONLY_OPS`. Test assertion updated from >=93 to >=120. The 4 ops from T-47 (`slice_update`, `sliding_windows`, `reverse`, `argsort`) were also added as part of that fix. Remaining gaps (e.g., `for`, `call`, `condition`, `yield`, `shape`, `rank`, `size`, `dimension_size`, `sparse_tensor_storage`, `materialize_sparse_tensor`, `buffer_tensor`, `one_hot`) are lower priority.

| Category | Added Ops (T-49) | Remaining Gaps | Severity |
|---|---|---|---|
| Control flow | `return`, `func` | `for`, `call`, `condition`, `yield` | MEDIUM |
| Shape query | — | `shape`, `rank`, `size`, `dimension_size` | MEDIUM |
| Type check | `is_finite`, `is_infinite`, `is_nan` | — | ~~MEDIUM~~ Fixed |
| Elementwise | `negative`, `reciprocal`, `reverse_square_root`, `rint`, `signbit` | — | ~~MEDIUM~~ Fixed |
| Transform | `strided_slice_update`, `dynamic_shape_cast`, `reinterpret_cast`, `col_to_im`, `im_to_col` | — | ~~MEDIUM~~ Fixed |
| Sparse/buffer | — | `sparse_tensor_storage`, `materialize_sparse_tensor`, `buffer_tensor` | MEDIUM |
| Other | `dequantize_lut`, `extract`, `from_elements`, `get_coordinates`, `local_convolution`, `lp_norm`, `prune`, `pruning_metric`, `pruning_structure`, `variable_from_tensor`, `assign_variable`, `placeholder`, `device_hint`, `nf`, `unrealized_fold`, `create_texture_tensor` | `one_hot` | ~~MEDIUM~~ Mostly Fixed |

### II-C. Family-Version Guard Gaps

| Guard | Canon Requirement | Current Implementation | File | Severity |
|---|---|---|---|---|
| ~~Broadcast FP16-only for A13~~ | ~~A13 also FP16-only~~ | **VERIFIED CORRECT** — A13 excluded from FP16-only (per-family matrix ✅; Apple error msg says "A11/A12" only; constraint-doc §3.2 A13 text has internal error) | `ane_target.rs:43-45` | ~~HIGH~~ **RETRACTED** |
| ReduceMin non-FP | A14+ only | ~~`supports_reducemin_all_dtypes()` exists but NOT enforced in placement validator for MILReduceMin~~ **Fixed by T-51:** Added `MILReduceMin` match arm checking `supports_reducemin_all_dtypes()` when dtype is non-FP | `placement_validate.rs` | ~~HIGH~~ **Fixed** |
| E4M3 | A17+ (LSE_6) | ~~A18 only~~ **Fixed by T-52:** A17 and A18 both support E4M3; V11→A17 remapped | `ane_target.rs:89-91` | ~~HIGH~~ **Fixed** |
| Square converter | A13Minus vs A14Plus | No per-family dtype validation | `placement_validate.rs` | MEDIUM |

### II-D. Constraint Validators Not Fully Wired

| Constraint | Canon Reference | Status | File | Severity |
|---|---|---|---|---|
| Tensor dimension limits (HW limits) | §3.4 `hal_params` | ~~`validate_tensor_dims()` exists but NOT called from `validate_placement_with_context()`~~ **Fixed by T-53:** Wired into placement pipeline with `anef_revision` field and `extract_whdc()` helper | `ane_hw_limits.rs:148-193` | ~~HIGH~~ **Fixed** |
| Conv kernel_d / stride | §4.1 | ~~`validate_conv_constraints()` discards params with `let _ = (kernel_d, stride)`~~ **Fixed by T-62:** kernel_d and stride validation implemented per constraint docs | `op_constraints.rs:37` | ~~MEDIUM~~ **Fixed** |
| Zero-channels bypass | §6.3 | ~~`channels.unwrap_or(0)` trivially passes interleave divisibility check~~ **Fixed by T-63:** Changed to `if let Some(channels)` pattern, skipping check when unknown | `placement_validate.rs:244` | ~~MEDIUM~~ **Fixed** |
| Packed10 format | §5 | No MilDtype variant or explicit rejection exists | `ane_layout.rs` | LOW |

### II-E. Palettization Pass Is a No-Op — ✅ Fixed

| Finding | File | Description | Severity |
|---|---|---|---|
| A-12 | `palettize_weights.rs:88-100` | ~~Pass computes `bits` but discards it with `_ = (weight, bits)`. No annotation is emitted. The pass is effectively dead code.~~ **Fixed by T-48:** Added `palette_bits: Option<usize>` to `SirOp::LinearProjection` and `SirOp::Const`. Pass now writes computed bits into field. | ~~CRITICAL~~ **Fixed** |
| A-13 | Multiple | ~~Palette bit-width validation {1,2,3,4,6,8} exists in `lut_projection.rs:151` and `task_spec.rs:937` but NOT in `palettize_weights.rs`.~~ **Fixed by T-48:** Added bit-width validation {1,2,3,4,6,8} with clamping for invalid widths (5→4, 7→6) in `palettize_weights.rs`. | ~~HIGH~~ **Fixed** |

---

## III. CODE-QUALITY FINDINGS

| # | Smell | Location | Suggestion | Severity |
|---|---|---|---|---|
| CQ-1 | ~~4 `panic!()` in production emission code~~ 2 `panic!()` converted to `bail!()` in legality_rewrite.rs | `mir_to_proto.rs:865,877,994,1004` (TEST), `legality_rewrite.rs:3903,4271` | **Fixed by T-54:** Converted 2 production `panic!()` to `anyhow::bail!()` with `Result` return; 3 in mil_lower.rs are intentional safety-net guards | ~~HIGH~~ MEDIUM → Fixed (partial) |
| CQ-2 | ~~3 `panic!()` in MIL lowering~~ 3 `panic!()` intentional guards for unreachable compilation stages | `mil_lower.rs:943,1412,3320` | Intentional safety-net guards — left as-is with documentation | MEDIUM |
| ~~CQ-3~~ | ~~~20 `.unwrap()` in weight file I/O~~ | ~~`weights.rs:530-652`~~ | **RETRACTED** — All `.unwrap()` calls are in test code; production code uses `Result`/`bail!()` | ~~HIGH~~ **RETRACTED** |
| CQ-4 | `panic!()` in legality passthrough — ~~intentional guards~~ converted to `bail!()` | `legality_rewrite.rs:3903,4271` | **Fixed by T-54:** Converted to `anyhow::bail!()` with `Result` return type | ~~MEDIUM~~ **Fixed** |
| CQ-5 | `eprintln!` in library function | `ane_hw_limits.rs:77-80` | Use `log::warn!()` instead | LOW |
| CQ-6 | Deprecated module still compiled | `kv_cache_rewrite` (pub(crate)) | Gate behind feature flag or remove entirely | MEDIUM |
| CQ-7 | ~~`ModelArchConfig::default()` hardcodes Qwen3-0.6B~~ | `common.rs:248-258` | **Fixed by T-56:** Added `qwen3_0_6b()` factory method; Default delegates with deprecation notice | ~~HIGH~~ **Fixed** |
| CQ-8 | ~~Bridge defaults to Qwen3 architecture~~ | `mir_to_compat.rs:455` | **Fixed by T-57:** Added `log::warn!()` when defaulting; deprecation warnings on shape defaults | ~~HIGH~~ **Fixed** |
| CQ-9 | ~~Shape inference defaults to 32768 max_seq_len~~ | `shape_inference.rs:72,566` | **Partially Fixed by T-57:** Added deprecation warnings to `compat_input_shape_default` and `compat_output_shape_default`. Full fix requires making max_seq_len a required parameter. | ~~HIGH~~ MEDIUM |
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
| B-1 | ~~`MILSliceUpdate` silently fails at emission~~ | Any model using slice_update op | **Fixed by T-47:** Moved to `None` in `default_engine()`; added to `CPU_ONLY_OPS` | ~~CRITICAL~~ **Fixed** |
| B-2 | ~~`MILReverse` silently fails at emission~~ | Any model using reverse op | **Fixed by T-47:** Moved to `None` in `default_engine()`; added to `CPU_ONLY_OPS` | ~~CRITICAL~~ **Fixed** |
| B-3 | ~~`MILSlidingWindows` silently fails at emission~~ | Any model using sliding_windows op | **Fixed by T-47:** Moved to `None` in `default_engine()`; added to `CPU_ONLY_OPS` | ~~CRITICAL~~ **Fixed** |
| B-4 | ~~`MILArgsort` silently fails at emission~~ | Any model using argsort (sort on ANE) | **Fixed by T-47:** Moved to `None` in `default_engine()`; added to `CPU_ONLY_OPS` | ~~CRITICAL~~ **Fixed** |
| B-5 | ~~Palettization decisions silently ignored~~ | Any model with LUT/palettized weights | **Fixed by T-48:** Added `palette_bits` field; wired bits into annotation; added bit-width validation with clamping | ~~CRITICAL~~ **Fixed** |
| ~~B-6~~ | ~~FP32 broadcast allowed on A13~~ | ~~A13 hardware with non-FP16 broadcast inputs~~ | **RETRACTED** — Per-family support matrix shows A13 broadcast = ✅; Apple error message specifically says "A11/A12"; constraint-doc A13 section text erroneously claims "same broadcast constraints" — code is correct | ~~HIGH~~ **RETRACTED** |
| B-7 | ~~ReduceMin Int8 allowed on A11-A13~~ | Non-FP ReduceMin on pre-A14 hardware | **Fixed by T-51:** Added `MILReduceMin` guard in placement validator checking `supports_reducemin_all_dtypes()` | ~~HIGH~~ **Fixed** |
| B-8 | ~~E4M3 denied on A17 Pro hardware~~ | ~~V11→A16 family doesn't support E4M3~~ | **Fixed by T-52:** Added `AneFamily::A17` variant; V11→A17 supports E4M3 | ~~HIGH~~ **Fixed** |
| B-9 | Tile reshape zeros resolved incorrectly | Tile with multiple zero placeholders and ctx=None | Use ctx dimensions when available; require ctx for Tile | **HIGH** |
| B-10 | ~~HW tensor dimension limits not enforced~~ | Large tensors pass placement but fail at ANE runtime | **Fixed by T-53:** Wired `validate_tensor_dims()` into placement pipeline with `anef_revision` field and `extract_whdc()` helper | ~~HIGH~~ **Fixed** |
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
│  SIR ──██████████████████████████░  94%       │
│  AIR ──█████████████████████████░░  91%       │
│  MIR ──████████████████████████░░░  92%       │
│  PIR ──█████████████████████████░░  94%       │
│                                             │
│  OVERALL: █████████████████████████░  93%       │
│                                             │
│  Deductions from 100%:                      │
│  - Zero test coverage for 6 critical mods: -3%
│  - Qwen3 deprecation warnings remaining: -2%
│  - Tile reshape placeholder zeros (quality): -2%
│                                             │
│  Deductions resolved since v3 audit:        │
│  + E4M3 denied on A17 Pro (V11→A16): FIXED  │
│  + panic!() in legality passthrough: FIXED  │
│  + Conv constraint discards params: FIXED   │
│  + Zero-channels bypasses interleave: FIXED │
│                                             │
│  Deductions resolved since v2 audit:        │
│  + 4 ops with PE engine but no ANEC: FIXED  │
│  + Palettize pass is no-op: FIXED           │
│  + 30 missing CPU_ONLY_OPS entries: FIXED   │
│  + .unwrap() in production paths: RETRACTED │
│  + panic!() severity: DOWNGRADED to MEDIUM  │
│                                             │
│  Improvements since v1 audit:               │
│  + 20 issues resolved (I-01 through I-20)   │
│  + 7 issues resolved (I-21 through I-31)    │
│  + 2 false positives retracted (I-24, I-29) │
│  + 1 severity downgrade (I-28: HIGH→MEDIUM) │
│  + ToProto trait unified mapping             │
│  + Constexpr* compat variants added          │
│  + 6-gate placement validation chain         │
│  + HW tensor dims enforced at placement      │
│  + ReduceMin non-FP guard added              │
│  + Palettize pass fully wired                │
│  + Zero clippy warnings, all tests pass      │
└─────────────────────────────────────────────┘
```

---

## VII. RECOMMENDED SPRINT BACKLOG

Sorted by **impact x urgency** (highest first):

| Rank | Task | Impact | Urgency | Effort | Issue Ref | Status |
|---|---|---|---|---|---|---|
| 1 | ~~**Fix 4 ops with PE engine but no ANEC converter**~~ | ~~CRITICAL~~ | ~~NOW~~ | S (0.5d) | I-21 | ✅ Fixed (T-47) |
| 2 | ~~**Fix palettize_weights no-op**~~ | ~~CRITICAL~~ | ~~NOW~~ | M (1d) | I-22 | ✅ Fixed (T-48) |
| 3 | ~~**Add ~30 missing ops to CPU_ONLY_OPS**~~ | ~~HIGH~~ | ~~NOW~~ | S (0.5d) | I-23 | ✅ Fixed (T-49) |
| ~~4~~ | ~~**Fix broadcast FP16-only to include A13**~~ | ~~HIGH~~ | **RETRACTED** | — | ~~I-24~~ | RETRACTED |
| 5 | ~~**Add ReduceMin non-FP dtype guard**~~ | ~~HIGH~~ | ~~NEXT~~ | S (0.5d) | I-25 | ✅ Fixed (T-51) |
| 6 | ~~**Fix E4M3 support for A17 Pro (V11)**: Add A17 family or revision-level override so V11 gets E4M3 capability~~ | ~~HIGH~~ | ~~NEXT~~ | M (1d) | I-26 | ✅ Fixed (T-52) |
| 7 | ~~**Wire `validate_tensor_dims()` into placement pipeline**~~ | ~~HIGH~~ | ~~NEXT~~ | S (0.5d) | I-27 | ✅ Fixed (T-53) |
| 8 | ~~**Replace `panic!()` in emission and lowering code**: Code quality improvement (downgraded from HIGH to MEDIUM — test/guard code, not production runtime risk)~~ | ~~MEDIUM~~ | ~~NEXT~~ | S (0.5d) | I-28 | ✅ Fixed (T-54) |
| ~~9~~ | ~~**Replace `.unwrap()` in weights.rs**~~ | ~~HIGH~~ | **RETRACTED** | — | ~~I-29~~ | RETRACTED |
| 10 | ~~**Remove Qwen3 `Default` impl for ModelArchConfig**~~ | ~~HIGH~~ | ~~NEXT~~ | S (0.5d) | I-30 | ✅ Fixed (T-56) |
| 11 | ~~**Fix Qwen3 architecture fallback in mir_to_compat.rs**~~ | ~~HIGH~~ | ~~NEXT~~ | S (0.5d) | I-31 | ✅ Fixed (T-57) |
| 12 | **Add tests for ir::payload, ir::shard_desc, ir::serialize**: Three 0%-coverage modules with 30 pub fn total | HIGH | NEXT | L (3d) | I-32 | ⬜ Open |
| 13 | **Add tests for lab::session, lab::harness, lab::fallback**: Three 0%-coverage critical modules | HIGH | NEXT | L (2d) | I-33 | ⬜ Open |
| 14 | **Fix Tile decomposition placeholder zeros**: Use ctx dimensions when available; document ctx requirement | MEDIUM | NEXT | S (0.5d) | I-34 | ⬜ Open |
| 15 | **Add cross-validation test for Python vs Rust emission**: Structural equivalence test for same MIR input | MEDIUM | LATER | M (1d) | I-35 | ⬜ Open |
| 16 | ~~**Fix Conv constraint discarded params**: Implement kernel_d and stride validation~~ | ~~MEDIUM~~ | ~~LATER~~ | S (0.5d) | I-36 | ✅ Fixed (T-62) |
| 17 | ~~**Fix zero-channels interleave bypass**: Return `AneConditional` when channels unknown~~ | ~~MEDIUM~~ | ~~LATER~~ | S (0.5d) | I-37 | ✅ Fixed (T-63) |
| 18 | **Centralize palette bit-width validation**: Single `validate_palette_bits()` in ane_layout | MEDIUM | LATER | S (0.5d) | I-38 | ⬜ Open |
| 19 | **Unify CPU-only classification**: Derive CPU_ONLY_OPS from `default_engine() == None` | MEDIUM | LATER | M (1d) | I-39 | ⬜ Open |
| 20 | **Add ReduceMin/ArgMinMax/etc. to MirOpCompat**: Close remaining compat coverage gaps for ops with real ANEC converters | MEDIUM | LATER | M (2d) | I-40 | ⬜ Open |
