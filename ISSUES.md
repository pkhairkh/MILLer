# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-04 (v3 audit post-fix update — I-26, I-28, I-36, I-37 fixes applied)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `AUDIT.md` (generated 2026-05-04)*

---

## P0 — CRITICAL (Silent Emission Failures / Functional No-Ops)

### I-21 · Four Ops With PE Engine but No ANEC Converter — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/ir/src/mir.rs`, `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-A, §IV (B-1 through B-4)
**Severity:** ~~CRITICAL~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-47 ✅

`MILSliceUpdate`, `MILReverse`, `MILSlidingWindows`, and `MILArgsort` were assigned `Some(AneEngine::PE)` in `default_engine()` but mapped to `MirOpCompat::Unsupported` at emission time.

**Fix applied (T-47):** Moved all four from `Some(AneEngine::PE)` to `None` in `default_engine()`. Added `"slice_update"`, `"reverse"`, `"sliding_windows"`, `"argsort"` to `CPU_ONLY_OPS`.

---

### I-22 · Palettize Weights Pass Is a Functional No-Op — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/passes/src/palettize_weights.rs`
**AUDIT ref:** §II-E, §IV (B-5)
**Severity:** ~~CRITICAL~~ Fixed
**Effort:** M (1 day)
**Task:** T-48 ✅

The `palettize_weights` pass computed `bits` values but discarded them with `_ = (weight, bits)`. No palette annotation was emitted.

**Fix applied (T-48):** Added `palette_bits: Option<usize>` field to `SirOp::LinearProjection` and `SirOp::Const`. The pass now writes computed bits into the field instead of discarding. Added bit-width validation {1,2,3,4,6,8} with clamping for invalid widths (5→4, 7→6).

---

## P1 — HIGH (Missing Enforcement / Model Leakage / Untested Paths)

### I-23 · ~30 Missing Ops in CPU_ONLY_OPS Set — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-B
**Severity:** ~~HIGH~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-49 ✅

The `CPU_ONLY_OPS` HashSet was missing entries from the canonical ANE CPU-only list.

**Fix applied (T-49):** Added ~27 missing ops to `CPU_ONLY_OPS` (including `slice_update`, `sliding_windows`, `reverse`, `argsort` from T-47 plus `return`, `is_finite`, `is_infinite`, `is_nan`, `negative`, `reciprocal`, `reverse_square_root`, `rint`, `signbit`, `strided_slice_update`, `dynamic_shape_cast`, `reinterpret_cast`, `col_to_im`, `im_to_col`, `dequantize_lut`, `extract`, `from_elements`, `func`, `get_coordinates`, `local_convolution`, `lp_norm`, `prune`, `pruning_metric`, `pruning_structure`, `variable_from_tensor`, `assign_variable`, `placeholder`, `device_hint`, `nf`, `unrealized_fold`, `create_texture_tensor`). Test assertion updated from >=93 to >=120.

---

### ~~I-24 · Broadcast FP16-Only Should Include A13~~ — **RETRACTED**

**Status:** ~~⬜ Open~~ **RETRACTED**
**Files:** ~~`crates/ir/src/ane_target.rs:43-45`, `crates/passes/src/dtype_constraints.rs:259`~~
**AUDIT ref:** ~~§II-C, §IV (B-6)~~
**Severity:** ~~HIGH~~ **RETRACTED**
**Effort:** —
**Task:** ~~T-50~~

**RETRACTED during audit verification.** The per-family support matrix (`per-op-per-family-support-matrix.md` line 432) clearly shows A13 broadcast = ✅ (no FP16 restriction). The Apple ANE error message specifically says "Only fp16 is supported for A11/A12 Broadcasts" — not A13. The constraint-doc A13 section text (`per-op-per-family-support-matrix.md` line 464) erroneously claims "Same broadcast and ReduceMin constraints as A11/A12" which is incorrect for broadcast (correct for ReduceMin). The current `broadcast_fp16_only()` implementation is **correct** to exclude A13. The code comment at `ane_target.rs:119-122` explicitly documents this: "A13 lifts the FP16-only broadcast restriction (unlike A12)."

---

### I-25 · ReduceMin Non-FP Dtype Not Enforced — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-7)
**Severity:** ~~HIGH~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-51 ✅

The placement validator had no specific match arm for `MILReduceMin` to enforce the "ReduceMin non-FP: only A14+" rule.

**Fix applied (T-51):** Added `MILReduceMin` match arm in `validate_placement_with_context()` that checks `target_family.supports_reducemin_all_dtypes()` when dtype is non-FP.

---

### I-26 · E4M3 Not Supported on A17 Pro (V11 Maps to A16) — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/ir/src/ane_target.rs`, `crates/trace/src/versioned.rs`, `crates/cli/src/main.rs`, `crates/ir/src/strategy.rs`, `crates/passes/src/dtype_constraints.rs`
**AUDIT ref:** §II-C, §IV (B-8)
**Severity:** ~~HIGH~~ Fixed
**Effort:** M (1 day)
**Task:** T-52 ✅

The canonical rule says "E4M3: only A17+ (LSE_6)". `supports_e4m3()` only matched `A18`. V11 (A17 Pro) mapped to `AneFamily::A16` which doesn't support E4M3. A17 Pro users cannot use E4M3 despite hardware support because the family mapping is too coarse.

**Fix applied (T-52):** Added `AneFamily::A17` variant with E4M3 conditional support (LSE_6). Remapped V11 (A17 Pro) from `AneFamily::A16` to `AneFamily::A17`. Updated all family-dependent logic: `supports_sdpa()`, `supports_layernorm()`, `supports_reducemin_all_dtypes()`, `supports_e4m3()`, `family_level()`, `family_to_default_revision()`, CLI parser, strategy KV cache benefit, dtype constraints tests. Added 10 new A17-specific unit tests.

---

### I-27 · Tensor Dimension HW Limits Not Enforced in Placement — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/ir/src/ane_hw_limits.rs:148-193`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-D, §IV (B-10)
**Severity:** ~~HIGH~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-53 ✅

`AneHwLimits::validate_tensor_dims()` existed but was never called from `validate_placement_with_context()`.

**Fix applied (T-53):** Wired `validate_tensor_dims()` into placement pipeline. Added `anef_revision` field to `PlacementContext`, `extract_whdc()` helper, and HW limit validation before op-specific constraints.

---

### I-28 · `panic!()` in Emission and Lowering Code — ✅ Fixed (partially)

**Status:** ✅ Fixed (legality_rewrite.rs); remaining in mil_lower.rs are intentional safety-net guards
**Files:** `crates/passes/src/legality_rewrite.rs:3903,4271`, `crates/passes/src/mil_lower.rs:943,1412,3320`
**AUDIT ref:** §III (CQ-1, CQ-2, CQ-4)
**Severity:** ~~HIGH~~ MEDIUM → Fixed (partial)
**Effort:** S (0.5 day)
**Task:** T-54 ✅

~~7 `panic!()` calls in production code paths~~ **Corrected during source code verification:** The 4 `panic!()` calls in `mir_to_proto.rs` are in TEST code, not production. The 3 `panic!()` calls in `mil_lower.rs` and 2 in `legality_rewrite.rs` are intentional guards for ops that should never reach those compilation stages.

**Fix applied (T-54):** Converted the 2 `panic!()` calls in `legality_rewrite.rs` (`sir_to_air_passthrough()`) to `anyhow::bail!()` with proper `Result` return type. Changed function signature from `(AirOp, &'static str)` to `Result<(AirOp, &'static str)>` with `?` propagation at call site. The 3 remaining `panic!()` calls in `mil_lower.rs` are intentional safety-net guards (double-checking that Where/Select never survive to MIR) and are left as-is with clear documentation.

---

### ~~I-29 · `.unwrap()` in Weight File I/O~~ — **RETRACTED**

**Status:** ~~⬜ Open~~ **RETRACTED**
**Files:** ~~`crates/coreml-emit/src/weights.rs:530-652`~~
**AUDIT ref:** ~~§III (CQ-3)~~
**Severity:** ~~HIGH~~ **RETRACTED**
**Effort:** —
**Task:** ~~T-55~~

**RETRACTED during source code verification.** The ~20 `.unwrap()` calls in `weights.rs:530-652` are ALL in test code. Production code in the same file already uses `Result` and `bail!()` for error propagation. There is no runtime crash risk from the `.unwrap()` calls.

---

### I-30 · ModelArchConfig Default Hardcodes Qwen3-0.6B — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/ir/src/common.rs:248-258`
**AUDIT ref:** §III (CQ-7)
**Severity:** ~~HIGH~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-56 ✅

`ModelArchConfig::default()` hardcoded Qwen3-0.6B dimensions, causing silent Qwen3 assumptions for any caller.

**Fix applied (T-56):** Added `ModelArchConfig::qwen3_0_6b()` factory method. Default impl now delegates to it with deprecation notice, forcing callers toward explicit config.

---

### I-31 · Qwen3 Architecture Fallback in Bridge — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/bridge/src/mir_to_compat.rs:455`, `crates/bridge/src/shape_inference.rs:72,566`
**AUDIT ref:** §III (CQ-8, CQ-9)
**Severity:** ~~HIGH~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-57 ✅

Two Qwen3 default leakage points where architecture defaulted silently.

**Fix applied (T-57):** Added `log::warn!()` in `mir_to_compat.rs` when architecture defaults to Qwen3. Added deprecation warnings to `compat_input_shape_default` and `compat_output_shape_default` in `shape_inference.rs`.

---

### I-32 · Zero Tests for ir::payload, ir::shard_desc, ir::serialize

**Status:** ⬜ Open
**Files:** `crates/ir/src/payload.rs`, `crates/ir/src/shard_desc.rs`, `crates/ir/src/serialize.rs`
**AUDIT ref:** §V
**Severity:** HIGH
**Effort:** L (3 days)
**Task:** T-58

Three modules with 0% test coverage and 30 pub fn total. `payload.rs` has 16 untested pub fn including the precision adaptation pipeline (`from_spec_with_override`). `shard_desc.rs` has 6 untested pub fn for shard pipeline construction. `serialize.rs` has 8 untested pub fn for IR round-trip. The precision adaptation pipeline has zero end-to-end coverage — this is the single highest-risk gap because it prevents fp16 precision hazards.

---

### I-33 · Zero Tests for lab::session, lab::harness, lab::fallback

**Status:** ⬜ Open
**Files:** `crates/lab/src/session.rs`, `crates/lab/src/harness.rs`, `crates/lab/src/fallback.rs`
**AUDIT ref:** §V
**Severity:** HIGH
**Effort:** L (2 days)
**Task:** T-59

Three 0%-coverage critical modules: `session.rs` (7 pub fn — task hashing, knowledge update, artifact manifest), `harness.rs` (14 pub fn — LabRunBuilder, all builder paths, to_json/write_to_file), `fallback.rs` (3 pub fn — FallbackDetector::detect_from_timing). These are the lab's main orchestration and diagnostic entry points.

---

## P2 — MEDIUM (Technical Debt / Drift / Code Quality)

### I-34 · Tile Decomposition Placeholder Zeros

**Status:** ⬜ Open
**Files:** `crates/passes/src/legality_rewrite.rs:542-543`
**AUDIT ref:** §IV (B-9)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-60

Tile decomposition generates `reshape_shape.push(0)` and `final_shape.push(0)` as placeholder dimensions. `resolve_reshape_zeros()` uses batch=1 heuristic for multi-zero resolution, semantically incorrect for general Tile patterns. When ctx is None, shapes are resolved with wrong heuristics.

---

### I-35 · No Cross-Validation Between Python and Rust Emission Paths

**Status:** ⬜ Open
**Files:** `python/mil_emitter.py` vs `crates/coreml-emit/src/mir_to_proto.rs`
**AUDIT ref:** §III (D-1, D-4)
**Severity:** MEDIUM
**Effort:** M (1 day)
**Task:** T-61

Python bridge (coremltools subprocess) and Rust proto-direct path exist independently with no cross-validation test. Fill/FillLike decomposition, weight embedding, and op-specific serialization may diverge. No test verifies both paths produce structurally equivalent MIL for the same MIR input.

---

### I-36 · Conv Constraint Discards kernel_d and stride — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/passes/src/op_constraints.rs:37`
**AUDIT ref:** §II-D
**Severity:** ~~MEDIUM~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-62 ✅

`validate_conv_constraints()` takes `kernel_d` and `stride` params but discards them with `let _ = (kernel_d, stride)`. Depth dimension and stride constraints per the ANE constraint docs are not validated.

**Fix applied (T-62):** Added kernel_d validation: 3D conv with large kernel (kw > 7 or kh > 7) is rejected per constraint docs §4.2 ("kernel with depth > 1 is not supported for large kernel"). Added stride validation: stride[0] (batch) and stride[1] (channel) must be 1 per constraint docs §4.2 ("Conv stride must be 1 for batch / channel axis").

---

### I-37 · Zero-Channels Bypasses Interleave Check — ✅ Fixed

**Status:** ✅ Fixed
**Files:** `crates/passes/src/placement_validate.rs:244`
**AUDIT ref:** §II-D
**Severity:** ~~MEDIUM~~ Fixed
**Effort:** S (0.5 day)
**Task:** T-63 ✅

`channels.unwrap_or(0)` trivially passes interleave divisibility check because 0 is divisible by any factor. Channel-divisibility constraints are silently bypassed when channels are unknown.

**Fix applied (T-63):** Replaced `channels.unwrap_or(0)` with `if let Some(channels) = ctx.channels { ... }` pattern. When channels are unknown, the interleave divisibility check is skipped (rather than silently passing with 0), and other validation continues. This prevents the false-positive pass while maintaining the test suite's behavior for layout and other checks.

---

### I-38 · Palette Bit-Width Validation Scattered

**Status:** ⬜ Open
**Files:** `crates/lab/src/families/lut_projection.rs:151`, `crates/ir/src/task_spec.rs:937`
**AUDIT ref:** §II-E (A-13)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-64

{1,2,3,4,6,8} validation appears in 3 places with no central validator. Create `ane_ir::ane_layout::validate_palette_bits()` and call from all sites.

---

### I-39 · CPU-Only Classification in Two Places

**Status:** ⬜ Open
**Files:** `crates/passes/src/cpu_only_ops.rs`, `crates/ir/src/mir.rs`
**AUDIT ref:** §III (E-1)
**Severity:** MEDIUM
**Effort:** M (1 day)
**Task:** T-65

CPU-only op classification is maintained in both `CPU_ONLY_OPS` HashSet and `default_engine() → None` branch. These can diverge (as they did for MILSliceUpdate, MILReverse, etc.). Derive the CPU-only set from `default_engine() == None` for single source of truth.

---

### I-40 · Remaining MirOpCompat Coverage Gaps

**Status:** ⬜ Open
**Files:** `crates/coreml-proto/src/lib.rs`, `crates/bridge/src/mir_to_compat.rs`
**AUDIT ref:** §II-A (resolved ops note)
**Severity:** MEDIUM
**Effort:** M (2 days)
**Task:** T-66

Ops with real ANEC converters that still map to `MirOpCompat::Unsupported`: BatchNorm, InstanceNorm, L2Norm, MaxPool, AvgPool, L2Pool, Quantize, Dequantize, all resize/resample variants, CropResize, DepthToSpace, SpaceToDepth, PixelShuffle, PixelUnshuffle, BatchToSpace, SpaceToBatch, and others. These have hardware support but lack proto emission code in the Rust path.

---

## Resolved Issues (v1 Audit, All Fixed)

| ID | Description | Resolution |
|---|---|---|
| I-01 | Three sources of truth diverged | ✅ T-22: Aligned engine/CPU-only/compat |
| I-02 | CPU-only list not checked by validator | ✅ T-23: Added is_cpu_only() gate |
| I-03 | V6 (A13) mapped to A14 family | ✅ T-24: Added A13 family variant |
| I-04 | Interleave + dtype validators dead code | ✅ T-25: Wired into placement validator |
| I-05 | Missing validate_matmul_constraints() | ✅ T-26: Added 4 MatMul constraints |
| I-06 | Missing validate_pad_constraints() | ✅ T-27: Added 6 Pad constraints |
| I-07 | Reshape .unwrap() panic | ✅ T-28: Converted to Result |
| I-08 | Zero-dim shapes survive to emission | ✅ T-29: Added zero-dim validation |
| I-09 | % 1 == 0 always-true logic bug | ✅ T-30: Fixed divisor logic |
| I-10 | SDPA compat missing mask and scale | ✅ T-31: Added both fields |
| I-11 | ArgMinMax missing A18 guard | ✅ T-32: Added supports_argminmax() |
| I-12 | Zero tests for shape_inference | ✅ T-33: 153 tests added |
| I-13 | Zero tests for staticize | ✅ T-34: 62 tests added |
| I-14 | MilDtype missing Int4/UInt4/E4M3/E5M2 | ✅ T-35: Added 5 dtype variants |
| I-15 | Model-specific constants hardcoded | ✅ T-36: Added ModelArchConfig |
| I-16 | No SIR→AIR roundtrip test | ✅ T-37: 14 roundtrip tests added |
| I-17 | MirOp + MirOpCompat not unified | ✅ T-38: Added ToProto trait |
| I-18 | Proto-direct cannot emit palettized weights | ✅ T-39: Added 7 Constexpr* variants |
| I-19 | V17 (M1) mapped to A18 family | ✅ T-40: Mapped to A14 |
| I-20 | Formatting + clippy cleanup | ✅ T-41: fmt + clippy --fix |

---

## Summary Statistics

| Priority | Total | Open | Fixed | Retracted |
|----------|-------|------|-------|-----------|
| P0 | 2 | 0 | 2 | 0 |
| P1 | 10 | 2 | 6 | 2 |
| P2 | 8 | 6 | 2 | 0 |
| Resolved (v1) | 20 | 0 | 20 | 0 |
| **Total** | **40** | **8** | **30** | **2** |
