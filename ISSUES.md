# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-04 (TABULA RASA v2 full audit refresh)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `AUDIT.md` (generated 2026-05-04)*

---

## P0 — CRITICAL (Silent Emission Failures / Functional No-Ops)

### I-21 · Four Ops With PE Engine but No ANEC Converter

**Status:** ⬜ Open
**Files:** `crates/ir/src/mir.rs`, `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-A, §IV (B-1 through B-4)
**Severity:** CRITICAL
**Effort:** S (0.5 day)
**Task:** T-47

`MILSliceUpdate`, `MILReverse`, `MILSlidingWindows`, and `MILArgsort` are assigned `Some(AneEngine::PE)` in `default_engine()` but map to `MirOpCompat::Unsupported` at emission time. They pass placement validation as ANE-legal but silently fail during proto emission, causing CPU fallback with synchronization stalls. None are in the `CPU_ONLY_OPS` set.

**Fix:** Move all four to `None` branch in `default_engine()`. Add `"slice_update"`, `"reverse"`, `"sliding_windows"`, `"argsort"` to `CPU_ONLY_OPS`.

---

### I-22 · Palettize Weights Pass Is a Functional No-Op

**Status:** ⬜ Open
**Files:** `crates/passes/src/palettize_weights.rs`
**AUDIT ref:** §II-E, §IV (B-5)
**Severity:** CRITICAL
**Effort:** M (1 day)
**Task:** T-48

The `palettize_weights` pass computes `bits` values (lines 88-95) but discards them with `_ = (weight, bits)` on line 100. No palette annotation is emitted — the pass is effectively dead code. Additionally, the `attention_bits + 2` computation can produce invalid widths (e.g., 7-bit), and there is no bit-width validation anywhere in the pass. Palette bit-width validation {1,2,3,4,6,8} exists only in lab/families code, not in the core compilation pass.

**Fix:** Wire `bits` into weight annotation. Add `validate_palette_bits()` ensuring bits ∈ {1,2,3,4,6,8}. Reject invalid widths with clear error messages.

---

## P1 — HIGH (Missing Enforcement / Model Leakage / Untested Paths)

### I-23 · ~30 Missing Ops in CPU_ONLY_OPS Set

**Status:** ⬜ Open
**Files:** `crates/passes/src/cpu_only_ops.rs`
**AUDIT ref:** §II-B
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-49

The `CPU_ONLY_OPS` HashSet is missing entries from the canonical ANE CPU-only list. Missing ops include: `for`, `call`, `condition`, `yield`, `return`, `shape`, `rank`, `size`, `dimension_size`, `is_finite`, `is_infinite`, `is_nan`, `negative`, `reciprocal`, `reverse_square_root`, `rint`, `signbit`, `strided_slice_update`, `one_hot`, and approximately 20 more. While the `is_cpu_only()` gate in the placement validator provides defense-in-depth, it only works for ops that are actually in the set.

**Fix:** Add all missing ops from the canonical CPU-only list to `CPU_ONLY_OPS` and corresponding `CPU_ONLY_OPS_DETAILED` entries.

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

### I-25 · ReduceMin Non-FP Dtype Not Enforced

**Status:** ⬜ Open
**Files:** `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-C, §IV (B-7)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-51

The canonical rule says "ReduceMin non-FP: only A14+ (LSE_3+)". `AneFamily::supports_reducemin_all_dtypes()` correctly implements this, but the placement validator has NO specific match arm for `MILReduceMin` to enforce it. On A11/A12/A13, ReduceMin with Int8/UInt8 dtypes passes validation but fails at ANE runtime.

**Fix:** Add `MILReduceMin` match arm in `validate_placement_with_context()` that checks `target_family.supports_reducemin_all_dtypes()` when `ctx.dtype` is non-FP.

---

### I-26 · E4M3 Not Supported on A17 Pro (V11 Maps to A16)

**Status:** ⬜ Open
**Files:** `crates/ir/src/ane_target.rs:89-91,130`
**AUDIT ref:** §II-C, §IV (B-8)
**Severity:** HIGH
**Effort:** M (1 day)
**Task:** T-52

The canonical rule says "E4M3: only A17+ (LSE_6)". `supports_e4m3()` only matches `A18`. V11 (A17 Pro) maps to `AneFamily::A16` which doesn't support E4M3. A17 Pro users cannot use E4M3 despite hardware support because the family mapping is too coarse.

**Fix:** Either add `AneFamily::A17` variant mapping V11→A17, or add a revision-level override in `supports_e4m3()` for V11, or extend the match to include A16 if A16 silicon supports E4M3.

---

### I-27 · Tensor Dimension HW Limits Not Enforced in Placement

**Status:** ⬜ Open
**Files:** `crates/ir/src/ane_hw_limits.rs:148-193`, `crates/passes/src/placement_validate.rs`
**AUDIT ref:** §II-D, §IV (B-10)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-53

`AneHwLimits::validate_tensor_dims()` exists but is never called from `validate_placement_with_context()`. Hardware tensor dimension limits (max_tensor_width, max_tensor_height, max_tensor_channels, etc.) are defined per-revision but not enforced at placement time. Oversized tensors pass validation but fail at ANE runtime.

**Fix:** Call `AneHwLimits::for_revision().validate_tensor_dims()` from `validate_placement_with_context()` before returning `AneAllowed`.

---

### I-28 · `panic!()` in Production Emission and Lowering Code

**Status:** ⬜ Open
**Files:** `crates/coreml-emit/src/mir_to_proto.rs:865,877,994,1004`, `crates/passes/src/mil_lower.rs:943,1412,3320`, `crates/passes/src/legality_rewrite.rs:3903,4271`
**AUDIT ref:** §III (CQ-1, CQ-2, CQ-4)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-54

7 `panic!()` calls in production code paths: 4 in mir_to_proto.rs (unexpected weight format), 3 in mil_lower.rs (unexpected AIR op patterns), 2 in legality_rewrite.rs (Select/Where/Tile passthrough assertions). These crash the compiler instead of returning proper errors.

**Fix:** Replace with `anyhow::bail!()` returning proper error types.

---

### I-29 · `.unwrap()` in Weight File I/O

**Status:** ⬜ Open
**Files:** `crates/coreml-emit/src/weights.rs:530-652`
**AUDIT ref:** §III (CQ-3)
**Severity:** HIGH
**Effort:** M (1 day)
**Task:** T-55

~20 `.unwrap()` calls in the binary weight file format parsing/writing path. A malformed weight file crashes the compiler instead of returning an error.

**Fix:** Replace with `?` operator for Result propagation.

---

### I-30 · ModelArchConfig Default Hardcodes Qwen3-0.6B

**Status:** ⬜ Open
**Files:** `crates/ir/src/common.rs:248-258`
**AUDIT ref:** §III (CQ-7)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-56

`ModelArchConfig::default()` hardcodes Qwen3-0.6B dimensions (vocab_size=151936, embed_dim=1024, num_heads=16, kv_heads=8, intermediate_size=2048, max_seq_len=32768, architecture=Qwen3). Any caller using `default()` gets Qwen3 assumptions silently — a correctness hazard for any other model architecture.

**Fix:** Remove `Default` impl or rename to `fn qwen3_0_6b()` factory method. Force callers to provide explicit config.

---

### I-31 · Qwen3 Architecture Fallback in Bridge

**Status:** ⬜ Open
**Files:** `crates/bridge/src/mir_to_compat.rs:455`, `crates/bridge/src/shape_inference.rs:72,566`
**AUDIT ref:** §III (CQ-8, CQ-9)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-57

Two Qwen3 default leakage points: (1) `mir_to_compat.rs:455` falls back to `ModelArchitecture::Qwen3` when none specified, silently assuming Qwen3 for any model without an architecture tag. (2) `shape_inference.rs:72,566` defaults to 32768 max_seq_len (Qwen3-0.6B's max_position_embeddings).

**Fix:** Return error when no architecture is provided instead of defaulting to Qwen3.

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

### I-36 · Conv Constraint Discards kernel_d and stride

**Status:** ⬜ Open
**Files:** `crates/passes/src/op_constraints.rs:37`
**AUDIT ref:** §II-D
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-62

`validate_conv_constraints()` takes `kernel_d` and `stride` params but discards them with `let _ = (kernel_d, stride)`. Depth dimension and stride constraints per the ANE constraint docs are not validated.

---

### I-37 · Zero-Channels Bypasses Interleave Check

**Status:** ⬜ Open
**Files:** `crates/passes/src/placement_validate.rs:244`
**AUDIT ref:** §II-D
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-63

`channels.unwrap_or(0)` trivially passes interleave divisibility check because 0 is divisible by any factor. Channel-divisibility constraints are silently bypassed when channels are unknown.

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
| P0 | 2 | 2 | 0 | 0 |
| P1 | 11 | 10 | 0 | 1 |
| P2 | 7 | 7 | 0 | 0 |
| Resolved (v1) | 20 | 0 | 20 | 0 |
| **Total** | **40** | **19** | **20** | **1** |
