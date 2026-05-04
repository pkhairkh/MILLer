# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-04 (v3 audit — I-41 through I-60 added from deep source re-audit)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `docs/audit/tabula-rasa-v3.md` (v3, generated 2026-05-04)*

---

## P0 — CRITICAL (Silent Emission Failures / Functional No-Ops)

### I-41 · MILNeg Passes CPU-Only Gate — Name Mismatch in CPU_ONLY_OPS

**Status:** ✅ Fixed (T-67)
**Files:** `crates/ir/src/mir.rs:1115`, `crates/passes/src/cpu_only_ops.rs:222`, `crates/passes/src/placement_validate.rs:540`
**AUDIT ref:** §II-A, §IV (B-13)
**Severity:** CRITICAL
**Effort:** S (0.5 day)
**Task:** T-67

`MILNeg` is assigned `Some(AneEngine::PE)` in `default_engine()` at mir.rs:1115, but per the per-op support matrix (Section 2.2, row 197), `mps.negative` has **no ANEC converter**. The `CPU_ONLY_OPS` set contains `"negative"` (cpu_only_ops.rs:222) but `MirOp::MILNeg`'s `mil_op_name()` returns `"neg"` (mir.rs:1345). The placement validator checks `is_cpu_only(op.mil_op_name())` which calls `is_cpu_only("neg")` → **false**. Combined with `default_engine().is_none()` returning `false` (it returns `Some(PE)`), MILNeg passes all gates and is classified as `AneAllowed` on the ANE. This will silently fail at ANE runtime since there is no `ConvertNegative` or similar converter in the ANEC dialect.

**Fix:** (1) Move `MILNeg` from the PE branch to the None branch in `default_engine()`. (2) Add `"neg"` to `CPU_ONLY_OPS`. (3) Remove the dead `"negative"` entry.

---

### I-42 · CPU_ONLY_OPS Name Mismatches for 5 T-49 Entries — Dead Code That Never Matches

**Status:** ✅ Fixed (T-67)
**Files:** `crates/passes/src/cpu_only_ops.rs:222-226`
**AUDIT ref:** §II-A
**Severity:** CRITICAL (for `"negative"`) / HIGH (for others)
**Effort:** S (0.5 day)
**Task:** T-67

The T-49 additions used MIL builder function names rather than `mil_op_name()` return values. Five entries have no corresponding `mil_op_name()` match:

| CPU_ONLY_OPS entry | Actual `mil_op_name()` | Match? | Impact |
|---|---|---|---|
| `"negative"` | `"neg"` | ❌ | MILNeg silently passes CPU-only gate (see I-41) |
| `"reverse_square_root"` | `"rsqrt"` | ❌ | Dead code — rsqrt IS ANE-legal (`anec.r_sqrt`), entry incorrectly marks it as CPU-only |
| `"reciprocal"` | No MirOp variant | ❌ | Dead code — no op produces this name |
| `"rint"` | `"round"` (MILRound) | ❌ | Dead code — `"rint"` never matches any `mil_op_name()` |
| `"signbit"` | No MirOp variant | ❌ | Dead code — no op produces this name |

**Fix:** (1) Add `"neg"` to `CPU_ONLY_OPS`, remove `"negative"`. (2) Remove `"reverse_square_root"` (rsqrt is ANE-legal). (3) Remove `"reciprocal"` and `"signbit"` (dead code). (4) Replace `"rint"` with `"round"`. (5) Add a test that verifies every CPU_ONLY_OPS entry matches at least one `mil_op_name()`.

---

## P1 — HIGH (Missing Enforcement / Model Leakage / Untested Paths)

### I-43 · `extract_whdc()` Swaps Depth and Channels for Rank-4 NCHW Tensors

**Status:** ✅ Fixed (T-68)
**Files:** `crates/passes/src/placement_validate.rs:155-169`
**AUDIT ref:** §II-D, §IV (B-14)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-68

`extract_whdc()` treats rank-4 shapes as `[channels, depth, height, width]` (CDHW) but Core ML / MIL uses `[batch, channels, height, width]` (NCHW). For a typical `[1, 64, 128, 128]` tensor:
- Current: w=128, h=128, d=64 (actually channels), c=1 (actually batch)
- Correct: w=128, h=128, d=1, c=64

The "depth" and "channels" values are swapped for ranks 3 and 4. This means:
1. `max_tensor_channels` limit is checked against the **batch** dimension (usually 1), silently bypassing channel limits for tensors with large channel counts
2. `max_tensor_depth` limit is checked against the **channel** dimension, potentially causing false rejections for tensors with large channels

**Fix:** For rank-4 NCHW: `(shape[3], shape[2], 1, shape[1])`. For rank-3 CHW: `(shape[2], shape[1], 1, shape[0])`. Update comment to document NCHW interpretation. Add unit tests verifying dimension extraction for NCHW shapes.

---

### I-44 · Pooling Kernel Size Constraint Discarded

**Status:** ✅ Fixed (T-69)
**Files:** `crates/passes/src/op_constraints.rs:160`
**AUDIT ref:** §II-B, §IV (B-15)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-69

`validate_pooling_constraints()` takes a `kernel_size` parameter but discards it with `let _ = kernel_size;` (line 160). The ANE constraint docs specify per-family pooling kernel limits (`pe_max_pooling_kh`, `pe_max_pooling_kw`, depth-specific limits). Pooling kernel sizes are never validated against hardware limits. This is the same pattern as I-36 (conv constraint discarding kernel_d and stride), which was fixed in T-62.

**Fix:** Add kernel_size validation against revision-specific HW limits. At minimum, validate that kernel_size is within a reasonable range per the constraint docs.

---

### I-45 · K/V Projection Alias Maps Silently Dropped

**Status:** ✅ Fixed (T-70)
**Files:** `crates/bridge/src/mir_to_compat.rs:470-471`
**AUDIT ref:** §III (CQ-16), §IV (B-17)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-70

`build_input_alias_map` resolves `k_proj_pattern()` and `v_proj_pattern()` but immediately discards them with `let _ = (k_proj, v_proj)`. Only the Q projection pattern is used (line 469) to build QKV-split aliases. For models with separate K/V projections (GQA architectures like Qwen3), the alias map only contains Q projection → QKV split mappings, while K/V tensor references are never properly wired through the alias map. This produces silently incorrect SSA references for K/V heads in the compat layer.

**Current behavior:** When a weight contains `q_proj`, ALL of `sir_qkv_split_q`, `sir_qkv_split_k`, and `sir_qkv_split_v` aliases point to the same Q projection linear node. This works for fused QKV but is incorrect for GQA models with separate K/V projections.

**Fix:** Used `k_proj`/`v_proj` patterns to build separate K/V alias entries.

---

### I-46 · `CoreMlDataType::Float64` Element Size Returns 4 Instead of 8

**Status:** ✅ Fixed (T-71)
**Files:** `crates/coreml-proto/src/lib.rs:124`
**AUDIT ref:** §IV (B-16)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-71

```rust
CoreMlDataType::Float32 | CoreMlDataType::Float64 => 4,
```

`Float64` (double precision) is 8 bytes per element, not 4. This function is used to compute weight data sizes in `weights.rs` and `mir_to_compat.rs`. Any `Float64` constant would produce a weight entry with half the expected bytes, causing buffer over-reads when Core ML tries to load the weight from `weight.bin`. The existing unit test (`test_coreml_data_type_element_size`) does not cover Float64.

**Fix:** Split the match arm: `CoreMlDataType::Float32 => 4, CoreMlDataType::Float64 => 8`. Add `assert_eq!(CoreMlDataType::Float64.element_size(), 8)` test.

---

### I-47 · Palettize Weights Pass Uses Qwen3-Specific Name Heuristics

**Status:** ✅ Fixed (T-72)
**Files:** `crates/passes/src/palettize_weights.rs:129-136`
**AUDIT ref:** §II-E, §IV
**Severity:** HIGH
**Effort:** M (1 day)
**Task:** T-72

The `run_palettize_weights_pass` uses node name patterns to classify weights:
```rust
let is_attention = node.name.contains("q_proj")
    || node.name.contains("k_proj")
    || node.name.contains("v_proj")
    || node.name.contains("o_proj")
    || node.name.contains("out_proj")
    || node.name.contains("qkv");
```

These are Qwen3/LLaMA-specific naming conventions. For other architectures (GPT-2, T5, BART), non-matching attention weights will be classified as MLP and receive `mlp_bits` instead of `attention_bits`, producing sub-optimal palettization decisions. This is the same model-leakage pattern as I-30/I-31 but in the palettization pass.

**Fix:** Added `run_palettize_weights_pass_with_arch()` using `ModelArchitecture` pattern methods.

---

### I-48 · `LM_HEAD_SHARD_SIZE = 19000` Hardcoded in SafetensorsResolver

**Status:** ✅ Fixed (T-73)
**Files:** `crates/bridge/src/safetensors_resolver.rs:293`
**AUDIT ref:** §III (CQ-17), §IV (B-19)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-73

The vocabulary-projection sharding size is hardcoded to 19000, which is specific to Qwen3-0.6B (vocab_size=151936, 151936/8≈18992). For other models with different vocab sizes, the shard size is incorrect and may cause ANE execution planner errors (-5) or sub-optimal sharding. Same pattern as I-30/I-31.

**Fix:** Shard size derived from `vocab_size / TARGET_SHARD_COUNT`.

---

### I-49 · `resolve_shard` Assumes FP16 Element Size (Hardcoded 2 bytes)

**Status:** ✅ Fixed (T-74)
**Files:** `crates/bridge/src/safetensors_resolver.rs:319`
**AUDIT ref:** §III (CQ-18), §IV (B-18)
**Severity:** HIGH
**Effort:** S (0.5 day)
**Task:** T-74

```rust
let bytes_per_row = hidden_size * 2; // FP16 = 2 bytes per element
```

The shard slicing assumes all weights are FP16 (2 bytes per element). F32 weights that pass through without conversion would be 4 bytes per element; INT8/UInt8 would be 1 byte. The byte offset calculation `start_row * bytes_per_row` would produce wrong offsets for non-FP16 weights, silently slicing the wrong portion of the weight data.

**Fix:** Element size derived from `data.len() / total_elements`.

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

**Status:** ✅ Fixed (T-65)
**Files:** `crates/passes/src/cpu_only_ops.rs`, `crates/ir/src/mir.rs`
**AUDIT ref:** §III (E-1)
**Severity:** MEDIUM
**Effort:** M (1 day)
**Task:** T-65

CPU-only op classification is maintained in both `CPU_ONLY_OPS` HashSet and `default_engine() → None` branch. These can diverge (as they did for MILSliceUpdate, MILReverse, etc., and now for MILNeg per I-41). Derive the CPU-only set from `default_engine() == None` for single source of truth.

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

### I-50 · `coreml_model_destroy` FFI Unsoundness

**Status:** ✅ Fixed (T-75)
**Files:** `crates/coreml-ffi/src/capi.rs:35-51,187-225`
**AUDIT ref:** §III (CQ-22)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-75

`coreml_model_destroy` calls `Box::from_raw(inner)` on a handle that was never allocated with `Box::new(ModelHandleInner)`. If a future implementation of `coreml_model_load` allocates the handle differently (e.g., as a raw Core ML `MLModel*` from the C API), calling `destroy` would cause undefined behavior (use-after-free, double-free, or memory corruption). This is a latent unsoundness — current code paths never trigger it because `load` always returns an error, but the FFI contract is broken.

**Fix:** Documented allocation contract on `ModelHandleInner` — `coreml_model_load` MUST use `Box::new(ModelHandleInner)` so that `coreml_model_destroy` can safely reconstruct with `Box::from_raw`. Added contract test verifying a Box-allocated handle can be destroyed without UB.

---

### I-51 · Zero Tests for coreml-ffi::api Module

**Status:** ✅ Fixed (T-76)
**Files:** `crates/coreml-ffi/src/api.rs`
**AUDIT ref:** §V
**Severity:** MEDIUM
**Effort:** M (1 day)
**Task:** T-76

The `api.rs` module has 5 `pub fn` methods (`is_available`, `version`, `compile_model`, `inspect_model_structure`, `inspect_compute_plan`) with zero test coverage. The high-level API's error handling, result construction, and platform detection logic are untested.

**Fix:** Added 11 new tests covering error type verification for all 5 methods, JSON serialization roundtrips for result types, and field-level validation.

---

### I-52 · PythonBridge `timeout_secs` Field Never Enforced

**Status:** ✅ Fixed (T-77)
**Files:** `crates/bridge/src/subprocess.rs:28,67-127`
**AUDIT ref:** §III (CQ-19)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-77

`PythonBridge` struct has a `timeout_secs: u64` field (defaults to 300) that is never used. The `execute_raw_payload` method calls `Command::new().output()` without any timeout, meaning a hung Python subprocess will block the compiler indefinitely. This is a production code path used by the CLI and lab commands.

**Fix:** Replaced `Command::output()` with `spawn` + poll-based timeout loop using `try_wait()` and `Instant`. On timeout, the child is killed and a timeout error is returned. No new dependencies required.

---

### I-53 · `compare_with_python_bridge` Is Dead Code Stub

**Status:** ✅ Fixed (T-78)
**Files:** `crates/coreml-emit/src/emitter.rs`
**AUDIT ref:** §III (CQ-20)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-78

The `compare_with_python_bridge()` method takes `_python_output_path` (unused) and always returns `None` for all fields. This method is dead code — never called by any production or test code. It should either be implemented (tying into I-35's cross-validation goal) or removed to avoid misleading callers.

**Fix:** Removed the method along with `ComparisonReport` and `WeightBinComparison` types.

---

### I-54 · Empty SafetensorsResolver Returned Without Warning

**Status:** ✅ Fixed (T-79)
**Files:** `crates/bridge/src/safetensors_resolver.rs:135-169`
**AUDIT ref:** §III (CQ-24)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-79

When all three resolution strategies (explicit paths, cache dir, HF model ID) fail, `from_traced_graph` returns `(Self::empty(), "no weights found")` without logging any warning. A silently-empty resolver means all weights become zero-filled placeholders, producing a model that compiles but produces garbage output at inference time — with no indication that weight resolution failed.

**Fix:** Added `log::warn!()` when all resolution strategies fail.

---

### I-55 · Fill Op `input_names()` Returns Empty Vec

**Status:** ✅ Fixed (T-80)
**Files:** `crates/coreml-proto/src/lib.rs:1078`
**AUDIT ref:** §III (CQ-23)
**Severity:** MEDIUM
**Effort:** S (0.5 day)
**Task:** T-80

```rust
MirOpCompat::Fill { .. } => vec![],
```

`Fill` takes a `shape: Vec<i32>` and `value: f32` — but `input_names()` returns an empty vec for it. The `Fill` op in Core ML's MIL format expects `shape` as an input tensor (not an attribute). When `input_names()` is used for SSA validation or dead-reference detection, the `Fill` op's shape input is not considered, potentially leading to false "unreferenced weight" warnings or missing weight materialization.

**Fix:** Now returns `vec![format!("{}_shape", name)]`.

---

## P3 — LOW (Minor Quality / Style / Documentation)

### I-56 · `compat_input_dtype` Uses String Matching for `input_ids` Detection

**Status:** ⬜ Open
**Files:** `crates/bridge/src/shape_inference.rs:33-38`
**AUDIT ref:** §III (CQ-25)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-81

```rust
if name.contains("input_ids") { MilDtypeCompat::Int32 } else { ... }
```

Uses string contains to detect input_ids tensors. If a tensor is named e.g. `my_input_ids_special`, it would be incorrectly typed as Int32 even if it's actually FP16. The correct approach would be to use the MIR node's declared dtype directly.

---

### I-57 · Dead-Code `mir_node_to_compat` With `#[allow(dead_code)]`

**Status:** ✅ Fixed (T-82)
**Files:** `crates/bridge/src/mir_to_compat.rs:590`
**AUDIT ref:** §III (CQ-26)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-82

The function `mir_node_to_compat` is marked as dead code, but the shape-aware version `mir_node_to_compat_with_shapes` IS used. This suggests incomplete migration — the original function should either be removed or the `#[allow(dead_code)]` should document why it's kept.

**Fix:** Removed `#[allow(dead_code)]` and gated with `#[cfg(test)]` since the function is only used in tests. Added documentation explaining when to use it vs. `mir_node_to_compat_with_shapes`.

---

### I-58 · BF16→FP16 Conversion Missing Edge-Case Tests

**Status:** ✅ Fixed (T-83)
**Files:** `crates/bridge/src/safetensors_resolver.rs`
**AUDIT ref:** §III (CQ-27)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-83

The BF16→FP16 conversion function uses `half::f16::from_f32()` which handles subnormals and NaN payloads correctly, but there are no tests verifying NaN preservation, Infinity preservation, subnormal flushing behavior, or zero-sign preservation. The only test (`test_f32_to_f16_roundtrip`) tests 6 simple positive values.

**Fix:** Added 7 edge-case tests: NaN preservation (quiet + signaling), infinity preservation (+/-), negative zero, subnormal handling, max finite overflow to +Inf, and bulk conversion pipeline test.

---

### I-59 · `eprintln!` in Library Function

**Status:** ✅ Fixed (T-84)
**Files:** `crates/ir/src/ane_hw_limits.rs:77-80`
**AUDIT ref:** §III (CQ-5)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-84

Use `log::warn!()` instead of `eprintln!()` in library code.

**Fix:** Replaced with `log::warn!` in `ane_hw_limits.rs`.

---

### I-60 · Deprecated `kv_cache_rewrite` Module Still Compiled

**Status:** ✅ Fixed (T-85)
**Files:** `crates/passes/src/kv_cache_rewrite.rs`
**AUDIT ref:** §III (CQ-6)
**Severity:** LOW
**Effort:** S (0.5 day)
**Task:** T-85

Gate behind feature flag or remove entirely.

**Fix:** Gated behind `deprecated-kv-cache-rewrite` feature flag.

---

## Resolved Issues (v1 + v2 Audits, All Fixed)

### v2 Audit (I-21 through I-40)

| ID | Description | Resolution |
|---|---|---|
| I-21 | Four ops with PE engine but no ANEC converter | ✅ T-47: Moved to None; added to CPU_ONLY_OPS |
| I-22 | Palettize weights pass is a functional no-op | ✅ T-48: Added palette_bits field with validation |
| I-23 | ~30 missing ops in CPU_ONLY_OPS set | ✅ T-49: Added ~27 missing ops (with name mismatches, see I-41/I-42) |
| I-24 | Broadcast FP16-only should include A13 | **RETRACTED** — Code is correct; constraint-doc text has error |
| I-25 | ReduceMin non-FP dtype not enforced | ✅ T-51: Added guard in placement validator |
| I-26 | E4M3 not supported on A17 Pro (V11→A16) | ✅ T-52: Added A17 family variant; V11→A17 remapped |
| I-27 | Tensor dimension HW limits not enforced | ✅ T-53: Wired into placement pipeline |
| I-28 | `panic!()` in emission and lowering code | ✅ T-54: Converted 2 to bail!(); 3 are intentional guards |
| I-29 | `.unwrap()` in weight file I/O | **RETRACTED** — All in test code; production uses Result/bail!() |
| I-30 | ModelArchConfig default hardcodes Qwen3-0.6B | ✅ T-56: Added qwen3_0_6b() factory with deprecation |
| I-31 | Qwen3 architecture fallback in bridge | ✅ T-57: Added log::warn!() and deprecation warnings |
| I-36 | Conv constraint discards kernel_d and stride | ✅ T-62: Added kernel_d and stride validation |
| I-37 | Zero-channels bypasses interleave check | ✅ T-63: Changed to if let Some(channels) pattern |

### v1 Audit (I-01 through I-20)

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
| P0 | 4 | 0 | 4 | 0 |
| P1 | 17 | 2 | 13 | 2 |
| P2 | 15 | 2 | 11 | 0 |
| P3 | 5 | 1 | 3 | 0 |
| Resolved (v1+v2) | 33 | 0 | 31 | 2 |
| **Total** | **74** | **2** | **68** | **4** |
