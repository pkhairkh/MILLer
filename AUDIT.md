# ♉ AUDIT.md — TABULA RASA Full-Spectrum Diagnostic (v3)

**Date:** 2026-05-04
**Scope:** Full repository sweep — 12 crates, ~1252 tests, 9 constraint documents
**Method:** Automated lint + deep source walk + canon cross-reference + drift analysis
**Prior audits:** v1 (2026-05-03, I-01 through I-20 all resolved), v2 (2026-05-04, I-21 through I-40 mostly resolved)
**Verification:** Source-code spot-check of ALL new CRITICAL/HIGH findings; I-24 and I-29 retracted as false positives in prior audit
**Toolchain:** cargo check ✓, cargo clippy (2 minor warnings), cargo test (1252 passed, 0 failed)

---

## I. EXECUTIVE JUDGEMENT

The MILLer compiler lattice has matured substantially across three audit cycles. All 20 v1 issues (I-01 through I-20) and 10 of 20 v2 issues (I-21 through I-31) have been resolved. The codebase compiles cleanly, passes all 1252 tests, and enforces a six-gate placement validation chain covering dtype, interleave, layout, blockwise-scale, asymmetric-quant, and per-op constraints. The `ToProto` trait unified MirOp-to-proto mapping across 167 variants, and the `Constexpr*` MirOpCompat variants closed the palettized-weight emission gap.

**However, the v3 audit has uncovered 20 new issues (I-41 through I-60)** — two CRITICAL, seven HIGH, seven MEDIUM, and four LOW. The most dangerous findings are:

1. **CPU_ONLY_OPS name mismatches (I-41, I-42):** Five entries added by T-49 used MIL builder function names instead of `mil_op_name()` return values. The most consequential is `"negative"` instead of `"neg"`, which means `MILNeg` (assigned `Some(AneEngine::PE)`) passes the `is_cpu_only()` gate and is classified as ANE-legal. At emission time, there is no ANEC converter for `MILNeg`, causing silent emission failure. Additionally, `"reverse_square_root"` incorrectly marks rsqrt as CPU-only when it IS ANE-legal per the per-op support matrix.

2. **`extract_whdc()` channel/depth swap for NCHW tensors (I-43):** The function treats rank-4 shapes as `[channels, depth, height, width]` (CDHW), but Core ML MIL uses `[batch, channels, height, width]` (NCHW). The result: `max_tensor_channels` is checked against the batch dimension (usually 1), silently bypassing channel limits for tensors with large channel counts, while `max_tensor_depth` is checked against the channel dimension, potentially causing false rejections.

3. **`Float64.element_size()` returns 4 instead of 8 (I-46):** This produces weight entries with half the required bytes for Float64 constants, causing corrupted model data at inference time. The existing unit test does not cover Float64.

4. **Pooling kernel_size discarded (I-44):** `validate_pooling_constraints()` takes a `kernel_size` parameter but discards it with `let _ = kernel_size;`, the same pattern as the previously-fixed I-36 (Conv constraint discarding kernel_d and stride).

5. **K/V projection alias maps silently dropped (I-45):** `build_input_alias_map` resolves `k_proj_pattern()` and `v_proj_pattern()` but immediately discards them with `let _ = (k_proj, v_proj)`. For models with separate K/V projections (GQA architectures), this means K/V tensor references are never properly wired through the alias map.

**IR Cleanliness Score drops from 93% to 87%** due to the new findings, primarily driven by the CPU_ONLY_OPS name mismatches, the extract_whdc dimensional swap, and the Float64 element_size bug — all of which represent silent correctness failures that pass validation but produce wrong results at runtime.

---

## II. ANE-CONSTRAINT VIOLATIONS

### II-A. CPU_ONLY_OPS Name Mismatches — 5 Dead Entries (T-49 Regression)

The T-49 fix added ~27 ops to `CPU_ONLY_OPS` using MIL builder function names rather than `mil_op_name()` return values. Five entries have no corresponding `mil_op_name()` match, making them dead code that never triggers the CPU-only gate:

| CPU_ONLY_OPS Entry | Actual `mil_op_name()` | Match? | Impact |
|---|---|---|---|
| `"negative"` | `"neg"` | ❌ | MILNeg passes CPU-only gate, classified as ANE-legal with no converter |
| `"reverse_square_root"` | `"rsqrt"` | ❌ | Dead code — rsqrt IS ANE-legal (`anec.r_sqrt`), entry should be removed |
| `"reciprocal"` | No MirOp variant exists | ❌ | Dead code — no op ever produces this name |
| `"rint"` | `"round"` (MILRound) | ❌ | Dead code — `"rint"` never matches any `mil_op_name()` |
| `"signbit"` | No MirOp variant exists | ❌ | Dead code — no op ever produces this name |

The most dangerous consequence is for `MILNeg`: it is assigned `Some(AneEngine::PE)` in `default_engine()`, but `is_cpu_only("neg")` returns `false` because the CPU_ONLY_OPS entry is `"negative"` not `"neg"`. This means MILNeg passes all placement gates and is classified as `AneAllowed`, but at emission time there is no ANEC converter for it (per per-op matrix row 197: `mps.negative → no converter; may decompose to multiply by -1`). Additionally, `"reverse_square_root"` incorrectly marks rsqrt as CPU-only when the per-op support matrix (row 18) shows `anec.r_sqrt` — this entry should be removed entirely since rsqrt IS ANE-legal.

### II-B. Pooling Kernel Size Constraint Discarded

`validate_pooling_constraints()` takes a `kernel_size` parameter but discards it with `let _ = kernel_size;` (op_constraints.rs:160). The ANE constraint docs specify per-family pooling kernel limits:
- `pe_max_pooling_kh`: `1 <= kh && kh <= pe_max_pooling_kh`
- `pe_max_pooling_kw`: `1 <= kw && kw <= pe_max_pooling_kw`
- Depth-specific limits for 3D pooling

Pooling kernel sizes are never validated against hardware limits. This is the same pattern as I-36 (conv constraint discarding kernel_d and stride), which was fixed in T-62.

### II-C. Family-Version Guard Gaps (Carried Forward + New)

| Guard | Canon Requirement | Current Implementation | File | Severity |
|---|---|---|---|---|
| ~~Broadcast FP16-only for A13~~ | ~~A13 also FP16-only~~ | **VERIFIED CORRECT** — A13 excluded from FP16-only (per-family matrix ✅; Apple error msg says "A11/A12" only; constraint-doc §3.2 A13 text has internal error) | `ane_target.rs:43-45` | ~~HIGH~~ **RETRACTED** |
| ReduceMin non-FP | A14+ only | **Fixed by T-51:** `MILReduceMin` match arm checking `supports_reducemin_all_dtypes()` | `placement_validate.rs` | ~~HIGH~~ **Fixed** |
| E4M3 | A17+ (LSE_6) | **Fixed by T-52:** A17 and A18 both support E4M3; V11→A17 remapped | `ane_target.rs:89-91` | ~~HIGH~~ **Fixed** |
| Square converter | A13Minus vs A14Plus | No per-family dtype validation | `placement_validate.rs` | MEDIUM |
| MILNeg negative op | No ANEC converter | **NOT enforced:** `is_cpu_only("neg")` = false due to name mismatch | `cpu_only_ops.rs:222` | CRITICAL |
| Pooling kernel_size | Per-family max limits | **NOT enforced:** `kernel_size` param discarded | `op_constraints.rs:160` | HIGH |

### II-D. Constraint Validators Not Fully Wired (Updated)

| Constraint | Canon Reference | Status | File | Severity |
|---|---|---|---|---|
| Tensor dimension limits (HW limits) | §3.4 `hal_params` | **Fixed by T-53:** Wired into placement pipeline | `ane_hw_limits.rs:148-193` | ~~HIGH~~ **Fixed** |
| Conv kernel_d / stride | §4.1 | **Fixed by T-62:** kernel_d and stride validation implemented | `op_constraints.rs:37` | ~~MEDIUM~~ **Fixed** |
| Zero-channels bypass | §6.3 | **Fixed by T-63:** Changed to `if let Some(channels)` pattern | `placement_validate.rs:244` | ~~MEDIUM~~ **Fixed** |
| Pooling kernel_size | §4.6 | **NOT enforced:** param discarded with `let _ = kernel_size` | `op_constraints.rs:160` | HIGH |
| `extract_whdc()` dimensional swap | §3.4 | **NOT enforced correctly:** Rank-4 NCHW shapes swap depth↔channels, bypassing `max_tensor_channels` limit | `placement_validate.rs:155-169` | HIGH |
| Packed10 format | §5 | No MilDtype variant or explicit rejection exists | `ane_layout.rs` | LOW |

### II-E. Palettization Name Heuristics (Model Leakage)

The `palettize_weights` pass uses node name patterns to classify weights:
```rust
let is_attention = node.name.contains("q_proj")
    || node.name.contains("k_proj")
    || node.name.contains("v_proj")
    || node.name.contains("o_proj")
    || node.name.contains("out_proj")
    || node.name.contains("qkv");
```

These are Qwen3/LLaMA-specific naming conventions. For other architectures:
- GPT-2: `"attn.c_attn"`, `"attn.c_proj"` — will NOT match
- T5: `"q"`, `"k"`, `"v"` — will NOT match
- BART: `"self_attn.q_proj"` — partially matches, `"encoder_attn"` will NOT match

Non-matching attention weights will be classified as MLP and receive `mlp_bits` instead of `attention_bits`, producing sub-optimal palettization decisions (potentially too aggressive quantization for attention projections). This is the same model-leakage pattern as I-30/I-31 but in the palettization pass.

---

## III. CODE-QUALITY FINDINGS

| # | Smell | Location | Suggestion | Severity |
|---|---|---|---|---|
| CQ-1 | ~~4 `panic!()` in production emission code~~ 2 converted to `bail!()` in legality_rewrite.rs | `mir_to_proto.rs` (TEST), `legality_rewrite.rs:3903,4271` | **Fixed by T-54:** Converted 2 production `panic!()` to `anyhow::bail!()`; 3 in mil_lower.rs are intentional safety-net guards | ~~HIGH~~ **Fixed** |
| CQ-2 | 3 `panic!()` in MIL lowering — intentional guards | `mil_lower.rs:943,1412,3320` | Intentional safety-net guards — left as-is with documentation | MEDIUM |
| ~~CQ-3~~ | ~~~20 `.unwrap()` in weight file I/O~~ | ~~`weights.rs:530-652`~~ | **RETRACTED** — All in test code; production code uses `Result`/`bail!()` | ~~HIGH~~ **RETRACTED** |
| CQ-4 | `panic!()` in legality passthrough | `legality_rewrite.rs:3903,4271` | **Fixed by T-54:** Converted to `bail!()` with `Result` return type | ~~MEDIUM~~ **Fixed** |
| CQ-5 | `eprintln!` in library function | `ane_hw_limits.rs:77-80` | Use `log::warn!()` instead | LOW |
| CQ-6 | Deprecated module still compiled | `kv_cache_rewrite` (pub(crate)) | Gate behind feature flag or remove entirely | MEDIUM |
| CQ-7 | ~~`ModelArchConfig::default()` hardcodes Qwen3-0.6B~~ | `common.rs:248-258` | **Fixed by T-56:** Added `qwen3_0_6b()` factory method; Default delegates with deprecation notice | ~~HIGH~~ **Fixed** |
| CQ-8 | ~~Bridge defaults to Qwen3 architecture~~ | `mir_to_compat.rs:455` | **Fixed by T-57:** Added `log::warn!()` when defaulting; deprecation warnings | ~~HIGH~~ **Fixed** |
| CQ-9 | ~~Shape inference defaults to 32768 max_seq_len~~ | `shape_inference.rs:72,566` | **Partially Fixed by T-57:** Deprecation warnings added. Full fix requires making max_seq_len a required parameter. | ~~HIGH~~ MEDIUM |
| CQ-10 | Hardcoded 896/768 in test fixtures | `staticize.rs:167`, `sir_build.rs:1753,2204` | Use named constants or ModelArchConfig | MEDIUM |
| CQ-11 | Model-specific precision hazard in code | `precision_policy.rs:247` | Move to knowledge seed file | MEDIUM |
| CQ-12 | Unused variables in test compilation | `mir_engine_test.rs:962`, `coreml-proto/src/lib.rs:7088` | Prefix with `_` or remove | LOW |
| CQ-13 | Unused imports in test files | `pipeline.rs:7-12`, `should_panic.rs:6`, `report_test.rs:6` | Remove unused imports | LOW |
| CQ-14 | Deprecated test still runs | `kv_cache_rewrite.rs:225-310` | Move to ignored or remove | LOW |
| CQ-15 | `cargo fmt` drift (3 files) | `bridge/mir_to_compat.rs`, `ir/mir.rs`, `coreml-proto/src/lib.rs` | Run `cargo fmt --all` | LOW |
| CQ-16 | K/V projection patterns fetched then discarded | `mir_to_compat.rs:470-471` | Either use K/V patterns for GQA alias mapping or remove the fetch entirely | HIGH |
| CQ-17 | `LM_HEAD_SHARD_SIZE = 19000` hardcoded | `safetensors_resolver.rs:293` | Derive from model vocab_size or ANE HW limits; same pattern as I-30/I-31 | HIGH |
| CQ-18 | `resolve_shard` assumes FP16 (2 bytes/element) | `safetensors_resolver.rs:319` | Use actual element size from the tensor's dtype; wrong for F32/INT8/UInt8 | HIGH |
| CQ-19 | PythonBridge `timeout_secs` field never enforced | `subprocess.rs:28,65-69` | Use `Command::new().stdin().stdout().stderr().spawn()` with timeout, or `process::Command` with deadline | MEDIUM |
| CQ-20 | `compare_with_python_bridge` is dead code stub | `emitter.rs:129-147` | Implement or remove; always returns `None` for all fields | MEDIUM |
| CQ-21 | `.unwrap()` on `write!` in `compute_task_hash` | `session.rs:104-110` | Use `write!` without unwrap or `.expect("hash string allocation")` | MEDIUM |
| CQ-22 | `coreml_model_destroy` FFI unsoundness | `capi.rs:186-192` | `Box::from_raw` on handle not allocated with `Box::new()`; latent UB if `coreml_model_load` is ever implemented | MEDIUM |
| CQ-23 | `MirOpCompat::Fill input_names()` returns empty vec | `lib.rs (coreml-proto):1078` | Fill's `shape` is an input in Core ML MIL; should return shape input name | MEDIUM |
| CQ-24 | Empty resolver returned without warning | `safetensors_resolver.rs:135-169` | Log `log::warn!()` when all resolution strategies fail; currently silently produces zero-filled weights | MEDIUM |
| CQ-25 | `compat_input_dtype` uses string matching for `input_ids` | `shape_inference.rs:33-38` | Use MIR node's declared dtype instead of name-based heuristics | LOW |
| CQ-26 | Dead-code `mir_node_to_compat` with `#[allow(dead_code)]` | `mir_to_compat.rs:565` | Remove or document why kept | LOW |
| CQ-27 | BF16→FP16 conversion missing edge-case tests | `safetensors_resolver.rs:370-386` | Add tests for NaN, Inf, subnormals, negative zero | LOW |

---

## IV. BUG REPORT

| # | Symptom | Trigger | Fix Direction | Severity |
|---|---|---|---|---|
| B-1 | ~~`MILSliceUpdate` silently fails at emission~~ | Any model using slice_update op | **Fixed by T-47:** Moved to `None`; added to CPU_ONLY_OPS | ~~CRITICAL~~ **Fixed** |
| B-2 | ~~`MILReverse` silently fails at emission~~ | Any model using reverse op | **Fixed by T-47** | ~~CRITICAL~~ **Fixed** |
| B-3 | ~~`MILSlidingWindows` silently fails at emission~~ | Any model using sliding_windows op | **Fixed by T-47** | ~~CRITICAL~~ **Fixed** |
| B-4 | ~~`MILArgsort` silently fails at emission~~ | Any model using argsort op | **Fixed by T-47** | ~~CRITICAL~~ **Fixed** |
| B-5 | ~~Palettization decisions silently ignored~~ | Any model with LUT/palettized weights | **Fixed by T-48:** Added `palette_bits` field with validation and clamping | ~~CRITICAL~~ **Fixed** |
| ~~B-6~~ | ~~FP32 broadcast allowed on A13~~ | ~~A13 hardware with non-FP16 broadcast~~ | **RETRACTED** — Per-family support matrix confirms A13 broadcast = ✅ | ~~HIGH~~ **RETRACTED** |
| B-7 | ~~ReduceMin Int8 allowed on A11-A13~~ | Non-FP ReduceMin on pre-A14 hardware | **Fixed by T-51:** Added guard in placement validator | ~~HIGH~~ **Fixed** |
| B-8 | ~~E4M3 denied on A17 Pro hardware~~ | V11→A16 family doesn't support E4M3 | **Fixed by T-52:** Added `AneFamily::A17` variant | ~~HIGH~~ **Fixed** |
| B-9 | Tile reshape zeros resolved incorrectly | Tile with multiple zero placeholders and ctx=None | Use ctx dimensions when available; require ctx for Tile | HIGH |
| B-10 | ~~HW tensor dimension limits not enforced~~ | Large tensors pass placement but fail at ANE runtime | **Fixed by T-53:** Wired `validate_tensor_dims()` into placement pipeline | ~~HIGH~~ **Fixed** |
| B-11 | Sampler Gather forces CPU fallback | Sampler decomposition produces Gather (CPU-only) | Replace Gather with SliceByIndex or mark Sampler CPU-only | MEDIUM |
| B-12 | Attention reshape placeholder zero | Batch dim > 1 with zero placeholder | Use ctx.batch_size when available | MEDIUM |
| B-13 | MILNeg passes CPU-only gate (name mismatch) | Any graph containing MILNeg on ANE path | Fix CPU_ONLY_OPS: add `"neg"`, remove `"negative"` | CRITICAL |
| B-14 | `extract_whdc()` swaps depth↔channels for NCHW | Rank-4 tensor [1, 64, 128, 128] → d=64, c=1 (should be d=1, c=64) | Fix NCHW interpretation: `(shape[3], shape[2], 1, shape[1])` for rank 4 | HIGH |
| B-15 | Pooling kernel_size never validated | Large kernel sizes pass placement but may fail at ANE runtime | Implement kernel_size validation against revision-specific HW limits | HIGH |
| B-16 | Float64 element_size returns 4 instead of 8 | Float64 weight constants produce 50%-size weight entries | Fix: `CoreMlDataType::Float64 => 8` | HIGH |
| B-17 | K/V alias maps dropped for GQA models | Non-Qwen3 models with separate K/V projections | Use k_proj/v_proj patterns to build K/V alias entries | HIGH |
| B-18 | `resolve_shard` assumes FP16 byte offsets | Non-FP16 weights (F32, INT8) get wrong byte slicing | Use actual dtype element_size for byte offset calculation | HIGH |
| B-19 | LM_HEAD_SHARD_SIZE=19000 is Qwen3-specific | Non-Qwen3 models get wrong shard sizes | Derive from model vocab_size or ANE HW limits | HIGH |

---

## V. TEST COVERAGE MAP

Updated with newly-discovered zero-coverage modules:

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
| **bridge::mir_to_compat::alias_map** | 1 | 0 | 0% | 🔴 Critical |
| **coreml-ffi::api** | 5 | 0 | 0% | 🔴 Critical |
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

**Total zero-coverage modules:** 10 (up from 6 in v2 audit)
**Total zero-coverage pub fn:** ~70 (up from ~48 in v2 audit)

---

## VI. IR CLEANLINESS SCORE

```
┌─────────────────────────────────────────────┐
│  IR CLEANLINESS SCORE                       │
│                                             │
│  SIR ──█████████████████████████░░  90%       │
│  AIR ──████████████████████████░░░  88%       │
│  MIR ──███████████████████████░░░░  86%       │
│  PIR ──█████████████████████████░░  90%       │
│                                             │
│  OVERALL: ████████████████████████░░  87%     │
│                                             │
│  Deductions from 100%:                      │
│  - CPU_ONLY_OPS name mismatches (T-49): -3% │
│  - extract_whdc channel/depth swap: -3%      │
│  - Float64 element_size bug: -2%             │
│  - K/V alias maps dropped: -2%              │
│  - Pooling kernel_size discarded: -1%        │
│  - Palettize Qwen3 name heuristics: -1%      │
│  - Zero test coverage for 10 mods: -1%       │
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
│  + Zero clippy errors, all tests pass        │
│  + 1252 tests (up from 566 in v1)           │
└─────────────────────────────────────────────┘
```

---

## VII. RECOMMENDED SPRINT BACKLOG

Sorted by **impact × urgency** (highest first):

| Rank | Task | Impact | Urgency | Effort | Issue Ref | Status |
|---|---|---|---|---|---|---|
| 1 | **Fix CPU_ONLY_OPS name mismatches**: Add `"neg"`, remove `"negative"` and `"reverse_square_root"`, fix `"rint"`→`"round"`, remove dead entries `"reciprocal"` and `"signbit"` | CRITICAL | NOW | S (0.5d) | I-41, I-42 | ⬜ Open |
| 2 | **Fix `extract_whdc()` NCHW dimensional swap**: Rank-4 NCHW [B,C,H,W] should extract `(W, H, 1, C)` not `(W, H, C, B)` | HIGH | NOW | S (0.5d) | I-43 | ⬜ Open |
| 3 | **Fix `Float64.element_size()` returning 4 instead of 8**: Add Float64 case to element_size() match arm | HIGH | NOW | S (0.5d) | I-46 | ⬜ Open |
| 4 | **Wire pooling kernel_size validation**: Remove `let _ = kernel_size;`, implement per-family kernel limit checks | HIGH | NOW | S (0.5d) | I-44 | ⬜ Open |
| 5 | **Fix K/V projection alias map drop**: Use k_proj/v_proj patterns to build separate alias entries for GQA models | HIGH | NEXT | S (0.5d) | I-45 | ⬜ Open |
| 6 | **Fix palettize Qwen3 name heuristics**: Use `ModelArchitecture` for weight classification instead of hardcoded name patterns | HIGH | NEXT | M (1d) | I-47 | ⬜ Open |
| 7 | **Fix `LM_HEAD_SHARD_SIZE` hardcoding**: Derive from model vocab_size or ANE HW limits | HIGH | NEXT | S (0.5d) | I-48 | ⬜ Open |
| 8 | **Fix `resolve_shard` FP16-only byte offsets**: Use actual dtype element_size | HIGH | NEXT | S (0.5d) | I-49 | ⬜ Open |
| 9 | **Add tests for ir::payload, ir::shard_desc, ir::serialize**: Three 0%-coverage modules with 30 pub fn total | HIGH | NEXT | L (3d) | I-32 | ⬜ Open |
| 10 | **Add tests for lab::session, lab::harness, lab::fallback**: Three 0%-coverage critical modules | HIGH | NEXT | L (2d) | I-33 | ⬜ Open |
| 11 | **Fix Tile decomposition placeholder zeros**: Use ctx dimensions when available; document ctx requirement | MEDIUM | NEXT | S (0.5d) | I-34 | ⬜ Open |
| 12 | **Add cross-validation test for Python vs Rust emission**: Structural equivalence test for same MIR input | MEDIUM | LATER | M (1d) | I-35 | ⬜ Open |
| 13 | **Centralize palette bit-width validation**: Single `validate_palette_bits()` in ane_layout | MEDIUM | LATER | S (0.5d) | I-38 | ⬜ Open |
| 14 | **Unify CPU-only classification**: Derive CPU_ONLY_OPS from `default_engine() == None` | MEDIUM | LATER | M (1d) | I-39 | ⬜ Open |
| 15 | **Add remaining MirOpCompat variants**: Close compat coverage gaps for ops with real ANEC converters | MEDIUM | LATER | M (2d) | I-40 | ⬜ Open |
| 16 | **Enforce PythonBridge timeout**: Wire `timeout_secs` into subprocess execution | MEDIUM | LATER | S (0.5d) | I-52 | ⬜ Open |
| 17 | **Fix FFI `coreml_model_destroy` unsoundness**: Ensure allocation matches deallocation contract | MEDIUM | LATER | S (0.5d) | I-50 | ⬜ Open |
| 18 | **Remove dead-code `compare_with_python_bridge`**: Implement or remove stub | MEDIUM | LATER | S (0.5d) | I-53 | ⬜ Open |
| 19 | **Log warning when safetensors resolver is empty**: Prevent silent zero-filled weights | MEDIUM | LATER | S (0.5d) | I-54 | ⬜ Open |
| 20 | **Fix Fill op `input_names()` returning empty vec**: Include shape input name | MEDIUM | LATER | S (0.5d) | I-55 | ⬜ Open |
