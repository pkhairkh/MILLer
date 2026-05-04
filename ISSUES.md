# MILLer Compiler — Issue Tracker

*Last updated: 2026-05-04 (NECROSCOPY forensic audit — I-66 through I-161 added from ane-violations.md 138-violation catalog)*
*Reference implementation: https://huggingface.co/pkhairkh/qwen3-coreml-palettized*
*Audit source: `docs/audit/ane-violations.md` (expanded 2026-05-08 with deep binary forensic evidence)*

---

## Resolved Issues — Archive Summary

Issues I-01 through I-65 from the v1/v2/v3 tabula-rasa audits are all resolved. See `CHANGELOG.md` for details.

| Audit | Issues | Status |
|-------|--------|--------|
| v1 (I-01 through I-20) | 20 | All fixed |
| v2 (I-21 through I-40) | 20 | 18 fixed, 2 retracted |
| v3 (I-41 through I-65) | 25 | All fixed |

---

## P0 — CRITICAL (Silent Miscompilation / Data Corruption)

### I-66 · Knowledge Seed Family Mismatches Grant Wrong Capabilities to Hardware

**Status:** ⬜ Open
**Files:** `knowledge/ane_hw_limits_seed.json:40-55,108-123`, `crates/ir/src/ane_target.rs`
**AUDIT ref:** V-001, V-002
**Severity:** CRITICAL
**Effort:** S (0.5 day)
**Task:** T-91

**Intent:** The `ane_hw_limits_seed.json` maps hardware revision V6 to family "A14" but Rust code (`ane_target.rs`) maps V6→A13. Similarly, V11→family "A16" in the JSON but V11→A17 in Rust. These mismatches grant A14-class capabilities to A13 hardware and miss A17 E4M3 support. Every model compiled for A13 or A17 hardware uses incorrect constraint data, potentially placing ops on the ANE that the hardware cannot execute, causing silent runtime failures.

**Current behavior:** JSON seed file's family assignments for V6 and V11 contradict the Rust source of truth. No test validates consistency between these two representations.

**Fix direction:** (1) Change V6 family to "A13" and V11 family to "A17" in `ane_hw_limits_seed.json` to match Rust. (2) Add CI test `test_hw_limits_seed_family_consistency()` that cross-validates every V→family mapping against `revision_to_family()`. (3) Add inline comment in JSON documenting that Rust code is the source of truth.

**Definition of Done:**
- [ ] V6→A13 and V11→A17 in JSON matching Rust code
- [ ] Cross-validation test passes for ALL V→family mappings
- [ ] No other V→family mismatches exist

---

### I-67 · Comparison Ops Wrongly Classified as CPU-Only

**Status:** ⬜ Open
**Files:** `knowledge/cpu_only_ops_seed.json:296-324`, `knowledge/ane_op_family_matrix.json`
**AUDIT ref:** V-003
**Severity:** CRITICAL
**Effort:** S (0.5 day)
**Task:** T-92

**Intent:** Comparison operations (equal, not_equal, greater, less, greater_equal, less_equal) are listed as CPU-only in `cpu_only_ops_seed.json`, but binary forensic evidence shows ConvertBinaryCompare ANEC converters exist for all of them with MinimumFamily=A11Legacy. These ops are ANE-legal on all hardware families from A11 onward, yet MILLer forces them to CPU, causing unnecessary CPU fallback and performance degradation for any model using comparison operations on the ANE path.

**Current behavior:** `cpu_only_ops_seed.json` contains comparison ops preventing ANE placement, contradicting binary evidence showing dedicated ANEC converters.

**Fix direction:** (1) Remove comparison ops from `cpu_only_ops_seed.json`. (2) Add comparison ops to `ane_op_family_matrix.json` with A14+ scope. (3) Add cross-validation test ensuring no op appears as both CPU-only and ANE-supported.

**Definition of Done:**
- [ ] Comparison ops removed from cpu_only_ops_seed.json
- [ ] Comparison ops added to ane_op_family_matrix.json with A14+ scope
- [ ] Cross-validation test passes

---

### I-68 · Phantom Logical AND/OR/NOT Support Claims

**Status:** ⬜ Open
**Files:** `knowledge/ane_op_family_matrix.json:1239-1285`
**AUDIT ref:** V-004
**Severity:** CRITICAL
**Effort:** S (0.25 day)
**Task:** T-92

**Intent:** Logical AND/OR/NOT are listed as "supported" A12+ in `ane_op_family_matrix.json` but have no ANEC converter — the per-op documentation confirms they never land on the ANE. Binary evidence shows `anec.equal_zero` covers NOT but no dedicated `logical_and`/`logical_or` converters exist. If these ops are placed on the ANE based on the family matrix claim, they will fail at ANEC compilation time with no prior warning.

**Current behavior:** The family matrix claims logical AND/OR/NOT are ANE-supported A12+, but no converter exists to lower them.

**Fix direction:** (1) Change logical_and/or/not from "supported" to "cpu_only" for all families. (2) Document that `anec.equal_zero` covers logical NOT for comparison-against-zero. (3) Add test validating no "supported" op lacks a converter.

**Definition of Done:**
- [ ] Logical AND/OR/NOT marked as cpu_only in ane_op_family_matrix.json
- [ ] anec.equal_zero NOT coverage documented
- [ ] Cross-validation test passes

---

### I-69 · Gather Wrongly Declared ANE-Illegal

**Status:** ⬜ Open
**Files:** `knowledge/legality_seed.json:62-75`
**AUDIT ref:** V-005
**Severity:** CRITICAL
**Effort:** S (0.25 day)
**Task:** T-92

**Intent:** `mb.gather` is declared ANE-illegal (ane_legal: false) in `legality_seed.json`, but `anec.gather` exists with ConvertGather converter (family-agnostic, MinimumFamily=A11Legacy). The blanket illegal claim is wrong — Gather IS ANE-legal when the axis is a constant. The current classification forces all Gather ops to CPU even when they could execute on the ANE, causing unnecessary performance degradation for embedding lookups and other constant-axis gather patterns.

**Current behavior:** `legality_seed.json` declares `ane_legal: false` for Gather, blocking it from ANE entirely.

**Fix direction:** (1) Change `mb.gather` to `ane_legal: true` with constraint tag `limited_index_range` and notes about constant-axis requirement. (2) Add validation that rejects dynamic-axis Gather targeting ANE while allowing constant-axis Gather.

**Definition of Done:**
- [ ] Gather marked as ane_legal:true with limited_index_range constraint
- [ ] Dynamic-axis Gather rejected at validation time
- [ ] Constant-axis Gather allowed on ANE

---

### I-70 · Zero-Fill Weight Placeholders Produce Silently Broken Models

**Status:** ⬜ Open
**Files:** `crates/bridge/src/mir_to_compat.rs:224-249`, `crates/bridge/src/safetensors_resolver.rs:135-169`
**AUDIT ref:** V-007
**Severity:** CRITICAL
**Effort:** M (1 day)
**Task:** T-95

**Intent:** When a weight cannot be resolved, MILLer silently produces zero-filled placeholder data. The model compiles and loads successfully but produces completely incorrect inference — all zero-weight outputs are meaningless. The only indication is a `log::warn!()` message that users miss. Partial failures (some weights resolved, others zero-filled) are even more dangerous because the model appears to work for some operations but produces wrong results for others.

**Current behavior:** `safetensors_resolver.rs` returns empty resolver, `mir_to_compat.rs` uses EmptyWeightResolver that fills all weights with zeros. No compilation error is raised.

**Fix direction:** (1) Change zero-fill fallback to return `Err(...)`. (2) Add `--allow-missing-weights` CLI flag (default: false) that restores old behavior. (3) Fail compilation with clear error listing unresolvable weights.

**Definition of Done:**
- [ ] Compilation fails with hard error when any weight resolves to zero-fill (default)
- [ ] `--allow-missing-weights` flag restores old behavior
- [ ] Error message lists all unresolvable weight names
- [ ] Test verifies default-fail and flag-allow paths

---

### I-71 · MILConcat Emission Violates Orion #1 — Concat Rejected by ANE Compiler

**Status:** ⬜ Open
**Files:** `crates/passes/src/mil_lower.rs:2842-2858`, `crates/passes/src/legality_rewrite.rs:3098,3622`, `python/mil_emitter.py:432`
**AUDIT ref:** V-098, V-130 (Orion #1)
**Severity:** CRITICAL
**Effort:** L (3 days)
**Task:** T-93

**Intent:** MILLer emits MILConcat in three critical paths: SDPA decomposition, RoPE rotate_half, and embedding gather. Orion #1 documents that concat is rejected by the ANE compiler. Binary forensic confirms extensive concat constraints: channel-axis-only, const-positive-axis, no interleaved on some dimensions, no symbolic shape. All models using SDPA decomposition will fail ANE compilation because the concat node cannot be lowered to ANEC dialect. This is the single most impactful violation.

**Current behavior:** `mil_lower.rs:2842-2858` emits concat in SDPA fallback, `legality_rewrite.rs:3098,3622` emits concat(-x2, x1, axis=-1) for RoPE, `mil_emitter.py:432` emits concat for embedding gather. All produce concat nodes rejected by ANEC.

**Fix direction:** (1) SDPA path: replace concat with reshape+stack or emit as fused SDPA for A16+. (2) RoPE: replace concat(-x2, x1, axis=-1) with reshape+transpose sequence. (3) Embedding gather: replace concat with reshape+stack or SliceByIndex. (4) Add test that no MILConcat survives in MIR after legality rewriting.

**Definition of Done:**
- [ ] No MILConcat emission in SDPA decomposition
- [ ] No MILConcat emission in RoPE rotate_half
- [ ] No MILConcat emission in embedding gather
- [ ] Test verifies concat-free MIR for standard topologies

---

### I-72 · Gelu EXACT Mode Unsupported by ANEC — SIR Builder Hardcodes Wrong Mode

**Status:** ⬜ Open
**Files:** `crates/trace/src/sir_build.rs:518,1415`, `crates/passes/src/role_mir.rs:252`, `python/mil_emitter.py:893,1142`
**AUDIT ref:** V-113, V-099 (Orion #10)
**Severity:** CRITICAL
**Effort:** M (1.5 days)
**Task:** T-94

**Intent:** The SIR builder hardcodes Gelu mode="EXACT" but ANEC's ConvertElementwiseUnary(Gelu) only supports tanh approximation. The SIR→AIR→MIR pipeline preserves the mode, so Rust-compiled models emit mb.gelu(mode="EXACT") which ANEC rejects. Meanwhile, Python emitter and role_mir.rs use "TANH_APPROXIMATION". The Rust and Python paths produce incompatible gelu modes. Orion #10 documents gelu is not a valid standalone MIL activation — it must use tanh approximation.

**Current behavior:** `sir_build.rs:518,1415` sets mode="EXACT". This passes through the entire pipeline and is rejected by ANEC.

**Fix direction:** (1) Change "EXACT" to "TANH_APPROXIMATION" in sir_build.rs:518,1415. (2) Add validation rejecting non-TANH_APPROXIMATION gelu for ANE targets. (3) Update test fixtures. (4) Document ANEC limitation in SIR reference.

**Definition of Done:**
- [ ] sir_build.rs uses "TANH_APPROXIMATION" for all gelu
- [ ] Validation rejects non-TANH_APPROXIMATION gelu for ANE
- [ ] Test fixtures updated
- [ ] Rust and Python paths produce consistent gelu mode

---

## P1 — HIGH (Missing Enforcement / Model Leakage / Untested Paths)

### I-73 · Conv/Pool Constraint Fields Defined But Never Validated
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_hw_limits.rs:148-193`, `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-009 | **Severity:** HIGH | **Effort:** S (as part of T-97) | **Task:** T-97

**Intent:** AneHwLimits defines 7 conv/pool/PE-specific constraint fields but `validate_tensor_dims()` never validates them. Conv/pool ops with oversized kernels pass validation but fail at ANE emission, providing no guidance about which constraint was violated.

**Fix direction:** Wire the 7 constraint fields into validate_tensor_dims() with per-op validation functions for conv, pool, and PE operations.

**Definition of Done:** All 7 AneHwLimits constraint fields validated; per-op validation functions for conv, pool, PE; clear error messages.

---

### I-74 · default_engine() Not Revision-Aware
**Status:** ⬜ Open | **Files:** `crates/ir/src/mir.rs:1061-1311` | **AUDIT ref:** V-010 | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-97 (partial)

**Intent:** `default_engine()` returns static engine assignment per op regardless of AneRevision. Ops assigned to PE may be placed on families that don't support them, causing silent emission failure.

**Fix direction:** Make default_engine() revision-aware by cross-referencing AneFamily capability methods. Return None for ops not supported on the target family.

**Definition of Done:** default_engine() considers AneFamily; ops without converter on a family return None; test verifies family-specific engine assignments.

---

### I-75 · Unknown Dtype String Silently Defaults to Fp16
**Status:** ⬜ Open | **Files:** `crates/ir/src/shard_desc.rs:95` | **AUDIT ref:** V-011 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** When parsing dtype strings in shard_desc.rs, unknown strings silently default to Fp16. Invalid dtype strings like "bf16" or "int8" produce wrong precision without error, causing data corruption when int8 data is interpreted as fp16.

**Fix direction:** Replace silent Fp16 default with explicit error listing valid dtype strings. Add Int8 to accepted list.

**Definition of Done:** Unknown dtype strings produce explicit errors; Int8 added to accepted list; error lists valid dtypes.

---

### I-76 · Generic Architecture Falls Back to Qwen3 Weight Patterns
**Status:** ⬜ Open | **Files:** `crates/ir/src/common.rs:297-308`, `crates/bridge/src/mir_to_compat.rs:458-468` | **AUDIT ref:** V-012, V-025 | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-101

**Intent:** When model architecture is not recognized, MILLer silently falls back to Qwen3 weight name patterns. Non-Qwen3/LLaMA models will have silently broken weight resolution — their weight tensors won't match Qwen3 patterns, producing models with undefined references. Same model-leakage pattern as I-30/I-31.

**Fix direction:** Remove Generic→Qwen3 fallback. Return error when architecture is unrecognized, requiring explicit specification.

**Definition of Done:** No Generic→Qwen3 fallback; unrecognized architectures produce explicit errors; ModelArchConfig::unspecified() available.

---

### I-77 · FamilyPayload Hardcodes stateful: false
**Status:** ⬜ Open | **Files:** `crates/ir/src/payload.rs:685-698` | **AUDIT ref:** V-013 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-103

**Intent:** FamilyPayload hardcodes `stateful: false` regardless of actual op. Stateful ops (DecodeStep with KV cache) emitted via generic path get wrong function descriptors. Non-stateful flag on stateful ops causes Core ML to not preserve state across invocations, breaking autoregressive generation.

**Fix direction:** Derive stateful flag from actual op type. Check for DecodeStep and other stateful ops. Add is_stateful() method.

**Definition of Done:** stateful flag derived from op type; DecodeStep returns stateful:true; test verifies flag for both stateful and non-stateful ops.

---

### I-78 · StaticizePass Is Pure Pass-Through (Phantom Pass)
**Status:** ⬜ Open | **Files:** `crates/passes/src/staticize.rs:43-46` | **AUDIT ref:** V-014 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-103

**Intent:** StaticizePass::run() is `Ok(input)` — pure pass-through doing nothing. Documentation claims it replaces symbolic dimensions, resolves variable-length sequences, and records decisions. None are implemented. This phantom pass wastes developer trust.

**Fix direction:** Either implement or remove from pipeline and document that static dimension resolution is not yet implemented.

**Definition of Done:** StaticizePass either implemented or removed; documentation clearly states gap; no phantom passes.

---

### I-79 · StateTopologyPass Only Logs Warnings, Never Returns Err
**Status:** ⬜ Open | **Files:** `crates/passes/src/state_topology.rs:43-96` | **AUDIT ref:** V-016 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-103

**Intent:** StateTopologyPass claims to "verify" and "ensure" state patterns but only logs warnings — never returns Err. Invalid state patterns pass through silently, causing undefined behavior at inference time.

**Fix direction:** Change return type to Result. Return Err for invalid patterns in strict mode. Add strict: bool flag.

**Definition of Done:** Returns Err for invalid patterns in strict mode; advisory mode documented; tests for both modes.

---

### I-80 · FunctionEntry TensorSpec Shapes Hardcoded as vec![1,1]
**Status:** ⬜ Open | **Files:** `crates/passes/src/shard_plan.rs:367-378` | **AUDIT ref:** V-017 | **Severity:** HIGH | **Effort:** M (as part of T-112) | **Task:** T-112

**Intent:** FunctionEntry shapes are hardcoded as vec![1,1] despite comments saying "derived from graph". Wrong shapes cause incorrect function descriptors in the compiled model, leading to shape mismatches at inference time.

**Fix direction:** Derive shapes from MIR graph. Walk the graph to extract actual batch/seq dimensions.

**Definition of Done:** Shapes derived from MIR graph; no hardcoded vec![1,1] in production paths; tests verify derived values.

---

### I-81 · MatMul Inner-Dim Mismatch Only Logs Warning
**Status:** ⬜ Open | **Files:** `crates/passes/src/mil_lower.rs:92-98` | **AUDIT ref:** V-018 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** MatMul inner-dimension mismatch only logs a warning and continues, producing a graph with wrong dimensions. Shape mismatch is a correctness violation, not an advisory warning. A MatMul with mismatched inner dimensions produces silently incorrect results.

**Fix direction:** Replace warning with bail!("MatMul inner dimension mismatch: {} vs {}", lhs_inner, rhs_inner).

**Definition of Done:** MatMul inner-dim mismatch causes compilation failure; error includes both dimensions; test verifies failure.

---

### I-82 · Gather in CPU_ONLY_OPS But Actively Emitted for ANE
**Status:** ⬜ Open | **Files:** `crates/passes/src/cpu_only_ops.rs:147-156`, `crates/passes/src/mil_lower.rs`, `crates/passes/src/legality_rewrite.rs` | **AUDIT ref:** V-019, V-136 | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-108

**Intent:** Gather is listed in CPU_ONLY_OPS but mil_lower actively emits MILGather for embedding lookup and legality_rewrite generates Gather for RoPE table lookups. Binary confirms non-constant axis gather is rejected. This contradiction forces all gather ops off the ANE or causes ANEC failures.

**Fix direction:** Remove Gather from CPU_ONLY_OPS. Add const-axis validation: constant-axis Gather is ANE-legal, dynamic is not. Replace RoPE Gather with const-axis or SliceByIndex.

**Definition of Done:** Gather removed from CPU_ONLY_OPS; const-axis Gather classified as ANE-legal; dynamic-axis rejected; RoPE and embedding paths use const-axis.

---

### I-83 · Interleave Constraints Skipped When Channels Unknown
**Status:** ⬜ Open | **Files:** `crates/passes/src/placement_validate.rs:272-292` | **AUDIT ref:** V-020 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** Interleave constraints are skipped entirely when channels is None, including non-channel-dependent checks (const→1, int4→8). Missing validation allows invalid dtype/interleave combinations.

**Fix direction:** Enforce non-channel-dependent checks (const→1, int4→8) even when channels is None. Only skip channel-count-dependent checks.

**Definition of Done:** Int4/UInt4 interleave==8 validated regardless of channels; const interleave==1 validated regardless.

---

### I-84 · Knowledge claims_agree Defaults True for 7/8 Types
**Status:** ⬜ Open | **Files:** `crates/knowledge/src/transfer.rs:152` | **AUDIT ref:** V-021 | **Severity:** HIGH | **Effort:** M (as part of T-110) | **Task:** T-110

**Intent:** claims_agree() defaults to true for 7/8 knowledge types without field comparison. Conflicting knowledge entries are treated as consistent, preventing detection of contradictions.

**Fix direction:** Implement claims_agree for all 8 types with field-level comparison. At minimum, add comparison for PrecisionHazard, SurvivalMatrixEntry, ShardTemplateKnowledge.

**Definition of Done:** claims_agree implemented for all 8 types; field-level comparison for at least 3 types; test validates conflict detection.

---

### I-85 · Knowledge Conflict Detection Not Symmetric
**Status:** ⬜ Open | **Files:** `crates/knowledge/src/store.rs:524-547` | **AUDIT ref:** V-022 | **Severity:** HIGH | **Effort:** S (as part of T-110) | **Task:** T-110

**Intent:** Conflict detection marks new entry B as ConflictedWith(A) but never back-patches A. Querying A shows no conflict indication. Half of all conflicts are invisible.

**Fix direction:** Make conflict detection symmetric. Mark both A and B as conflicted. Add conflict_group linking all mutually conflicting entries.

**Definition of Done:** Both conflicting entries marked; conflict group links mutual conflicts; test verifies symmetry.

---

### I-86 · I/O Node Fallback Produces Wrong Shapes and Dtypes
**Status:** ⬜ Open | **Files:** `crates/bridge/src/mir_to_compat.rs:275-413` | **AUDIT ref:** V-023 | **Severity:** HIGH | **Effort:** M (as part of T-104) | **Task:** T-104

**Intent:** When input/output nodes are missing from MIR graph, fallback to shape vec![1] and dtype Fp16. These defaults are almost certainly wrong for any real model. Using wrong shapes produces incorrect I/O descriptors.

**Fix direction:** Replace fallback with hard errors. Fail compilation rather than emitting wrong descriptors.

**Definition of Done:** Missing I/O nodes cause compilation failure; error identifies missing node; no silent wrong defaults.

---

### I-87 · F32 Weight Data Not Converted to FP16
**Status:** ⬜ Open | **Files:** `crates/bridge/src/safetensors_resolver.rs:196-199` | **AUDIT ref:** V-026 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-105

**Intent:** F32 weight data passed through without FP16 conversion. BF16 gets converted but F32 does not. F32 bytes written as-is but declared as FP16 in proto produces 50% data corruption.

**Fix direction:** Add F32→FP16 conversion when target dtype is FP16. Ensure data format matches proto declaration.

**Definition of Done:** F32→FP16 conversion when target is FP16; proto matches data; test verifies conversion.

---

### I-88 · Bool/Float64/Unknown Dtype Silently Mapped to Float32 Blob
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/src/weights.rs:116-119` | **AUDIT ref:** V-027 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-105

**Intent:** Bool, Float64, and Unknown dtypes silently mapped to Float32 blob format. Data-corrupting for Bool (1-bit packed as 4-byte float), incorrect for Float64 (8-byte as 4-byte), meaningless for Unknown.

**Fix direction:** Map Bool to dedicated blob or reject. Map Float64 to 8-byte blob. Reject Unknown with clear error.

**Definition of Done:** Bool has dedicated format or is rejected; Float64 uses 8-byte blob; Unknown rejected; no silent Float32 fallback.

---

### I-89 · State Declarations Default to Empty Shape + Fp16
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/src/mir_to_proto.rs:339-356` | **AUDIT ref:** V-028 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** State declarations default to empty shape + Fp16 when only a write op is present. Core ML rejects proto with empty-dimension state tensors. Affects stateful ops where initial state shape cannot be inferred from a read.

**Fix direction:** Derive state shape from ReadState op. If no ReadState exists, require explicit shape specification. Fail if shape cannot be determined.

**Definition of Done:** State shape derived from ReadState when available; explicit shape required otherwise; no empty-dimension states.

---

### I-90 · Softmax/InstanceNorm Architecture-Conditional Rejection Not Modeled
**Status:** ⬜ Open | **Files:** `knowledge/ane_op_family_matrix.json:806-821,951-965` | **AUDIT ref:** V-029, V-030, V-101 | **Severity:** HIGH | **Effort:** M (as part of T-96) | **Task:** T-96

**Intent:** Softmax and InstanceNorm listed as "supported" for all families but binary evidence shows architecture-conditional rejection. Converters exist (family-agnostic) but specific architecture variants reject them. This nuance is not captured.

**Fix direction:** Add architecture_conditional: true flag for both. Document which architectures reject them. Update constraint model.

**Definition of Done:** Both ops have architecture_conditional flag; documented architecture rejections; constraint model handles conditional support.

---

### I-91 · Conv 32K-Channel Limit Not Enforced (Orion #16)
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_hw_limits.rs:178`, `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-103 (Orion #16) | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-97

**Intent:** max_tensor_channels allows 65536 but conv-specific limit is 32768 per Orion #16. Models with conv channels 32768-65536 pass MILLer validation but fail at ANEC.

**Fix direction:** Add conv-specific 32K-channel limit in op_constraints.rs, distinct from general max_tensor_channels.

**Definition of Done:** Conv channels > 32768 rejected; separate from general limit; test verifies rejection.

---

### I-92 · Multi-Output/Input Surface Ordering and Uniformity Not Enforced (Orion #2, #3, #18, #19)
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/`, `crates/bridge/src/mir_to_compat.rs` | **AUDIT ref:** V-105, V-117, V-118, V-119, V-120 | **Severity:** HIGH | **Effort:** L (2 days) | **Task:** T-99

**Intent:** Four Orion constraints unenforced: output buffer uniformity (0x1d error), output surface alphabetical ordering (silent data corruption), input surface alphabetical ordering (silent corruption), input surface uniformity (0x1d error). Ordering violations are the most dangerous — correct values written to wrong tensors.

**Fix direction:** (1) Sort surfaces alphabetically before emission. (2) Validate uniform sizes. (3) Fail with error if non-uniform. (4) Add tests.

**Definition of Done:** Surfaces sorted alphabetically; non-uniform sizes cause error; tests for both constraints.

---

### I-93 · ANEC Attribute Shape Validation Missing
**Status:** ⬜ Open | **Files:** `python/mil_emitter.py`, `crates/coreml-proto/proto/coreml/MIL.proto` | **AUDIT ref:** V-107 | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-113

**Intent:** ANEC schema defines precise attribute shapes for all 98 operations. MILLer doesn't validate that emitted attribute shapes match ANEC expectations. Wrong-shaped attributes fail at ANEC time with cryptic errors.

**Fix direction:** Add attribute shape validation to emission pipeline. Validate stride, padding, dilation, kernel_size have correct element counts per ANEC schema.

**Definition of Done:** Attribute shape validation for conv, pool, deconv, SDPA; clear errors for mismatches; tests.

---

### I-94 · Conv Quantized Weight Attributes Not Modeled
**Status:** ⬜ Open | **Files:** `crates/ir/src/mir.rs:59-68`, `crates/coreml-proto/proto/coreml/MIL.proto:116-121` | **AUDIT ref:** V-110 | **Severity:** HIGH | **Effort:** L (as part of T-107) | **Task:** T-107

**Intent:** ANEC convolution has kernel_scale, kernel_zero_point, kernel_palettized_LUT attributes for quantized/palettized weights. MILLer's MILConv and MilConvOp proto don't carry these, blocking the entire quantized convolution pipeline.

**Fix direction:** Add quantized weight attributes to MILConv and MilConvOp proto. Wire through compat layer.

**Definition of Done:** MILConv has kernel_scale, kernel_zero_point, kernel_palettized_LUT; proto updated; compat layer wires data; test verifies.

---

### I-95 · Large Kernel Mode Constraints Not Enforced
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs`, `crates/ir/src/ane_hw_limits.rs` | **AUDIT ref:** V-115 | **Severity:** HIGH | **Effort:** M (as part of T-97) | **Task:** T-97

**Intent:** 12+ ANEC constraints for large kernel mode (W/H > threshold) not enforced: W/H multiple of 8, stride 1-2, zero padding only, no depth>1, no palettized weights, no grouped conv, no dynamic shape, no dilation. Models pass validation but fail at ANEC.

**Fix direction:** Add large kernel mode detection and validate at least 6 constraints.

**Definition of Done:** Large kernel mode detected; at least 6 constraints validated; tests for valid and invalid configs.

---

### I-96 · Deconvolution Constraints Not Enforced
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs`, `crates/passes/src/placement_validate.rs:516` | **AUDIT ref:** V-116, V-048 | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-98

**Intent:** ConvTranspose passes placement validation unconditionally. Binary shows: SOx==2, no large kernel, no vector palettization, no dilation. ANEC rejects violating deconvolutions.

**Fix direction:** Add deconv-specific validation: SOx==2, no large kernel, no vector palettization, no dilation, stride>2+depth>1 rejection.

**Definition of Done:** ConvTranspose no longer passes unconditionally; all 5 constraints enforced; tests.

---

### I-97 · ANE Flat Buffer Layout Packed [1,C,1,S] Not Validated (Orion #20)
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/` | **AUDIT ref:** V-121 (Orion #20) | **Severity:** HIGH | **Effort:** M (as part of T-99) | **Task:** T-99

**Intent:** ANE reads data as packed [1,C,1,S]. Data in wrong layout produces silently incorrect inference — the most dangerous failure class. No layout validation exists.

**Fix direction:** Add buffer layout validation ensuring packed [1,C,1,S] format. Add layout transformation if needed.

**Definition of Done:** Buffer layout validated; transformation if needed; test verifies correct layout.

---

### I-98 · BF16/F16 Cross-Type Operations Not Validated
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs` | **AUDIT ref:** V-125 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-100

**Intent:** ANEC rejects BF16/F16 cross-type operations (9 constraint strings from binary). MILLer has no cross-type validation. Each operand's dtype validated independently.

**Fix direction:** Add validate_cross_type_compatibility() checking input/output dtype pairs. Reject all 9 documented cross-type combinations.

**Definition of Done:** BF16/F16 cross-type rejected; all 9 constraints enforced; tests.

---

### I-99 · FP32 Architecture-Conditional Rejection Not Checked
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs` | **AUDIT ref:** V-126 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-100

**Intent:** FP32 rejected on some architectures but is_dtype_ane_legal() approves FP32 for all families. MILLer approves FP32 where ANEC rejects it. "May be downcast" comment but no downcast enforced.

**Fix direction:** Add architecture-conditional check for FP32. At minimum, warn when approved without architecture verification.

**Definition of Done:** FP32 architecture-conditional check; known-rejecting families return Err; others warn.

---

### I-100 · Dilated Pooling and Stencil Not Rejected
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-128 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** ANEC rejects dilated pooling and dilated stencil. MILLer has no dilation check for either. Models with dilated pooling/stencil pass validation but fail at ANEC.

**Fix direction:** Add dilation check rejecting pooling and stencil with dilation > 1.

**Definition of Done:** Dilated pooling rejected; dilated stencil rejected; tests.

---

### I-101 · Conv Kernel Power-of-2 Not Validated
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs:38-51` | **AUDIT ref:** V-132 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-97

**Intent:** ANEC requires kernel W/H/D be power of 2. MILLer validates range 1-7 only. Sizes 3, 5, 6, 7 pass but are ANEC-illegal. 3x3 convolutions (most common) require decomposition.

**Fix direction:** Add power-of-2 validation for conv kernel W, H, D. Add is_power_of_two() helper. Document 3x3 decomposition requirement.

**Definition of Done:** is_power_of_two() added; sizes 3,5,6,7 rejected; 3x3 decomposition documented; tests.

---

### I-102 · Asymmetric Quantization Not Rejected for ANE
**Status:** ⬜ Open | **Files:** `crates/passes/src/palettize_weights.rs` | **AUDIT ref:** V-134 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** ANEC constraint: "Asym quantization is not supported". No check prevents asymmetric quantization in ANE path.

**Fix direction:** Add asymmetric quantization rejection check for ANE path. Check symmetry before emitting to ANEC.

**Definition of Done:** Asymmetric quantization rejected; clear error; test.

---

### I-103 · Vector Palettization at-Cout Constraint Not Enforced
**Status:** ⬜ Open | **Files:** `crates/passes/src/palettize_weights.rs` | **AUDIT ref:** V-133 | **Severity:** HIGH | **Effort:** S (0.5 day) | **Task:** T-109

**Intent:** "vector palettization is only supported at Cout for ANE" not enforced. Additional: zero-point not supported for vector palettized, palettize size=256 not supported.

**Fix direction:** Add vector palettization at-Cout constraint. Reject at non-Cout dimensions. Add zero-point and palettize-size checks.

**Definition of Done:** Vector palettization at non-Cout rejected; zero-point for vector palettized rejected; size=256 rejected; tests.

---

### I-104 · MILLinear→MILMatMul Defeats Conv1x1 3x Performance (Orion #17)
**Status:** ⬜ Open | **Files:** `crates/passes/src/mil_lower.rs:3268-3307`, `crates/passes/src/legality_rewrite.rs:354-370` | **AUDIT ref:** V-114 (Orion #17) | **Severity:** HIGH | **Effort:** M (1 day) | **Task:** T-106

**Intent:** Orion #17: conv 1x1 is 3x faster than matmul. Pipeline creates Conv1x1AsLinear in AIR but then converts ALL MILLinear to MILMatMul. Binary shows both ConvertLayer (97 instances) and ConvertMatMul (8) are available.

**Fix direction:** Replace MILLinear→MILMatMul with MILLinear→MILConv(1x1) for ANE targets. Keep MatMul as fallback.

**Definition of Done:** MILLinear emits as Conv1x1; MatMul only when Conv1x1 impossible; performance test.

---

## P2 — MEDIUM (Technical Debt / Drift / Code Quality)

### I-105 · V26 Fabricated Limits Without Warning
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_hw_limits.rs:144-146` | **AUDIT ref:** V-031, V-088 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-111

**Intent:** V26 limits are fabricated (inherits A18 + num_nes=16). No hardware spec exists. for_revision(V26) returns plausible but fictional limits without any warning. Developers may use these limits thinking they're real.

**Fix direction:** Add explicit "speculative — not based on any hardware" warning in for_revision(V26) return. Document in code comment.

**Definition of Done:** V26 returns speculative warning; code comment documents fabrication; test verifies warning.

---

### I-106 · KvCacheLayout::Paged Constructible But Unimplemented
**Status:** ⬜ Open | **Files:** `crates/ir/src/sir.rs:1052-1054` | **AUDIT ref:** V-034 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-111

**Intent:** KvCacheLayout::Paged is a full enum variant documented as "not yet implemented" but constructible via serde deserialization. Code paths receiving Paged layout will fail at runtime.

**Fix direction:** Gate behind feature flag or add serde validation rejecting Paged on deserialization.

**Definition of Done:** Paged gated behind feature flag or rejected by serde; clear error when used.

---

### I-107 · ModelArchConfig::default() Still Callable Despite Deprecation
**Status:** ⬜ Open | **Files:** `crates/ir/src/common.rs:241-254` | **AUDIT ref:** V-035 | **Severity:** MEDIUM | **Effort:** S (as part of T-101) | **Task:** T-101

**Intent:** ModelArchConfig::default() silently assumes Qwen3-0.6B. Deprecated but still callable, producing wrong defaults for other architectures.

**Fix direction:** Remove Default impl or make it return error. Add ModelArchConfig::unspecified() for placeholder.

**Definition of Done:** Default removed or returns error; unspecified() available; test verifies.

---

### I-108 · PIR Decode-Step Claims StateWriteRead But Uses Linear Projection
**Status:** ⬜ Open | **Files:** `crates/ir/src/pir.rs:787-790` | **AUDIT ref:** V-036 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-112

**Intent:** Decode-step claims StateWriteRead handoff but comment says emission uses linear projection. PIR claims runtime semantics not delivered.

**Fix direction:** Change handoff kind to accurate descriptor (DirectPassThrough) or implement stateful KV cache semantics.

**Definition of Done:** Handoff kind matches actual emission; no false claims.

---

### I-109 · Hardcoded iOS18 Opset and Deployment Target
**Status:** ⬜ Open | **Files:** `crates/ir/src/lib.rs:17`, `crates/passes/src/shard_plan.rs:561-562` | **AUDIT ref:** V-037, V-046 | **Severity:** MEDIUM | **Effort:** M (1 day) | **Task:** T-111

**Intent:** DEFAULT_OPSET_VERSION and minimum_deployment_target hardcoded to "iOS18". Models fail on older iOS at load time. Wrong for A11/A12 (iOS 16-era hardware).

**Fix direction:** Make opset version and deployment target configurable from CLI or task spec.

**Definition of Done:** Opset and target configurable; not hardcoded; test verifies.

---

### I-110 · PIR Tensor Specs Hardcode Fp16 Dtype
**Status:** ⬜ Open | **Files:** `crates/ir/src/shard_desc.rs:363-389` | **AUDIT ref:** V-038 | **Severity:** MEDIUM | **Effort:** S (as part of T-112) | **Task:** T-112

**Intent:** PIR tensor specs hardcoded to dtype "fp16" ignoring actual task spec dtype. Wrong for fp32/int4 tasks.

**Fix direction:** Use actual dtype from task spec for PIR tensor specs.

**Definition of Done:** PIR tensor specs use actual dtype; no hardcoded "fp16" in production paths.

---

### I-111 · AIR Legality/Fallback Risk Fields Unvalidated
**Status:** ⬜ Open | **Files:** `crates/ir/src/air.rs:885-895` | **AUDIT ref:** V-039 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-110

**Intent:** legality_confidence, fallback_risk, drift_risk fields have no validation of value ranges, no documented semantics, no producers/consumers.

**Fix direction:** Add value range validation. Document semantics. Add producers.

**Definition of Done:** Value ranges validated; semantics documented; at least one producer.

---

### I-112 · Conv Kernel Range 1-7 Contradicts Large Kernel Threshold of 16
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs:38-51` | **AUDIT ref:** V-041 | **Severity:** MEDIUM | **Effort:** S (as part of T-97) | **Task:** T-97

**Intent:** Conv kernel range 1-7 contradicts later grouped/dilated threshold of 16. Either 1-7 is too restrictive or 16-check is dead code. Binary shows large kernel mode uses a threshold corresponding to the 16-check.

**Fix direction:** Resolve contradiction: expand range or remove dead 16-threshold code. Binary suggests threshold corresponds to large kernel mode.

**Definition of Done:** No contradictory kernel range checks; documented relationship between range and large kernel threshold.

---

### I-113 · Broadcast Shape Fallback Only Logs Warning
**Status:** ⬜ Open | **Files:** `crates/passes/src/mil_lower.rs:156-175` | **AUDIT ref:** V-043 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** Broadcast incompatibility falls back to x's shape with only warning. Produces wrong MIR output shapes. Binary also shows "Only fp16 is supported for A11/A12 Broadcasts" — additional dtype constraint not enforced.

**Fix direction:** Replace fallback with error. Add A11/A12 broadcast FP16-only constraint.

**Definition of Done:** Broadcast incompatibility causes error; A11/A12 FP16-only enforced; tests.

---

### I-114 · PIR context_length Always Zero
**Status:** ⬜ Open | **Files:** `crates/passes/src/shard_plan.rs:559` | **AUDIT ref:** V-045 | **Severity:** MEDIUM | **Effort:** S (as part of T-112) | **Task:** T-112

**Intent:** PIR context_length always 0. Semantically important for KV cache models but never derived from graph or task spec.

**Fix direction:** Derive context_length from graph or task spec.

**Definition of Done:** context_length derived from graph or task spec; not always 0; test verifies.

---

### I-115 · KV Cache Default Shape Is Arbitrary
**Status:** ⬜ Open | **Files:** `crates/passes/src/shard_plan.rs:400,527` | **AUDIT ref:** V-047 | **Severity:** MEDIUM | **Effort:** S (as part of T-112) | **Task:** T-112

**Intent:** KV cache default shape fallback vec![2,1,1,1,1] is arbitrary. Batch=2 and all-1 dimensions almost certainly wrong for any real model.

**Fix direction:** Add configuration for KV cache default shape. Derive from model architecture when available.

**Definition of Done:** KV cache shape configurable; derived from architecture when available; no arbitrary defaults.

---

### I-116 · FP32 ANE-Legal Without Downcast Enforcement
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs:73` | **AUDIT ref:** V-049 | **Severity:** MEDIUM | **Effort:** S (as part of T-100) | **Task:** T-100

**Intent:** FP32 allowed as ANE-legal with comment "may be downcast" but downcast not enforced. ANE does not natively compute in FP32.

**Fix direction:** Add downcast enforcement or architecture-conditional check.

**Definition of Done:** Downcast enforced or architecture-conditional check; no unconditional FP32 approval.

---

### I-117 · Int4/UInt4 Interleave==8 Deferred to Caller
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs:79-81` | **AUDIT ref:** V-050 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** Int4/UInt4 return Ok(()) with "caller must also check interleave==8" comment. Critical constraint deferred to caller with no enforcement.

**Fix direction:** Add interleave==8 validation directly in is_dtype_ane_legal() for Int4/UInt4.

**Definition of Done:** Int4/UInt4 interleave==8 enforced in is_dtype_ane_legal(); not deferred to caller.

---

### I-118 · E5M2 Accepted by Quantize Validator But Universally Rejected by ANEC
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs:180-182` | **AUDIT ref:** V-051, V-111 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** Quantize validator accepts E5M2 as output dtype but is_dtype_ane_legal() rejects it. Binary confirms E5M2 is universally "not supported" — not just architecture-conditional. The quantize validator's acceptance is definitively wrong.

**Fix direction:** Reject E5M2 as quantize output dtype. Binary evidence confirms universal rejection.

**Definition of Done:** E5M2 rejected as quantize output; consistent with is_dtype_ane_legal(); test.

---

### I-119 · Canonicalize Catch-All Produces Dangling References
**Status:** ⬜ Open | **Files:** `crates/passes/src/canonicalize.rs:294` | **AUDIT ref:** V-052 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** Wildcard catch-all silently copies unrecognized SirOp variants without rewriting SirNodeId references, producing dangling references for new variants.

**Fix direction:** Replace catch-all with explicit variant list. Add log::warn! for unrecognized variants.

**Definition of Done:** No catch-all in canonicalize; unrecognized variants logged; tests.

---

### I-120 · Knowledge Doc/Code Mismatches
**Status:** ⬜ Open | **Files:** `crates/knowledge/src/update.rs:87-89`, `crates/knowledge/src/transfer.rs:86-89`, `crates/knowledge/src/lib.rs:38-39` | **AUDIT ref:** V-053, V-055, V-056 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-110

**Intent:** Doc says "never start above 0.5" but CompileFailure starts at 0.7. Doc says transfer scales "0.5-0.8" but code uses 0.65. ComputePlan doc says "confidence always 0.9" but code accepts any value.

**Fix direction:** Fix documentation to match code behavior. Update doc comments with correct values.

**Definition of Done:** Documentation matches code; no misleading claims; tests verify documented values.

---

### I-121 · Knowledge Seed Schema Mismatch
**Status:** ⬜ Open | **Files:** `knowledge/` (all seeds) | **AUDIT ref:** V-071 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-110

**Intent:** Knowledge schema defines unit wrapper with provenance, conflict_status, conflict_priority fields. No seed JSON follows this schema — all use flat fields without unit wrapper.

**Fix direction:** Either align seed JSONs with schema or update schema to match actual format.

**Definition of Done:** Seed JSONs follow schema or schema matches actual format; consistency test.

---

### I-122 · EmptyWeightResolver Doc Says "Returns Some" But Returns None
**Status:** ⬜ Open | **Files:** `crates/bridge/src/mir_to_compat.rs:92-104` | **AUDIT ref:** V-059 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-116

**Intent:** Doc comment says "returns Some with empty data" but implementation returns None. Misleading documentation.

**Fix direction:** Fix doc comment to match implementation.

**Definition of Done:** Doc matches implementation; test verifies behavior.

---

### I-123 · Python Subprocess Failure Returns Ok Instead of Err
**Status:** ⬜ Open | **Files:** `crates/bridge/src/subprocess.rs:73-89` | **AUDIT ref:** V-060 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-114

**Intent:** Python subprocess failure returns Ok(BridgeResult{status:"error"}) instead of Err(). Forces callers to manually check status string.

**Fix direction:** Return Err when Python subprocess reports error status.

**Definition of Done:** Subprocess failure returns Err; callers don't need to check status string; test.

---

### I-124 · Shape Inference Catch-All Returns Empty Shape
**Status:** ⬜ Open | **Files:** `crates/bridge/src/shape_inference.rs:530` | **AUDIT ref:** V-062 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** Catch-all returns empty shape for unrecognized MirOp variants. Core ML inference not guaranteed to succeed.

**Fix direction:** Add log::warn!() for unrecognized variants. Document which ops lack shape inference.

**Definition of Done:** Unrecognized variants produce warning; documented gap; test.

---

### I-125 · MirOpCompat::Unsupported Constructible But Always Rejected
**Status:** ⬜ Open | **Files:** `crates/coreml-proto/src/lib.rs:930-938` | **AUDIT ref:** V-065 | **Severity:** MEDIUM | **Effort:** S (as part of T-107) | **Task:** T-107

**Intent:** MirOpCompat::Unsupported is constructible but always rejected at emission gate. Type-level capability that can never produce output.

**Fix direction:** Document clearly or make Unsupported non-constructible in production paths.

**Definition of Done:** Unsupported either non-constructible or clearly documented; test.

---

### I-126 · Fill/Select/Where Rejected But Type-System Allows
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/src/mir_to_proto.rs:94-121` | **AUDIT ref:** V-066 | **Severity:** MEDIUM | **Effort:** S (as part of T-107) | **Task:** T-107

**Intent:** Fill, Select, Where are valid MirOpCompat variants but always rejected as "ANE-illegal." Error says "should have been replaced earlier" but type system allows them through.

**Fix direction:** Document replacement requirement or add compile-time prevention.

**Definition of Done:** Replacement documented; type system prevents or warns; test.

---

### I-127 · HW Limit Seed Incomplete
**Status:** ⬜ Open | **Files:** `knowledge/ane_hw_limits_seed.json` | **AUDIT ref:** V-067 | **Severity:** MEDIUM | **Effort:** M (1 day) | **Task:** T-110

**Intent:** Only ~14 of ~40+ documented hardware limit parameters captured. No validation for conv kernel sizes, pooling limits, padding limits, PE/NE engine constraints.

**Fix direction:** Add missing constraint parameters to seed. At minimum, add conv kernel size limits, pooling limits, and PE/NE engine constraints from binary evidence.

**Definition of Done:** Seed covers at least 25 of 40+ parameters; missing parameters documented.

---

### I-128 · Device Class ↔ AneFamily Mapping Missing
**Status:** ⬜ Open | **Files:** `knowledge/` (all seeds) | **AUDIT ref:** V-069 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-96

**Intent:** Knowledge seeds use device_classes ["M2","M3"] but compiler operates on AneFamily. No documented device_class→AneFamily mapping exists.

**Fix direction:** Add device_class_to_family mapping document in knowledge/. Add validation.

**Definition of Done:** Mapping document exists; validation test passes.

---

### I-129 · SDPA Support Contradiction
**Status:** ⬜ Open | **Files:** `knowledge/ane_op_family_matrix.json:86-101` | **AUDIT ref:** V-072 | **Severity:** MEDIUM | **Effort:** S (as part of T-96) | **Task:** T-96

**Intent:** SDPA marked "unreliable" for A12-A15 but Rust binary-classifies as not-supported (A16+ only). Binary shows ConvertScaledDotProductAttention is family-agnostic with MinimumFamily=A11Legacy.

**Fix direction:** Change "unreliable" to "supported" for A13+ with architecture_conditional: true.

**Definition of Done:** SDPA support scope reflects binary evidence; no "unreliable" category.

---

### I-130 · Precision Hazard Seed Single-Model Bias
**Status:** ⬜ Open | **Files:** `knowledge/precision_hazard_seed.json` | **AUDIT ref:** V-073 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-110

**Intent:** All 4 entries derive from single model (Qwen3). Claims general rules based on 3 evidence points from one model. No cross-validation.

**Fix direction:** Document limitation. Add cross-model validation when additional data available.

**Definition of Done:** Limitation documented; cross-validation TODO noted.

---

### I-131 · Palettization Constraints Dual Min Bits Without Conditional Context
**Status:** ⬜ Open | **Files:** `knowledge/palettization_constraints_seed.json:5-9` | **AUDIT ref:** V-075 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** Dual conv min bits (standard:4, alternate:2) without conditional context. Binary shows version-conditional constraints: "3-bit palettization is only supported from version {1}".

**Fix direction:** Add version-conditional context for palettization constraints. Document which versions support which bit widths.

**Definition of Done:** Version-conditional palettization constraints documented; seed updated.

---

### I-132 · device_backed() Always Returns HostOnly
**Status:** ⬜ Open | **Files:** `crates/lab/src/device_meta.rs:125-140` | **AUDIT ref:** V-076 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-114

**Intent:** device_backed() always returns HostOnly on all platforms including macOS. Method name implies device-backed metadata but never produces it.

**Fix direction:** Document as stub with TODO for macOS implementation.

**Definition of Done:** Documented as stub; TODO noted.

---

### I-133 · coremltools_available Hardcoded False
**Status:** ⬜ Open | **Files:** `crates/lab/src/harness.rs:70` | **AUDIT ref:** V-077 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-114

**Intent:** coremltools_available hardcoded to false. Python bridge detection result never folded back. Every LabRun permanently reports false.

**Fix direction:** Add Python bridge detection logic to set dynamically.

**Definition of Done:** coremltools_available dynamically detected; not hardcoded.

---

### I-134 · FallbackLogEvidence Never Constructed
**Status:** ⬜ Open | **Files:** `crates/lab/src/fallback.rs:165-181` | **AUDIT ref:** V-096 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-114

**Intent:** FallbackLogEvidence defined with full serde support but never constructed. "Reserved for future use" — phantom capability.

**Fix direction:** Document as reserved or remove until implemented.

**Definition of Done:** Either documented as reserved with TODO or removed.

---

### I-135 · LabRun Timing on HostOnlyInspection Allowed
**Status:** ⬜ Open | **Files:** `crates/lab/src/harness.rs:408-412` | **AUDIT ref:** V-079 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-114

**Intent:** LabRunBuilder allows timing on HostOnlyInspection runs despite doc saying "MUST be None." No enforcement.

**Fix direction:** Add runtime enforcement: reject timing on HostOnlyInspection runs.

**Definition of Done:** Timing rejected on HostOnlyInspection; test.

---

### I-136 · CoreMlApi/Model/CAPI Stub Functions
**Status:** ⬜ Open | **Files:** `crates/coreml-ffi/src/api.rs:43-71`, `crates/coreml-ffi/src/model.rs:92-111`, `crates/coreml-ffi/src/capi.rs:204-224` | **AUDIT ref:** V-080, V-081, V-082 | **Severity:** MEDIUM | **Effort:** S (as part of T-103) | **Task:** T-103

**Intent:** version() returns "unknown", compile_model() returns Err on macOS, load() returns Ok(None), coreml_model_info() writes zeroed info with Ok. All are stub-mimic functions that suggest functionality that doesn't exist.

**Fix direction:** Return specific NotAvailableOnPlatform errors. Document as stubs.

**Definition of Done:** Stubs return specific error types; documented; load() returns Err instead of Ok(None).

---

### I-137 · FFI Validation Rejects Valid No-Weight Models
**Status:** ⬜ Open | **Files:** `crates/coreml-ffi/src/capi.rs:383-391` | **AUDIT ref:** V-083 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-114

**Intent:** Validation rejects packages without weight.bin, but not all models require it. Inline-const-only models are valid but fail validation.

**Fix direction:** Make weight.bin optional. Only require when model declares external weights.

**Definition of Done:** weight.bin optional; inline-const models pass validation; test.

---

### I-138 · QualityContract Hardcoded
**Status:** ⬜ Open | **Files:** `crates/trace/src/sir_build.rs:169-172` | **AUDIT ref:** V-084 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-114

**Intent:** QualityContract hardcoded: max_perplexity_delta=0.1, max_latency_ms=50.0. Overly restrictive for 7B models. Not per-model configurable.

**Fix direction:** Make QualityContract configurable from task spec or CLI.

**Definition of Done:** QualityContract configurable; not hardcoded; test.

---

### I-139 · SIR Input Shape Silent Fallback
**Status:** ⬜ Open | **Files:** `crates/trace/src/sir_build.rs:84-94` | **AUDIT ref:** V-085 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-104

**Intent:** Missing input shape falls back to (1,32) silently. Wrong for models with different expected shapes.

**Fix direction:** Add log::warn!() when fallback used. Consider making shape required.

**Definition of Done:** Warning emitted on fallback; documented limitation.

---

### I-140 · --seed CLI Parameter Discarded
**Status:** ⬜ Open | **Files:** `crates/cli/src/main.rs:696,954` | **AUDIT ref:** V-087 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-111

**Intent:** --seed parameter accepted by CLI but silently discarded. SPEC requires deterministic compilation but seed is unused.

**Fix direction:** Either wire --seed through compile pipeline or remove from CLI.

**Definition of Done:** --seed either wired or removed; no discarded parameters.

---

### I-141 · Minimum IOSurface Size Not Validated (Orion #4)
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/`, `crates/coreml-emit/src/weights.rs` | **AUDIT ref:** V-104, V-122 | **Severity:** MEDIUM | **Effort:** S (as part of T-113) | **Task:** T-113

**Intent:** Minimum IOSurface size (~49 KB) for eval not validated (Orion #4). Smaller buffers cause 0x1d runtime error.

**Fix direction:** Add minimum IOSurface size validation in emission pipeline.

**Definition of Done:** Minimum buffer size validated; error when too small; test.

---

### I-142 · Compilation Count Per Process Not Tracked (Orion #5)
**Status:** ⬜ Open | **Files:** `crates/bridge/src/subprocess.rs`, `crates/coreml-emit/` | **AUDIT ref:** V-106, V-123 | **Severity:** MEDIUM | **Effort:** S (as part of T-113) | **Task:** T-113

**Intent:** ~119 compilation limit per process (Orion #5) not tracked. Exceeding causes silent crash. No counter or warning.

**Fix direction:** Add compilation count tracking. Warn at ~100, error at ~119.

**Definition of Done:** Count tracked; warning at threshold; test.

---

### I-143 · Missing AneRevision Variants
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_hw_limits.rs` | **AUDIT ref:** V-108 | **Severity:** MEDIUM | **Effort:** S (as part of T-115) | **Task:** T-115

**Intent:** Binary shows 14 hardware versions but MILLer only defines 11 revisions. Missing V0-V3 (pre-A11) with dedicated compiler code paths.

**Fix direction:** Add missing AneRevision variants for pre-A11 hardware with minimal capabilities.

**Definition of Done:** Pre-A11 variants added; marked with minimal capabilities; test.

---

### I-144 · AneFamily Enum Incomplete — 8 Binary Families vs 6 MILLer
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_target.rs` | **AUDIT ref:** V-109 | **Severity:** MEDIUM | **Effort:** M (1 day) | **Task:** T-115

**Intent:** Binary MinimumFamily enum has 8 values (0-7) but MILLer only models 6. Families 6-7 unmapped — MILLer will misattribute ops on future hardware.

**Fix direction:** Extend AneFamily to cover all 8 binary-defined families. Add A17Pro and A18Pro variants.

**Definition of Done:** 8 AneFamily variants matching binary; test; documentation.

---

### I-145 · Missing MILInputView for Negative Strides
**Status:** ⬜ Open | **Files:** `crates/ir/src/mir.rs` | **AUDIT ref:** V-112 | **Severity:** MEDIUM | **Effort:** M (1 day) | **Task:** T-107

**Intent:** ANEC's anec.input_view supports negative strides (step=i64) but MILLer doesn't model it. Certain ANE-legal patterns (reverse, crop/resize) cannot be correctly expressed.

**Fix direction:** Add MirOp::MILInputView or equivalent supporting negative-stride tensor views.

**Definition of Done:** InputView variant added; negative stride support; test.

---

### I-146 · Weight Dict Initialization Not Guaranteed (Orion #11)
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/src/mir_to_proto.rs` | **AUDIT ref:** V-124 | **Severity:** MEDIUM | **Effort:** S (as part of T-113) | **Task:** T-113

**Intent:** Weight dict must be @{} not nil (Orion #11). Nil weight dict crashes ANEC at compile time. MILLer doesn't verify initialization.

**Fix direction:** Add weight dict initialization check ensuring dict is @{} (not nil) before ANEC compilation.

**Definition of Done:** Weight dict initialization guaranteed; test.

---

### I-147 · Pooling Stride-3 Avg-Only Not Enforced
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-127 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-102

**Intent:** "Pool with strides of 3 is only supported with Avg mode." MILLer allows stride-3 MaxPool without error. Also: "Large stride Min/Max pool with padding is not supported."

**Fix direction:** Add stride-3 Avg-only check. Reject MaxPool/L2Pool with stride 3. Add large-stride MaxPool+padding rejection.

**Definition of Done:** MaxPool stride-3 rejected; large-stride MaxPool+padding rejected; tests.

---

### I-148 · HW Sub-Variants Not Modeled
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_hw_limits.rs` | **AUDIT ref:** V-131 | **Severity:** MEDIUM | **Effort:** M (1 day) | **Task:** T-115

**Intent:** Binary shows NE(12), PE(13), DMA(27) engine-specific sub-variant descriptors per version. MILLer only maps 11 top-level revisions, cannot express engine-specific constraints.

**Fix direction:** Document sub-variant structure. Future work: add sub-variant constraints.

**Definition of Done:** Sub-variant structure documented in comments; TODO for implementation.

---

### I-149 · Stencil Constraints Not Enforced
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-137 | **Severity:** MEDIUM | **Effort:** S (as part of T-109) | **Task:** T-109

**Intent:** 5 ANEC stencil constraints not enforced: 5D stencil rejected, non-4D kernel rejected, non-sum reduction rejected, dilated stencil rejected, strided stencil rejected.

**Fix direction:** Add stencil constraint validation for all 5.

**Definition of Done:** All 5 stencil constraints enforced; tests.

---

### I-150 · Width Wrap Axis Architecture Check Missing
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs` | **AUDIT ref:** V-135 | **Severity:** MEDIUM | **Effort:** S (as part of T-109) | **Task:** T-109

**Intent:** "Width wrap axis is not supported on this architecture" — architecture-conditional constraint not enforced.

**Fix direction:** Add width wrap axis architecture check.

**Definition of Done:** Width wrap axis check added; architecture-conditional; test.

---

### I-151 · LayerNorm Epsilon FP16 Truncation
**Status:** ⬜ Open | **Files:** `python/mil_emitter.py:904` | **AUDIT ref:** V-097 | **Severity:** MEDIUM | **Effort:** S (0.25 day) | **Task:** T-116

**Intent:** LayerNorm epsilon np_dtype(1e-5) truncates for FP16 (becomes ~9.98e-6). Emitted model epsilon differs from programmer intent.

**Fix direction:** Document truncation or compute epsilon in FP32 before casting.

**Definition of Done:** Truncation documented or avoided; test.

---

### I-152 · MirOpCompat Missing 70+ ANEC Operations
**Status:** ⬜ Open | **Files:** `crates/coreml-proto/src/lib.rs`, `crates/ir/src/mir.rs` | **AUDIT ref:** V-100, V-138 | **Severity:** MEDIUM | **Effort:** L (3 days) | **Task:** T-107

**Intent:** ANEC has 98+ operations but MirOpCompat models ~30. 70+ ANEC ops including 27 ConvertElementwiseUnary variants (Elu, LeakyRelu, Sqr, Rsqrt, Sign, etc.) have no MILLer equivalents. These ops have dedicated converters and specific hardware behavior.

**Fix direction:** Add MirOpCompat variants for 27 ConvertElementwiseUnary operations and RingBufferReader/Writer. Each needs conversion, input_names, remap_inputs, rename_output, tests.

**Definition of Done:** 27+ new variants; full conversion paths; tests.

---

### I-153 · MatMul Transpose Flags as Immediate Bools
**Status:** ⬜ Open | **Files:** `crates/coreml-proto/src/lib.rs:3854-3860` | **AUDIT ref:** V-129 | **Severity:** MEDIUM | **Effort:** S (0.5 day) | **Task:** T-116

**Intent:** Orion #12: matmul transpose flags need named const nodes, not immediate bools. Current code uses make_immediate_bool_value. Binary shows ConvertMatMul is family-scoped (8 instantiations) — immediate bools may not be accepted by all family implementations.

**Fix direction:** Emit matmul transpose flags as named const nodes instead of immediate bool values.

**Definition of Done:** Transpose flags as named const nodes; test.

---

## P3 — LOW (Minor Quality / Style / Documentation)

### I-154 · Hardcoded Fallback Dimensions in role_mir
**Status:** ⬜ Open | **Files:** `crates/passes/src/role_mir.rs:131,146,210` | **AUDIT ref:** V-089 | **Severity:** LOW | **Effort:** S (0.25 day) | **Task:** T-116
**Intent:** Multiple unwrap_or fallbacks use hardcoded dimension values (64, 48, 32). Silently produce wrong MIR shapes if spec incomplete.
**Fix direction:** Replace with explicit configuration or fail-closed behavior.
**Definition of Done:** No hardcoded dimension fallbacks; explicit config or error.

---

### I-155 · Canonicalization Cycle Limit Without Diagnostic
**Status:** ⬜ Open | **Files:** `crates/passes/src/canonicalize.rs:113-119` | **AUDIT ref:** V-090 | **Severity:** LOW | **Effort:** S (0.25 day) | **Task:** T-116
**Intent:** Chain resolution loop has magic limit of 100 steps with no diagnostic when hit. Incomplete substitution for deep chains.
**Fix direction:** Add diagnostic when limit is hit.
**Definition of Done:** Diagnostic emitted when cycle limit hit.

---

### I-156 · UInt16/Bool "Limited Support" Undefined
**Status:** ⬜ Open | **Files:** `crates/passes/src/dtype_constraints.rs:105,113` | **AUDIT ref:** V-091, V-092 | **Severity:** LOW | **Effort:** S (0.5 day) | **Task:** T-116
**Intent:** UInt16 and Bool marked ANE-legal with "limited support" but no constraint checks or documentation of what "limited" means.
**Fix direction:** Add constraint documentation specifying which ops/families support UInt16 and Bool.
**Definition of Done:** Limited support documented with specific ops/families.

---

### I-157 · CPU_ONLY_OPS_DETAILED Incomplete
**Status:** ⬜ Open | **Files:** `crates/passes/src/cpu_only_ops.rs:256-319` | **AUDIT ref:** V-093 | **Severity:** LOW | **Effort:** M (1 day) | **Task:** T-116
**Intent:** CPU_ONLY_OPS_DETAILED has ~30 entries vs 120+ in CPU_ONLY_OPS. ~90 ops have no documented reason for being CPU-only.
**Fix direction:** Expand to cover all 120+ ops with reason codes.
**Definition of Done:** All CPU-only ops have documented reasons.

---

### I-158 · Knowledge Consistency Binary All-or-Nothing
**Status:** ⬜ Open | **Files:** `crates/knowledge/src/compute_plan_verify.rs:337` | **AUDIT ref:** V-094 | **Severity:** LOW | **Effort:** S (0.5 day) | **Task:** T-116
**Intent:** knowledge_consistent is binary all-or-nothing. Single mismatch makes entire proof "not consistent." Discards nuance.
**Fix direction:** Replace with ratio or graded score.
**Definition of Done:** Graded consistency score; not binary.

---

### I-159 · UUID Not Globally Unique
**Status:** ⬜ Open | **Files:** `crates/coreml-emit/src/package.rs:155-195` | **AUDIT ref:** V-095 | **Severity:** LOW | **Effort:** S (0.25 day) | **Task:** T-116
**Intent:** UUIDs generated via v5 with fixed namespace + function name. Two models with same function name produce identical UUIDs.
**Fix direction:** Use model-specific salt in UUID generation.
**Definition of Done:** UUIDs include model-specific salt; test verifies uniqueness.

---

### I-160 · ArgMinMax A18+ Dropping Confirmed by Binary
**Status:** ⬜ Open | **Files:** `crates/ir/src/ane_target.rs:145` | **AUDIT ref:** V-102 | **Severity:** LOW | **Effort:** S (0.25 day) | **Task:** documentation
**Intent:** ConvertReductionArg has exactly 7 family instantiations (0-6), confirming ArgMinMax NOT available on A18+. Update documentation to remove "unverified" qualifier.
**Fix direction:** Update documentation for V-032 and V-102 with high-confidence binary evidence.
**Definition of Done:** Documentation reflects confirmed binary evidence.

---

### I-161 · Conv Kernel Range Contradicts Large Kernel Threshold
**Status:** ⬜ Open | **Files:** `crates/passes/src/op_constraints.rs:38-51` | **AUDIT ref:** V-041 | **Severity:** LOW | **Effort:** S (as part of T-97) | **Task:** T-97
**Intent:** Conv kernel range 1-7 contradicts grouped/dilated threshold of 16. Either 1-7 too restrictive or 16-check is dead code.
**Fix direction:** Resolve: expand range or remove dead threshold. Binary suggests threshold corresponds to large kernel mode activation.
**Definition of Done:** No contradictory kernel range checks; relationship documented.

---

## Summary Statistics

| Priority | Total | Open | Fixed | Retracted |
|----------|-------|------|-------|-----------|
| P0 (CRITICAL) | 7 | 7 | 0 | 0 |
| P1 (HIGH) | 32 | 32 | 0 | 0 |
| P2 (MEDIUM) | 49 | 49 | 0 | 0 |
| P3 (LOW) | 8 | 8 | 0 | 0 |
| Resolved (v1+v2+v3) | 65 | 0 | 61 | 4 |
| **Total** | **161** | **96** | **61** | **4** |

---

## Issue → Violation Cross-Reference

| Issue | Violation(s) | Classification | Severity |
|-------|-------------|----------------|----------|
| I-66 | V-001, V-002 | ABERRANT | CRITICAL |
| I-67 | V-003 | ABERRANT | CRITICAL |
| I-68 | V-004 | PHANTOM | CRITICAL |
| I-69 | V-005 | ABERRANT | CRITICAL |
| I-70 | V-007 | LACUNA | CRITICAL |
| I-71 | V-098, V-130 | ABERRANT | CRITICAL |
| I-72 | V-113, V-099 | ABERRANT | CRITICAL |
| I-73 | V-009 | LACUNA | HIGH |
| I-74 | V-010 | LACUNA | HIGH |
| I-75 | V-011 | STUB-MIMIC | HIGH |
| I-76 | V-012, V-025 | ABERRANT | HIGH |
| I-77 | V-013 | ABERRANT | HIGH |
| I-78 | V-014 | PHANTOM | HIGH |
| I-79 | V-016 | STUB-MIMIC | HIGH |
| I-80 | V-017 | LACUNA | HIGH |
| I-81 | V-018 | LACUNA | HIGH |
| I-82 | V-019, V-136 | ABERRANT | HIGH |
| I-83 | V-020 | LACUNA | HIGH |
| I-84 | V-021 | LACUNA | HIGH |
| I-85 | V-022 | LACUNA | HIGH |
| I-86 | V-023 | LACUNA | HIGH |
| I-87 | V-026 | LACUNA | HIGH |
| I-88 | V-027 | LACUNA | HIGH |
| I-89 | V-028 | LACUNA | HIGH |
| I-90 | V-029, V-030, V-101 | UNVERIFIED/ABERRANT | HIGH |
| I-91 | V-103 | LACUNA | HIGH |
| I-92 | V-105, V-117, V-118, V-119, V-120 | LACUNA | HIGH |
| I-93 | V-107 | LACUNA | HIGH |
| I-94 | V-110 | LACUNA | HIGH |
| I-95 | V-115 | LACUNA | HIGH |
| I-96 | V-116, V-048 | LACUNA | HIGH |
| I-97 | V-121 | LACUNA | HIGH |
| I-98 | V-125 | ABERRANT | HIGH |
| I-99 | V-126 | LACUNA | HIGH |
| I-100 | V-128 | LACUNA | HIGH |
| I-101 | V-132 | LACUNA | HIGH |
| I-102 | V-134 | LACUNA | HIGH |
| I-103 | V-133 | LACUNA | HIGH |
| I-104 | V-114 | ABERRANT | HIGH |
| I-105 | V-031, V-088 | PHANTOM | MEDIUM |
| I-106 | V-034 | PHANTOM | MEDIUM |
| I-107 | V-035 | ABERRANT | MEDIUM |
| I-108 | V-036 | ABERRANT | MEDIUM |
| I-109 | V-037, V-046 | UNVERIFIED/LACUNA | MEDIUM |
| I-110 | V-038 | ABERRANT | MEDIUM |
| I-111 | V-039 | LACUNA | MEDIUM |
| I-112 | V-041 | ABERRANT | MEDIUM |
| I-113 | V-043 | LACUNA | MEDIUM |
| I-114 | V-045 | LACUNA | MEDIUM |
| I-115 | V-047 | UNVERIFIED | MEDIUM |
| I-116 | V-049 | UNVERIFIED | MEDIUM |
| I-117 | V-050 | LACUNA | MEDIUM |
| I-118 | V-051, V-111 | ABERRANT | MEDIUM |
| I-119 | V-052 | LACUNA | MEDIUM |
| I-120 | V-053, V-055, V-056 | ABERRANT | MEDIUM |
| I-121 | V-071 | ABERRANT | MEDIUM |
| I-122 | V-059 | ABERRANT | MEDIUM |
| I-123 | V-060 | LACUNA | MEDIUM |
| I-124 | V-062 | LACUNA | MEDIUM |
| I-125 | V-065 | PHANTOM | MEDIUM |
| I-126 | V-066 | PHANTOM | MEDIUM |
| I-127 | V-067 | LACUNA | MEDIUM |
| I-128 | V-069 | LACUNA | MEDIUM |
| I-129 | V-072 | UNVERIFIED | MEDIUM |
| I-130 | V-073 | UNVERIFIED | MEDIUM |
| I-131 | V-075 | LACUNA | MEDIUM |
| I-132 | V-076 | PHANTOM | MEDIUM |
| I-133 | V-077 | LACUNA | MEDIUM |
| I-134 | V-096 | PHANTOM | MEDIUM |
| I-135 | V-079 | LACUNA | MEDIUM |
| I-136 | V-080, V-081, V-082 | STUB-MIMIC | MEDIUM |
| I-137 | V-083 | ABERRANT | MEDIUM |
| I-138 | V-084 | LACUNA | MEDIUM |
| I-139 | V-085 | LACUNA | MEDIUM |
| I-140 | V-087 | LACUNA | MEDIUM |
| I-141 | V-104, V-122 | LACUNA | MEDIUM |
| I-142 | V-106, V-123 | LACUNA | MEDIUM |
| I-143 | V-108 | LACUNA | MEDIUM |
| I-144 | V-109 | LACUNA | MEDIUM |
| I-145 | V-112 | LACUNA | MEDIUM |
| I-146 | V-124 | LACUNA | MEDIUM |
| I-147 | V-127 | LACUNA | MEDIUM |
| I-148 | V-131 | LACUNA | MEDIUM |
| I-149 | V-137 | LACUNA | MEDIUM |
| I-150 | V-135 | LACUNA | MEDIUM |
| I-151 | V-097 | LACUNA | MEDIUM |
| I-152 | V-100, V-138 | LACUNA | MEDIUM |
| I-153 | V-129 | LACUNA | MEDIUM |
| I-154 | V-089 | UNVERIFIED | LOW |
| I-155 | V-090 | UNVERIFIED | LOW |
| I-156 | V-091, V-092 | UNVERIFIED | LOW |
| I-157 | V-093 | LACUNA | LOW |
| I-158 | V-094 | LACUNA | LOW |
| I-159 | V-095 | UNVERIFIED | LOW |
| I-160 | V-102 | ABERRANT | LOW |
| I-161 | V-041 | ABERRANT | LOW |
