# ANEVIOLATIONS.md — MILLer Constraint-Grounded Violation Report

---

## I. EXECUTIVE ABSTRACT

### Files Examined

This audit examined the following files and directories by filename and structural role within the MILLer open-source compiler project (repository: `pkhairkh/MILLer`, branch `main`):

**Rust crates (12 workspace members):**
- `crates/ir/` — IR core: `sir.rs`, `air.rs`, `mir.rs`, `pir.rs`, `kir.rs`, `common.rs`, `ane_target.rs`, `ane_hw_limits.rs`, `ane_layout.rs`, `ane_engine.rs`, `task_spec.rs`, `linear_slice.rs`, `shard_desc.rs`, `strategy.rs`, `payload.rs`, `prof_ir.rs`, `shape_ops.rs`, `serialize.rs`, `toproto.rs`
- `crates/passes/` — Compilation passes: `legality_rewrite.rs`, `mil_lower.rs`, `op_constraints.rs`, `dtype_constraints.rs`, `placement_validate.rs`, `cpu_only_ops.rs`, `canonicalize.rs`, `staticize.rs`, `precision_policy.rs`, `shard_plan.rs`, `risk_annotate.rs`, `palettize_weights.rs`, `kv_cache_rewrite.rs`, `knowledge_query.rs`, `slanc_scales.rs`, `static_tables.rs`, `role_mir.rs`
- `crates/bridge/` — Python bridge + proto-direct emission: `mir_to_compat.rs`, `shape_inference.rs`, `safetensors_resolver.rs`, `static_table_resolver.rs`, `proto_direct.rs`, `subprocess.rs`
- `crates/coreml-emit/` — mlpackage emission: `mir_to_proto.rs`, `weights.rs`, `package.rs`, `emitter.rs`
- `crates/coreml-proto/` — Protobuf definitions: `lib.rs`, `build.rs`, proto files
- `crates/coreml-ffi/` — C API FFI: `capi.rs`, `api.rs`, `model.rs`, `error.rs`
- `crates/knowledge/` — Knowledge store: `store.rs`, `query.rs`, `update.rs`, `conflict.rs`, `confidence.rs`, `transfer.rs`, `shard_template.rs`, `snapshot.rs`, `compute_plan_verify.rs`, `util.rs`
- `crates/lab/` — Execution lab: `session.rs`, `baseline.rs`, `harness.rs`, `mir_compare.rs`, `task_gen.rs`, `host_inspect.rs`, `run_dir.rs`, `drift.rs`, `device_meta.rs`, `fallback.rs`, `families/`
- `crates/trace/` — HuggingFace model tracing: `sir_build.rs`, `versioned.rs`, `graph.rs`, `subprocess.rs`, `config.rs`, `discovery.rs`
- `crates/artifacts/` — Manifest, hashing, packaging: `manifest.rs`, `packaging.rs`, `hashing.rs`
- `crates/report/` — JSON and Markdown reporting: `markdown.rs`, `json_report.rs`
- `crates/cli/` — CLI entry point: `main.rs`

**Python bridge:**
- `python/mil_emitter.py`, `python/bridge.py`, `python/verify.py`, `python/trace_model.py`, `python/model_structure.py`, `python/compute_plan.py`, `python/program_builder.py`, `python/converter.py`, `python/profiler.py`, `python/palettize.py`, `python/common.py`

**Knowledge store (JSON seed files):**
- `knowledge/ane_op_family_matrix.json`, `knowledge/ane_hw_limits_seed.json`, `knowledge/cpu_only_ops_seed.json`, `knowledge/legality_seed.json`, `knowledge/precision_hazard_seed.json`, `knowledge/shard_template_seed.json`, `knowledge/decode_step_shard_template_seed.json`, `knowledge/palettization_constraints_seed.json`

**Documentation:**
- `docs/architecture.md`, `docs/ir_reference.md`, `docs/knowledge_schema.md`, `docs/bridge_protocol.md`, `docs/profiling_methodology.md`
- `ane-constraints-docs/` (8 files across 6 subdirectories)
- `SPEC.md`, `README.md`, `CHANGELOG.md`

### Audit Scope

The audit covered all Rust source files (~133,000 lines), Python bridge code (~9,000 lines), knowledge store JSON files (~3,000 lines), and project documentation (~9,000 lines). The focus was on: ANE constraint enforcement, hardware-version gating, operation legality, dimensional limits, data-type masks, fusion rules, descriptor requirements, emission correctness, and documentation fidelity.

### Methodology

1. **Structural mapping**: Complete file inventory and dependency analysis of all 12 workspace crates.
2. **Pattern-based scanning**: Systematic search for `todo!()`, `unimplemented!()`, permissive defaults, missing validation, phantom capabilities, and stub-mimic functions.
3. **Cross-reference verification**: Comparison of constraint claims across source code, knowledge store JSON, tests, and documentation.
4. **Conservative local metadata review**: Non-invasive, abstract-only analysis of local reference artefacts for compatibility vocabulary hints.
5. **Classification**: Each finding classified as SUPPORTED, UNVERIFIED, PHANTOM, LACUNA, ABERRANT, or STUB-MIMIC with confidence and severity ratings.

### High-Level Findings

- **75 violations** identified across 6 classification categories.
- **5 CRITICAL** violations: LayerNorm family gate missing, KV-cache rewrite generating ANE-illegal ops, proto-direct validator using wrong path, auto-materialized weight dtype hardcoded to Fp16, and cross-type compatibility validation being a complete stub.
- **13 HIGH** violations: Missing ConvTranspose constraint validation, UInt16/Bool dtype gates bypassed, FP32 compute allowed on A11/A12, knowledge store contradicting source code on gather/neg/select/where/erf, zero-filled weight emission on unresolvable constants, and CAPI stubs returning success with wrong data.
- **25 MEDIUM** violations: Documentation contradictions, permissive validation warnings that should be errors, Qwen3-specific defaults applied universally, and unverified hardware limits.
- **32 LOW** violations: Cosmetic issues, minor documentation gaps, dead code, and hardcoded constants that should come from the knowledge store.

### Statement on Raw Local Artefacts

Local reference materials used during this audit are stored in a `forensics/` directory that is excluded from the repository by `.gitignore`. No raw local artefacts, binary strings, disassembly, or proprietary implementation details are included in this report or in the repository.

---

## II. COMPATIBILITY CONSTRAINT TABLES

### II-A. Claimed Supported Operation Categories

| Operation Category | MILLer Claim | ANE Family Scope | Evidence Basis | Confidence |
|---|---|---|---|---|
| Convolution (1x1 as linear) | ANE-legal, PE engine | All families | Source: legality_rewrite.rs, mil_lower.rs | High |
| Convolution (general) | ANE-legal, NE engine | All families | Source: op_constraints.rs, knowledge store | High |
| ConvTranspose | ANE-legal | A12+ (unsupported A11Legacy) | Source: placement_validate.rs; Knowledge: ane_op_family_matrix.json | Medium — no constraint validation wired |
| MaxPool / AvgPool | ANE-legal, NE engine | All families | Source: mil_lower.rs; Knowledge: matrix | High |
| GlobalAvgPool | ANE-legal | All families | Source: mil_lower.rs | High |
| LinearProjection | ANE-legal, PE engine | All families | Source: legality_rewrite.rs, mil_lower.rs | High |
| LUT Projection (palettized linear) | ANE-legal, PE engine | All families | Source: legality_rewrite.rs, mil_lower.rs | High |
| Elementwise (add, mul, sub, div, max, min) | ANE-legal, PE engine | All families | Source: legality_rewrite.rs | High |
| Scaled Dot-Product Attention | ANE-legal, NE engine | A16+ only | Source: mir.rs, ane_target.rs `supports_sdpa()` | High |
| LayerNorm | ANE-legal, PE engine | A15+ only | Source: ane_target.rs `supports_layernorm()` | High — but engine gate missing (V-001) |
| BatchNorm / InstanceNorm | CPU-only | N/A | Source: cpu_only_ops.rs | High |
| ArgMin / ArgMax | ANE-legal, PE engine | A17+ (dropped on A18) | Source: mir.rs, ane_target.rs | High |
| Softmax | ANE-legal | All families (with decomposition) | Source: legality_rewrite.rs | High |
| Reduce L2 Norm | ANE-legal | A13+ (unsupported A11/A12) | Source: mir.rs | Medium |
| Gather / GatherND | CPU-only | N/A | Source: cpu_only_ops.rs (contradicts knowledge store) | Medium — V-009 |
| Select / Where | CPU-only (decomposed to arithmetic) | N/A | Source: cpu_only_ops.rs, legality_rewrite.rs | High |
| Neg | CPU-only | N/A | Source: cpu_only_ops.rs (contradicts knowledge store) | Medium — V-012 |
| Erf | ANE-legal, PE engine | All families | Source: cpu_only_ops.rs (removed from CPU-only) | High |
| Quantize / Dequantize | CPU-only | N/A | Source: cpu_only_ops.rs | High |
| Pad / ReflectivePad | ANE-legal (with constraints) | All families | Source: legality_rewrite.rs | High |
| Reshape / Transpose | ANE-legal | All families | Source: mil_lower.rs | High |
| Const / Constexpr | ANE-legal | All families | Source: mir_to_compat.rs | High |
| Einsum | CPU-only | N/A | Source: cpu_only_ops.rs | High |
| Slice / Concat | ANE-legal | All families | Source: mil_lower.rs | High |

### II-B. Claimed Dimensional Limits

| Parameter | A11Legacy | A12 | A13/A14 | A15 | A16 | A17 | A18 | Evidence Basis | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| max_tensor_width | 16384 | 16384* | 32768 | 65536† | 65536† | 131072 | 131072 | Source: ane_hw_limits.rs | High |
| max_tensor_height | 16384 | 16384* | 16384 | 16384† | 16384† | 16384 | 16384 | Source: ane_hw_limits.rs | High |
| max_tensor_channels | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | 65536 | Source: ane_hw_limits.rs | High |
| max_conv_channels | 32768 | 32768 | 32768 | 32768 | 32768 | 32768 | 32768 | Source: ane_hw_limits.rs | High |
| num_nes | 1 | 1* | 1 | 2 | 2 | 2 | 2 | Source: ane_hw_limits.rs | High |
| max_pooling_kernel_dim | 27 | 27 | 27 | 27 | 27 | 27 | 27 | Source: op_constraints.rs, knowledge store | High |
| ne_transpose_c_max | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 | 16384 | Source: ane_hw_limits.rs | Medium — never validated |

*A12 limits are unverified copies of A11 (V-002).
†A15 limits inherit from A14 without width increase — possibly incorrect (V-012-ir).

### II-C. Claimed Alignment and Layout Requirements

| Constraint | Value | Scope | Evidence Basis | Confidence |
|---|---|---|---|---|
| Channel-last interleave factor | 8 (for Int4/UInt4) | All families | Source: ane_layout.rs, placement_validate.rs | High |
| Palette bit-widths | {1, 2, 3, 4, 6, 8} | All families | Source: ane_layout.rs | High |
| 3-bit palettization | A13+ only | A13 through A18 | Source: ane_layout.rs `validate_palette_bits_for_family()` | High |
| 6-bit palettization | A13+ only | A13 through A18 | Source: ane_layout.rs | High |
| IOSurface eval buffer minimum | ~49 KB | All families | Source: mir_to_proto.rs T-119 | Medium — warning only (V-011) |
| ANE flat buffer layout | [1, C, 1, S] | All families | Source: mir_to_proto.rs | Medium — warning only (V-013) |
| Uniform buffer sizes | Required | All families | Source: mir_to_proto.rs | Medium — warning only (V-012-emit) |

### II-D. Claimed Data-Type Masks

| Data Type | ANE Compute | ANE I/O | Evidence Basis | Confidence |
|---|---|---|---|---|
| FP16 | Allowed | Allowed | Source: dtype_constraints.rs | High |
| FP32 | A13+ only | Allowed | Source: dtype_constraints.rs `is_fp32_compute_supported()` | Medium — gate never called (V-003) |
| Int8 | Allowed (with constraints) | Allowed | Source: dtype_constraints.rs | High |
| UInt8 | Allowed (with constraints) | Allowed | Source: dtype_constraints.rs | High |
| Int4 | Allowed (interleave==8) | Allowed | Source: dtype_constraints.rs, placement_validate.rs | Medium — None bypass (V-004) |
| UInt4 | Allowed (interleave==8) | Allowed | Source: dtype_constraints.rs | Medium — None bypass (V-004) |
| Int32 | Limited (embedding tables) | Allowed | Source: dtype_constraints.rs | High |
| UInt16 | Constrained (op-context) | Allowed | Source: dtype_constraints.rs | Low — context check never enforced (V-002-dt) |
| Bool | Constrained (op-context) | Allowed | Source: dtype_constraints.rs | Low — context check never enforced (V-002-dt) |
| E4M3 | Not supported on ANE | Not supported on ANE | Source: ane_target.rs `supports_e4m3()` | High |
| E5M2 | Not supported on ANE | Not supported on ANE | Source: common.rs comment | Medium — no validation gate (V-007) |
| BF16 | Not in MilDtype | N/A | Source: common.rs | High — BF16 cross-type checks impossible |

### II-E. Claimed Hardware-Version Gates

| Gate | Implementation | Enforced? | Evidence Basis | Confidence |
|---|---|---|---|---|
| `supports_sdpa()` → A16+ | `AneFamily::A16.level() >= level` | Yes, in `default_engine_for_revision()` | Source: mir.rs, ane_target.rs | High |
| `supports_layernorm()` → A15+ | `AneFamily::A15.level() >= level` | **No** — missing from `default_engine_for_revision()` (V-001) | Source: ane_target.rs, mir.rs | High |
| `supports_argminmax()` → A17+ (dropped A18) | Explicit check in `default_engine_for_revision()` | Yes | Source: mir.rs | High |
| `supports_e4m3()` → A16+ | Capability check exists | Yes | Source: ane_target.rs | High |
| FP32 compute → A13+ | `is_fp32_compute_supported()` exists | **No** — never called in placement pipeline (V-003) | Source: dtype_constraints.rs | High |
| 3/6-bit palette → A13+ | `validate_palette_bits_for_family()` | **Partial** — skipped when family is None (V-005) | Source: ane_layout.rs | High |
| ConvTranspose A11Legacy restriction | In knowledge store matrix | **No** — not checked in placement (V-007-place) | Source: ane_op_family_matrix.json | High |

### II-F. Claimed Descriptor Requirements

| Descriptor Type | Claimed Requirement | Evidence Basis | Confidence |
|---|---|---|---|
| MIL program descriptor | op_type, dtype, shape per input/output | Source: mir_to_compat.rs, mil_emitter.py | High |
| Weight descriptor | data, shape, dtype, weight_type (nonzero/const/lut) | Source: mir_to_compat.rs, weights.rs | Medium — auto-dtype hardcoded to Fp16 (V-002-emit) |
| Conv descriptor | kernel_size, stride, padding, dilation, groups | Source: mir.rs, op_constraints.rs | High |
| Pool descriptor | kernel_size, stride, padding | Source: mir.rs | High |
| SDPA descriptor | head_dim, num_heads, is_causal | Source: mir.rs | High |
| Palettization descriptor | bits, lut_data, axis | Source: mir.rs, ane_layout.rs | High |

---

## III. FAITHFULNESS VIOLATIONS

Sorted by severity (CRITICAL → HIGH → MEDIUM → LOW).

### V-001: MILLayerNorm missing family-specific engine override

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/mir.rs:1169, 1354–1414` |
| **Class** | LACUNA |
| **Description** | `MILLayerNorm` assigns `Some(AneEngine::PE)` in `base_engine()` but `default_engine_for_revision()` has no override for pre-A15 families. `AneFamily::supports_layernorm()` correctly returns `false` for A14 and earlier but is never consulted for engine assignment. The same pattern was correctly implemented for SDPA (`supports_sdpa()`) and ArgMinMax (`supports_argminmax()`). Code calling `default_engine_for_revision(Some(V7))` (A14) will get `Some(PE)` for LayerNorm, incorrectly implying ANE placement is legal, causing silent ANEC compile-time failures on A14-class hardware (including M1). |
| **Evidence basis** | Source: ane_target.rs line 96–98, mir.rs line 1169, lines 1369–1411 |
| **Confidence** | High |
| **Severity** | CRITICAL |
| **Section II ref** | II-E: `supports_layernorm()` gate |

### V-002: kv_cache_rewrite generates ANE-illegal Where ops with dangling node references

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/kv_cache_rewrite.rs:128–141, 206` |
| **Class** | ABERRANT |
| **Description** | The `MaskedBlend` path generates `SirOp::Where` ops with synthetic `SirNodeId` values (`"valid_mask_0"`, `"inv_mask_0"`, `"pos_mask_0"`) that reference nodes never added to the graph. The module doc warns this pass is ANE-illegal, but the `RingBuffer` code path falls through to the same `MaskedBlend` logic. If accidentally invoked, the result is a broken SIR graph with dangling node references. |
| **Evidence basis** | Source: kv_cache_rewrite.rs lines 129, 143, 153 |
| **Confidence** | High |
| **Severity** | CRITICAL |
| **Section II ref** | II-A: Select/Where |

### V-003: bridge.py validate_proto_direct uses wrong directory path

| Field | Value |
|---|---|
| **Location** | `python/bridge.py:730, 733–734, 765` |
| **Class** | ABERRANT |
| **Description** | `handle_validate_proto_direct()` checks for `Model/com.apple.CoreML/model.mlmodel`, but the proto-direct emitter writes to `Data/com.apple.CoreML/model.mlmodel` (package.rs line 83). Apple's mlpackage format uses `Data/`. The validator will always report the model directory as "missing" for proto-direct emitted packages, making the validation useless. |
| **Evidence basis** | Source: bridge.py:730 vs package.rs:83 |
| **Confidence** | High |
| **Severity** | CRITICAL |
| **Section II ref** | II-F: MIL program descriptor |

### V-004: mir_to_compat auto-materializes resolved weights with hardcoded MilDtypeCompat::Fp16

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/mir_to_compat.rs:253` |
| **Class** | UNVERIFIED |
| **Description** | When a weight is resolved from the safetensors resolver, the auto-materialized Const op always gets `MilDtypeCompat::Fp16` regardless of the actual dtype. This is incorrect for Int32 weight tensors (e.g., embedding lookup tables containing integer indices). The safetensors resolver passes through non-FP16/BF16/F32 dtypes as-is, so Int32 data would be tagged as Fp16, producing a type mismatch in weight.bin. |
| **Evidence basis** | Source: mir_to_compat.rs:253, safetensors_resolver.rs:217–220 |
| **Confidence** | High |
| **Severity** | CRITICAL |
| **Section II ref** | II-D: FP16, Int32; II-F: Weight descriptor |

### V-005: validate_cross_type_compatibility is a complete stub

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/dtype_constraints.rs:440–469` |
| **Class** | STUB-MIMIC |
| **Description** | `validate_cross_type_compatibility()` claims in its doc comment to validate BF16/F16 cross-type rejections (9 documented ANEC constraint strings), but every code path returns `Ok(())`. The FP16-to-FP32 path logs a warning but still passes. No BF16 check exists because `MilDtype` lacks a BF16 variant. The function presents itself as a validation gate but never rejects anything. |
| **Evidence basis** | Source: dtype_constraints.rs:444–469 |
| **Confidence** | High |
| **Severity** | CRITICAL |
| **Section II ref** | II-D: BF16, cross-type constraints |

### V-006: ConvTranspose unconditionally allowed on ANE without constraint validation

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/placement_validate.rs:589` |
| **Class** | LACUNA |
| **Description** | `MILConvTranspose` unconditionally returns `PlacementDecision::AneAllowed` with no validation. `validate_deconv_constraints()` exists in `op_constraints.rs` (5 constraints: no dilation, SOx==2, no large kernel, no vector palettization, stride>2 with depth>1) but is never called. The knowledge store shows `conv_transpose` is unsupported on A11Legacy. A dilated deconv or deconv with SOx!=2 will pass placement and fail at ANEC compile time. |
| **Evidence basis** | Source: placement_validate.rs:589, op_constraints.rs:340–396, ane_op_family_matrix.json |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: ConvTranspose; II-E: ConvTranspose A11Legacy gate |

### V-007: UInt16/Bool dtype gates bypassed in placement validation

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/dtype_constraints.rs:153, 161; crates/passes/src/placement_validate.rs:267` |
| **Class** | LACUNA |
| **Description** | `is_dtype_ane_legal()` returns `Ok(())` for `UInt16` and `Bool` with the comment "caller must also validate op context". However, the placement validator uses `is_dtype_ane_legal()` as the sole gate and never makes follow-up UInt16/Bool constraint checks. Ops using these dtypes will be allowed on ANE without the documented context restrictions. |
| **Evidence basis** | Source: dtype_constraints.rs:153,161; placement_validate.rs:266–274 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-D: UInt16, Bool |

### V-008: FP32 compute allowed on A11Legacy/A12 without gate

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/dtype_constraints.rs:121; crates/passes/src/placement_validate.rs` |
| **Class** | LACUNA |
| **Description** | `is_dtype_ane_legal()` returns `Ok(())` unconditionally for `Fp32`, referencing `is_fp32_compute_supported()` for compute-specific checks. However, `placement_validate.rs` uses `is_dtype_ane_legal()` as the sole gate and never calls `is_fp32_compute_supported()`. On A11Legacy/A12, FP32 compute is rejected by ANEC but the placement validator allows it. |
| **Evidence basis** | Source: dtype_constraints.rs:121,487–498; placement_validate.rs |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-D: FP32; II-E: FP32 compute gate |

### V-009: Knowledge store declares gather as `ane_legal: true`, contradicting CPU-only classification

| Field | Value |
|---|---|
| **Location** | `knowledge/legality_seed.json:94–115; crates/passes/src/cpu_only_ops.rs:162` |
| **Class** | ABERRANT |
| **Description** | The legality seed declares `mb.gather` as `ane_legal: true` with confidence 0.7, while `cpu_only_ops.rs` lists `gather`, `gather_along_axis`, and `gather_nd` in CPU_ONLY_OPS with the note "Gather has ANE plannability score ~0.26, causing frequent CPU fallback." This is a direct contradiction between the knowledge store and the codebase's hard constraint. |
| **Evidence basis** | Source: legality_seed.json, cpu_only_ops.rs |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: Gather |

### V-010: Knowledge store lists erf as CPU-only, contradicting source code removal

| Field | Value |
|---|---|
| **Location** | `knowledge/cpu_only_ops_seed.json:302–304; crates/passes/src/cpu_only_ops.rs:178–179` |
| **Class** | ABERRANT |
| **Description** | The seed JSON lists `erf` with reason_code "Miscellaneous", but the Rust code explicitly notes "erf IS ANE-legal per the per-op support matrix... Removed from CPU_ONLY." The source code correctly omits `erf`, but the knowledge seed still includes it. Any system bootstrapping from the seed would incorrectly classify `erf` as CPU-only. |
| **Evidence basis** | Source: cpu_only_ops_seed.json, cpu_only_ops.rs:178–179 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: Erf |

### V-011: cpu_only_ops_seed.json missing 70+ ops present in Rust code

| Field | Value |
|---|---|
| **Location** | `knowledge/cpu_only_ops_seed.json` (entire file) |
| **Class** | LACUNA |
| **Description** | The seed JSON has ~84 entries while the Rust code's `CPU_ONLY_OPS` set has >=154 entries. Missing ops include relu6, sigmoid_hard, thresholded_relu, einsum, slice_update, sliding_windows, reverse, argsort, return, is_finite, is_nan, neg, round, strided_slice_update, and many more. Any consumer relying solely on the seed JSON has an incomplete CPU-only catalog. |
| **Evidence basis** | Source: cpu_only_ops_seed.json, cpu_only_ops.rs:198–262 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: Operation categories |

### V-012: Op family matrix lists neg/gather as ANE-supported, contradicting CPU-only classification

| Field | Value |
|---|---|
| **Location** | `knowledge/ane_op_family_matrix.json:715–732, 1238–1255; crates/passes/src/cpu_only_ops.rs` |
| **Class** | ABERRANT |
| **Description** | The matrix lists `neg` as "supported" on A11Legacy through A18, and `gather` as "supported" on A12+. However, `cpu_only_ops.rs` lists both in CPU_ONLY_OPS. The matrix reflects theoretical ANEC converter existence but not practical ANE placement reality, creating contradictory signals. |
| **Evidence basis** | Source: ane_op_family_matrix.json, cpu_only_ops.rs |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: Neg, Gather |

### V-013: Op family matrix lists select/where as ANE-supported (A12+), contradicting CPU-only classification

| Field | Value |
|---|---|
| **Location** | `knowledge/ane_op_family_matrix.json:957–991; crates/passes/src/cpu_only_ops.rs:150–154` |
| **Class** | ABERRANT |
| **Description** | The matrix lists `select` and `where` as "supported" on A12+ with ANEC dialect `PEFUSEDSelect`. However, `cpu_only_ops.rs` lists both as CPU-only with the note: "Despite per-op matrix row 69 listing ConvertSelect, empirical testing shows mb.select causes CPU fallback in practice." The legality_rewrite pass correctly decomposes both to arithmetic, but any system reading the matrix directly would bypass the decomposition. |
| **Evidence basis** | Source: ane_op_family_matrix.json, cpu_only_ops.rs |
| **Confidence** | Medium |
| **Severity** | HIGH |
| **Section II ref** | II-A: Select, Where |

### V-014: mir_op_to_compat silently fills zero bytes for unresolvable MILConst

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/mir_to_compat.rs:939–944` |
| **Class** | LACUNA |
| **Description** | When the WeightResolver returns `None` for a `MILConst` value_path, the code creates `vec![0u8; total_elements * element_size]` — zero-filled weight data — without any error or warning. This is not the `allow_missing_weights` path. Unresolvable weights silently emit garbage, which would produce silently incorrect model outputs. |
| **Evidence basis** | Source: mir_to_compat.rs:940–943 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-F: Weight descriptor |

### V-015: default_engine() bypasses all family-specific checks

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/mir.rs:1429–1431` |
| **Class** | LACUNA |
| **Description** | `default_engine()` calls `default_engine_for_revision(None)`, which returns the base engine without any family-specific overrides (line 1363: `None => return base`). Any caller using `default_engine()` gets incorrect engine assignments for ops with family-specific restrictions. The test file `mir_engine_test.rs` exclusively tests `default_engine()`, meaning no test covers the revision-aware path. |
| **Evidence basis** | Source: mir.rs:1361–1363, 1429–1431 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-E: Hardware-version gates |

### V-016: ALLOWED_DIVERGENCES test is a no-op

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/cpu_only_ops.rs:508–553` |
| **Class** | LACUNA |
| **Description** | 22 ops are listed in `ALLOWED_DIVERGENCES` (ops that have `default_engine() != None` but are in CPU_ONLY_OPS). The test `test_no_ops_in_cpu_only_with_engine_assignment` does nothing — it just iterates over names and discards them with `let _ = name`. The dual-source-of-truth problem is acknowledged but unverified. |
| **Evidence basis** | Source: cpu_only_ops.rs:547–552 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A: Operation categories |

### V-017: CAPI coreml_model_info returns zeroed struct with Ok status

| Field | Value |
|---|---|
| **Location** | `crates/coreml-ffi/src/capi.rs:246–257` |
| **Class** | STUB-MIMIC |
| **Description** | On macOS, `coreml_model_info` returns `CoreMlModelInfo { function_count: 0, has_state: false, spec_version: 0 }` with status `CoreMlStatus::Ok`. A caller checking `status == Ok` would believe the model has zero functions and no state, which is always wrong. |
| **Evidence basis** | Source: capi.rs:252–253 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | N/A (FFI API surface) |

### V-018: CAPI coreml_version returns "unknown" on macOS

| Field | Value |
|---|---|
| **Location** | `crates/coreml-ffi/src/capi.rs:132–143` |
| **Class** | STUB-MIMIC |
| **Description** | On macOS where Core ML is available, the function returns C string "unknown" instead of querying the actual framework version. The doc comment says "On macOS, we would query the actual framework version" but doesn't. |
| **Evidence basis** | Source: capi.rs:139 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | N/A (FFI API surface) |

### V-019: A12 (V5) hardware limits are unverified copies of A11

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_hw_limits.rs:83–91` |
| **Class** | UNVERIFIED |
| **Description** | `AneHwLimits::a12()` constructs limits via `Self { revision: AneRevision::V5, ..Self::a11_legacy() }` — an exact copy of A11 values with only the revision changed. The code acknowledges this with `log::warn!` but the limits are consumed as if authoritative. Any compilation targeting A12 will use A11's limits without verification. |
| **Evidence basis** | Source: ane_hw_limits.rs:90 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-B: A12 dimensional limits |

### V-020: V26 (future) limits are fabricated speculation

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_hw_limits.rs:153–166` |
| **Class** | UNVERIFIED |
| **Description** | `AneHwLimits::future()` inherits from `a18_max()` with `num_nes=16`. This is a fabricated configuration for hardware that doesn't exist. The `log::warn!` acknowledges this, but the struct still passes all `validate_tensor_dims()` checks as if real. |
| **Evidence basis** | Source: ane_hw_limits.rs:153–166 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-B: Future dimensional limits |

### V-021: emit_mir_graph_proto_direct always passes allow_missing_weights=true

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/proto_direct.rs:186–188` |
| **Class** | LACUNA |
| **Description** | When a real WeightResolver is provided (with actual weight data), the code still passes `allow_missing_weights=true`. The comment says "Production callers should check resolver.is_empty() before calling" but there is no such check. Production compilation paths could silently emit zero-filled weights without any error. |
| **Evidence basis** | Source: proto_direct.rs:188 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-F: Weight descriptor |

### V-022: mir_to_proto falls back to empty shape + Float16 for missing I/O descriptors

| Field | Value |
|---|---|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:411–416, 428–433` |
| **Class** | LACUNA |
| **Description** | When an input or output name from the graph isn't found in `input_descs`/`output_descs`, the fallback creates `TensorDesc { shape: vec![], dtype: CoreMlDataType::Float16 }`. An empty shape means "unknown" and Float16 is a default that may be wrong (e.g., Int32 outputs from Argmax). |
| **Evidence basis** | Source: mir_to_proto.rs:413–414 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-D: FP16, Int32; II-F: Descriptor requirements |

### V-023: Qwen3 defaults applied universally when architecture unspecified

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/common.rs:241–254; crates/passes/src/palettize_weights.rs:148–157; crates/bridge/src/mir_to_compat.rs:496–506` |
| **Class** | ABERRANT |
| **Description** | Three separate locations default to Qwen3-specific values when architecture is unspecified: `ModelArchConfig::default()` returns Qwen3-0.6B parameters; `palettize_weights` defaults to `ModelArchitecture::Qwen3`; `build_input_alias_map` defaults to Qwen3 weight patterns. For non-Qwen3 models (LLaMA, Mistral, Phi, GPT-2), these defaults silently produce wrong configurations. |
| **Evidence basis** | Source: common.rs:243–253, palettize_weights.rs:148–157, mir_to_compat.rs:503–504 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-A, II-D, II-F |

### V-024: max_seq_len defaults to 32768 (Qwen3-specific)

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/mir_to_compat.rs:169–175; crates/bridge/src/shape_inference.rs:75–80, 569–580` |
| **Class** | UNVERIFIED |
| **Description** | Three functions silently default to `32768` when `max_seq_len` is not provided, producing wrong shapes for models with different max_position_embeddings. |
| **Evidence basis** | Source: mir_to_compat.rs:169–175, shape_inference.rs:75–80 |
| **Confidence** | High |
| **Severity** | HIGH |
| **Section II ref** | II-F: Descriptor requirements |

### V-025: validate_palette_bits_for_family(None) bypasses version checks

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_layout.rs:216–243` |
| **Class** | LACUNA |
| **Description** | When `family` is `None`, `validate_palette_bits_for_family()` only performs the basic validity check and returns `Ok(())` for 3-bit and 6-bit palettization even on A11Legacy/A12/A13 hardware. Any caller that doesn't provide a family gets a permissive result that would cause ANEC compile-time failures on older hardware. |
| **Evidence basis** | Source: ane_layout.rs:224–240 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: 3/6-bit palette; II-E: Palette version gate |

### V-026: AIR legality_confidence/fallback_risk/drift_risk always hardcoded to ideal values

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/air.rs:890–893; crates/ir/src/serialize.rs:114–117` |
| **Class** | STUB-MIMIC |
| **Description** | `AirNode` carries `legality_confidence`, `fallback_risk`, and `drift_risk` fields that appear to represent probabilistic assessment of ANE placement legality. These are always set to 1.0, 0.0, 0.0 respectively — claiming perfect legality confidence with zero fallback risk. Downstream code reading these values gets a false sense of certainty. |
| **Evidence basis** | Source: serialize.rs:114–117 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Descriptor requirements |

### V-027: KvCacheLayout::Paged is a phantom capability

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/sir.rs:1053–1054` |
| **Class** | PHANTOM |
| **Description** | `KvCacheLayout::Paged` is defined as an enum variant with the comment "Not yet implemented; reserved for future paged-attention support." It is part of the public API, serializable, and can be constructed by any caller. Downstream code that matches on this variant has no implementation to handle it. |
| **Evidence basis** | Source: sir.rs:1053 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-A: Operation categories |

### V-028: E5M2 dtype is a phantom capability

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/common.rs:39–40` |
| **Class** | PHANTOM |
| **Description** | `MilDtype::E5M2` exists with the comment "NOT supported on ANE" but there is no validation gate preventing E5M2 from being used in ANE-targeted compilation. Unlike E4M3 (which has `supports_e4m3()` checks), E5M2 has no corresponding capability check. |
| **Evidence basis** | Source: common.rs:39–40 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-D: E5M2 |

### V-029: StaticizePass is a documented phantom no-op

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/staticize.rs:1–63` |
| **Class** | PHANTOM |
| **Description** | The module documents itself as "REMOVED FROM PIPELINE" with `run()` returning `Ok(input)` and marked `#[deprecated]`. While mitigated by deprecation, the code still exists and could be accidentally reintroduced. |
| **Evidence basis** | Source: staticize.rs:11–12, 59–61 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (dead pass) |

### V-030: IOSurface size validation is warn-only

| Field | Value |
|---|---|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:533–556` |
| **Class** | LACUNA |
| **Description** | ANE fails with 0x1d runtime error for undersized IOSurface buffers (~49 KB minimum). The validation only logs a warning and returns `Ok(())`, allowing silently broken models to pass validation. |
| **Evidence basis** | Source: mir_to_proto.rs:541 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: IOSurface minimum |

### V-031: Surface uniformity validation is warn-only

| Field | Value |
|---|---|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:569–626` |
| **Class** | LACUNA |
| **Description** | ANE requires uniform buffer sizes and produces 0x1d errors when they're not uniform. The code only warns, allowing silently broken models to pass. |
| **Evidence basis** | Source: mir_to_proto.rs:586–594 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: Uniform buffer sizes |

### V-032: Flat buffer layout validation is warn-only

| Field | Value |
|---|---|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:639–663` |
| **Class** | LACUNA |
| **Description** | Tensors not conforming to ANE flat buffer layout [1,C,1,S] may be "silently misinterpreted" (producing incorrect results) but the validation only warns. |
| **Evidence basis** | Source: mir_to_proto.rs:649–657 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: Flat buffer layout |

### V-033: risk_annotate uses catch-all "unknown" pattern for AIR ops

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/risk_annotate.rs:115` |
| **Class** | LACUNA |
| **Description** | The `op_pattern` match has `_ => "unknown"` catch-all. Any new AIR op variant not listed gets "unknown" as its pattern, causing the knowledge query to return `None` and falling back to `DEFAULT_FALLBACK_RISK` (0.1), silently assigning low risk to ops that may have high fallback risk. |
| **Evidence basis** | Source: risk_annotate.rs:115, 124–127 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Descriptor requirements |

### V-034: Int4/UInt4 interleave check bypassed when PlacementContext.interleave is None

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/placement_validate.rs:281; crates/passes/src/dtype_constraints.rs:127, 129` |
| **Class** | LACUNA |
| **Description** | `is_dtype_ane_legal()` returns `Ok(())` for Int4/UInt4 with the comment "caller must also check interleave==8". The placement validator only checks interleave when `ctx.interleave` is `Some(...)`. If a caller provides `dtype=Int4` but `interleave=None`, the dtype passes but the interleave check is silently skipped. |
| **Evidence basis** | Source: placement_validate.rs:281, dtype_constraints.rs:127,129 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: Int4/UInt4 interleave; II-D: Int4, UInt4 |

### V-035: palette_bits construction lacks validation

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/sir.rs:48–53, 156–162` |
| **Class** | LACUNA |
| **Description** | `SirOp::LinearProjection` and `SirOp::Const` have `palette_bits: Option<usize>` with doc "Valid values: {1,2,3,4,6,8}" but no construction-time validation. An invalid value (e.g., `Some(5)`) can be silently constructed and will only fail later during emission. |
| **Evidence basis** | Source: sir.rs:53 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: Palette bit-widths |

### V-036: clamp_to_valid_palette_bits silently rounds without warning

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_layout.rs:259–265` |
| **Class** | ABERRANT |
| **Description** | `clamp_to_valid_palette_bits()` silently rounds invalid bit-widths down (e.g., 5 to 4, 7 to 6) without any logging. A user requesting 7-bit palettization gets 6-bit without knowing their request was modified, causing subtle accuracy degradation. |
| **Evidence basis** | Source: ane_layout.rs:259–265 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-C: Palette bit-widths |

### V-037: validate_conv_channels is separate from validate_tensor_dims

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_hw_limits.rs:168–229` |
| **Class** | LACUNA |
| **Description** | `validate_tensor_dims()` checks `max_tensor_channels` (65536) but NOT `max_conv_channels` (32768). A separate `validate_conv_channels()` exists. Callers that validate tensor dimensions for convolution using only `validate_tensor_dims()` will miss the conv channel constraint. A conv with 40000 channels passes `validate_tensor_dims()` but fails at ANEC. |
| **Evidence basis** | Source: ane_hw_limits.rs:21–25 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-B: max_conv_channels |

### V-038: MILConvTranspose lacks quantization fields present on MILConv

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/mir.rs:59–89` |
| **Class** | LACUNA |
| **Description** | `MILConv` has `kernel_scale`, `kernel_zero_point`, and `kernel_palettized_lut` fields. `MILConvTranspose` has none of these. If ConvTranspose weights are quantized or palettized, there's no way to express the quantization metadata at the MIR level. |
| **Evidence basis** | Source: mir.rs:68–77 vs 79–89 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Conv descriptor |

### V-039: PIR decode-step Interior-to-Exit handoff uses StateWriteRead for tensor passthrough

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/pir.rs:767–789` |
| **Class** | ABERRANT |
| **Description** | The Interior-to-Exit handoff carries `tensor_name: "attn_out"` with `handoff_kind: HandoffKind::StateWriteRead`. However, `StateWriteRead` is documented for "KV-cache-style persistence." The attention output is a direct tensor, not persistent state. The handoff should likely be `TensorPassThrough`. |
| **Evidence basis** | Source: pir.rs:779–789 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (IR semantics) |

### V-040: ModelArchConfig::default() silently assumes Qwen3-0.6B

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/common.rs:241–254` |
| **Class** | ABERRANT |
| **Description** | `ModelArchConfig::default()` returns `Self::qwen3_0_6b()` with Qwen3-specific `vocab_size=151936`, `head_dim=128`, etc. The doc comment marks it as "Deprecated" but it remains the `Default` impl. Any code deriving or using `Default` gets Qwen3-specific values wrong for any other model. |
| **Evidence basis** | Source: common.rs:243–253 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-A: Operation categories |

### V-041: Unknown model_type defaults to Generic with Qwen3 weight patterns

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/common.rs:297–308` |
| **Class** | ABERRANT |
| **Description** | `ModelArchConfig::from_model_config()` maps unknown `model_type` strings to `ModelArchitecture::Generic` with Qwen3's weight patterns (`.self_attn.q_proj.weight`, etc.) as defaults. For models with different weight naming, these patterns won't match, causing silent weight resolution failures. |
| **Evidence basis** | Source: common.rs:299–308 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Weight descriptor |

### V-042: resolve_reshape_shape batch=1 heuristic

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/mir_to_compat.rs:884–893` |
| **Class** | UNVERIFIED |
| **Description** | When a reshape target has two or more zero dimensions, the heuristic sets all but the last zero to 1 (batch dimension heuristic), assuming batch size is always 1. For batch > 1 inference, this produces silently incorrect reshape targets. |
| **Evidence basis** | Source: mir_to_compat.rs |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Descriptor requirements |

### V-043: shape_inference uses name-based heuristic for input_ids shape

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/shape_inference.rs:60–64, 107–109` |
| **Class** | UNVERIFIED |
| **Description** | `compat_input_shape` and `compat_output_shape` use `name.contains("input_ids")` as a fallback to return `vec![1, max_seq_len]`. A tensor named something like "my_input_ids_transform" that isn't actually the model's input_ids would get a wrong shape. |
| **Evidence basis** | Source: shape_inference.rs:60–64 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Descriptor requirements |

### V-044: Python mil_emitter.py _resolve_dtype only handles fp16/fp32

| Field | Value |
|---|---|
| **Location** | `python/mil_emitter.py:62–70` |
| **Class** | LACUNA |
| **Description** | `_resolve_dtype` only recognizes "fp16" and "fp32" strings. All other dtype strings (e.g., "bf16", "int8", "int32") silently fall through to `np.float32`, which is wrong for integer types. |
| **Evidence basis** | Source: mil_emitter.py:70 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | II-D: Data-type masks |

### V-045: Python bridge profile command uses median as mean

| Field | Value |
|---|---|
| **Location** | `python/bridge.py:497` |
| **Class** | STUB-MIMIC |
| **Description** | The profile result claims `"mean_ms"` but returns `median_ns / 1_000_000.0` and `"std_dev_ms": 0.0`. Downstream consumers that use these fields for statistical analysis get wrong results. |
| **Evidence basis** | Source: bridge.py:497–498 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (profiling) |

### V-046: ir_reference.md still lists StaticizePass as pipeline step

| Field | Value |
|---|---|
| **Location** | `docs/ir_reference.md:66` |
| **Class** | ABERRANT |
| **Description** | The IR reference doc lists `StaticizePass: SIR to SIR` as a pipeline step, but the pass is documented as a removed phantom pass (T-107). |
| **Evidence basis** | Source: ir_reference.md:66 vs staticize.rs:14 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (documentation) |

### V-047: bridge_protocol.md claims "No multifunction emission yet" but code has multifunction support

| Field | Value |
|---|---|
| **Location** | `docs/bridge_protocol.md:414` |
| **Class** | ABERRANT |
| **Description** | The limitations section states "No multifunction emission yet" but the codebase has `emit_multifunction`, `emit_multifunction_shared_weights`, `build_multifunction_shared_weights_mir`, and `emit_proto_direct_multifunction`. |
| **Evidence basis** | Source: bridge_protocol.md:414 vs mil_emitter.py:53–54, proto_direct.rs:95–119 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (documentation) |

### V-048: architecture.md claims "No todo!() stubs remain" but CAPI has stubs

| Field | Value |
|---|---|
| **Location** | `docs/architecture.md:93` |
| **Class** | ABERRANT |
| **Description** | The architecture doc states "No `todo!()` stubs remain in these modules" but `capi.rs` has multiple stub implementations (coreml_model_load returns ErrorModelLoad on macOS, coreml_model_info returns zeroed info, coreml_version returns "unknown"). While not `todo!()` macros, they are stubs presenting API surfaces without real implementations. |
| **Evidence basis** | Source: architecture.md:93 vs capi.rs:138,178,252,291,332 |
| **Confidence** | High |
| **Severity** | MEDIUM |
| **Section II ref** | N/A (documentation) |

### V-049: A15 limits inherit from A14 without width increase — possibly incorrect

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_hw_limits.rs:109–111` |
| **Class** | UNVERIFIED |
| **Description** | `AneHwLimits::a15()` is `Self { revision: AneRevision::V8, num_nes: 2, ..Self::a14() }`, inheriting A14's `max_tensor_width=65536`. Given that A14 doubled from A13 and A16 doubles again, A15 might also have different limits. No comment acknowledges this uncertainty. |
| **Evidence basis** | Source: ane_hw_limits.rs:110 |
| **Confidence** | Low |
| **Severity** | MEDIUM |
| **Section II ref** | II-B: A15 dimensional limits |

### V-050: output_dim_for_weight returns 0 for unknown projections

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/legality_rewrite.rs:207–239` |
| **Class** | UNVERIFIED |
| **Description** | `output_dim_for_weight` returns `0` when the weight name doesn't match any known pattern. This flows into `AirOp::Conv1x1AsLinear { output_dim: 0 }` and then through shape inference heuristics. Unknown projections silently get `output_dim=0` with no error. |
| **Evidence basis** | Source: legality_rewrite.rs:222–224, 233–234, 237 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-F: Conv descriptor |

### V-051: ne_transpose_c_max limit exists but is never validated

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_hw_limits.rs:29` |
| **Class** | LACUNA |
| **Description** | `AneHwLimits` has a `ne_transpose_c_max` field (e.g., 16384 for A11) but no corresponding validation method. Transpose operations exceeding this channel limit pass IR-level validation but fail at ANEC compile time. |
| **Evidence basis** | Source: ane_hw_limits.rs:29 |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | II-B: ne_transpose_c_max |

### V-052: AneFamily::family_level() claims total ordering but A18 is not a superset

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_target.rs:188–190` |
| **Class** | ABERRANT |
| **Description** | `family_level()` is used for `>=` comparisons but A18 drops ArgMinMax support, making it not a superset of A17. The doc warns about this but the API invites misuse. |
| **Evidence basis** | Source: ane_target.rs:186–190 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | II-E: Hardware-version gates |

### V-053: Conv1x1AsLinear uses magic zero for unknown output_dim

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/air.rs:44–52` |
| **Class** | ABERRANT |
| **Description** | `AirOp::Conv1x1AsLinear` has `output_dim: usize` with "When 0, the output dim is unknown." Using 0 as a sentinel is error-prone — a legitimate output_dim of 0 would be indistinguishable from "unknown." Should use `Option<usize>`. |
| **Evidence basis** | Source: air.rs:50 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | II-F: Conv descriptor |

### V-054: ShardTemplate::context_length is zero for linear pipelines

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/pir.rs:650` |
| **Class** | ABERRANT |
| **Description** | In `three_shard_linear()`, `context_length` is set to `0`. While linear pipelines may not need a context length, this zero could be misinterpreted by downstream code using `context_length` for KV cache sizing. |
| **Evidence basis** | Source: pir.rs:650 |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | N/A (IR semantics) |

### V-055: LARGE_KERNEL_THRESHOLD hardcoded at 16

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/op_constraints.rs:32` |
| **Class** | UNVERIFIED |
| **Description** | `LARGE_KERNEL_THRESHOLD` is hardcoded as 16, not loaded from the knowledge store. No `large_kernel_threshold` key exists in any knowledge JSON file. If a future ANE revision changes this threshold, the constant would be wrong. |
| **Evidence basis** | Source: op_constraints.rs:32 |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | II-B: Dimensional limits |

### V-056: MAX_POOLING_KERNEL_DIM hardcoded at 27

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/op_constraints.rs:583` |
| **Class** | UNVERIFIED |
| **Description** | `MAX_POOLING_KERNEL_DIM` is hardcoded as 27. The value matches the knowledge store for all current revisions but is not loaded from it. If a future revision changes this limit, the code would need modification. |
| **Evidence basis** | Source: op_constraints.rs:583, ane_hw_limits_seed.json |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | II-B: max_pooling_kernel_dim |

### V-057: StaticizePass dead code still exists

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/staticize.rs:1–63` |
| **Class** | PHANTOM |
| **Description** | Already documented as removed and deprecated, but the code persists. See V-029. |
| **Evidence basis** | Source: staticize.rs |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-058: AirOp::StaticLUTProjection is legacy dead weight

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/air.rs:876–883` |
| **Class** | PHANTOM |
| **Description** | `AirOp::StaticLUTProjection` is marked "Legacy: kept for backward compat." It has no corresponding SIR or MIR op and has been superseded by `ConstexprLutToDense`. Dead variant increasing AIR match surface area. |
| **Evidence basis** | Source: air.rs:876 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-059: canonicalize.rs has undocumented catch-all for new SirOp variants

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/canonicalize.rs:303–306` |
| **Class** | LACUNA |
| **Description** | The `rewrite_op_refs` function has a catch-all `_ => op.clone()` that passes through any SirOp variant not explicitly handled. Any new SirOp variant containing `SirNodeId` references won't have those references rewritten. |
| **Evidence basis** | Source: canonicalize.rs:303–306 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-060: shard_plan.rs has unimplemented TODO for IO shard index

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/shard_plan.rs:428` |
| **Class** | LACUNA |
| **Description** | `let io_shard_idx: usize = 0; // TODO: currently always 0; will shift decoder later when IO shard placement changes`. The IO shard index is hardcoded to 0. |
| **Evidence basis** | Source: shard_plan.rs:428 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-061: shard_plan.rs computes _has_lm_head but never uses it

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/shard_plan.rs:486` |
| **Class** | LACUNA |
| **Description** | `let _has_lm_head = gather_indices.len() >= 2; // TODO: computed but unused`. Planned but unimplemented feature for separate LM head shard handling. |
| **Evidence basis** | Source: shard_plan.rs:486 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-062: Python mil_emitter.py defines _DEFAULT_OPSET_MAP but never uses it

| Field | Value |
|---|---|
| **Location** | `python/mil_emitter.py:57–60` |
| **Class** | ABERRANT |
| **Description** | `_DEFAULT_OPSET_MAP` is defined but never referenced. The actual opset resolution uses `program_builder.resolve_opset_target()`. Dead constant could mislead developers. |
| **Evidence basis** | Source: mil_emitter.py:57–60 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-063: safetensors_resolver uses eprintln! instead of log framework

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/safetensors_resolver.rs:66, 495, 543` |
| **Class** | ABERRANT |
| **Description** | Several diagnostic messages use `eprintln!` instead of the `log` framework used everywhere else. These messages cannot be suppressed or redirected by log level configuration. |
| **Evidence basis** | Source: safetensors_resolver.rs:66, 495, 543 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-064: dirs_home_cache falls back to /tmp/.cache

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/safetensors_resolver.rs:556–569` |
| **Class** | LACUNA |
| **Description** | When neither `XDG_CACHE_HOME` nor `HOME` is set, the cache directory falls back to `/tmp/.cache` — a non-standard location that HuggingFace would never check. |
| **Evidence basis** | Source: safetensors_resolver.rs:568 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-065: T-38 TODOs remain as technical debt

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/mir_to_compat.rs:468, 601, 616; crates/coreml-emit/src/mir_to_proto.rs:841` |
| **Class** | LACUNA |
| **Description** | Four `TODO(T-38)` comments about removing wrapper functions once callers use `MirOpCompat` methods directly. API migration is incomplete. |
| **Evidence basis** | Source: mir_to_compat.rs, mir_to_proto.rs |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-066: CAPI coreml_model_compile returns ErrorModelCompile on macOS

| Field | Value |
|---|---|
| **Location** | `crates/coreml-ffi/src/capi.rs:291` |
| **Class** | STUB-MIMIC |
| **Description** | On macOS where model compilation should work, the function returns `ErrorModelCompile` instead of actually compiling. |
| **Evidence basis** | Source: capi.rs:291 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-067: CAPI coreml_model_predict returns ErrorPrediction on macOS

| Field | Value |
|---|---|
| **Location** | `crates/coreml-ffi/src/capi.rs:332` |
| **Class** | STUB-MIMIC |
| **Description** | On macOS where prediction should work, the function returns `ErrorPrediction` instead of actually running prediction. |
| **Evidence basis** | Source: capi.rs:332 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-068: CanonicalizePass has documented future work items not implemented

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/canonicalize.rs:38–42` |
| **Class** | LACUNA |
| **Description** | Doc lists "Future work": CSE, dead code elimination, linear+bias fusion, naming standardization, elementwise merge. None are implemented. The pass only does identity elimination. These are documented as future work, not claimed capabilities. |
| **Evidence basis** | Source: canonicalize.rs:38–42 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-069: is_fp32_compute_supported() missing M1/A18 sub-revision distinction

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/dtype_constraints.rs:487–498` |
| **Class** | UNVERIFIED |
| **Description** | The function lists `AneFamily::A13` through `AneFamily::A18` as FP32-compute-supported. However, the `A18` family covers three different revisions (V19, V20, V26) per `ane_hw_limits_seed.json`. The function does not distinguish between these sub-revisions. |
| **Evidence basis** | Source: dtype_constraints.rs:487–498, ane_hw_limits_seed.json:139–213 |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | II-D: FP32 |

### V-070: WeightBinBuilder estimated_size calculation may be inaccurate

| Field | Value |
|---|---|
| **Location** | `crates/coreml-emit/src/weights.rs:484–501` |
| **Class** | UNVERIFIED |
| **Description** | `estimated_size()` computes per-entry padding using `e.size` directly without accounting for the metadata header offset. The estimate may be slightly off for models with many small weights. |
| **Evidence basis** | Source: weights.rs:493 |
| **Confidence** | Low |
| **Severity** | LOW |
| **Section II ref** | II-F: Weight descriptor |

### V-071: kv_cache_rewrite.rs uses fragile layer index parsing

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/kv_cache_rewrite.rs:216` |
| **Class** | UNVERIFIED |
| **Description** | `parse_layer_idx` splits on `_` and takes the first `usize`-parseable segment. For state IDs like `"kv_cache_group_2_layer_3_key"`, it would extract `2` instead of `3`. The `unwrap_or(0)` fallback silently assigns layer 0 for unrecognized formats. Pass is documented as dead code. |
| **Evidence basis** | Source: kv_cache_rewrite.rs:216 |
| **Confidence** | Medium |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-072: Resolved weight data may contain non-FP16 data tagged as Fp16

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/safetensors_resolver.rs:217–220` |
| **Class** | UNVERIFIED |
| **Description** | The `_ => { raw_data.to_vec() }` branch passes through Int8, UInt8, and other dtypes without conversion. But auto-materialization in mir_to_compat.rs tags all resolved weights as `MilDtypeCompat::Fp16` (see V-004). Int8 weight data stored as-is but tagged as Fp16 causes incorrect dtype metadata. |
| **Evidence basis** | Source: safetensors_resolver.rs:217–220 (overlaps with V-004) |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-D: Data-type masks; II-F: Weight descriptor |

### V-073: ane_target.rs test panic on unknown revision string

| Field | Value |
|---|---|
| **Location** | `crates/ir/src/ane_target.rs:582` |
| **Class** | ABERRANT |
| **Description** | In `test_hw_limits_seed_family_consistency`, the match arm `_ => panic!("Unknown revision string: {}", rev_str)` will panic if a new revision is added to the expected list but the match isn't updated. |
| **Evidence basis** | Source: ane_target.rs:582 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | N/A |

### V-074: PrecisionPolicyPass silently defaults to Fp16 without knowledge

| Field | Value |
|---|---|
| **Location** | `crates/passes/src/precision_policy.rs:21, 369–410` |
| **Class** | UNVERIFIED |
| **Description** | When `NoKnowledge` is used, every op gets `fp16` with no adaptations. This is documented as intentional but creates a gap: newly-discovered precision hazards in the knowledge store won't affect compilation until the store is populated. |
| **Evidence basis** | Source: precision_policy.rs:21 |
| **Confidence** | High |
| **Severity** | LOW |
| **Section II ref** | II-D: FP16 |

### V-075: Safetensors non-FP16 dtype pass-through without tag correction

| Field | Value |
|---|---|
| **Location** | `crates/bridge/src/safetensors_resolver.rs:217–220` |
| **Class** | UNVERIFIED |
| **Description** | Overlaps with V-004 and V-072. The safetensors resolver passes through non-FP16/BF16/F32 dtypes, but downstream code assumes all resolved data is Fp16. This creates a dtype tagging mismatch that affects weight.bin correctness. |
| **Evidence basis** | Source: safetensors_resolver.rs:217–220 |
| **Confidence** | Medium |
| **Severity** | MEDIUM |
| **Section II ref** | II-D: Data-type masks |

---

## IV. ABSENTEE CAPABILITIES

Operations or features relevant to MILLer's stated scope that are absent or incomplete:

| Capability | Status | Notes |
|---|---|---|
| Paged KV-cache attention | **Declared but not implemented** | `KvCacheLayout::Paged` exists as enum variant with no implementation (V-027) |
| BF16 dtype support | **Absent from type system** | `MilDtype` lacks BF16; cross-type validation stub claims BF16 checks but cannot deliver (V-005) |
| E5M2 dtype rejection gate | **Absent** | `MilDtype::E5M2` exists but no validation gate prevents ANE-targeted compilation (V-028) |
| CSE (Common Subexpression Elimination) | **Declared but not implemented** | Listed as future work in canonicalize.rs (V-068) |
| Dead code elimination | **Declared but not implemented** | Listed as future work in canonicalize.rs (V-068) |
| Linear+bias fusion in canonicalize | **Declared but not implemented** | Listed as future work in canonicalize.rs (V-068) |
| Elementwise merge in canonicalize | **Declared but not implemented** | Listed as future work in canonicalize.rs (V-068) |
| ConvTranspose quantization metadata | **Partially implemented** | MILConv has quant fields; MILConvTranspose lacks them (V-038) |
| Deconv constraint validation | **Implemented but not wired** | `validate_deconv_constraints()` exists with 5 checks but is never called (V-006) |
| FP32 compute gating | **Implemented but not wired** | `is_fp32_compute_supported()` exists but is never called in placement pipeline (V-008) |
| LayerNorm family gating | **Implemented but not wired** | `supports_layernorm()` exists but not consulted in engine assignment (V-001) |
| Hardware limit verification flag | **Absent** | A12/V26 limits are unverified but no `verified: bool` field exists (V-019, V-020) |
| Unified knowledge store schema | **Partially implemented** | Migration to `KnowledgeEntry` documented but not yet implemented; 3 different seed formats coexist |
| ConvTranspose A11Legacy restriction | **In knowledge store only** | `ane_op_family_matrix.json` marks unsupported but placement validator doesn't check (V-006) |
| Batch > 1 reshape inference | **Out of scope (heuristic assumes batch=1)** | V-042 |
| CAPI real implementations on macOS | **Out of scope (stubs)** | coreml_model_info, coreml_version, coreml_model_compile, coreml_model_predict all stubs (V-017, V-018, V-066, V-067) |
| Runtime execution testing | **Out of scope** | Requires Apple hardware; project honestly distinguishes Linux-verifiable vs hardware-verifiable |

---

## V. REMEDIATION ROADMAP

Ordered by priority. Each action addresses one or more violations.

### Phase 1: Critical Fixes (Silent Miscompilation Risk)

1. **Add MILLayerNorm family override to `default_engine_for_revision()`** (V-001)
   - In `mir.rs`, add a check for `family.supports_layernorm()` in `default_engine_for_revision()`, returning `None` for pre-A15 families. Same pattern as SDPA at lines 1404–1408.

2. **Gate or remove `kv_cache_rewrite` pass** (V-002)
   - Either remove the pass entirely (it's documented as dead code) or add a hard guard that prevents it from being invoked for ANE-targeted compilation.

3. **Fix bridge.py proto-direct validation path** (V-003)
   - Change `Model/com.apple.CoreML` to `Data/com.apple.CoreML` in `handle_validate_proto_direct()`.

4. **Derive auto-materialized Const dtype from resolved WeightData** (V-004, V-072, V-075)
   - Add a dtype field to `WeightData` or infer from data length and shape. Replace hardcoded `MilDtypeCompat::Fp16` with the actual dtype.

5. **Implement or remove `validate_cross_type_compatibility()`** (V-005)
   - Either implement the 9 documented cross-type checks (requires adding BF16 to MilDtype) or remove the function and update all callers to fail explicitly.

### Phase 2: High-Priority Validation Gaps

6. **Wire `validate_deconv_constraints()` into placement validator** (V-006)
   - Replace unconditional `AneAllowed` with a call to `validate_deconv_constraints()` and family check.

7. **Enforce UInt16/Bool context constraints in placement validator** (V-007)
   - Add follow-up calls to `validate_uint16_constraints()` and `validate_bool_constraints()` after `is_dtype_ane_legal()`.

8. **Wire `is_fp32_compute_supported()` into placement pipeline** (V-008)
   - Add FP32 compute gating after `is_dtype_ane_legal()` for compute operations.

9. **Fix knowledge store contradictions** (V-009, V-010, V-011, V-012, V-013)
   - Update `legality_seed.json`: set gather `ane_legal: false`.
   - Remove `erf` from `cpu_only_ops_seed.json`.
   - Add missing CPU-only ops to `cpu_only_ops_seed.json`.
   - Add `practical_note` or `empirical_status` field to `ane_op_family_matrix.json` entries for neg, gather, select, where to indicate CPU-only reality despite theoretical ANEC support.

10. **Add error on unresolvable MILConst** (V-014)
    - Change `mir_op_to_compat` to return an error when the WeightResolver returns `None` for a MILConst, rather than silently emitting zero-filled data.

11. **Deprecate `default_engine()`; force revision-aware calls** (V-015)
    - Mark `default_engine()` as deprecated with a warning. Add documentation that it returns incorrect results for family-restricted ops.

12. **Implement ALLOWED_DIVERGENCES test** (V-016)
    - Replace the no-op test with actual assertions verifying the dual-source-of-truth relationship is consistent.

13. **Fix CAPI stubs to return errors instead of Ok with wrong data** (V-017, V-018)
    - Change `coreml_model_info` and `coreml_version` to return `ErrorUnknown` instead of `Ok` with zeroed/fake data.

14. **Mark unverified hardware limits** (V-019, V-020)
    - Add `verified: bool` field to `AneHwLimits`. Make `for_revision()` return `Result` or add a runtime check that prevents compilation with unverified limits unless explicitly opted in.

15. **Fix allow_missing_weights=true for production paths** (V-021)
    - Change `proto_direct.rs` to pass `allow_missing_weights=false` when a real WeightResolver is provided. Add the missing `is_empty()` check.

16. **Fix mir_to_proto I/O descriptor fallback** (V-022)
    - Return an error when an I/O name is missing from the descriptors, rather than defaulting to empty shape + Float16.

17. **Remove Qwen3-specific defaults** (V-023, V-024, V-040, V-041)
    - Remove `Default` impl for `ModelArchConfig`. Make architecture a required parameter in `palettize_weights` and `mir_to_compat`. Make `max_seq_len` a required parameter with no default.

### Phase 3: Medium-Priority Constraint Enforcement

18. **Promote ANE constraint violations from warnings to errors** (V-030, V-031, V-032)
    - IOSurface size, surface uniformity, and flat buffer layout violations should be hard errors since the ANE will reject these models at runtime.

19. **Make `validate_palette_bits_for_family()` require family** (V-025)
    - Change signature to `family: AneFamily` (not `Option`). Force callers to provide the family.

20. **Replace AIR risk fields with status enum** (V-026)
    - Replace `legality_confidence: f32` + `fallback_risk: f32` + `drift_risk: f32` with `legality_status: LegalityStatus` enum (`Verified`/`Unverified`/`LikelyFallback`).

21. **Add `validate_conv_dims()` convenience method** (V-037)
    - Combine `validate_tensor_dims()` + `validate_conv_channels()` into a single method, or add op-type parameter to `validate_tensor_dims()`.

22. **Add E5M2 validation gate** (V-028)
    - Add `supports_e5m2()` method to `AneFamily` (always returns `false`). Add validation in placement pipeline.

23. **Remove or gate `KvCacheLayout::Paged`** (V-027)
    - Either remove the variant or add `#[non_exhaustive]` and a validation gate preventing construction for ANE targets.

24. **Fix PIR handoff semantics** (V-039)
    - Change Interior-to-Exit `attn_out` handoff from `StateWriteRead` to `TensorPassThrough`.

25. **Add ne_transpose_c_max validation** (V-051)
    - Add `validate_transpose_c_max()` method to `AneHwLimits` and call it during transpose op placement.

### Phase 4: Documentation Alignment

26. **Update ir_reference.md** (V-046) — Remove StaticizePass from the pipeline listing.

27. **Update bridge_protocol.md** (V-047) — Remove or update "No multifunction emission yet" limitation.

28. **Update architecture.md** (V-048) — Qualify "no stubs" claim to exclude CAPI layer, or implement CAPI functions.

### Phase 5: Low-Priority Cleanup

29. **Move hardcoded constants to knowledge store** (V-055, V-056) — Add `large_kernel_threshold` and `max_pooling_kernel_dim` fields to hardware limits and load from knowledge store.

30. **Replace magic zero with Option<usize>** (V-053) — Change `Conv1x1AsLinear::output_dim` to `Option<usize>`.

31. **Remove dead code** (V-057, V-058, V-062) — Remove `StaticizePass`, `AirOp::StaticLUTProjection`, `_DEFAULT_OPSET_MAP`.

32. **Fix logging consistency** (V-063) — Replace `eprintln!` with `log::warn!` in safetensors_resolver.

33. **Fix profile reporting** (V-045) — Rename `mean_ms` to `median_ms` or compute actual mean.

---

## VI. FORENSIC NOTE

Local reference materials were used during this audit for non-invasive compatibility vocabulary extraction. These materials are stored in a `forensics/` directory at the repository root, which is excluded from the repository by `.gitignore`. The forensic summary contains only abstract, conservative audit notes: observed public and interface-level vocabulary, broad compatibility hints, possible operation categories, possible descriptor categories, possible validation categories, uncertainty markers, and evidence quality assessments. No raw local artefacts, binary strings, disassembly snippets, proprietary implementation details, or links to local forensic material are included in this report or in the repository.
