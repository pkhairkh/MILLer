# ANEVIOLATIONS.md — MILLer Constraint-Grounded Violation Report

**Operation**: NECROSCOPY — Compatibility Audit  
**Date**: 2026-05-04  
**Scope**: Full MILLer source tree, knowledge seeds, documentation, and non-invasive local reference inventory

---

## I. Executive Abstract

### Files Examined

This audit examined the following source categories by filename:

**Rust crates (12)**: `ir`, `passes`, `knowledge`, `lab`, `bridge`, `trace`, `coreml-proto`, `coreml-emit`, `coreml-ffi`, `artifacts`, `report`, `cli` — approximately 95 source files.

**Python bridge (11)**: `bridge.py`, `mil_emitter.py`, `trace_model.py`, `converter.py`, `compute_plan.py`, `model_structure.py`, `verify.py`, `profiler.py`, `palettize.py`, `program_builder.py`, `common.py`.

**Knowledge seeds (8)**: `legality_seed.json`, `ane_hw_limits_seed.json`, `ane_op_family_matrix.json`, `precision_hazard_seed.json`, `palettization_constraints_seed.json`, `shard_template_seed.json`, `decode_step_shard_template_seed.json`, `cpu_only_ops_seed.json`.

**Documentation (16)**: `SPEC.md`, `STATUS.md`, `README.md`, `ISSUES.md`, `AUDIT.md`, `CHANGELOG.md`, `docs/architecture.md`, `docs/ir_reference.md`, `docs/bridge_protocol.md`, `docs/knowledge_schema.md`, `docs/profiling_methodology.md`, and 5 files under `ane-constraints-docs/`.

**Configuration (6)**: `Cargo.toml`, `pyproject.toml`, `rust-toolchain.toml`, `clippy.toml`, `rustfmt.toml`, `requirements-dev.txt`.

### Audit Scope

The audit scope covered: (1) internal consistency of MILLer's constraint model against its own source code, tests, and documentation; (2) cross-referencing of claimed capabilities against non-invasive local reference metadata; (3) identification of phantom capabilities, stub-mimic functions, missing validation (lacunae), aberrant claims, and unverified assertions.

### Methodology

All source files were read line-by-line. Pattern searches were performed for `todo!()`, `unimplemented!()`, `FIXME`, `HACK`, hardcoded limits, silent fallbacks, permissive defaults, and fake validation. Knowledge seed JSONs were cross-referenced against Rust type definitions, documentation, and conservative local metadata signals. All local reference artefacts were examined using non-invasive techniques only (file identity, container metadata, public symbol names, printable string triage for coarse vocabulary).

### High-Level Findings

The audit identified **97 violations** across 6 classifications:

| Classification | Count | Description |
|---------------|-------|-------------|
| ABERRANT | 16 | Source contradicts its own docs, tests, configs, or conservative compatibility evidence |
| PHANTOM | 7 | Source assumes a capability for which there is no adequate evidence |
| LACUNA | 38 | Source omits validation for a constraint it claims to respect |
| STUB-MIMIC | 10 | Function presents itself as real logic but uses no-ops, fake success, or permissive fallback |
| UNVERIFIED | 26 | Plausible but not backed by sufficient source evidence or tests |

Severity distribution: **7 CRITICAL**, **22 HIGH**, **43 MEDIUM**, **25 LOW**.

### Raw Local Artefacts

Raw local reference artefacts are stored in `forensics/` and `binaries/` within the local working tree. These directories are excluded from the repository by `.gitignore`. No raw binary strings, disassembly, proprietary implementation details, or links to local forensic material appear in this report.

---

## II. Compatibility Constraint Tables

### II-A. Claimed Supported Operation Categories

| Operation Category | Claimed Support Scope | Evidence Basis | Confidence |
|---|---|---|---|
| Convolution (standard, depthwise, dilated, grouped) | All A11+ families | source (ane_op_family_matrix.json, op_constraints.rs) | medium |
| Deconvolution / ConvTranspose | All A11+ families | source (ane_op_family_matrix.json) | medium |
| Average Pool, Max Pool, L2Norm Pool | All A11+ families | source (ane_op_family_matrix.json) | medium |
| ArgMinMax (global, windowed) | A14+ only; dropped for A18 | source (ane_target.rs:145) | medium |
| Reduce (avg, max, min, sum) | All A11+ families | source (ane_op_family_matrix.json) | medium |
| Softmax | All families (matrix); architecture-dependent (per-op doc) | source + local metadata | low |
| InstanceNorm | All families (matrix); architecture-dependent (per-op doc) | source + local metadata | low |
| LayerNorm | A14+ only | source (ane_target.rs) | medium |
| Linear / Fully-Connected | All A11+ families | source (ane_op_family_matrix.json) | high |
| MatMul | All A11+ families | source (ane_op_family_matrix.json) | high |
| Concat, Tile, Transpose, Reshape, Flatten, Unflatten | All A11+ families | source (ane_op_family_matrix.json) | medium |
| Gather, GatherND | A12+ (matrix); illegal (legality seed) | source (contradictory seeds) | low |
| Padding, Resample, Resize, CropResize | A12+ families | source (ane_op_family_matrix.json) | medium |
| PixelShuffle, PixelUnshuffle | A14+ | source (ane_op_family_matrix.json) | medium |
| BatchToSpace, SpaceToBatch, ChannelToSpace, SpaceToChannel | A12+ | source (ane_op_family_matrix.json) | medium |
| Elementwise unary (relu, sigmoid, tanh, erf, invert) | All A11+ families | source (ane_op_family_matrix.json) | medium |
| Elementwise binary (add, mul, sub, div, max, min) | All A11+ families | source (ane_op_family_matrix.json) | high |
| Comparison (equal, not_equal, greater, less, etc.) | CPU-only (cpu_only seed); A14+ ANE (matrix) | source (contradictory seeds) | low |
| Logical (and, or, not) | A12+ ANE (matrix); CPU-only (cpu_only seed); no converter (per-op doc) | source (three-way contradiction) | low |
| SDPA (scaled-dot-product attention) | A16+ (Rust); unreliable A12-A15 (matrix) | source (ane_target.rs) | medium |
| PReLU, Softsign | A12+ ANE (matrix); CPU-only (cpu_only seed) | source (contradictory seeds) | low |

### II-B. Claimed Dimensional Limits

| Constraint | Claimed Value (per AneHwLimits) | Evidence Basis | Confidence |
|---|---|---|---|
| max_tensor_width | 16 384 (V4) → 262 144 (V26) | source (ane_hw_limits.rs, ane_hw_limits_seed.json) | medium |
| max_tensor_height | 16 384 (V4) → 262 144 (V26) | source | medium |
| max_tensor_depth | 256 (V4) → 2 048 (V26) | source | medium |
| max_tensor_channels | 16 384 (V4) → 65 536 (V26) | source | medium |
| max_tensor_rank | 5 (all revisions) | source | medium |
| A12 limits | Copied from A11 (unverified) | source (ane_hw_limits.rs:66-82, self-documented) | low |
| V26 limits | Fabricated from A18 + num_nes=16 | source (ane_hw_limits.rs:144-146) | low |
| ne_transpose_c_max | 16 384 (all revisions, suspiciously uniform) | source (ane_hw_limits_seed.json) | low |

### II-C. Claimed Alignment and Layout Requirements

| Constraint | Claimed Value | Evidence Basis | Confidence |
|---|---|---|---|
| DRAM alignment | 64-byte row stride alignment | local metadata (ANECompiler strings) | medium |
| Power-of-2 alignment requirement | Enforced for memory/cache | local metadata (ANECompiler strings) | medium |
| Tile DMA granularity | Calculated from dram_alignment per HW version | local metadata (ANECompiler strings) | medium |
| L2 cache alignment | Computed from tensor format | source (ane_hw_limits.rs, ZinIrHalParameters) | medium |

### II-D. Claimed Data-Type Masks

| Data Type | ANE Legality Claim | Evidence Basis | Confidence |
|---|---|---|---|
| FP16 | Legal for compute, default dtype | source (dtype_constraints.rs) | high |
| FP32 | Legal (may be downcast) | source (dtype_constraints.rs:73) | medium |
| Int8 | Legal for weights | source (dtype_constraints.rs) | high |
| UInt8 | Legal for weights | source (dtype_constraints.rs) | high |
| Int4 | Legal with interleave==8 (caller must check) | source (dtype_constraints.rs:79-81) | medium |
| UInt4 | Legal with interleave==8 (caller must check) | source (dtype_constraints.rs:79-81) | medium |
| E4M3 | Architecture-conditional (rejected on some, supported on A17+) | local metadata (ANECompiler strings) | medium |
| E5M2 | Rejected by is_dtype_ane_legal(); accepted by quantize validator | source (contradictory checks) | low |
| UInt16 | "Limited support" (no constraints defined) | source (dtype_constraints.rs:105) | low |
| Bool | "Limited support" (no constraints defined) | source (dtype_constraints.rs:113) | low |
| BF16 | Not listed as valid dtype | source (absent from MilDtype enum) | high |

### II-E. Claimed Hardware-Version Gates

| Gate | Claimed Scope | Evidence Basis | Confidence |
|---|---|---|---|
| AneRevision V4–V26 | 11 revisions defined | source (ane_hw_limits.rs) | high |
| AneFamily A11Legacy–A18 | 6 families defined | source (ane_target.rs) | high |
| V6→A13 (Rust) vs V6→A14 (JSON seed) | Family mismatch | source (contradictory mappings) | low |
| V11→A17 (Rust) vs V11→A16 (JSON seed) | Family mismatch | source (contradictory mappings) | low |
| V20 (M4 Mac) → A18 | ArgMinMax dropped; unverified for Mac | source (ane_target.rs:145) | low |
| V26 → future (invented limits) | No hardware specification | source (ane_hw_limits.rs:144-146) | low |

### II-F. Claimed Descriptor Requirements

| Requirement | Claim | Evidence Basis | Confidence |
|---|---|---|---|
| Opset version | iOS18 (hardcoded constant) | source (ir/src/lib.rs:17) | medium |
| Minimum deployment target | iOS18 (hardcoded in shard_plan) | source (shard_plan.rs:561-562) | medium |
| Interleave factor for Int4/UInt4 | Must be 8 (caller-enforced, not validated) | source (dtype_constraints.rs:79-81) | low |
| Conv kernel range | 1–7 (op_constraints.rs) | source | medium |
| Palette bits | Valid: {1,2,3,4,6,8} (documented, not enforced) | source (sir.rs:48-53) | low |

---

## III. Faithfulness Violations

Sorted by severity. Each entry includes a cross-reference to Section II.

### CRITICAL

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-001 | knowledge/ane_hw_limits_seed.json:40-55 | ABERRANT | V6 mapped to family "A14" but Rust code maps V6→A13. Knowledge seed grants A14-class capabilities to A13 hardware. | source | high | II-E |
| V-002 | knowledge/ane_hw_limits_seed.json:108-123 | ABERRANT | V11 mapped to family "A16" but Rust code maps V11→A17. Knowledge seed misses A17 E4M3 support distinction. | source | high | II-E |
| V-003 | knowledge/cpu_only_ops_seed.json:296-324 | ABERRANT | Comparison ops (equal, not_equal, greater, etc.) listed as CPU-only but have ConvertBinaryCompare ANEC converters on A14+. | source + local metadata | high | II-A |
| V-004 | knowledge/ane_op_family_matrix.json:1239-1285 | PHANTOM | logical_and/or/not listed as "supported" A12+ but have no ANEC converter (per-op doc confirms never land on ANE). | source + local metadata | high | II-A |
| V-005 | knowledge/legality_seed.json:62-75 | ABERRANT | mb.gather declared ANE-illegal (ane_legal: false) but anec.gather exists with ConvertGather converter. Blanket illegal claim is wrong. | source + local metadata | high | II-A |
| V-006 | crates/coreml-proto/src/lib.rs:124 | ABERRANT | Float64 element_size() returns 4 instead of 8. All byte-size calculations for Float64 weights will be wrong. | source | high | II-D |
| V-007 | crates/bridge/src/mir_to_compat.rs:224-249 | LACUNA | Zero-filled weight placeholders silently produce models that compile and load but produce completely incorrect inference. Only indication is stderr warning. | source | high | II-D |

### HIGH

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-008 | crates/ir/src/ane_hw_limits.rs:66-82 | UNVERIFIED | A12 hardware limits are unverified copies of A11 values with self-documented WARNING, but used in production constraint validation. | source | high | II-B |
| V-009 | crates/ir/src/ane_hw_limits.rs:148-193 | LACUNA | 7 conv/pool/PE-specific constraint fields defined but never validated by validate_tensor_dims. Conv/pool ops with oversized kernels pass validation but fail at ANE emission. | source | high | II-B |
| V-010 | crates/ir/src/mir.rs:1061-1311 | LACUNA | default_engine() returns static engine assignment per op regardless of AneRevision. Ops assigned to PE may be placed on families that don't support them. | source | high | II-E |
| V-011 | crates/ir/src/shard_desc.rs:95 | STUB-MIMIC | Unknown dtype string silently defaults to Fp16. Invalid dtype strings like "bf16" or "int8" produce wrong precision without error. | source | high | II-D |
| V-012 | crates/ir/src/common.rs:297-308 | ABERRANT | Generic model architecture falls back to Qwen3 weight patterns. Non-Qwen3/LLaMA models will have silently broken weight resolution. | source | high | II-A |
| V-013 | crates/ir/src/payload.rs:685-698 | ABERRANT | FamilyPayload hardcodes stateful: false regardless of actual op. Stateful ops emitted via generic path get wrong function descriptors. | source | high | II-F |
| V-014 | crates/passes/src/staticize.rs:43-46 | PHANTOM | Entire StaticizePass::run() is Ok(input) — pure pass-through. Doc claims it replaces symbolic dims, resolves variable-length sequences, records decisions. None implemented. | source | high | II-B |
| V-015 | crates/passes/src/precision_policy.rs:94-118 | LACUNA | Only 14 of ~167 SIR op types query precision hazards. All others silently use default fp16 even if stored knowledge indicates a hazard. | source | high | II-D |
| V-016 | crates/passes/src/state_topology.rs:43-96 | STUB-MIMIC | Pass claims to "verify" and "ensure" state patterns but only logs eprintln warnings — never returns Err. Claims to validate naming conventions and state capacity but implements neither. | source | high | II-F |
| V-017 | crates/passes/src/shard_plan.rs:367-378 | LACUNA | FunctionEntry TensorSpec shapes hardcoded as vec![1,1] throughout. Comments say "derived from graph" but derivation not implemented. | source | high | II-B |
| V-018 | crates/passes/src/mil_lower.rs:92-98 | LACUNA | MatMul inner-dim mismatch only logs eprintln warning and continues, producing graph with wrong dimensions. | source | high | II-B |
| V-019 | crates/passes/src/cpu_only_ops.rs:147-156 | ABERRANT | gather listed in CPU_ONLY_OPS but mil_lower actively emits MILGather for embedding lookup and legality_rewrite generates Gather for RoPE table lookups. | source | high | II-A |
| V-020 | crates/passes/src/placement_validate.rs:272-292 | LACUNA | Interleave constraints skipped entirely when channels unknown, including non-channel-dependent checks (const→1, int4→8). | source | high | II-D |
| V-021 | crates/knowledge/src/transfer.rs:152 | LACUNA | claims_agree defaults to true for 7/8 knowledge types. No contradiction detection for PrecisionHazard, SurvivalMatrixEntry, MotifCatalog, FallbackSignature, DeviceFingerprint, ShardTemplateKnowledge, StateTopologyOutcome. | source | high | II-A |
| V-022 | crates/knowledge/src/store.rs:524-547 | LACUNA | Conflict detection marks new entry as ConflictedWith(existing) but never back-patches existing entry. Querying only existing entries misses mutual conflicts. | source | high | II-F |
| V-023 | crates/bridge/src/mir_to_compat.rs:275-413 | LACUNA | Input/output node fallback to shape vec![1] and dtype Fp16 when node missing from MIR graph. Almost certainly wrong for any real model. | source | high | II-B, II-D |
| V-024 | crates/bridge/src/subprocess.rs:28,55-69 | PHANTOM | timeout_secs field stored but never enforced. Command::output() blocks indefinitely. Timeout is a phantom capability. | source | high | II-F |
| V-025 | crates/bridge/src/mir_to_compat.rs:458-468 | LACUNA | Default architecture silently defaults to Qwen3 weight name patterns. Wrong patterns mean wrong input remapping, producing models with undefined references. | source | high | II-A |
| V-026 | crates/bridge/src/safetensors_resolver.rs:196-199 | LACUNA | F32 weight data passed through without FP16 conversion. BF16 gets converted; F32 does not. F32 bytes written as-is but may be declared as FP16 in proto. | source | high | II-D |
| V-027 | crates/coreml-emit/src/weights.rs:116-119 | LACUNA | Bool, Float64, Unknown data types silently mapped to Float32 blob format. Data-corrupting for Bool tensors; incorrect for Float64. | source | high | II-D |
| V-028 | crates/coreml-emit/src/mir_to_proto.rs:339-356 | LACUNA | State declarations default to empty shape + Fp16 when only write op present. Core ML will reject proto with empty-dimension state. | source | high | II-B, II-D |
| V-029 | knowledge/ane_op_family_matrix.json:806-821 | UNVERIFIED | Softmax listed as "supported" for all families including A11Legacy, but per-op doc shows architecture-dependent. Will cause compilation failures on A11/A12 ANE. | source | high | II-A |
| V-030 | knowledge/ane_op_family_matrix.json:951-965 | UNVERIFIED | InstanceNorm listed as "supported" for all families but per-op doc shows architecture-dependent on older families. | source | high | II-A |

### MEDIUM

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-031 | crates/ir/src/ane_hw_limits.rs:144-146 | PHANTOM | V26 "future" limits are fabricated (inherits A18 + num_nes=16). No hardware spec exists; no warning emitted. | source | high | II-B |
| V-032 | crates/ir/src/ane_target.rs:145 | UNVERIFIED | V20 (M4 Mac) mapped to A18 family; ArgMinMax dropped. Mac ANE hardware may differ from mobile A18. | source | medium | II-E |
| V-033 | crates/ir/src/sir.rs:48-53 | LACUNA | palette_bits documented as valid {1,2,3,4,6,8} but no validation enforces this. Out-of-range values accepted silently. | source | high | II-F |
| V-034 | crates/ir/src/sir.rs:1052-1054 | PHANTOM | KvCacheLayout::Paged is a full enum variant documented as "not yet implemented" but constructible via serde deserialization. | source | high | II-A |
| V-035 | crates/ir/src/common.rs:241-254 | ABERRANT | ModelArchConfig::default() silently assumes Qwen3-0.6B. Deprecated but still callable, producing wrong defaults for other architectures. | source | high | II-A |
| V-036 | crates/ir/src/pir.rs:787-790 | ABERRANT | Decode-step claims StateWriteRead handoff but comment says emission uses linear projection. PIR claims runtime semantics not delivered. | source | high | II-F |
| V-037 | crates/ir/src/lib.rs:17 | UNVERIFIED | DEFAULT_OPSET_VERSION = "iOS18" hardcoded without target validation. Models will fail on older iOS at load time. | source | medium | II-F |
| V-038 | crates/ir/src/shard_desc.rs:363-389 | ABERRANT | PIR tensor specs hardcoded to dtype "fp16" ignoring actual task spec dtype. Wrong for fp32/int4 tasks. | source | high | II-D |
| V-039 | crates/ir/src/air.rs:885-895 | LACUNA | legality_confidence, fallback_risk, drift_risk fields have no validation of value ranges, no documented semantics, no producers/consumers. | source | medium | II-F |
| V-040 | crates/passes/src/kv_cache_rewrite.rs:1-313 | ABERRANT | Deprecated pass generates ANE-illegal Where ops. Still compilable with working tests that produce illegal graphs. | source | high | II-A |
| V-041 | crates/passes/src/op_constraints.rs:38-51 | ABERRANT | Conv kernel range 1–7 contradicts later grouped/dilated threshold of 16. Either 1–7 is too restrictive or the 16-check is dead code. | source | medium | II-B |
| V-042 | crates/passes/src/op_constraints.rs:160-161 | LACUNA | Pooling validation accepts kernel_size parameter but immediately discards it (let _ = kernel_size). No kernel size limits enforced. | source | high | II-B |
| V-043 | crates/passes/src/mil_lower.rs:156-175 | LACUNA | Broadcast incompatibility falls back to x's shape with only eprintln warning. Produces wrong MIR output shapes. | source | high | II-B |
| V-044 | crates/passes/src/legality_rewrite.rs:533-543 | LACUNA | Tile decomposition emits reshape_shape and final_shape with 0 placeholders. Zeros propagate through shape inference. | source | high | II-B |
| V-045 | crates/passes/src/shard_plan.rs:559 | LACUNA | PIR context_length always 0. Semantically important for KV cache models but never derived from graph or task spec. | source | high | II-B |
| V-046 | crates/passes/src/shard_plan.rs:561-562 | LACUNA | PIR opset_version and minimum_deployment_target hardcoded to "iOS18" regardless of target. Wrong for A11/A12 (iOS 16-era). | source | high | II-F |
| V-047 | crates/passes/src/shard_plan.rs:400,527 | UNVERIFIED | KV cache default shape fallback vec![2,1,1,1,1] is arbitrary. Batch=2 and all-1 dimensions almost certainly wrong for any real model. | source | high | II-B |
| V-048 | crates/passes/src/placement_validate.rs:516 | LACUNA | ConvTranspose always passes placement validation unconditionally. No kernel size, stride, or group checks. | source | high | II-B |
| V-049 | crates/passes/src/dtype_constraints.rs:73 | UNVERIFIED | FP32 allowed as ANE-legal with comment "may be downcast" but downcast not enforced. ANE does not natively compute in FP32. | source | medium | II-D |
| V-050 | crates/passes/src/dtype_constraints.rs:79-81 | LACUNA | Int4/UInt4 return Ok(()) with comment "caller must also check interleave==8." Critical constraint deferred to caller with no enforcement. | source | high | II-D |
| V-051 | crates/passes/src/dtype_constraints.rs:180-182 | ABERRANT | Quantize validator accepts E5M2 as output dtype, but is_dtype_ane_legal() rejects E5M2 on all families. Two checks are contradictory. | source | high | II-D |
| V-052 | crates/passes/src/canonicalize.rs:294 | LACUNA | Wildcard catch-all silently copies unrecognized SirOp variants without rewriting SirNodeId references, producing dangling references for new variants. | source | high | II-F |
| V-053 | crates/knowledge/src/update.rs:87-89 | ABERRANT | Doc says "never start above 0.5" but CompileFailure starts at 0.7, LoadFailure at 0.8, ComputePlan at 0.9. | source | high | II-F |
| V-054 | crates/knowledge/src/confidence.rs:13-25 | PHANTOM | decay_confidence is "currently only used in tests." Advertised temporal decay mechanism has zero production integration. | source | high | II-F |
| V-055 | crates/knowledge/src/transfer.rs:86-89 | ABERRANT | Doc says pattern-level transfer scales "0.5–0.8 depending on similarity" but code uses hardcoded midpoint 0.65. No similarity metric computed. | source | high | II-F |
| V-056 | crates/knowledge/src/lib.rs:38-39 | ABERRANT | ComputePlanObservation doc says "confidence always 0.9" but code accepts any [0,1] value from caller. | source | high | II-F |
| V-057 | crates/knowledge/src/compute_plan_verify.rs:169-242 | UNVERIFIED | default_known_placements claims Apple documentation source but all entries marked "empirical" with hardcoded confidence. No cross-check infrastructure. | source | high | II-A |
| V-058 | crates/knowledge/src/shard_template.rs:377-388 | LACUNA | parse_evidence_source silently maps unknown sources to ManualEntry, changing semantic meaning and reliability interpretation. | source | high | II-F |
| V-059 | crates/bridge/src/mir_to_compat.rs:92-104 | ABERRANT | EmptyWeightResolver doc says "returns Some with empty data" but implementation returns None. | source | high | II-D |
| V-060 | crates/bridge/src/subprocess.rs:73-89 | LACUNA | Python subprocess failure returns Ok(BridgeResult{status:"error"}) instead of Err(). Forces callers to manually check status string. | source | high | II-F |
| V-061 | crates/bridge/src/safetensors_resolver.rs:319 | LACUNA | Shard weight slicing assumes FP16 (bytes_per_row = hidden_size * 2). F32 or BF16 base weights would produce corrupted data. | source | high | II-D |
| V-062 | crates/bridge/src/shape_inference.rs:530 | LACUNA | Catch-all returns empty shape for unrecognized MirOp variants. Core ML inference not guaranteed to succeed for all ops. | source | medium | II-B |
| V-063 | crates/bridge/src/shape_inference.rs:74-80 | UNVERIFIED | Deprecated shape functions hardcode Qwen3 max_seq_len=32768. Still callable, not feature-gated. | source | high | II-B |
| V-064 | crates/coreml-emit/src/emitter.rs:129-148 | PHANTOM | compare_with_python_bridge is an unimplemented stub that always returns None. Doc describes full comparison workflow. | source | high | II-F |
| V-065 | crates/coreml-proto/src/lib.rs:930-938 | PHANTOM | MirOpCompat::Unsupported is constructible but always rejected at emission gate. Type-level capability that can never produce output. | source | high | II-A |
| V-066 | crates/coreml-emit/src/mir_to_proto.rs:94-121 | PHANTOM | Fill, Select, Where are valid MirOpCompat variants but always rejected as "ANE-illegal." Error says "should have been replaced earlier" but type system allows them. | source | high | II-A |
| V-067 | knowledge/ane_hw_limits_seed.json (all revisions) | LACUNA | Only ~14 of ~40+ documented hardware limit parameters captured. No validation for conv kernel sizes, pooling limits, padding limits, PE/NE engine constraints. | source | high | II-B |
| V-068 | knowledge/ane_hw_limits_seed.json (ne_transpose_c_max) | UNVERIFIED | ne_transpose_c_max identical (16384) across all 11 revisions despite 16x variation in other parameters. Possibly copied from V4 defaults. | local metadata | medium | II-B |
| V-069 | knowledge/ (all seeds) | LACUNA | Knowledge seeds use device_classes ["M2","M3"] but compiler operates on AneFamily. No documented device_class→AneFamily mapping exists. | source | high | II-E |
| V-070 | knowledge/legality_seed.json:48-60 | UNVERIFIED | Synthetic evidence at confidence 0.9 exceeds real-model evidence at 0.6. Schema's own ×0.7 synthetic transfer penalty not applied to stored raw confidence. | source | medium | II-F |
| V-071 | knowledge/ (all seeds) | ABERRANT | Knowledge schema defines unit wrapper with provenance, conflict_status, conflict_priority fields. No seed JSON follows this schema — all use flat fields without unit wrapper. | source | high | II-F |
| V-072 | knowledge/ane_op_family_matrix.json:86-101 | UNVERIFIED | SDPA marked "unreliable" for A12-A15 but Rust binary-classifies as not-supported (A16+ only). "Unreliable" has no operational semantics. | source | medium | II-A |
| V-073 | knowledge/precision_hazard_seed.json | UNVERIFIED | All 4 entries derive from single model (Qwen3). Claims general rules based on 3 evidence points from one model. No cross-validation. | source | medium | II-D |
| V-074 | knowledge/shard_template_seed.json:18-19 | UNVERIFIED | known_good: true with perplexity_delta: -0.57 (worse quality). No documented threshold for acceptable quality delta. | source | medium | II-F |
| V-075 | knowledge/palettization_constraints_seed.json:5-9 | LACUNA | Dual conv min bits (standard:4, alternate:2) without conditional context. Compiler cannot determine which minimum applies for a given hardware version or conv subtype. | source | medium | II-D |
| V-076 | crates/lab/src/device_meta.rs:125-140 | PHANTOM | device_backed() always returns HostOnly on all platforms including macOS. Method name implies device-backed metadata but never produces it. | source | high | II-F |
| V-077 | crates/lab/src/harness.rs:70 | LACUNA | coremltools_available hardcoded to false. Python bridge detection result never folded back. Every LabRun permanently reports false. | source | high | II-F |
| V-078 | crates/lab/src/fallback.rs:135-162 | ABERRANT | assess_overall_level() returns Unavailable for device-backed runs with no baseline, even though device evidence IS available. | source | medium | II-F |
| V-079 | crates/lab/src/harness.rs:408-412 | LACUNA | LabRunBuilder allows timing on HostOnlyInspection runs despite doc saying "MUST be None." No compile-time or runtime enforcement. | source | high | II-F |
| V-080 | crates/coreml-ffi/src/api.rs:43-71 | STUB-MIMIC | CoreMlApi::version() returns Ok("unknown"); compile_model() returns Err on macOS after passing is_available() check. API surface suggests functionality that doesn't exist. | source | high | II-F |
| V-081 | crates/coreml-ffi/src/model.rs:92-111 | STUB-MIMIC | FfiModel::load() returns Ok with handle: None on macOS. "Loaded" model where is_loaded() returns false. Subsequent calls fail but not at load time. | source | high | II-F |
| V-082 | crates/coreml-ffi/src/capi.rs:204-224 | STUB-MIMIC | coreml_model_info() writes zeroed info with CoreMlStatus::Ok. C consumer would interpret Ok as successful metadata retrieval. | source | high | II-F |
| V-083 | crates/coreml-ffi/src/capi.rs:383-391 | ABERRANT | Validation rejects packages without weight.bin, but not all models require it. Inline-const-only models are perfectly valid but fail validation. | source | high | II-F |
| V-084 | crates/trace/src/sir_build.rs:169-172 | LACUNA | QualityContract hardcoded: max_perplexity_delta=0.1, max_latency_ms=50.0. Not per-model configurable. Overly restrictive for 7B models. | source | high | II-F |
| V-085 | crates/trace/src/sir_build.rs:84-94 | LACUNA | Missing input shape falls back to (1,32) silently. Wrong for models with different expected shapes. No warning when fallback used. | source | high | II-B |
| V-086 | crates/bridge/src/mir_to_compat.rs (30+ eprintln) | LACUNA | Critical constraint violations reported via eprintln instead of structured logging or error returns. Cannot be suppressed, tested, or consumed by downstream. | source | high | II-F |
| V-087 | crates/cli/src/main.rs:696,954 | LACUNA | --seed parameter accepted by CLI but silently discarded (_seed prefix). SPEC requires deterministic compilation but seed is unused. | source | high | II-F |

### LOW

| ID | Location | Class | Description | Evidence | Confidence | Ref |
|----|----------|-------|-------------|----------|------------|-----|
| V-088 | crates/ir/src/ane_hw_limits.rs:144-146 | PHANTOM | V26 "future" limits fabricated. Currently unreachable from normal paths but for_revision(V26) returns plausible-looking but fictional limits. | source | high | II-B |
| V-089 | crates/passes/src/role_mir.rs:131,146,210 | UNVERIFIED | Multiple unwrap_or fallbacks use hardcoded dimension values (64, 48, 32). Silently produce wrong MIR shapes if spec incomplete. | source | high | II-B |
| V-090 | crates/passes/src/canonicalize.rs:113-119 | UNVERIFIED | Chain resolution loop has magic limit of 100 steps with no diagnostic when hit. Graphs with 100+ chained identity nodes produce incomplete substitution. | source | high | II-F |
| V-091 | crates/passes/src/dtype_constraints.rs:105 | UNVERIFIED | UInt16 marked ANE-legal with "limited support" but no constraint checks or documentation of what "limited" means. | source | medium | II-D |
| V-092 | crates/passes/src/dtype_constraints.rs:113 | UNVERIFIED | Bool marked ANE-legal with "limited support" but no constraint checks. | source | medium | II-D |
| V-093 | crates/passes/src/cpu_only_ops.rs:256-319 | LACUNA | CPU_ONLY_OPS_DETAILED has ~30 entries vs 120+ in CPU_ONLY_OPS. ~90 ops have no documented reason for being CPU-only. | source | high | II-A |
| V-094 | crates/knowledge/src/compute_plan_verify.rs:337 | LACUNA | knowledge_consistent is binary all-or-nothing. Single mismatch out of hundreds makes entire proof "not consistent." Discards nuance. | source | medium | II-F |
| V-095 | crates/coreml-emit/src/package.rs:155-195 | UNVERIFIED | UUIDs generated via v5 with fixed namespace + function name. Two models with same function name produce identical UUIDs. Not globally unique. | source | medium | II-F |
| V-096 | crates/lab/src/fallback.rs:165-181 | PHANTOM | FallbackLogEvidence defined with full serde support but never constructed. "Reserved for future use" — phantom capability. | source | high | II-F |
| V-097 | python/mil_emitter.py:904 | LACUNA | LayerNorm epsilon np_dtype(1e-5) truncates for FP16 (becomes ~9.98e-6). Emitted model epsilon differs from programmer intent. | source | high | II-D |

---

## IV. Absentee Capabilities

| Capability | Status | Notes |
|---|---|---|
| Static dimension resolution | Declared but not implemented | StaticizePass exists as pure pass-through (V-014) |
| Paged KV-cache attention | Declared but not implemented | KvCacheLayout::Paged variant exists but no code path implements it (V-034) |
| Temporal confidence decay | Declared but not in production | decay_confidence function exists but is test-only (V-054) |
| Python bridge comparison | Declared but not implemented | compare_with_python_bridge always returns None (V-064) |
| Timeout enforcement on Python bridge | Declared but not implemented | timeout_secs stored but never used (V-024) |
| Device-backed metadata on macOS | Declared but not implemented | device_backed() always returns HostOnly (V-076) |
| Core ML FFI compilation on macOS | Declared but not functional | compile_model() always returns Err (V-080) |
| Core ML model version query on macOS | Declared but returns "unknown" | version() returns Ok("unknown") (V-080) |
| Reproducibility seed in compile pipeline | Declared but not wired | --seed accepted but discarded (V-087) |
| Hardware-specific conv/pool kernel validation | Partially declared | Constraint fields exist in AneHwLimits but never validated (V-009) |
| Precision hazard coverage for all ops | Partially declared | Only 14/167 op types query hazards (V-015) |
| BF16 data type support | Out of scope (absent) | Not listed in MilDtype enum; silently defaults to Fp16 in shard_desc (V-011) |
| Conflict detection for most knowledge types | Partially declared | claims_agree defaults true for 7/8 types (V-021) |
| Compute plan verification against real hardware | Inferred but unverified | Hardcoded empirical mappings with no cross-check (V-057) |
| Int8 dtype in shard descriptor lowering | Declared but silently treated as Fp16 | Unknown dtype falls through to Fp16 default (V-011) |
| Stateful function descriptors via generic path | Declared but always false | FamilyPayload hardcodes stateful: false (V-013) |

---

## V. Remediation Roadmap

Ordered by priority; each action references the violation ID(s) it addresses.

### Phase 1 — Critical Fixes (Silent Miscompilation or Data Corruption)

1. **V-006**: Fix `Float64` element_size to return 8 instead of 4 in `coreml-proto/src/lib.rs`. All downstream byte-size calculations depend on this.

2. **V-007**: Replace zero-filled weight placeholders with hard errors. When a weight cannot be resolved, fail the compilation rather than producing a silently broken model. Add `--allow-missing-weights` flag for intentional zero-fill scenarios.

3. **V-001, V-002**: Align `ane_hw_limits_seed.json` family assignments with Rust code: V6→A13 (not A14), V11→A17 (not A16). Add automated test that checks JSON→Rust mapping consistency.

4. **V-003, V-004, V-005**: Resolve three-way contradictions in knowledge seeds:
   - Comparison ops: move from `cpu_only_ops_seed.json` to `ane_op_family_matrix.json` with A14+ scope.
   - Logical AND/OR/NOT: mark as CPU-only in `ane_op_family_matrix.json`; remove phantom "supported" entries.
   - Gather: change `legality_seed.json` to `ane_legal: true` with `limited_index_range` constraint.

5. **V-011**: Replace silent Fp16 default for unknown dtype with explicit error listing valid dtype strings. Add Int8 to accepted dtype list.

### Phase 2 — High-Priority Validation Gaps

6. **V-008**: Mark A12 limits as UNVERIFIED in the knowledge store with reduced confidence. Add test that logs a warning when A12 limits are used in constraint validation. Begin empirical measurement on A12 hardware.

7. **V-009**: Implement validation for the 7 unused conv/pool/PE constraint fields in `validate_tensor_dims()`. Add per-op validation functions for convolution, pooling, and PE operations.

8. **V-010**: Make `default_engine()` revision-aware by cross-referencing `AneFamily` capability methods. Return `None` (no engine) for ops not supported on the target family.

9. **V-012**: Remove Generic→Qwen3 fallback. Return an error when model architecture is not recognized, requiring explicit architecture specification.

10. **V-013**: Derive `stateful` flag from the actual op type in `FamilyPayload::from_spec_with_override()`. Check for DecodeStep and other stateful ops.

11. **V-014**: Either implement StaticizePass or remove it from the pipeline and document its absence. Current phantom pass wastes developer trust.

12. **V-015**: Expand precision_policy coverage to all SIR op types. At minimum, add the top-30 most common op types and mark the remainder with explicit "coverage gap" markers.

13. **V-016**: Replace `eprintln!` warnings in StateTopologyPass with `Result::Err` returns for invalid state patterns, or clearly document that the pass is advisory-only.

14. **V-017**: Derive FunctionEntry shapes from the MIR graph instead of hardcoding `vec![1,1]`. Walk the graph to extract actual batch/seq dimensions.

15. **V-018**: Replace eprintln+continue for MatMul inner-dim mismatch with `Err`. Shape mismatch is a correctness violation, not a warning condition.

16. **V-019**: Resolve Gather contradiction: either remove from CPU_ONLY_OPS (with appropriate scoping for embedding-only), or replace Gather emission with SliceByIndex in all paths.

17. **V-020**: Restructure interleave validation to enforce non-channel-dependent checks (const→1, int4→8) even when channels is None.

18. **V-021**: Implement claims_agree logic for all 8 knowledge types. At minimum, add field-level comparison for PrecisionHazard and SurvivalMatrixEntry.

19. **V-022**: Make conflict detection symmetric: when new entry B conflicts with existing A, mark both A and B as conflicted.

20. **V-023, V-025**: Replace fallback shapes/dtypes with hard errors when nodes are missing from MIR graph. Fail compilation rather than emitting wrong descriptors.

21. **V-024**: Implement timeout using `Command::new().timeout()` or `child.wait_timeout()`. Remove the phantom field or make it functional.

22. **V-026**: Add FP32→FP16 conversion in safetensors_resolver when target dtype is FP16. Ensure data format matches proto declaration.

23. **V-027**: Map Bool to a dedicated Bool blob type or reject at validation time. Map Float64 to 8-byte Float64 blob (after fixing V-006). Reject Unknown dtype early.

24. **V-028**: Derive state shape from the corresponding ReadState op. If no ReadState exists, require explicit shape specification rather than defaulting to empty.

25. **V-029, V-030**: Add architecture-dependent qualifiers to Softmax and InstanceNorm matrix entries. Use "conditional" or "architecture-dependent" support status with explicit family caveats.

### Phase 3 — Medium-Priority Cleanup

26. **V-031, V-088**: Either remove V26 revision or add explicit "speculative — not based on any hardware" warning in for_revision() return value.

27. **V-033**: Add palette_bits validation to SirOp construction or deserialization, rejecting values outside {1,2,3,4,6,8}.

28. **V-034**: Gate KvCacheLayout::Paged behind a feature flag or add serde validation that rejects Paged on deserialization.

29. **V-035**: Remove `Default` impl for `ModelArchConfig` or make it return an error. Add `ModelArchConfig::unspecified()` for cases that need a placeholder.

30. **V-036**: Either implement stateful KV cache semantics in the emission path or change the handoff kind to a more accurate descriptor (e.g., `DirectPassThrough`).

31. **V-037, V-046**: Make opset version and deployment target configurable from the CLI or task spec rather than hardcoded.

32. **V-038**: Use actual dtype from task spec for PIR tensor specs instead of hardcoded "fp16".

33. **V-040**: Remove deprecated kv_cache_rewrite from the codebase or gate it behind a feature flag with explicit "ANE-illegal" warning in the module doc.

34. **V-041**: Resolve conv kernel range contradiction: either expand the 1–7 range or remove the dead 16-threshold code.

35. **V-042**: Implement pooling kernel_size validation using hardware limits from AneHwLimits.

36. **V-050**: Add interleave validation directly in is_dtype_ane_legal() for Int4/UInt4 instead of deferring to caller.

37. **V-051**: Align quantize validator with dtype validator — reject E5M2 as quantize output since it is ANE-illegal.

38. **V-053, V-055, V-056**: Fix documentation to match actual code behavior (confidence start values, transfer scaling, ComputePlan confidence).

39. **V-071**: Align seed JSON format with knowledge_schema.md, or update schema to match actual seed format.

40. **V-083**: Make weight.bin optional in validation. Only require it when the model declares external weights.

41. **V-087**: Wire --seed parameter through the compile pipeline, or remove it from the CLI.

### Phase 4 — Low-Priority and Cosmetic

42. **V-089**: Replace hardcoded unwrap_or defaults with explicit configuration or fail-closed behavior.

43. **V-090**: Add diagnostic when canonicalization cycle limit is hit.

44. **V-091, V-092**: Add constraint documentation for UInt16 and Bool "limited support" — specify which ops/families support them.

45. **V-093**: Expand CPU_ONLY_OPS_DETAILED to cover all 120+ CPU-only ops with reason codes.

46. **V-094**: Replace binary knowledge_consistent with a ratio or graded score.

47. **V-095**: Use model-specific salt in UUID generation to improve uniqueness.

48. **V-097**: Document FP16 epsilon truncation or compute epsilon in FP32 before casting.

---

## VI. Forensic Note

Local reference materials were used during this audit for non-invasive
compatibility vocabulary extraction only. These materials are stored in
the `forensics/` directory within the local working tree and are excluded
from the repository by `.gitignore`. No assembly, raw disassembly
snippets, long verbatim binary strings, proprietary implementation
details, raw private/local forensic artefacts, links or paths to local
forensic material, or claims of authorization from any third party
appear in this report. All findings are expressed in abstract,
source-facing terms using the author's own wording.
