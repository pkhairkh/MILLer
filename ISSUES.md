# ISSUES.md — MILLer Compatibility & Correctness Issue Tracker

> Auto-generated from `ANEVIOLATIONS.md` forensic audit and binary forensics.
> Format: Agentic AI issue specification (structured, machine-parseable, GitHub-compatible).
> Issues are classified by domain, severity, and evidence strength.
> Each issue is independently resolvable by an AI coding agent with repository access.

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique issue identifier matching ANEVIOLATIONS.md (`V-NNN`) or forensic finding (`F-NNN`) |
| `title` | Concise, GitHub-style issue title |
| `domain` | Primary concern area |
| `severity` | `critical` / `high` / `medium` / `low` |
| `class` | Violation classification from audit |
| `status` | `open` / `in-progress` / `resolved` |
| `labels` | GitHub-style labels for filtering |
| `violation_ref` | Cross-reference to ANEVIOLATIONS.md |
| `forensic_ref` | Cross-reference to forensic constraint-summary.md section |
| `affects` | What is impacted (compilation, emission, runtime, documentation) |
| `reproduce` | Steps or conditions that trigger the issue |
| `fix_hint` | Suggested approach for resolution |
| `task_ref` | Reference to TASKS.md task |

---

## Critical Issues

### V-001: LayerNorm silently misplaces on pre-A15 hardware

| Field | Value |
|-------|-------|
| `id` | V-001 |
| `title` | MILLayerNorm missing family-specific engine override — silently assigned to PE on A14 and earlier |
| `domain` | constraint-enforcement |
| `severity` | critical |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `silicon-gate`, `silent-miscompilation`, `mir`, `engine-assignment` |
| `violation_ref` | V-001 |
| `forensic_ref` | §2.3 (anec.layer_norm), §8.2 |
| `affects` | Compilation — produces models that silently fail at ANEC compile time on A14/M1 hardware |
| `reproduce` | 1) Create a MILLayerNorm op; 2) Call `default_engine_for_revision(Some(V7))` (A14); 3) Observe `Some(PE)` instead of `None` |
| `fix_hint` | Add `family.supports_layernorm()` check in `default_engine_for_revision()`, returning `None` for pre-A15. Mirror the SDPA pattern at mir.rs ~1404. |
| `task_ref` | T-P1-01 |

---

### V-002: KV-cache rewrite generates broken SIR graphs

| Field | Value |
|-------|-------|
| `id` | V-002 |
| `title` | kv_cache_rewrite generates ANE-illegal Where ops with dangling SirNodeId references |
| `domain` | pass-correctness |
| `severity` | critical |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `sir-integrity`, `dead-code-risk`, `dangling-reference` |
| `violation_ref` | V-002 |
| `forensic_ref` | N/A |
| `affects` | SIR graph integrity — if pass is accidentally invoked, produces broken graphs |
| `reproduce` | 1) Enable `deprecated-kv-cache-rewrite` feature; 2) Run the pass on a model with KV cache; 3) Observe SirNodeId references to nodes not in the graph |
| `fix_hint` | Remove the pass or add graph integrity assertions. The RingBuffer fallthrough to MaskedBlend is especially dangerous. |
| `task_ref` | T-P1-02 |

---

### V-003: Proto-direct validation always fails

| Field | Value |
|-------|-------|
| `id` | V-003 |
| `title` | bridge.py validate_proto_direct checks wrong directory (Model/ instead of Data/) |
| `domain` | emission-correctness |
| `severity` | critical |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `mlpackage-format`, `validation-bypass`, `python-bridge` |
| `violation_ref` | V-003 |
| `forensic_ref` | N/A |
| `affects` | Validation — proto-direct emitted packages always report "missing" model directory |
| `reproduce` | 1) Emit a model using proto-direct path; 2) Run `handle_validate_proto_direct()`; 3) Observe validation failure for `Model/com.apple.CoreML/` which doesn't exist |
| `fix_hint` | Change `Model/com.apple.CoreML` to `Data/com.apple.CoreML` at bridge.py ~line 730. |
| `task_ref` | T-P1-03 |

---

### V-004: Auto-materialized weights always tagged as Fp16

| Field | Value |
|-------|-------|
| `id` | V-004 |
| `title` | mir_to_compat auto-materializes resolved weights with hardcoded MilDtypeCompat::Fp16 |
| `domain` | dtype-correctness |
| `severity` | critical |
| `class` | UNVERIFIED |
| `status` | resolved |
| `labels` | `weight-dtype`, `type-mismatch`, `safetensors` |
| `violation_ref` | V-004, V-072, V-075 |
| `forensic_ref` | §2.3 (anec.cast for dtype mapping) |
| `affects` | Emission — Int32 embedding weights incorrectly tagged as Fp16 in weight.bin |
| `reproduce` | 1) Load a model with Int32 embedding weights via safetensors; 2) Emit via proto-direct; 3) Observe weight data tagged as Fp16 despite being Int32 bytes |
| `fix_hint` | Add `dtype: MilDtypeCompat` to WeightData. Derive from safetensors tensor dtype. Replace hardcoded Fp16 with actual dtype. |
| `task_ref` | T-P1-04 |

---

### V-005: Cross-type compatibility validation is a no-op

| Field | Value |
|-------|-------|
| `id` | V-005 |
| `title` | validate_cross_type_compatibility always returns Ok — claims 9 checks, delivers none |
| `domain` | validation |
| `severity` | critical |
| `class` | STUB-MIMIC |
| `status` | resolved |
| `labels` | `stub`, `dtype-validation`, `bf16-gap` |
| `violation_ref` | V-005 |
| `forensic_ref` | §6.6 (E4M3/E5M2 constraints) |
| `affects` | Validation — mixed-dtype operations that ANEC would reject are allowed through |
| `reproduce` | 1) Create a model with FP16→FP32 cross-type operation; 2) Call `validate_cross_type_compatibility()`; 3) Observe `Ok(())` with only a log warning |
| `fix_hint` | Either implement the 9 documented checks (requires BF16 in MilDtype) or remove the function and make callers fail explicitly. |
| `task_ref` | T-P1-05 |

---

## High Issues

### V-006: ConvTranspose placement skips all constraint checks

| Field | Value |
|-------|-------|
| `id` | V-006 |
| `title` | MILConvTranspose unconditionally allowed on ANE — 5 deconv constraints and A11Legacy restriction not enforced |
| `domain` | constraint-enforcement |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `silicon-gate`, `conv-transpose`, `unvalidated-constraints` |
| `violation_ref` | V-006 |
| `forensic_ref` | §6.3 (deconv constraints verified in binary), §8.3 (validation gap) |
| `affects` | Compilation — dilated deconv, SOx!=2 deconv, A11Legacy deconv pass placement and fail at ANEC |
| `reproduce` | 1) Create MILConvTranspose with dilation > 1; 2) Run placement validation; 3) Observe `AneAllowed` instead of `CpuOnly` |
| `fix_hint` | Wire `validate_deconv_constraints()` into placement_validate.rs. Add A11Legacy family check from knowledge store. |
| `task_ref` | T-P2-01 |

---

### V-007: UInt16/Bool dtype context constraints never enforced

| Field | Value |
|-------|-------|
| `id` | V-007 |
| `title` | is_dtype_ane_legal() returns Ok for UInt16/Bool but follow-up context checks are never called |
| `domain` | dtype-validation |
| `severity` | high |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `dtype-gate`, `placement`, `bypassed-validation` |
| `violation_ref` | V-007 |
| `forensic_ref` | N/A |
| `affects` | Placement — UInt16/Bool ops placed on ANE without op-context validation |
| `reproduce` | 1) Create an op with UInt16 dtype; 2) Run `is_dtype_ane_legal()` — observe `Ok(())`; 3) Placement validator allows ANE without checking context |
| `fix_hint` | Add follow-up `validate_uint16_constraints()` / `validate_bool_constraints()` calls in placement_validate.rs. |
| `task_ref` | T-P2-02 |

---

### V-008: FP32 compute allowed on A11/A12 without gate

| Field | Value |
|-------|-------|
| `id` | V-008 |
| `title` | is_fp32_compute_supported() exists but is never called — FP32 compute allowed on A11Legacy/A12 |
| `domain` | dtype-validation |
| `severity` | high |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `silicon-gate`, `fp32`, `a11-a12` |
| `violation_ref` | V-008 |
| `forensic_ref` | §6.10 (architecture-gated constraints) |
| `affects` | Compilation — FP32 compute ops on A11/A12 pass placement but fail at ANEC |
| `reproduce` | 1) Target A11Legacy; 2) Create FP32 compute op; 3) Placement allows ANE — should reject |
| `fix_hint` | Wire `is_fp32_compute_supported()` into placement pipeline after `is_dtype_ane_legal()`. |
| `task_ref` | T-P2-03 |

---

### V-009: Knowledge store contradicts code on gather legality

| Field | Value |
|-------|-------|
| `id` | V-009 |
| `title` | legality_seed.json declares gather as ane_legal: true while cpu_only_ops.rs classifies it CPU-only |
| `domain` | knowledge-store |
| `severity` | high |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `knowledge-contradiction`, `gather`, `dual-source-of-truth` |
| `violation_ref` | V-009 |
| `forensic_ref` | §6.9 (gather constraints in binary) |
| `affects` | Knowledge queries — legality rewrite and risk annotate produce overly optimistic scores for gather |
| `reproduce` | 1) Query knowledge store for gather legality — observe `ane_legal: true`; 2) Check CPU_ONLY_OPS — observe `"gather"` present |
| `fix_hint` | Set gather `ane_legal: false` in legality_seed.json. Add `empirical_note` to ane_op_family_matrix.json. |
| `task_ref` | T-P2-04 |

---

### V-010: Knowledge seed lists erf as CPU-only (code removed it)

| Field | Value |
|-------|-------|
| `id` | V-010 |
| `title` | cpu_only_ops_seed.json lists erf as CPU-only but source code correctly omits it |
| `domain` | knowledge-store |
| `severity` | high |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `knowledge-contradiction`, `erf`, `seed-staleness` |
| `violation_ref` | V-010 |
| `forensic_ref` | §2.3 (anec.erf present), §9.2 (ActivationV7 includes GetErfLut) |
| `affects` | Knowledge bootstrapping — systems seeding from JSON incorrectly classify erf as CPU-only |
| `reproduce` | 1) Load `cpu_only_ops_seed.json`; 2) Find `"mil_name": "erf"` with `"reason_code": "Miscellaneous"` |
| `fix_hint` | Remove `erf` entry from cpu_only_ops_seed.json. Binary confirms anec.erf exists across all families. |
| `task_ref` | T-P2-04 |

---

### V-011: Knowledge seed missing 70+ CPU-only ops

| Field | Value |
|-------|-------|
| `id` | V-011 |
| `title` | cpu_only_ops_seed.json has ~84 entries vs >=154 in Rust code — 70+ ops missing from seed |
| `domain` | knowledge-store |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `knowledge-gap`, `cpu-only-ops`, `incomplete-seed` |
| `violation_ref` | V-011 |
| `forensic_ref` | N/A |
| `affects` | Knowledge bootstrapping — incomplete CPU-only catalog |
| `reproduce` | 1) Count entries in cpu_only_ops_seed.json (~84); 2) Count CPU_ONLY_OPS in cpu_only_ops.rs (>=154); 3) Observe gap |
| `fix_hint` | Regenerate seed from the Rust code's CPU_ONLY_OPS set. Add all T-22, T-47, T-49 additions. |
| `task_ref` | T-P2-04 |

---

### V-012: Op family matrix contradicts code on neg/gather

| Field | Value |
|-------|-------|
| `id` | V-012 |
| `title` | ane_op_family_matrix.json lists neg/gather as ANE-supported, contradicting cpu_only_ops.rs |
| `domain` | knowledge-store |
| `severity` | high |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `knowledge-contradiction`, `neg`, `gather`, `empirical-vs-theoretical` |
| `violation_ref` | V-012 |
| `forensic_ref` | §2.3 (anec dialect ops), §8.3 |
| `affects` | Knowledge queries — theoretical ANEC converter existence vs practical ANE placement |
| `reproduce` | 1) Check matrix for `neg` — "supported" on all families; 2) Check CPU_ONLY_OPS — `"neg"` present |
| `fix_hint` | Add `practical_status: "cpu_only"` field to matrix entries. Document the empirical vs theoretical distinction. |
| `task_ref` | T-P2-04 |

---

### V-014: Zero-filled weight emission on unresolvable MILConst

| Field | Value |
|-------|-------|
| `id` | V-014 |
| `title` | mir_op_to_compat silently emits zero bytes for unresolvable MILConst without error |
| `domain` | emission-correctness |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `silent-failure`, `weight-emission`, `zero-fill` |
| `violation_ref` | V-014 |
| `forensic_ref` | N/A |
| `affects` | Emission — models with missing weights emit silently incorrect zero-filled data |
| `reproduce` | 1) Create MILConst with value_path that doesn't resolve; 2) Emit via mir_to_compat; 3) Observe zero-filled weight data without any error |
| `fix_hint` | Return `Err(BridgeError::UnresolvedWeight)` instead of zero-filling. Only `allow_missing_weights` should permit zero-fill. |
| `task_ref` | T-P2-05 |

---

### V-015: default_engine() bypasses all family checks

| Field | Value |
|-------|-------|
| `id` | V-015 |
| `title` | default_engine() returns base engine without family-specific overrides — tests only cover this path |
| `domain` | engine-assignment |
| `severity` | high |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `engine-assignment`, `test-coverage`, `api-hazard` |
| `violation_ref` | V-015 |
| `forensic_ref` | §2.3, §5.2 |
| `affects` | Engine assignment — callers using default_engine() get wrong results for family-restricted ops |
| `reproduce` | 1) Call `MILLayerNorm::default_engine()` — observe `Some(PE)` regardless of family; 2) No test covers `default_engine_for_revision(Some(rev))` |
| `fix_hint` | Deprecate `default_engine()`. Add revision-aware test coverage. |
| `task_ref` | T-P2-06 |

---

### V-017: CAPI coreml_model_info returns zeroed struct with Ok

| Field | Value |
|-------|-------|
| `id` | V-017 |
| `title` | CAPI coreml_model_info returns zeroed info struct with Ok status on macOS |
| `domain` | ffi-api |
| `severity` | high |
| `class` | STUB-MIMIC |
| `status` | resolved |
| `labels` | `ffi-stub`, `false-success`, `capi` |
| `violation_ref` | V-017 |
| `forensic_ref` | N/A |
| `affects` | FFI callers — believe model has 0 functions and no state, which is always wrong |
| `reproduce` | 1) Call `coreml_model_info()` on macOS; 2) Observe `CoreMlStatus::Ok` with zeroed struct |
| `fix_hint` | Return `CoreMlStatus::ErrorUnknown` instead of `Ok` with fabricated data. |
| `task_ref` | T-P2-07 |

---

### V-019: A12 hardware limits are unverified copies of A11

| Field | Value |
|-------|-------|
| `id` | V-019 |
| `title` | A12 (V5) hardware limits are exact copies of A11 — no verification, consumed as authoritative |
| `domain` | hardware-limits |
| `severity` | high |
| `class` | UNVERIFIED |
| `status` | resolved |
| `labels` | `unverified-limits`, `a12`, `speculative-data` |
| `violation_ref` | V-019 |
| `forensic_ref` | §5.1 (H12 HAL variant exists), §5.2 (E1 MinimumFamily exists) |
| `affects` | Compilation — A12-targeted models use A11 limits that may be incorrect |
| `reproduce` | 1) Check `AneHwLimits::a12()` — observe `..Self::a11_legacy()` with only revision changed |
| `fix_hint` | Add `verified: bool` field to AneHwLimits. Mark A12 as unverified. Gate compilation with visible warning. |
| `task_ref` | T-P2-08 |

---

### V-020: V26 future limits are fabricated

| Field | Value |
|-------|-------|
| `id` | V-020 |
| `title` | AneHwLimits::future() uses fabricated V26 limits (num_nes=16) for hardware that doesn't exist |
| `domain` | hardware-limits |
| `severity` | high |
| `class` | UNVERIFIED |
| `status` | resolved |
| `labels` | `speculative-limits`, `future-hardware`, `v26` |
| `violation_ref` | V-020 |
| `forensic_ref` | §5.2 (no E8+ family discriminant observed) |
| `affects` | Compilation — V26-targeted models use fabricated limits that may be completely wrong |
| `reproduce` | 1) Check `AneHwLimits::future()` — observe `num_nes=16` and warning comment |
| `fix_hint` | Mark as unverified. Consider requiring explicit opt-in for future hardware targets. |
| `task_ref` | T-P2-08 |

---

### V-021: Production paths silently allow missing weights

| Field | Value |
|-------|-------|
| `id` | V-021 |
| `title` | emit_mir_graph_proto_direct always passes allow_missing_weights=true even with real resolver |
| `domain` | emission-correctness |
| `severity` | high |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `weight-emission`, `silent-failure`, `production-safety` |
| `violation_ref` | V-021 |
| `forensic_ref` | N/A |
| `affects` | Production compilation — missing weights emit zero-filled data without error |
| `reproduce` | 1) Compile with a resolver that's missing some weights; 2) Observe zero-filled weights emitted without error |
| `fix_hint` | Pass `allow_missing_weights=false` when resolver is not EmptyWeightResolver. Add `is_empty()` check. |
| `task_ref` | T-P2-09 |

---

### V-022: Missing I/O descriptors silently default to Float16

| Field | Value |
|-------|-------|
| `id` | V-022 |
| `title` | mir_to_proto falls back to empty shape + Float16 for I/O names missing from descriptors |
| `domain` | emission-correctness |
| `severity` | high |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `dtype-default`, `descriptor-gap`, `proto-emission` |
| `violation_ref` | V-022 |
| `forensic_ref` | N/A |
| `affects` | Proto emission — Int32 outputs (e.g., Argmax) silently typed as Float16 |
| `reproduce` | 1) Create a graph with an output not in output_descs; 2) Emit via mir_to_proto; 3) Observe `TensorDesc { shape: [], dtype: Float16 }` |
| `fix_hint` | Return error on missing I/O descriptor instead of defaulting. |
| `task_ref` | T-P2-10 |

---

### V-023: Qwen3 defaults applied universally

| Field | Value |
|-------|-------|
| `id` | V-023 |
| `title` | Three locations default to Qwen3-specific values when architecture is unspecified |
| `domain` | model-config |
| `severity` | high |
| `class` | ABERRANT |
| `status` | in-progress |
| `labels` | `qwen3-default`, `model-architecture`, `wrong-default` |
| `violation_ref` | V-023 |
| `forensic_ref` | N/A |
| `affects` | Compilation — non-Qwen3 models get wrong vocab_size, weight patterns, and palettization |
| `reproduce` | 1) Compile a LLaMA model without specifying architecture; 2) Observe Qwen3 weight patterns applied |
| `fix_hint` | Remove Default impl for ModelArchConfig. Make architecture and max_seq_len required parameters. |
| `task_ref` | T-P2-11 |

---

## Medium Issues

### V-025: Palette version check bypassed when family is None

| Field | Value |
|-------|-------|
| `id` | V-025 |
| `title` | validate_palette_bits_for_family(None) skips A13+ check for 3-bit and 6-bit palettization |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `palettization`, `version-gate`, `option-bypass` |
| `violation_ref` | V-025 |
| `forensic_ref` | §6.7 (vector palettization constraints) |
| `affects` | Validation — 3/6-bit palette on A11/A12 passes validation but fails at ANEC |
| `task_ref` | T-P3-02 |

---

### V-026: AIR risk fields are stub values

| Field | Value |
|-------|-------|
| `id` | V-026 |
| `title` | AIR legality_confidence/fallback_risk/drift_risk always hardcoded to ideal values (1.0/0.0/0.0) |
| `domain` | risk-assessment |
| `severity` | medium |
| `class` | STUB-MIMIC |
| `status` | in-progress |
| `labels`| `stub`, `risk-metrics`, `false-certainty` |
| `violation_ref` | V-026 |
| `forensic_ref` | N/A |
| `affects` | Risk assessment — downstream code gets false certainty about ANE placement legality |
| `task_ref` | T-P3-03 |

---

### V-027: Paged KV-cache is a phantom enum variant

| Field | Value |
|-------|-------|
| `id` | V-027 |
| `title` | KvCacheLayout::Paged is defined in public API but not implemented — no handling downstream |
| `domain` | phantom-capability |
| `severity` | medium |
| `class` | PHANTOM |
| `status` | resolved |
| `labels` | `phantom`, `paged-attention`, `unimplemented-variant` |
| `violation_ref` | V-027 |
| `forensic_ref` | §2.1 (RingBufferReaderUnit/WriterUnit in binary — compiler-internal) |
| `affects` | API — callers can construct Paged variant but no code handles it |
| `task_ref` | T-P3-06 |

---

### V-028: E5M2 dtype has no validation gate

| Field | Value |
|-------|-------|
| `id` | V-028 |
| `title` | MilDtype::E5M2 exists with "NOT supported on ANE" comment but no validation prevents ANE-targeted use |
| `domain` | dtype-validation |
| `severity` | medium |
| `class` | PHANTOM |
| `status` | resolved |
| `labels` | `phantom-dtype`, `e5m2`, `ane-unsupported` |
| `violation_ref` | V-028 |
| `forensic_ref` | §6.6 ("E4M3 or E5M2 format not supported" in binary) |
| `affects` | Compilation — E5M2 can be used in ANE-targeted compilation without rejection |
| `task_ref` | T-P3-05 |

---

### V-030: IOSurface validation is warn-only

| Field | Value |
|-------|-------|
| `id` | V-030 |
| `title` | ANE will fail with 0x1d for undersized IOSurface buffers — validation only warns |
| `domain` | runtime-constraint |
| `severity` | medium |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `warn-only`, `iosurface`, `runtime-failure` |
| `violation_ref` | V-030 |
| `forensic_ref` | N/A |
| `affects` | Runtime — models pass validation but fail at ANE execution |
| `task_ref` | T-P3-01 |

---

### V-031: Surface uniformity validation is warn-only

| Field | Value |
|-------|-------|
| `id` | V-031 |
| `title` | Non-uniform buffer sizes cause 0x1d runtime errors — validation only warns |
| `domain` | runtime-constraint |
| `severity` | medium |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `warn-only`, `surface-uniformity`, `runtime-failure` |
| `violation_ref` | V-031 |
| `forensic_ref` | N/A |
| `affects` | Runtime — non-uniform buffer models pass validation but fail at ANE execution |
| `task_ref` | T-P3-01 |

---

### V-034: Int4/UInt4 interleave bypassed when None

| Field | Value |
|-------|-------|
| `id` | V-034 |
| `title` | Int4/UInt4 interleave check skipped when PlacementContext.interleave is None |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | resolved |
| `labels` | `interleave`, `option-bypass`, `int4` |
| `violation_ref` | V-034 |
| `forensic_ref` | §6.13 (interleave factor constraints in binary) |
| `affects` | Placement — Int4/UInt4 ops without interleave context pass validation |
| `task_ref` | T-P2-02 |

---

### V-037: Conv channel limit not checked with tensor dims

| Field | Value |
|-------|-------|
| `id` | V-037 |
| `title` | validate_tensor_dims() checks 65536 channel limit but convs need 32768 — separate method easy to miss |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `conv-channels`, `api-design`, `validation-gap` |
| `violation_ref` | V-037 |
| `forensic_ref` | N/A |
| `affects` | Validation — convs with 32K-64K channels pass general validation but fail at ANEC |
| `task_ref` | T-P3-04 |

---

### F-CROSS-01: Cross-constraint combinations not validated

| Field | Value |
|-------|-------|
| `id` | F-CROSS-01 |
| `title` | Binary enforces constraint combinations that MILLer doesn't check (dilation+vector_palettize, aliasing+vector_palettize, shuffle+per-channel_palettize, palettize+large_kernel_stride) |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `cross-constraint`, `binary-verified`, `validation-gap` |
| `violation_ref` | Forensic §6.7, §6.11 |
| `forensic_ref` | §6.7 (vector palettization combos), §6.11 (dilation combos) |
| `affects` | Compilation — ops with invalid constraint combinations pass placement but fail at ANEC |
| `task_ref` | T-P3-09 |

---

### F-ARCH-01: Architecture-gated constraints not per-family validated

| Field | Value |
|-------|-------|
| `id` | F-ARCH-01 |
| `title` | Binary contains per-family rejection strings (Softmax on old HW, LRN, depth-axis broadcast, A14 resize) that MILLer doesn't enforce |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `architecture-gated`, `binary-verified`, `per-family` |
| `violation_ref` | Forensic §6.10 |
| `forensic_ref` | §6.10 (architecture-gated constraint strings) |
| `affects` | Compilation — ops that are architecture-restricted pass placement on wrong family |
| `task_ref` | T-P3-10 |

---

### F-HAL-01: 9 HAL sub-variants and 7 non-Hxx targets not modeled

| Field | Value |
|-------|-------|
| `id` | F-HAL-01 |
| `title` | Binary has 24 HAL variants (H11–H18 with sub-variants, M9/M11/M12, T0/T1, U1/U2) but MILLer only models 8 base families |
| `domain` | hardware-modeling |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `hal-variants`, `hardware-modeling`, `sub-variant` |
| `violation_ref` | Forensic §5.1, §8.1 |
| `forensic_ref` | §5.1 (HAL variant table), §8.1 (AneFamily coverage) |
| `affects` | Compilation — sub-variant-specific constraints may be missed; Mac chip targets not modeled |
| `task_ref` | T-P4-07 |

---

### F-OPS-01: 12 ANEC operations have no MILLer MirOp mapping

| Field | Value |
|-------|-------|
| `id` | F-OPS-01 |
| `title` | Binary exposes 12 genuinely unmapped ANEC operations (broadcast, scaled_elementwise, global_arg_min_max, etc.) |
| `domain` | operation-coverage |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `op-coverage`, `unmapped-ops`, `anec-dialect` |
| `violation_ref` | Forensic §2.3 |
| `forensic_ref` | §2.3 (ANEC dialect operation table) |
| `affects` | Coverage — models using these operations cannot be compiled through MILLer |
| `task_ref` | T-P4-08 |

---

### F-FW-01: Multi-ANE/firmware capabilities not modeled

| Field | Value |
|-------|-------|
| `id` | F-FW-01 |
| `title` | Binary reveals multi-ANE device enumeration, 4 firmware images, subType matching, program chaining — none modeled by MILLer |
| `domain` | hardware-modeling |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `multi-ane`, `firmware`, `chaining`, `not-modeled` |
| `violation_ref` | Forensic §4.5, §7 |
| `forensic_ref` | §4.5 (firmware and multi-ANE), §7 (firmware paths, chaining API) |
| `affects` | Device management — multi-ANE systems and firmware loading not represented |
| `task_ref` | None (out of current scope, document for future) |

---

## Low Issues

### V-029: StaticizePass dead code persists

| Field | Value |
|-------|-------|
| `id` | V-029 |
| `title` | StaticizePass exists as deprecated no-op — could be accidentally reintroduced |
| `domain` | dead-code |
| `severity` | low |
| `class` | PHANTOM |
| `status` | resolved |
| `labels` | `dead-code`, `deprecated-pass` |
| `violation_ref` | V-029 |
| `task_ref` | T-P4-03 |

---

### V-046: ir_reference.md lists removed StaticizePass

| Field | Value |
|-------|-------|
| `id` | V-046 |
| `title` | Documentation lists StaticizePass as pipeline step but it was removed as phantom no-op |
| `domain` | documentation |
| `severity` | low |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `stale-docs`, `ir-reference` |
| `violation_ref` | V-046 |
| `task_ref` | T-P4-06 |

---

### V-047: bridge_protocol.md claims no multifunction emission

| Field | Value |
|-------|-------|
| `id` | V-047 |
| `title` | Documentation says "No multifunction emission yet" but code has emit_multifunction support |
| `domain` | documentation |
| `severity` | low |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `stale-docs`, `bridge-protocol`, `multifunction` |
| `violation_ref` | V-047 |
| `task_ref` | T-P4-06 |

---

### V-048: architecture.md claims no stubs remain

| Field | Value |
|-------|-------|
| `id` | V-048 |
| `title` | Documentation claims "No todo!() stubs remain" but CAPI layer has multiple stub implementations |
| `domain` | documentation |
| `severity` | low |
| `class` | ABERRANT |
| `status` | resolved |
| `labels` | `stale-docs`, `architecture`, `ffi-stubs` |
| `violation_ref` | V-048 |
| `task_ref` | T-P4-06 |

---

### V-051: ne_transpose_c_max never validated

| Field | Value |
|-------|-------|
| `id` | V-051 |
| `title` | AneHwLimits has ne_transpose_c_max field but no validation method exists |
| `domain` | constraint-enforcement |
| `severity` | low |
| `class` | LACUNA |
| `status` | open |
| `labels` | `transpose`, `unvalidated-limit` |
| `violation_ref` | V-051 |
| `task_ref` | T-P3-08 |

---

### V-055: LARGE_KERNEL_THRESHOLD hardcoded

| Field | Value |
|-------|-------|
| `id` | V-055 |
| `title` | LARGE_KERNEL_THRESHOLD=16 hardcoded instead of loaded from knowledge store |
| `domain` | hardcoded-constant |
| `severity` | low |
| `class` | UNVERIFIED |
| `status` | open |
| `labels` | `hardcoded`, `knowledge-store-gap` |
| `violation_ref` | V-055 |
| `task_ref` | T-P4-01 |

---

### V-056: MAX_POOLING_KERNEL_DIM hardcoded

| Field | Value |
|-------|-------|
| `id` | V-056 |
| `title` | MAX_POOLING_KERNEL_DIM=27 hardcoded instead of loaded from knowledge store |
| `domain` | hardcoded-constant |
| `severity` | low |
| `class` | UNVERIFIED |
| `status` | open |
| `labels` | `hardcoded`, `knowledge-store-gap` |
| `violation_ref` | V-056 |
| `task_ref` | T-P4-01 |

---

## Issue Statistics

| Domain | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| constraint-enforcement | 1 | 2 | 3 | 1 | 7 |
| dtype-validation | 1 | 2 | 1 | 0 | 4 |
| emission-correctness | 2 | 2 | 0 | 0 | 4 |
| knowledge-store | 0 | 4 | 0 | 0 | 4 |
| hardware-limits | 0 | 2 | 0 | 0 | 2 |
| pass-correctness | 1 | 0 | 0 | 0 | 1 |
| validation | 1 | 0 | 0 | 0 | 1 |
| engine-assignment | 0 | 1 | 0 | 0 | 1 |
| ffi-api | 0 | 1 | 0 | 0 | 1 |
| model-config | 0 | 1 | 0 | 0 | 1 |
| risk-assessment | 0 | 0 | 1 | 0 | 1 |
| phantom-capability | 0 | 0 | 1 | 0 | 1 |
| runtime-constraint | 0 | 0 | 2 | 0 | 2 |
| hardware-modeling | 0 | 0 | 2 | 0 | 2 |
| operation-coverage | 0 | 0 | 1 | 0 | 1 |
| dead-code | 0 | 0 | 0 | 1 | 1 |
| documentation | 0 | 0 | 0 | 3 | 3 |
| hardcoded-constant | 0 | 0 | 0 | 2 | 2 |
| **Total** | **5** | **15** | **14** | **7** | **41** |

---

## Forensic Confidence Summary

| Evidence Quality | Issue Count | Notes |
|-----------------|-------------|-------|
| Strong (symbol names, exported API) | 8 | HAL variants, ANEC ops, CAPI stubs, converter evidence |
| Medium (error strings, constraint strings) | 15 | Constraint vocabulary, architecture-gated strings |
| Weak (naming inference, absence claims) | 12 | Hardware target mapping, absence proofs |
| Unknown (sub-variant semantics) | 6 | HAL sub-variant differences |

---

## Additional Issues — MLIR-Method Violations (M-prefix)

> From MLIRVIOLATIONS.md: Applying MLIR compiler-design discipline as a conceptual lens.

### M-001: Zero-Filled "Production" Models via allow_missing_weights=true
- **Severity:** CRITICAL | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/bridge/src/proto_direct.rs:164–189`
- **Status:** RESOLVED | **Remediation:** T-P2-09
- **Description:** Production path passes `allow_missing_weights=true`, silently producing zero-filled weights.

### M-002: MatMul Inner Dimension Mismatch Treated as Warning
- **Severity:** CRITICAL | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:92–98`
- **Status:** RESOLVED | **Remediation:** T-P5-02
- **Description:** Mismatched inner dims produce garbage output instead of hard error.

### M-003: Shape Inference Returns Empty Vec for Unknown Operations
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:401`; `crates/bridge/src/shape_inference.rs:528–531`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** `infer_shape()` returns `Ok(vec![])` as fallback; unknown shapes silently propagate.

### M-004: StaticizePass Is a Pure Pass-Through (Stub Mimic)
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/passes/src/staticize.rs:52–63`
- **Status:** RESOLVED | **Remediation:** T-P4-03
- **Description:** Returns `Ok(input)` unchanged with 1500+ line test suite.

### M-005: slanc_scales Pass Inserts Const+Mul with Uncomputed Scale Values
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/passes/src/slanc_scales.rs:63–123`
- **Status:** OPEN | **Remediation:** T-P3-09
- **Description:** Inserts Mul ops with uninitialized scale factors, silently corrupting the graph.

### M-006: shard_plan.rs Falls Back to Dimension=1 with Only Warning
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/shard_plan.rs:239–247`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** Falls back to [1,1,1,1] when shape unavailable, producing wrong PIR specs.

### M-007: dtype_constraints.rs Returns Ok(()) for Constrained Dtypes
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/dtype_constraints.rs:127–161`
- **Status:** RESOLVED | **Remediation:** T-P2-02
- **Description:** Int4/UInt4/UInt16/Bool gates return Ok with deferred "caller must check" comments.

### M-008: validate_cross_type_compatibility Effectively No-Op
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/dtype_constraints.rs:440–470`
- **Status:** RESOLVED | **Remediation:** T-P1-05
- **Description:** Documents ANEC constraints but never rejects anything.

### M-009: MirOpCompat::Unsupported Invisible to Weight Materialization
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/coreml-proto/src/lib.rs:1043–1051, 1337`
- **Status:** OPEN | **Remediation:** T-P5-10
- **Description:** Unsupported ops return `input_names(): vec![]`, invisible to weight materialization.

### M-010: kv_cache_rewrite Generates ANE-Illegal Ops
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/passes/src/kv_cache_rewrite.rs:1–313`
- **Status:** RESOLVED | **Remediation:** T-P1-02
- **Description:** Generates SirOp::Where (ANE-illegal); deprecated but not removed.

### M-011: AIR legality_confidence Field Not Enforced
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/ir/src/air.rs:890`; SPEC §5.2
- **Status:** OPEN | **Remediation:** T-P5-03, T-P3-03
- **Description:** legality_confidence defaults to 0.0 with no rejection gate; SPEC claims "hard invariant."

### M-012: SPEC Claims SQLite Knowledge Store; Implementation Is JSON
- **Severity:** HIGH | **Class:** DOC-CODE-DRIFT | **Confidence:** HIGH
- **Location:** `SPEC.md:304, 516–520`; `crates/knowledge/src/store.rs:8–10`
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes SQLite; implementation uses JSON with linear-scan queries.

### M-013: Knowledge Seed Files Not Loaded by Any Runtime Crate
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `knowledge/ane_op_family_matrix.json`; `knowledge/palettization_constraints_seed.json`
- **Status:** OPEN | **Remediation:** T-P2-04
- **Description:** Seed files validated in tests but never loaded at runtime; data hardcoded in Rust.

### M-014: MILLinear/MILConv Shape Inference Propagates Input Shape
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:139, 376`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** Propagates input shape instead of computing correct output shape.

### M-015–M-044: See MLIRVIOLATIONS.md for full MEDIUM/LOW issues
- Includes: ANE constraints leak into lowering (M-015), name-based dtype heuristics (M-016), LegalityRewrite entirely ANE-specific (M-017), name-based shape heuristics (M-018–M-041), Python bridge unverifiable (M-020, M-032), doc-code drift (M-029–M-044).

---

## Additional Issues — Binary-Research Forensic Findings (N-prefix)

> From cross-referencing MILLer source code against ANEC binary research in ane-constraints-docs/.

### N-001: 1-bit and 2-bit Palettization Allowed but ANEC Rejects Them
- **Severity:** CRITICAL | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_layout.rs:164`
- **Status:** RESOLVED | **Remediation:** T-P5-01
- **Description:** VALID_PALETTE_BITS includes 1 and 2; ANEC rejects "1 and 2 bit palettes not supported."

### N-002: 35+ ANEC hal_params Not Modeled — 70% of Hardware Constraints Missing
- **Severity:** CRITICAL | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_hw_limits.rs:10–30`
- **Status:** OPEN | **Remediation:** T-P6-01
- **Description:** Only 15 of 50+ hal_params modeled. Missing: kernel depth, padding limits, PE/NE limits, alignment.

### N-003: M1 Hardware Limits Inherit from A17 Despite A14 Family
- **Severity:** HIGH | **Class:** ABERRANT | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_hw_limits.rs:133–135`
- **Status:** OPEN | **Remediation:** T-P6-02
- **Description:** m1() uses ..Self::a17() but M1 is A14-family; may allow oversized tensors.

### N-004: ValidateLayer Equivalent Entirely Absent (Layer 1 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs` (entire file)
- **Status:** OPEN | **Remediation:** T-P6-03
- **Description:** 40+ ANEC ValidateLayer instantiations have no MILLer equivalent.

### N-005: MLIR Placement Dialect Entirely Absent (Layer 2 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no placement dialect module
- **Status:** OPEN | **Remediation:** Long-term infrastructure
- **Description:** No region-based placement, no force-ane-placement, no boundary ops.

### N-006: Fusability Checks Entirely Absent (Layer 4 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no fusability module
- **Status:** OPEN | **Remediation:** T-P6-05
- **Description:** No IsFusable checks; ops may pass placement but fail to fuse into engine layers.

### N-007: Memory Pressure / L2 Legalization Absent (Layer 5 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no L2/memory module
- **Status:** OPEN | **Remediation:** T-P6-06
- **Description:** No L2 budget modeling; individually legal ops may collectively exceed limits.

### N-008: Missing AneRevision::Vu1 for uANE
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** MEDIUM
- **Location:** `crates/ir/src/ane_target.rs:200–213`
- **Status:** OPEN | **Remediation:** T-P6-04
- **Description:** uANE variant not modeled; potential constraint differences unknown.

### N-009: MILTile Assigned PE Engine but Tile Has No ANEC Converter
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/ir/src/mir.rs:1181`
- **Status:** OPEN | **Remediation:** T-P5-07
- **Description:** Tile assigned PE engine but not in CPU_ONLY_OPS; legality_rewrite decomposes it.

### N-010: ne_palette_lut_size_in_bytes Not Modeled
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_layout.rs` (entire file)
- **Status:** OPEN | **Remediation:** T-P6-01
- **Description:** No LUT size overflow detection; large palettes may exceed hardware limit.

### N-011: Deconv Validator Exists but Not Wired; Missing 12+ Binary Constraints
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs:589`; `op_constraints.rs:340–396`
- **Status:** OPEN | **Remediation:** T-P2-01
- **Description:** validate_deconv_constraints() checks 5 constraints; binary documents 12+ additional.

### N-012: Conv Kernel Constraints Not Wired into Placement
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs` — no MILConv match arm
- **Status:** OPEN | **Remediation:** T-P2-01
- **Description:** validate_conv_constraints() exists but placement never calls it.

### N-013: Pooling Constraints Not Wired into Placement
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs` — no pool match arm
- **Status:** OPEN | **Remediation:** T-P2-01
- **Description:** validate_pooling_constraints() exists but placement never calls it.

### N-014: Missing AneFamily Capability for A13 Pool-to-Conv Restriction
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_target.rs:73–191`
- **Status:** OPEN | **Remediation:** T-P6-04
- **Description:** "Pool-to-Conv not supported for A13 and below" has no capability method.

### N-015–N-038: See ISSUES.md full version for MEDIUM/LOW binary-research findings
- Includes: packed10 rejection (N-015), ChannelLast DRAM constraints (N-016), broadcast shape constraints (N-017), normalization format constraints (N-018–N-019), SDPA mask gaps (N-020), NMS OS-version dependency (N-021), PixelShuffle validation (N-022), Reshape dynamic-shape rejection (N-023), Gather wrong constraints (N-024), Sort/TopK format (N-025), conv kernel depth (N-026), InstanceNorm architecture gate (N-027), palettization+stride (N-028), palettization+compression (N-029), InputView constraints (N-030), dynamic shape kill switch (N-031), 2xInt8 (N-032), non-constant axes (N-033), mutable weights (N-034), stencil gaps (N-035), conv padding limits (N-036), ArgMinMax padding (N-037), Softmax architecture gate (N-038).
