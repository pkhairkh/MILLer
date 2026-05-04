# TASKS.md — MILLer Remediation Task Board

> Auto-generated from `ANEVIOLATIONS.md` forensic audit.
> Format: Agentic AI task specification (structured, machine-parseable, human-readable).
> Each task is independently executable by an AI coding agent with access to this repository.

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique task identifier (`T-<phase><seq>`) |
| `title` | Imperative, agent-actionable summary |
| `phase` | Execution phase (P1=immediate, P2=high-priority, P3=medium, P4=low) |
| `severity` | Worst violation severity if task is skipped |
| `depends_on` | Tasks that must complete first |
| `files` | Primary files to modify |
| `violation_refs` | Cross-references to ANEVIOLATIONS.md entries |
| `acceptance_criteria` | Verifiable conditions for task completion |
| `agent_hints` | Specific guidance for AI agent execution |

---

## Phase 1 — Critical Fixes (Silent Miscompilation)

### T-P1-01: Add LayerNorm family-specific engine override

| Field | Value |
|-------|-------|
| `id` | T-P1-01 |
| `title` | Add MILLayerNorm family override in default_engine_for_revision() |
| `phase` | P1 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `crates/ir/src/mir.rs` |
| `violation_refs` | [V-001] |
| `acceptance_criteria` | 1) `default_engine_for_revision(Some(rev))` returns `None` for LayerNorm when `AneFamily::from_revision(rev).supports_layernorm() == false`; 2) Existing tests pass; 3) New test covers pre-A15 LayerNorm returning None; 4) `default_engine()` deprecated or documented as unsafe |
| `agent_hints` | Mirror the SDPA pattern at mir.rs ~line 1404–1408. Insert a check for `family.supports_layernorm()` before the final `Some(base)` return. When `supports_layernorm()` is false, return `None`. Also deprecate `default_engine()` or add `#[deprecated]` note. |

---

### T-P1-02: Gate or remove kv_cache_rewrite pass

| Field | Value |
|-------|-------|
| `id` | T-P1-02 |
| `title` | Remove or hard-guard the kv_cache_rewrite pass to prevent ANE-illegal Where emission |
| `phase` | P1 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `crates/passes/src/kv_cache_rewrite.rs`, `crates/passes/src/lib.rs` |
| `violation_refs` | [V-002] |
| `acceptance_criteria` | 1) The pass is either removed from the crate or guarded by `#[cfg(feature = "kv-cache-rewrite")]` with a compile-time warning; 2) No code path can generate SirOp::Where with dangling SirNodeId references; 3) `cargo test` passes |
| `agent_hints` | The pass is already feature-gated (`deprecated-kv-cache-rewrite`) and documented as dead code. Remove the `RingBuffer` fallthrough to `MaskedBlend` or remove the entire module. If keeping, add an assertion that generated SirNodeIds are present in the graph. |

---

### T-P1-03: Fix bridge.py proto-direct validation path

| Field | Value |
|-------|-------|
| `id` | T-P1-03 |
| `title` | Correct the mlpackage directory path in handle_validate_proto_direct() |
| `phase` | P1 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `python/bridge.py` |
| `violation_refs` | [V-003] |
| `acceptance_criteria` | 1) `handle_validate_proto_direct()` checks `Data/com.apple.CoreML/model.mlmodel` instead of `Model/com.apple.CoreML/model.mlmodel`; 2) Proto-direct emitted packages pass validation; 3) Python tests pass |
| `agent_hints` | One-line fix: change `Model` to `Data` at line ~730. Verify against `crates/coreml-emit/src/package.rs` line 83 which writes to `Data/`. Apple's mlpackage format uses `Data/` for model descriptors. |

---

### T-P1-04: Derive auto-materialized weight dtype from resolved WeightData

| Field | Value |
|-------|-------|
| `id` | T-P1-04 |
| `title` | Replace hardcoded MilDtypeCompat::Fp16 with actual dtype in auto-materialized Const ops |
| `phase` | P1 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `crates/bridge/src/mir_to_compat.rs`, `crates/bridge/src/safetensors_resolver.rs` |
| `violation_refs` | [V-004, V-072, V-075] |
| `acceptance_criteria` | 1) Auto-materialized Const ops use the actual dtype from WeightData instead of hardcoded Fp16; 2) Int32 embedding weights are tagged as Int32, not Fp16; 3) WeightData struct carries dtype information; 4) `cargo test` passes; 5) New test verifies non-FP16 weight dtype round-trip |
| `agent_hints` | Add a `dtype: MilDtypeCompat` field to `WeightData` in `safetensors_resolver.rs`. Populate it based on the safetensors tensor dtype. In `mir_to_compat.rs` line ~253, use `wd.dtype` instead of `MilDtypeCompat::Fp16`. Handle the Fp16/BF16/Fp32 conversion case separately. |

---

### T-P1-05: Implement or remove validate_cross_type_compatibility

| Field | Value |
|-------|-------|
| `id` | T-P1-05 |
| `title` | Implement cross-type compatibility checks or remove the stub function |
| `phase` | P1 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `crates/passes/src/dtype_constraints.rs` |
| `violation_refs` | [V-005] |
| `acceptance_criteria` | 1) Either the function implements the 9 documented cross-type rejection checks (requires BF16 in MilDtype) OR the function is removed and all callers fail explicitly; 2) No code path claims to validate cross-type compatibility without actually doing so; 3) `cargo test` passes |
| `agent_hints` | The doc comment lists 9 ANEC constraint strings. Two options: (A) Add `MilDtype::Bf16` variant and implement all 9 checks, or (B) Remove the function and replace its call sites with `Err(CrossTypeNotSupported)` for any mixed-dtype scenario. Option B is simpler and more honest. |

---

## Phase 2 — High-Priority Validation Gaps

### T-P2-01: Wire deconv constraint validation into placement validator

| Field | Value |
|-------|-------|
| `id` | T-P2-01 |
| `title` | Call validate_deconv_constraints() from placement_validate for MILConvTranspose |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/passes/src/placement_validate.rs`, `crates/passes/src/op_constraints.rs` |
| `violation_refs` | [V-006] |
| `acceptance_criteria` | 1) MILConvTranspose match arm calls `validate_deconv_constraints()`; 2) A11Legacy family restriction is checked; 3) Tests for each of the 5 deconv constraints pass; 4) Dilated deconv returns `CpuOnly` instead of `AneAllowed` |
| `agent_hints` | Replace `MirOp::MILConvTranspose { .. } => PlacementDecision::AneAllowed` at line ~589 with a validation block. Import `validate_deconv_constraints` from `op_constraints`. Also check `ane_op_family_matrix` for A11Legacy exclusion. |

---

### T-P2-02: Enforce UInt16/Bool context constraints in placement

| Field | Value |
|-------|-------|
| `id` | T-P2-02 |
| `title` | Add follow-up UInt16/Bool constraint checks after is_dtype_ane_legal() |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/passes/src/placement_validate.rs`, `crates/passes/src/dtype_constraints.rs` |
| `violation_refs` | [V-007] |
| `acceptance_criteria` | 1) After `is_dtype_ane_legal()` returns `Ok(())` for UInt16 or Bool, the placement validator calls `validate_uint16_constraints()` / `validate_bool_constraints()`; 2) UInt16/Bool ops without valid context are rejected from ANE placement |
| `agent_hints` | In `placement_validate.rs`, after the dtype check at line ~267, add a match on the dtype. For UInt16/Bool, call the corresponding validation functions from `dtype_constraints.rs`. If they fail, return `CpuOnly`. |

---

### T-P2-03: Wire FP32 compute gating into placement pipeline

| Field | Value |
|-------|-------|
| `id` | T-P2-03 |
| `title` | Call is_fp32_compute_supported() when FP32 dtype is used for compute operations |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/passes/src/placement_validate.rs` |
| `violation_refs` | [V-008] |
| `acceptance_criteria` | 1) When dtype is FP32 and the op is a compute op (not just I/O), `is_fp32_compute_supported(family)` is checked; 2) FP32 compute on A11Legacy/A12 returns `CpuOnly`; 3) Test verifies FP32 rejection on A11/A12 |
| `agent_hints` | After `is_dtype_ane_legal()` returns `Ok(())` for FP32, check if the op is a compute op and if so call `is_fp32_compute_supported(family)`. Requires the PlacementContext to carry the AneFamily. |

---

### T-P2-04: Fix knowledge store contradictions

| Field | Value |
|-------|-------|
| `id` | T-P2-04 |
| `title` | Synchronize knowledge store JSON with source code ground truth |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `knowledge/legality_seed.json`, `knowledge/cpu_only_ops_seed.json`, `knowledge/ane_op_family_matrix.json` |
| `violation_refs` | [V-009, V-010, V-011, V-012, V-013] |
| `acceptance_criteria` | 1) `legality_seed.json`: gather `ane_legal` set to `false`; 2) `cpu_only_ops_seed.json`: `erf` removed, 70+ missing ops added from `cpu_only_ops.rs`; 3) `ane_op_family_matrix.json`: `neg`, `gather`, `select`, `where` entries gain `practical_status: "cpu_only"` field with explanation; 4) Seed validation tests pass |
| `agent_hints` | Run `cpu_only_ops.rs` test to get the full CPU_ONLY_OPS set. Use that as the ground truth. For the matrix, add an `empirical_note` field rather than changing `supported` to avoid losing theoretical converter info. |

---

### T-P2-05: Error on unresolvable MILConst instead of zero-filling

| Field | Value |
|-------|-------|
| `id` | T-P2-05 |
| `title` | Return error when WeightResolver returns None for MILConst |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [T-P1-04] |
| `files` | `crates/bridge/src/mir_to_compat.rs` |
| `violation_refs` | [V-014] |
| `acceptance_criteria` | 1) When WeightResolver returns `None` for a MILConst value_path, the function returns `Err(...)` instead of creating zero-filled data; 2) `allow_missing_weights` gate is the only path that permits zero-fill; 3) New test verifies error on missing weight |
| `agent_hints` | Change the `None` branch at line ~939 to return `Err(BridgeError::UnresolvedWeight { path })` unless `allow_missing_weights` is true. The `allow_missing_weights` path should still exist but must be explicitly opted into. |

---

### T-P2-06: Deprecate default_engine() and add revision-aware tests

| Field | Value |
|-------|-------|
| `id` | T-P2-06 |
| `title` | Deprecate default_engine(), add tests for default_engine_for_revision() |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [T-P1-01] |
| `files` | `crates/ir/src/mir.rs`, `crates/ir/src/mir_engine_test.rs` |
| `violation_refs` | [V-015] |
| `acceptance_criteria` | 1) `default_engine()` is marked `#[deprecated]` with explanation; 2) `mir_engine_test.rs` adds tests for `default_engine_for_revision(Some(rev))` for each AneRevision; 3) All family-specific overrides are tested |
| `agent_hints` | Add `#[deprecated(since = "0.x", note = "Use default_engine_for_revision(Some(rev)) instead. This method returns incorrect results for family-restricted ops.")]` to `default_engine()`. Add a test matrix: for each (MirOp, AneRevision) pair, verify the engine is correct. |

---

### T-P2-07: Fix CAPI stubs to return errors instead of Ok with wrong data

| Field | Value |
|-------|-------|
| `id` | T-P2-07 |
| `title` | Replace CAPI stub Ok responses with honest error returns |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/coreml-ffi/src/capi.rs` |
| `violation_refs` | [V-017, V-018, V-066, V-067] |
| `acceptance_criteria` | 1) `coreml_model_info` returns `ErrorUnknown` instead of zeroed struct with `Ok`; 2) `coreml_version` returns `ErrorUnknown` instead of "unknown"; 3) `coreml_model_compile` and `coreml_model_predict` return proper error codes; 4) No CAPI function returns `Ok` with fabricated data |
| `agent_hints` | Change the return values in capi.rs for the macOS paths. `coreml_model_info`: return `CoreMlStatus::ErrorUnknown` with zeroed struct. `coreml_version`: return `CoreMlStatus::ErrorUnknown` with empty string. Better to fail honestly than to lie. |

---

### T-P2-08: Mark unverified hardware limits

| Field | Value |
|-------|-------|
| `id` | T-P2-08 |
| `title` | Add verified flag to AneHwLimits and gate compilation with unverified limits |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [V-019, V-020] |
| `acceptance_criteria` | 1) `AneHwLimits` has a `verified: bool` field; 2) `for_revision()` returns `Result<AneHwLimits, UnverifiedLimitsWarning>` or logs a structured warning; 3) A12 and V26 limits are marked `verified: false`; 4) CLI emits a visible warning when compiling with unverified limits |
| `agent_hints` | Add `pub verified: bool` to the struct. Set `true` for all revisions with confirmed limits (A11, A13–A18). Set `false` for A12 and V26. In `for_revision()`, if `!verified`, emit a `log::warn!` with the revision name. Consider adding `require_verified_limits: bool` to the compilation config. |

---

### T-P2-09: Fix allow_missing_weights for production paths

| Field | Value |
|-------|-------|
| `id` | T-P2-09 |
| `title` | Pass allow_missing_weights=false when real WeightResolver is provided |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/bridge/src/proto_direct.rs` |
| `violation_refs` | [V-021] |
| `acceptance_criteria` | 1) `emit_mir_graph_proto_direct_with_resolver` passes `allow_missing_weights=false` when resolver is not `EmptyWeightResolver`; 2) `resolver.is_empty()` check is added before calling; 3) Production compilation with missing weights returns an error |
| `agent_hints` | At line ~188, change `true` to `resolver.is_empty()` (or `false` if resolver is real). Add the missing `is_empty()` method to the resolver trait if it doesn't exist. |

---

### T-P2-10: Fix mir_to_proto I/O descriptor fallback

| Field | Value |
|-------|-------|
| `id` | T-P2-10 |
| `title` | Error on missing I/O descriptors instead of defaulting to empty shape + Float16 |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/coreml-emit/src/mir_to_proto.rs` |
| `violation_refs` | [V-022] |
| `acceptance_criteria` | 1) Missing input/output names from descriptors return `Err(...)` instead of creating fallback `TensorDesc`; 2) No silent default to empty shape + Float16; 3) Test verifies error on missing descriptor |
| `agent_hints` | Change the fallback at lines ~411–416 and ~428–433 from `TensorDesc { shape: vec![], dtype: Float16 }` to `Err(EmissionError::MissingIODescriptor { name })`. The caller should ensure all I/O names are present before emission. |

---

### T-P2-11: Remove Qwen3-specific defaults

| Field | Value |
|-------|-------|
| `id` | T-P2-11 |
| `title` | Make architecture and max_seq_len required parameters, remove Qwen3 defaults |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/ir/src/common.rs`, `crates/passes/src/palettize_weights.rs`, `crates/bridge/src/mir_to_compat.rs`, `crates/bridge/src/shape_inference.rs` |
| `violation_refs` | [V-023, V-024, V-040, V-041] |
| `acceptance_criteria` | 1) `ModelArchConfig` has no `Default` impl (or returns error); 2) `palettize_weights` requires `ModelArchitecture` parameter; 3) `build_input_alias_map` requires `ModelArchitecture`; 4) `max_seq_len` has no default — callers must provide it; 5) CLI updated to require these flags |
| `agent_hints` | Remove `impl Default for ModelArchConfig`. Change `Option<ModelArchitecture>` to `ModelArchitecture` in function signatures. For `max_seq_len`, change `Option<usize>` to `usize` in the three functions. Update `crates/cli/src/main.rs` to require `--architecture` and `--max-seq-len` flags. |

---

### T-P2-12: Implement ALLOWED_DIVERGENCES test

| Field | Value |
|-------|-------|
| `id` | T-P2-12 |
| `title` | Replace no-op ALLOWED_DIVERGENCES test with real assertions |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/passes/src/cpu_only_ops.rs` |
| `violation_refs` | [V-016] |
| `acceptance_criteria` | 1) Test asserts that each op in ALLOWED_DIVERGENCES has `default_engine() != None` AND is in CPU_ONLY_OPS; 2) Test documents why each divergence exists; 3) Test fails if the dual-source-of-truth becomes inconsistent |
| `agent_hints` | Replace `let _ = name;` with actual assertions. For each op, verify: (1) it appears in CPU_ONLY_OPS, (2) `MirOp::from_name(name).default_engine().is_some()`, (3) add a comment explaining the divergence reason. |

---

## Phase 3 — Medium-Priority Constraint Enforcement

### T-P3-01: Promote ANE constraint violations from warnings to errors

| Field | Value |
|-------|-------|
| `id` | T-P3-01 |
| `title` | Change IOSurface size, surface uniformity, and flat buffer layout validations from warn to error |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/coreml-emit/src/mir_to_proto.rs` |
| `violation_refs` | [V-030, V-031, V-032] |
| `acceptance_criteria` | 1) `validate_iosurface_sizes` returns `Err(...)` for undersized buffers; 2) `validate_surface_uniformity` returns `Err(...)` for non-uniform sizes; 3) `validate_flat_buffer_layout` returns `Err(...)` for non-[1,C,1,S] shapes; 4) Add `allow_invalid_surface` escape hatch flag if needed for testing |
| `agent_hints` | Replace `log::warn!(...) + Ok(())` with `Err(EmissionError::...)` in all three validation functions. ANE will reject these models at runtime (0x1d error), so failing early is correct. Add a `ValidationPolicy { strict: bool }` config if a soft mode is needed. |

---

### T-P3-02: Make validate_palette_bits_for_family require family

| Field | Value |
|-------|-------|
| `id` | T-P3-02 |
| `title` | Change validate_palette_bits_for_family signature to require AneFamily |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_layout.rs` |
| `violation_refs` | [V-025] |
| `acceptance_criteria` | 1) `validate_palette_bits_for_family(family: AneFamily, ...)` — no Option; 2) All callers provide a concrete AneFamily; 3) 3-bit/6-bit palette on A11Legacy/A12/A13 is rejected; 4) Tests updated |
| `agent_hints` | Change `family: Option<AneFamily>` to `family: AneFamily`. Find all callers and ensure they pass a real family. For contexts where family is truly unavailable, the caller should fail rather than silently accepting. |

---

### T-P3-03: Replace AIR risk fields with status enum

| Field | Value |
|-------|-------|
| `id` | T-P3-03 |
| `title` | Replace legality_confidence/fallback_risk/drift_risk with LegalityStatus enum |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/air.rs`, `crates/ir/src/serialize.rs`, `crates/passes/src/risk_annotate.rs` |
| `violation_refs` | [V-026, V-033] |
| `acceptance_criteria` | 1) `AirNode` has `legality_status: LegalityStatus` enum instead of three f32 fields; 2) `LegalityStatus` variants: `Verified`, `Unverified`, `LikelyFallback`, `Unknown`; 3) `risk_annotate` populates from knowledge query; 4) No downstream code relies on the old fields |
| `agent_hints` | Define `enum LegalityStatus { Verified, Unverified, LikelyFallback, Unknown }`. Replace `legality_confidence/fallback_risk/drift_risk` in `AirNode`. Update `serialize.rs` to map old values to new enum. Update `risk_annotate.rs` to use the enum. |

---

### T-P3-04: Add validate_conv_dims convenience method

| Field | Value |
|-------|-------|
| `id` | T-P3-04 |
| `title` | Add validate_conv_dims() combining validate_tensor_dims() + validate_conv_channels() |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [V-037] |
| `acceptance_criteria` | 1) `validate_conv_dims()` method exists that calls both `validate_tensor_dims()` and `validate_conv_channels()`; 2) All conv-placement callers use `validate_conv_dims()` instead of `validate_tensor_dims()` alone; 3) Conv with 40000 channels is rejected |
| `agent_hints` | Add `pub fn validate_conv_dims(&self, ...) -> Result<(), ...>` that calls both methods. Find all call sites in `placement_validate.rs` and `op_constraints.rs` that validate conv dimensions and update them. |

---

### T-P3-05: Add E5M2 validation gate

| Field | Value |
|-------|-------|
| `id` | T-P3-05 |
| `title` | Add supports_e5m2() to AneFamily (always false) and validate in placement |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_target.rs`, `crates/passes/src/dtype_constraints.rs`, `crates/passes/src/placement_validate.rs` |
| `violation_refs` | [V-028] |
| `acceptance_criteria` | 1) `AneFamily::supports_e5m2()` exists and returns `false` for all families; 2) `is_dtype_ane_legal()` rejects E5M2 for ANE-targeted compilation; 3) Test verifies E5M2 is rejected |
| `agent_hints` | Add `pub fn supports_e5m2(&self) -> bool { false }` to `AneFamily`. In `is_dtype_ane_legal()`, add a match arm for `MilDtype::E5M2` that returns `Err(...)`. |

---

### T-P3-06: Remove or gate KvCacheLayout::Paged

| Field | Value |
|-------|-------|
| `id` | T-P3-06 |
| `title` | Add #[non_exhaustive] to KvCacheLayout and validate Paged variant |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/sir.rs` |
| `violation_refs` | [V-027] |
| `acceptance_criteria` | 1) `KvCacheLayout` is marked `#[non_exhaustive]`; 2) Downstream match arms handle the `Paged` variant with `todo!()` or `unimplemented!()` rather than silently accepting; 3) SIR construction rejects Paged for ANE targets |
| `agent_hints` | Add `#[non_exhaustive]` to the enum. In the SIR builder, add a validation that rejects `KvCacheLayout::Paged` for ANE-targeted compilation with a clear error message. |

---

### T-P3-07: Fix PIR handoff semantics

| Field | Value |
|-------|-------|
| `id` | T-P3-07 |
| `title` | Change Interior→Exit attn_out handoff from StateWriteRead to TensorPassThrough |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/pir.rs` |
| `violation_refs` | [V-039] |
| `acceptance_criteria` | 1) The `attn_out` handoff uses `HandoffKind::TensorPassThrough`; 2) KV-cache persistence remains modeled through `state_declarations`; 3) Decode-step shard test passes |
| `agent_hints` | At line ~779, change `handoff_kind: HandoffKind::StateWriteRead` to `HandoffKind::TensorPassThrough` for the `attn_out` tensor. The KV cache state should only use `StateWriteRead` in the separate `state_declarations` structure. |

---

### T-P3-08: Add ne_transpose_c_max validation

| Field | Value |
|-------|-------|
| `id` | T-P3-08 |
| `title` | Add validate_transpose_c_max() method and call in placement |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_hw_limits.rs`, `crates/passes/src/op_constraints.rs` |
| `violation_refs` | [V-051] |
| `acceptance_criteria` | 1) `validate_transpose_c_max()` method exists; 2) Transpose operations with channels exceeding the limit are rejected from ANE; 3) Test verifies rejection |
| `agent_hints` | Add `pub fn validate_transpose_c_max(&self, channels: u64) -> Result<(), ...>` to `AneHwLimits`. Call it from the transpose validation in `op_constraints.rs`. |

---

### T-P3-09: Add cross-constraint combination validations

| Field | Value |
|-------|-------|
| `id` | T-P3-09 |
| `title` | Validate cross-constraint combinations identified from forensic analysis |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [T-P2-01] |
| `files` | `crates/passes/src/op_constraints.rs` |
| `violation_refs` | [V-006, forensic §6.7, §6.11] |
| `acceptance_criteria` | 1) Dilation + vector_palettize combination is rejected; 2) Aliasing + vector_palettize combination is rejected; 3) Shuffle + per-channel_palettize combination is rejected; 4) Palettized weight + large_kernel_stride combination is rejected; 5) Z-dilation and X-dilation factor rejection added; 6) Deconv depth-axis stride rejection added |
| `agent_hints` | These constraint combinations are present in the ANECompiler binary but not validated by MILLer. Add checks in `op_constraints.rs` for each combination. The binary error strings provide the exact rejection messages to match against. |

---

### T-P3-10: Add architecture-gated constraint validations

| Field | Value |
|-------|-------|
| `id` | T-P3-10 |
| `title` | Add per-family validation for architecture-gated constraints from forensic analysis |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/passes/src/op_constraints.rs`, `crates/passes/src/dtype_constraints.rs` |
| `violation_refs` | [forensic §6.10] |
| `acceptance_criteria` | 1) Softmax on old HW (pre-A13) is rejected or documented as risky; 2) LRN on unsupported architectures is rejected; 3) Depth-axis broadcast on affected architectures is rejected; 4) A14-class resize alignCorners=true is rejected; 5) H13+ floor for MIL compilation is documented/enforced |
| `agent_hints` | The forensic analysis found strings like "Softmax is not supported by this ANE architecture" and "LRN is not supported on this architecture". Add per-family gates for these. Most of these are already CPU-only but the architecture gating adds an extra safety layer. |

---

### T-P3-11: Add palette_bits construction-time validation

| Field | Value |
|-------|-------|
| `id` | T-P3-11 |
| `title` | Validate palette_bits at construction time in SirOp::LinearProjection and SirOp::Const |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/sir.rs` |
| `violation_refs` | [V-035] |
| `acceptance_criteria` | 1) `palette_bits: Option<usize>` values are validated against {1,2,3,4,6,8} at construction; 2) Invalid values like `Some(5)` are rejected; 3) `clamp_to_valid_palette_bits()` logs a warning when clamping |
| `agent_hints` | Add a `validate_palette_bits()` call in the constructors or use a newtype `PaletteBits(usize)` with a `TryFrom<usize>` impl. Also add `log::warn!` to `clamp_to_valid_palette_bits()`. |

---

## Phase 4 — Low-Priority Cleanup and Polish

### T-P4-01: Move hardcoded constants to knowledge store

| Field | Value |
|-------|-------|
| `id` | T-P4-01 |
| `title` | Move LARGE_KERNEL_THRESHOLD and MAX_POOLING_KERNEL_DIM to ane_hw_limits_seed.json |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/passes/src/op_constraints.rs`, `knowledge/ane_hw_limits_seed.json`, `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [V-055, V-056] |
| `acceptance_criteria` | 1) `ane_hw_limits_seed.json` has `large_kernel_threshold` and `max_pooling_kernel_dim` fields; 2) `AneHwLimits` struct includes these fields; 3) `op_constraints.rs` loads them from the limits struct instead of using hardcoded constants |
| `agent_hints` | Add fields to `AneHwLimits`, populate from JSON seed, and update `op_constraints.rs` to accept `&AneHwLimits` instead of using constants. |

---

### T-P4-02: Replace magic zero with Option<usize>

| Field | Value |
|-------|-------|
| `id` | T-P4-02 |
| `title` | Change Conv1x1AsLinear::output_dim from usize to Option<usize> |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/ir/src/air.rs` |
| `violation_refs` | [V-053] |
| `acceptance_criteria` | 1) `output_dim: Option<usize>` with `None` meaning "unknown"; 2) All match arms updated; 3) Shape inference handles `None` correctly |
| `agent_hints` | Change the field type and update all construction and match sites. `None` is cleaner than `0` as a sentinel. |

---

### T-P4-03: Remove dead code

| Field | Value |
|-------|-------|
| `id` | T-P4-03 |
| `title` | Remove StaticizePass, AirOp::StaticLUTProjection, _DEFAULT_OPSET_MAP |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/passes/src/staticize.rs`, `crates/passes/src/lib.rs`, `crates/ir/src/air.rs`, `python/mil_emitter.py` |
| `violation_refs` | [V-029, V-058, V-062] |
| `acceptance_criteria` | 1) `staticize.rs` removed from crate; 2) `AirOp::StaticLUTProjection` removed from enum; 3) `_DEFAULT_OPSET_MAP` removed from `mil_emitter.py`; 4) All references updated; 5) `cargo test` passes |
| `agent_hints` | Remove the files/enums/constants. Update `lib.rs` to remove the module import. Update all match arms that reference `StaticLUTProjection`. Remove dead Python constant. |

---

### T-P4-04: Fix logging consistency

| Field | Value |
|-------|-------|
| `id` | T-P4-04 |
| `title` | Replace eprintln! with log::warn! in safetensors_resolver |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/bridge/src/safetensors_resolver.rs` |
| `violation_refs` | [V-063] |
| `acceptance_criteria` | 1) All `eprintln!` calls replaced with `log::warn!` or `log::error!`; 2) Messages are consistent with the rest of the codebase's logging style |
| `agent_hints` | Replace `eprintln!("Warning: ...")` with `log::warn!("...")` and `eprintln!("  ...")` with `log::debug!("...")`. |

---

### T-P4-05: Fix profile reporting

| Field | Value |
|-------|-------|
| `id` | T-P4-05 |
| `title` | Rename mean_ms to median_ms or compute actual mean in bridge.py |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `python/bridge.py` |
| `violation_refs` | [V-045] |
| `acceptance_criteria` | 1) Either `mean_ms` is renamed to `median_ms` or actual mean is computed; 2) `std_dev_ms` is either computed or renamed to `std_dev_ms: null`; 3) Downstream consumers updated |
| `agent_hints` | Rename `"mean_ms"` to `"median_ms"` and set `"std_dev_ms": None` (or compute it from the raw data if available). |

---

### T-P4-06: Update documentation

| Field | Value |
|-------|-------|
| `id` | T-P4-06 |
| `title` | Update ir_reference.md, bridge_protocol.md, architecture.md |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [T-P4-03] |
| `files` | `docs/ir_reference.md`, `docs/bridge_protocol.md`, `docs/architecture.md` |
| `violation_refs` | [V-046, V-047, V-048] |
| `acceptance_criteria` | 1) StaticizePass removed from pipeline listing; 2) Multifunction support documented in bridge_protocol.md; 3) "No stubs" claim in architecture.md qualified to exclude CAPI; 4) No stale documentation contradicts implementation |
| `agent_hints` | Remove StaticizePass from the pipeline table in ir_reference.md. Update the limitations section in bridge_protocol.md to document multifunction support. Qualify the stubs claim in architecture.md. |

---

### T-P4-07: Add HAL sub-variant modeling

| Field | Value |
|-------|-------|
| `id` | T-P4-07 |
| `title` | Research and model HAL sub-variants (H13g, H14c/g, H15c/g, H16c/g/s, H17a) |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [T-P2-08] |
| `files` | `crates/ir/src/ane_target.rs`, `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [forensic §5.1, §8.1] |
| `acceptance_criteria` | 1) Each HAL sub-variant is represented in AneRevision or a new AneSubVariant enum; 2) Constraint differences between sub-variants are documented; 3) Compilation targeting specific sub-variants uses correct limits |
| `agent_hints` | This requires Apple hardware testing to determine actual constraint differences. Start by adding the sub-variant names to the type system with the same limits as their parent. Mark as unverified. Document the need for hardware validation. |

---

### T-P4-08: Add unmapped ANEC operation stubs

| Field | Value |
|-------|-------|
| `id` | T-P4-08 |
| `title` | Add MirOp variants for 12 genuinely unmapped ANEC operations |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/ir/src/mir.rs`, `crates/passes/src/cpu_only_ops.rs` |
| `violation_refs` | [forensic §2.3] |
| `acceptance_criteria` | 1) MirOp has variants for: broadcast, scaled_elementwise, global_arg_min_max, degamma, dirac, gain_offset_control, n_relu, high_precision_sigmoid, log2, trunc, invert, unflatten, channel_to_space, space_to_channel; 2) Each is classified as ANE-legal or CPU-only; 3) Default engine assignments are correct per family |
| `agent_hints` | Add new MirOp variants. Most of these are niche operations. Classify them based on the forensic analysis: `scaled_elementwise` and `global_arg_min_max` may be ANE-legal on some families; others are likely CPU-only. Start with CPU-only classification for safety. |

---

## Task Dependency Graph

```
T-P1-01 ──→ T-P2-06
T-P1-04 ──→ T-P2-05
T-P2-01 ──→ T-P3-09
T-P2-08 ──→ T-P4-07
T-P4-03 ──→ T-P4-06
```

## Execution Order Recommendation

1. **T-P1-01 through T-P1-05** — can be executed in parallel (no interdependencies)
2. **T-P2-01 through T-P2-12** — can be mostly parallel; T-P2-05 depends on T-P1-04, T-P2-06 depends on T-P1-01
3. **T-P3-01 through T-P3-11** — can be mostly parallel; T-P3-09 depends on T-P2-01
4. **T-P4-01 through T-P4-08** — can be executed in any order; T-P4-06 depends on T-P4-03, T-P4-07 depends on T-P2-08
