# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Format: Agentic AI task specification (structured, machine-parseable, human-readable).
> Each task is independently executable by an AI coding agent with access to this repository.

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique task identifier (`T-<phase><seq>`) |
| `title` | Imperative, agent-actionable summary |
| `phase` | Execution phase (P2=high-priority, P3=medium, P4=low, P5=architectural) |
| `severity` | Worst violation severity if task is skipped |
| `depends_on` | Tasks that must complete first |
| `files` | Primary files to modify |
| `violation_refs` | Cross-references to ISSUES.md entries |
| `acceptance_criteria` | Verifiable conditions for task completion |
| `agent_hints` | Specific guidance for AI agent execution |

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

### T-P2-04: Fix knowledge store contradictions

| Field | Value |
|-------|-------|
| `id` | T-P2-04 |
| `title` | Synchronize knowledge store JSON with source code ground truth |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `knowledge/legality_seed.json`, `knowledge/cpu_only_ops_seed.json`, `knowledge/ane_op_family_matrix.json` |
| `violation_refs` | [V-011, M-013] |
| `acceptance_criteria` | 1) `cpu_only_ops_seed.json`: all ops from `cpu_only_ops.rs` CPU_ONLY_OPS set included; 2) `ane_op_family_matrix.json`: entries with empirical CPU-only status gain `practical_status: "cpu_only"` field; 3) Seed validation tests pass |
| `agent_hints` | Run `cpu_only_ops.rs` test to get the full CPU_ONLY_OPS set. Use that as the ground truth. For the matrix, add an `empirical_note` field rather than changing `supported` to avoid losing theoretical converter info. |

---

### T-P2-05: Error on unresolvable MILConst instead of zero-filling

| Field | Value |
|-------|-------|
| `id` | T-P2-05 |
| `title` | Return error when WeightResolver returns None for MILConst |
| `phase` | P2 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/bridge/src/mir_to_compat.rs` |
| `violation_refs` | [V-014] |
| `acceptance_criteria` | 1) When WeightResolver returns `None` for a MILConst value_path, the function returns `Err(...)` instead of creating zero-filled data; 2) `allow_missing_weights` gate is the only path that permits zero-fill; 3) New test verifies error on missing weight |
| `agent_hints` | Change the `None` branch at line ~939 to return `Err(BridgeError::UnresolvedWeight { path })` unless `allow_missing_weights` is true. The `allow_missing_weights` path should still exist but must be explicitly opted into. |

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
| `violation_refs` | [V-023, M-019] |
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

### T-P3-03: Replace AIR risk fields with status enum

| Field | Value |
|-------|-------|
| `id` | T-P3-03 |
| `title` | Replace legality_confidence/fallback_risk/drift_risk with LegalityStatus enum |
| `phase` | P3 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/air.rs`, `crates/ir/src/serialize.rs`, `crates/passes/src/risk_annotate.rs` |
| `violation_refs` | [V-026, M-011] |
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
| `violation_refs` | [V-006, F-CROSS-01, M-005] |
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
| `violation_refs` | [F-ARCH-01] |
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
| `acceptance_criteria` | 1) `palette_bits: Option<usize>` values are validated against {3,4,6,8} at construction; 2) Invalid values like `Some(5)` are rejected; 3) `clamp_to_valid_palette_bits()` logs a warning when clamping |
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

### T-P4-07: Add HAL sub-variant modeling

| Field | Value |
|-------|-------|
| `id` | T-P4-07 |
| `title` | Research and model HAL sub-variants (H13g, H14c/g, H15c/g, H16c/g/s, H17a) |
| `phase` | P4 |
| `severity` | LOW |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_target.rs`, `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [F-HAL-01] |
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
| `violation_refs` | [F-OPS-01] |
| `acceptance_criteria` | 1) MirOp has variants for: broadcast, scaled_elementwise, global_arg_min_max, degamma, dirac, gain_offset_control, n_relu, high_precision_sigmoid, log2, trunc, invert, unflatten, channel_to_space, space_to_channel; 2) Each is classified as ANE-legal or CPU-only; 3) Default engine assignments are correct per family |
| `agent_hints` | Add new MirOp variants. Most of these are niche operations. Classify them based on the forensic analysis: `scaled_elementwise` and `global_arg_min_max` may be ANE-legal on some families; others are likely CPU-only. Start with CPU-only classification for safety. |

---

## Phase 5 — MLIR-Method Architectural Remediation

### T-P5-03: Implement per-IR-layer verify() methods

| Field | Value |
|-------|-------|
| `id` | T-P5-03 |
| `title` | Add SirGraph::verify(), AirGraph::verify(), MirGraph::verify() |
| `phase` | P5 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/ir/src/sir.rs`, `crates/ir/src/air.rs`, `crates/ir/src/mir.rs` |
| `violation_refs` | [M-003, M-011, M-014] |
| `acceptance_criteria` | 1) Each verify() checks node reference integrity, required fields, dtype legality; 2) AirGraph::verify() rejects legality_confidence==0.0 in strict mode; 3) MirGraph::verify() requires non-empty shapes or Dynamic; 4) verify() called after each pass; 5) Pipeline aborts on failure |
| `agent_hints` | Add `pub fn verify(&self) -> Result<(), Vec<VerifyError>>` to each graph type. Check: all node IDs resolve, all dtype values are valid, required fields non-empty. For AIR, check legality_confidence>0.0. For MIR, check shapes are non-empty. Call from pipeline after each pass. |

---

### T-P5-04: Replace empty-shape fallbacks with explicit errors

| Field | Value |
|-------|-------|
| `id` | T-P5-04 |
| `title` | infer_shape() and compat_output_shape() must not return Ok(vec![]) |
| `phase` | P5 |
| `severity` | HIGH |
| `depends_on` | [T-P5-03] |
| `files` | `crates/passes/src/mil_lower.rs`, `crates/bridge/src/shape_inference.rs` |
| `violation_refs` | [M-003, M-006, M-033] |
| `acceptance_criteria` | 1) `infer_shape()` returns Err for unknown variants; 2) `compat_output_shape()` returns Err for unhandled MirOp; 3) `shard_plan.rs` returns Err instead of [1,1,1,1]; 4) Downstream handles Shape::Dynamic explicitly or fails |
| `agent_hints` | Change `Ok(vec![])` to `Err(...)` in infer_shape and compat_output_shape. Change derive_primary_shapes to bail! instead of defaulting. Add a `Shape::Dynamic` variant for intentional dynamic shapes. |

---

### T-P5-05: Rename LegalityRewritePass to AneLegalityRewritePass

| Field | Value |
|-------|-------|
| `id` | T-P5-05 |
| `title` | Rename pass to reflect ANE-specific nature |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/passes/src/legality_rewrite.rs`, `crates/passes/src/lib.rs`, `docs/ir_reference.md` |
| `violation_refs` | [M-017] |
| `acceptance_criteria` | 1) Pass renamed; 2) All references updated; 3) docs/ir_reference.md updated |
| `agent_hints` | Rename the struct and module. Update all use statements. Update the pipeline construction. Update the IR reference doc. |

---

### T-P5-06: Move ANE-specific validation out of mil_lower.rs

| Field | Value |
|-------|-------|
| `id` | T-P5-06 |
| `title` | Extract validate_sdpa_constraints() from MilLowerPass |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [T-P2-01] |
| `files` | `crates/passes/src/mil_lower.rs`, `crates/passes/src/placement_validate.rs` |
| `violation_refs` | [M-015] |
| `acceptance_criteria` | 1) MilLowerPass::run() does not call validate_sdpa_constraints(); 2) Constraints checked in placement; 3) MilLowerPass is pure AIR→MIR mapping |
| `agent_hints` | Move `validate_sdpa_constraints()` to placement_validate.rs. Remove the call from mil_lower.rs. Add a call in the SDPA placement validation block. |

---

### T-P5-07: Move engine assignment out of MirOp::base_engine()

| Field | Value |
|-------|-------|
| `id` | T-P5-07 |
| `title` | Create ane_placement.rs for target-parameterized engine mapping |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/ir/src/mir.rs`, new `crates/ir/src/ane_placement.rs` |
| `violation_refs` | [M-028, N-009] |
| `acceptance_criteria` | 1) New module maps MirOp→AneEngine parameterized by AneFamily; 2) base_engine() removed or replaced; 3) iOS18 hardcoding replaced with target-derived value; 4) Tests updated |
| `agent_hints` | Create ane_placement.rs with `fn engine_for_op(op: &MirOp, family: AneFamily) -> Option<AneEngine>`. Move the default_engine_for_revision logic there. Mark base_engine() deprecated. Make opset_version configurable. |

---

### T-P5-08: Remove ANE-specific attributes from SIR and MIR

| Field | Value |
|-------|-------|
| `id` | T-P5-08 |
| `title` | Move palette_bits, kernel_scale, kernel_zero_point, kernel_palettized_lut to target layer |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [T-P5-07] |
| `files` | `crates/ir/src/sir.rs`, `crates/ir/src/mir.rs` |
| `violation_refs` | [M-028] |
| `acceptance_criteria` | 1) SirOp::LinearProjection no longer has palette_bits; 2) MirOp::MILConv no longer has ANEC attributes; 3) Target-specific layer adds these during ANE lowering; 4) SIR can represent non-ANE targets |
| `agent_hints` | Move palette_bits from SirOp to a separate SirOpMetadata or target annotation. Move kernel_scale/kernel_zero_point/kernel_palettized_lut from MirOp to ane_placement metadata. This is a significant refactor — break into smaller PRs. |

---

### T-P5-09: Replace name-based heuristics with explicit annotations

| Field | Value |
|-------|-------|
| `id` | T-P5-09 |
| `title` | Remove name.contains("input_ids"), ends_with("_ids"), "__placeholder__" heuristics |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [T-P5-04] |
| `files` | `crates/bridge/src/shape_inference.rs`, `crates/passes/src/mil_lower.rs` |
| `violation_refs` | [M-016, M-018] |
| `acceptance_criteria` | 1) compat_input_shape() does not use name heuristics; 2) mil_lower dtype inference does not use name heuristics; 3) Shape/dtype carried as explicit fields from SIR→AIR→MIR; 4) Returns Err when unavailable |
| `agent_hints` | Replace name-based fallbacks with Err returns. Add explicit shape/dtype fields to MirOp variants. The goal is to carry information in the type system, not in naming conventions. |

---

### T-P5-10: Fix MirOpCompat::Unsupported weight materialization gap

| Field | Value |
|-------|-------|
| `id` | T-P5-10 |
| `title` | Make MirOpCompat::Unsupported visible to weight materialization or reject it |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `crates/coreml-proto/src/lib.rs` |
| `violation_refs` | [M-009] |
| `acceptance_criteria` | 1) Unsupported ops either return correct input_names() or are rejected at emission with a clear error; 2) No silent weight materialization gap |
| `agent_hints` | Either populate `input_names()` for Unsupported from the op's actual inputs, or reject Unsupported ops at the emission boundary with `bail!("Unsupported op: {}", name)`. |

---

### T-P5-11: Fix SPEC-implementation drift for knowledge store

| Field | Value |
|-------|-------|
| `id` | T-P5-11 |
| `title` | Update SPEC to match JSON knowledge store implementation or implement SQLite |
| `phase` | P5 |
| `severity` | MEDIUM |
| `depends_on` | [] |
| `files` | `SPEC.md`, `crates/knowledge/src/store.rs`, `crates/knowledge/src/confidence.rs` |
| `violation_refs` | [M-012, M-029, M-030] |
| `acceptance_criteria` | 1) SPEC accurately describes JSON backend (not SQLite); 2) Confidence decay is either implemented as described or SPEC updated to match; 3) Knowledge pruning is either implemented or SPEC section marked as planned; 4) No SPEC claims contradict implementation |
| `agent_hints` | The simplest path: update SPEC.md to describe the actual JSON implementation. Mark SQLite, confidence decay, and pruning as "planned" sections. Remove or qualify claims that don't match current implementation. |

---

## Phase 6 — Forensic Infrastructure Gaps (Long-Term)

### T-P6-01: Model remaining ANEC hal_params

| Field | Value |
|-------|-------|
| `id` | T-P6-01 |
| `title` | Add 35+ missing hal_params to AneHwLimits |
| `phase` | P6 |
| `severity` | CRITICAL |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [N-002, N-010] |
| `acceptance_criteria` | 1) All 50+ hal_params from ANEC binary research are modeled; 2) LUT size overflow detection added; 3) Validation uses complete parameter set |
| `agent_hints` | This requires extensive binary research. Add fields incrementally, starting with the most impactful: kernel depth limits, padding limits, PE/NE per-engine limits. Mark each as verified/unverified. |

---

### T-P6-02: Fix M1 hardware limits

| Field | Value |
|-------|-------|
| `id` | T-P6-02 |
| `title` | Fix M1() to inherit from A14 instead of A17 |
| `phase` | P6 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [N-003] |
| `acceptance_criteria` | 1) `m1()` uses `..Self::a14()` or correct M1-specific limits; 2) Tensor dimensions validated against correct M1 limits |
| `agent_hints` | Change the `..Self::a17()` to `..Self::a14()` in the m1() constructor. Verify against Apple documentation or empirical testing. |

---

### T-P6-03: Implement ValidateLayer-equivalent constraints

| Field | Value |
|-------|-------|
| `id` | T-P6-03 |
| `title` | Add MILLer equivalents for 40+ ANEC ValidateLayer instantiations |
| `phase` | P6 |
| `severity` | HIGH |
| `depends_on` | [T-P6-01] |
| `files` | `crates/passes/src/placement_validate.rs`, `crates/passes/src/op_constraints.rs` |
| `violation_refs` | [N-004] |
| `acceptance_criteria` | 1) Each ANEC ValidateLayer constraint has a MILLer validation equivalent; 2) Invalid configurations caught at placement time instead of ANEC compile time |
| `agent_hints` | This is a large task. Start by mapping each ValidateLayer instantiation to its constraint and adding validation methods. Prioritize by frequency of failure. |

---

### T-P6-04: Add uANE AneRevision variant

| Field | Value |
|-------|-------|
| `id` | T-P6-04 |
| `title` | Add AneRevision::Vu1 for uANE hardware |
| `phase` | P6 |
| `severity` | HIGH |
| `depends_on` | [] |
| `files` | `crates/ir/src/ane_target.rs`, `crates/ir/src/ane_hw_limits.rs` |
| `violation_refs` | [N-008] |
| `acceptance_criteria` | 1) `AneRevision::Vu1` variant exists; 2) uANE-specific limits (if different) are modeled; 3) CLI can target uANE |
| `agent_hints` | Add the variant. Research uANE constraints — they may differ from standard ANE. Start with conservative limits matching A17 until hardware testing confirms. |

---

### T-P6-05: Add fusability checks

| Field | Value |
|-------|-------|
| `id` | T-P6-05 |
| `title` | Implement IsFusable-equivalent checks for ANE layer fusion |
| `phase` | P6 |
| `severity` | HIGH |
| `depends_on` | [T-P6-03] |
| `files` | New module `crates/passes/src/fusability.rs` |
| `violation_refs` | [N-006] |
| `acceptance_criteria` | 1) Fusability check module exists; 2) Ops that individually pass placement but fail to fuse are caught; 3) ANEC fusion constraints modeled |
| `agent_hints` | This requires understanding ANEC's fusion rules from binary research. Create a fusability module that checks if adjacent ops can be fused into a single engine layer. |

---

### T-P6-06: Add L2 memory budget modeling

| Field | Value |
|-------|-------|
| `id` | T-P6-06 |
| `title` | Implement L2 memory budget modeling and legalization |
| `phase` | P6 |
| `severity` | HIGH |
| `depends_on` | [T-P6-01] |
| `files` | New module `crates/passes/src/l2_budget.rs` |
| `violation_refs` | [N-007] |
| `acceptance_criteria` | 1) L2 memory budget modeled per AneFamily; 2) Individually legal ops that collectively exceed budget are caught; 3) Legalization splits or reorders ops to fit |
| `agent_hints` | This is advanced infrastructure. Start by modeling L2 cache sizes per family. Add a budget check after placement validation. Legalization (splitting/reordering) is a follow-up. |
