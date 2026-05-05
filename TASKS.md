# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Recently completed: T-P2-01, T-P2-05, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P5-05.
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

## Phase 3 — Medium-Priority Constraint Enforcement

(All P3 tasks completed.)

---

## Phase 4 — Low-Priority Cleanup and Polish

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
| `acceptance_criteria` | 1) Each verify() checks node reference integrity, required fields, dtype legality; 2) AirGraph::verify() rejects LegalityStatus::Unknown in strict mode; 3) MirGraph::verify() requires non-empty shapes or Dynamic; 4) verify() called after each pass; 5) Pipeline aborts on failure |
| `agent_hints` | Add `pub fn verify(&self) -> Result<(), Vec<VerifyError>>` to each graph type. Check: all node IDs resolve, all dtype values are valid, required fields non-empty. For AIR, check legality_status is not Unknown in strict mode. For MIR, check shapes are non-empty. Call from pipeline after each pass. |

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
