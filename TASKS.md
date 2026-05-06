# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Recently completed: T-P2-01, T-P2-04, T-P2-05, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P4-07, T-P5-03, T-P5-04, T-P5-05, T-P5-06, T-P5-07, T-P5-09, T-P5-10, T-P5-11, T-P6-02, T-P6-04.
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
