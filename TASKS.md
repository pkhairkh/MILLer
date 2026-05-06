# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Recently completed: T-P2-01, T-P2-04, T-P2-05, T-P2-11, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P4-07, T-P4-08, T-P5-03, T-P5-04, T-P5-05, T-P5-06, T-P5-07, T-P5-09, T-P5-10, T-P5-11, T-P6-01, T-P6-02, T-P6-04, T-P6-06.
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

(All P2 tasks completed.)

---

## Phase 3 — Medium-Priority Constraint Enforcement

(All P3 tasks completed.)

---

## Phase 4 — Low-Priority Cleanup and Polish

(All P4 tasks completed.)

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
