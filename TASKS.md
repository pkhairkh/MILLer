# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Recently completed: T-P2-01, T-P2-04, T-P2-05, T-P2-11, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P4-07, T-P4-08, T-P5-03, T-P5-04, T-P5-05, T-P5-06, T-P5-07, T-P5-08, T-P5-09, T-P5-10, T-P5-11, T-P5-12, T-P6-01, T-P6-02, T-P6-03, T-P6-04, T-P6-05, T-P6-06, T-P7-01, T-P7-02, T-P7-03, T-P7-04, T-P7-05, T-P7-07, T-P7-08.
> Last audit: 2026-05-06 (codebase verification against all tracked issues)
> Format: Agentic AI task specification (structured, machine-parseable, human-readable).
> Each task is independently executable by an AI coding agent with access to this repository.

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique task identifier (`T-<phase><seq>`) |
| `title` | Imperative, agent-actionable summary |
| `phase` | Execution phase (P2=high-priority, P3=medium, P4=low, P5=architectural, P6=forensic, P7=remaining) |
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

(All P5 tasks completed.)

---

## Phase 6 — Forensic Infrastructure Gaps

(All P6 tasks completed.)

---

## Phase 7 — Remaining Code Quality Issues

### T-P7-06: Migrate callers to explicit shape API (M-019 remainder)
- **Severity:** LOW
- **Depends_on:** T-P5-09 (completed)
- **Files:** `crates/bridge/src/shape_inference.rs`, `crates/bridge/src/mir_to_compat.rs`, `crates/cli/src/main.rs`
- **Violation_refs:** M-019
- **Status:** PARTIAL — `compat_input_shape_explicit()` and `compat_output_shape_explicit()` added; legacy callers not yet migrated
- **Acceptance_criteria:**
  1. All callers of `compat_input_shape` / `compat_output_shape` migrated to explicit variants
  2. Legacy functions removed or marked `#[deprecated]`
- **Agent_hints:** Search for all call sites of the legacy functions. Pass `explicit_shape` from available MIR node shape or AirOp shape_hint.

### T-P7-09: Populate HAL sub-variant-specific hardware limits
- **Severity:** LOW
- **Depends_on:** None
- **Files:** `crates/ir/src/ane_hw_limits.rs`
- **Violation_refs:** F-HAL-01
- **Status:** PARTIAL — structural scaffolding done; per-variant data needed from hardware testing
- **Acceptance_criteria:**
  1. At least one sub-variant (e.g., H16s) has verified limits that differ from parent
  2. `verified: true` set for populated sub-variants
- **Agent_hints:** Requires hardware testing or binary research. Even partial data for one sub-variant would establish the pattern.

### T-P7-10: Add MIR-to-proto emission for high-priority ANEC ops
- **Severity:** LOW
- **Depends_on:** None
- **Files:** `crates/coreml-emit/src/mir_to_proto.rs`, `crates/ir/src/mir.rs`
- **Violation_refs:** F-OPS-01
- **Status:** PARTIAL — CPU-only safety stubs done; emission mappings needed
- **Acceptance_criteria:**
  1. `anec_fused_conv_activate` and `anec_scaled_elementwise` have proto emission paths
  2. Ops removed from CPU_ONLY_OPS and placed on ANE when legal
- **Agent_hints:** Start with fused_conv_activate since it's the most commonly used. Study the binary's ANEC dialect for the proto format.

---

## Deferred Tasks (Long-Term)

- **T-D-01**: Add structural verification to Python subprocess bridge (M-020) — requires Python-side graph verification
- **T-D-02**: Add MIR specification verification to Python MIL emitter (M-032) — requires Python-side schema validation
- **T-D-03**: Implement MLIR placement dialect (N-005) — major infrastructure project
- **T-D-04**: Model multi-ANE/firmware capabilities (F-FW-01) — out of current scope
