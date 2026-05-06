# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> All tasks have been completed — see git history for the full audit trail.
> Completed: T-P2-01, T-P2-04, T-P2-05, T-P2-11, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P4-07, T-P4-08, T-P5-03, T-P5-04, T-P5-05, T-P5-06, T-P5-07, T-P5-08, T-P5-09, T-P5-10, T-P5-11, T-P5-12, T-P6-01, T-P6-02, T-P6-03, T-P6-04, T-P6-05, T-P6-06, T-P7-01, T-P7-02, T-P7-03, T-P7-04, T-P7-05, T-P7-06, T-P7-07, T-P7-08, T-P7-09, T-P7-10, T-D-01, T-D-02, T-D-03, T-D-04.
> Last audit: 2026-05-06 (all tracked issues resolved)
> Format: Agentic AI task specification (structured, machine-parseable, human-readable).

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique task identifier (`T-<phase><seq>`) |
| `title` | Imperative, agent-actionable summary |
| `phase` | Execution phase (P2=high-priority, P3=medium, P4=low, P5=architectural, P6=forensic, P7=remaining, D=deferred) |
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

(All P7 tasks completed, including T-P7-06, T-P7-09, T-P7-10.)

---

## Deferred Tasks (Completed)

- **T-D-01** (M-020): `BridgeVerifier` added to `subprocess.rs` with strict/lenient/default modes; integrated into `execute_raw_payload()`; 14 tests
- **T-D-02** (M-032): `pre_emit_verification()` and `verify_mir_spec_compliance()` added to `python/verify.py`; integrated into `mil_emitter.py`; 9 Python tests
- **T-D-03** (N-005): `placement_dialect.rs` module in `ane-ir` crate; PlacementRegion, ForceAnePlacement, BoundaryOp, PlacementAnnotation, validate_placement_annotations(); 19 tests
- **T-D-04** (F-FW-01): `multi_ane.rs` module in `ane-ir` crate; AneDevice, AneFirmwareSet, FirmwareImage, SubTypeDescriptor, ChainedProgram, InterDeviceTransfer, TransferMethod, MultiAneSystem; 20 tests
