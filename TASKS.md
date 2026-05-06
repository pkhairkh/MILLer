# TASKS.md — MILLer Remediation Task Board

> Consolidated remediation task board for open and in-progress issues.
> Completed tasks have been removed — see git history for the full audit trail.
> Recently completed: T-P2-01, T-P2-04, T-P2-05, T-P2-11, T-P2-12, T-P3-03, T-P3-04, T-P3-07, T-P3-08, T-P3-09, T-P3-10, T-P3-11, T-P4-01, T-P4-02, T-P4-03, T-P4-04, T-P4-05, T-P4-07, T-P4-08, T-P5-03, T-P5-06, T-P5-07, T-P5-08, T-P5-10, T-P6-01, T-P6-02, T-P6-04, T-P6-05, T-P6-06.
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

### T-P5-04: Fix shape inference fallbacks and propagation
- **Severity:** HIGH
- **Depends_on:** None
- **Files:** `crates/bridge/src/shape_inference.rs`, `crates/passes/src/mil_lower.rs`, `crates/bridge/src/mir_to_compat.rs`
- **Violation_refs:** M-003, M-014, M-033
- **Status:** PARTIAL — `shard_plan.rs` fixed; shape inference propagation and empty-vec fallbacks remain
- **Acceptance_criteria:**
  1. `infer_shape()` returns `Err` for unknown operations instead of `Ok(vec![])`
  2. MILLinear/MILConv compute correct output shapes instead of propagating input shape
  3. Multi-zero reshape handles batch > 1 correctly
- **Agent_hints:** Start with adding proper output shape computation for Linear and Conv. The reshape fix needs batch dimension tracking.

### T-P5-05: Refactor LegalityRewritePass for target parameterization
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/legality_rewrite.rs`
- **Violation_refs:** M-017
- **Status:** PARTIAL — documentation added; pass remains entirely ANE-specific
- **Acceptance_criteria:**
  1. Pass accepts a target parameter (at minimum, ANE vs CPU)
  2. ANE-specific logic is gated behind target selection
  3. Pass name or module path reflects ANE specificity
- **Agent_hints:** Add a `Target` enum parameter to the pass. Wrap ANE-specific match arms in target checks. Consider renaming to `AneLegalityRewritePass`.

### T-P5-09: Replace name-based dtype/shape heuristics with type signatures
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/mil_lower.rs`, `crates/bridge/src/shape_inference.rs`
- **Violation_refs:** M-016, M-018
- **Status:** PARTIAL — `dtype_hint` override added to `AirOp::Identity`; name-based fallbacks remain
- **Acceptance_criteria:**
  1. `mil_lower.rs` uses type signatures or explicit hints instead of `ends_with("_ids")` / `contains("mask")`
  2. `shape_inference.rs` uses type signatures instead of `contains("input_ids")`
  3. Name-based heuristics removed, not just supplemented
- **Agent_hints:** Extend `dtype_hint` to all relevant AirOp variants. Add shape_hint mechanism for compat_input_shape. Remove name-matching code paths.

### T-P5-11: Implement confidence decay and knowledge pruning
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/knowledge/src/confidence.rs`, `crates/knowledge/src/store.rs`, `SPEC.md`
- **Violation_refs:** M-029, M-030
- **Status:** PARTIAL — SPEC updated to acknowledge JSON approach; decay and pruning not implemented
- **Acceptance_criteria:**
  1. Time-based confidence decay implemented (linear 1% per 30 days or as SPEC describes)
  2. Knowledge pruning mechanism removes stale/low-confidence entries
  3. SPEC and implementation agree on decay/pruning behavior
- **Agent_hints:** Implement `apply_time_decay()` on knowledge entries. Add `prune_below_threshold()` to KnowledgeStore. Update SPEC to match chosen implementation.

### T-P5-12: Fix StateTopologyPass — transform or rename
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/state_topology.rs`
- **Violation_refs:** M-024
- **Status:** OPEN — pass validates but never transforms
- **Acceptance_criteria:**
  1. Either implement graph transformations (reorder state reads/writes for optimal topology)
  2. Or rename to `StateTopologyValidator` and register as a validator, not a transform pass
- **Agent_hints:** Renaming is the simpler fix. If transforming, look at state read/write patterns and optimize placement.

---

## Phase 6 — Forensic Infrastructure Gaps

### T-P6-03: Complete ValidateLayer-equivalent coverage
- **Severity:** HIGH
- **Depends_on:** None
- **Files:** `crates/passes/src/placement_validate.rs`
- **Violation_refs:** N-004
- **Status:** PARTIAL — many constraints wired; 40+ ANEC ValidateLayer instantiations not fully covered
- **Acceptance_criteria:**
  1. Audit all ANEC ValidateLayer instantiations against placement_validate.rs match arms
  2. Add missing constraint checks for uncovered operation types
  3. Document coverage gap (which ValidateLayer checks are missing)
- **Agent_hints:** Extract ValidateLayer check names from binary research notes. Cross-reference with existing match arms in placement_validate.rs. Add stubs for missing checks.

---

## Phase 7 — Remaining Code Quality Issues

### T-P7-01: Remove slanc_scales stub pass
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/slanc_scales.rs`, `crates/passes/src/lib.rs`
- **Violation_refs:** M-005
- **Acceptance_criteria:**
  1. Remove `slanc_scales.rs` entirely (deprecated, not wired, stub-mimic)
  2. Remove from `mod.rs` / `lib.rs` registration
  3. No compilation errors or test failures
- **Agent_hints:** Simple deletion task. Verify no other code references the pass.

### T-P7-02: Wire knowledge seed files into runtime pipeline
- **Severity:** HIGH
- **Depends_on:** None
- **Files:** `crates/knowledge/src/store.rs`, `crates/cli/src/main.rs`
- **Violation_refs:** M-013
- **Acceptance_criteria:**
  1. CLI calls `KnowledgeStore::load_seeds_from_directory()` during initialization
  2. Seed data is used instead of or as fallback for hardcoded Rust data
  3. Missing seed files produce clear error messages
- **Agent_hints:** The `load_seeds_from_directory()` method already exists. Wire it into the CLI startup path.

### T-P7-03: Make circular substitution limit a hard error
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/canonicalize.rs`
- **Violation_refs:** M-025
- **Acceptance_criteria:**
  1. 100-step circular substitution limit returns `Err` instead of logging warning and returning partial result
  2. Error message identifies the circular chain
- **Agent_hints:** Change `log::warn!()` + `break` to `bail!()` or `return Err()`.

### T-P7-04: Make palettize_weights reject unknown projections
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/palettize_weights.rs`
- **Violation_refs:** M-027
- **Acceptance_criteria:**
  1. Unknown projection types return an error instead of silently defaulting to `mlp_bits`
  2. User must explicitly specify bit-width for non-standard architectures
- **Agent_hints:** Replace the `warn!()` + default path with `bail!()`. Add a config option to allow explicit opt-in for defaults.

### T-P7-05: Add automatic legality gate in compilation pipeline
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/passes/src/lib.rs`, `crates/cli/src/main.rs`
- **Violation_refs:** M-011
- **Acceptance_criteria:**
  1. Compilation pipeline rejects `LegalityStatus::Unknown` nodes by default
  2. Opt-in flag allows proceeding with Unknown legality (for development/debugging)
- **Agent_hints:** Add a pipeline step after placement that checks all nodes. Use the existing `verify_strict()` method.

### T-P7-06: Propagate ModelArchitecture through shape inference paths
- **Severity:** MEDIUM
- **Depends_on:** T-P5-09
- **Files:** `crates/bridge/src/shape_inference.rs`, `crates/bridge/src/mir_to_compat.rs`
- **Violation_refs:** M-019
- **Acceptance_criteria:**
  1. All shape inference functions accept `ModelArchitecture` parameter
  2. Name-based heuristics for Qwen3 patterns replaced with architecture-aware logic
- **Agent_hints:** Thread `ModelArchitecture` through `compat_input_shape()` and related functions.

### T-P7-07: Extract remaining ANE constraints from proto emission
- **Severity:** MEDIUM
- **Depends_on:** None
- **Files:** `crates/coreml-emit/src/mir_to_proto.rs`, `crates/passes/src/placement_validate.rs`
- **Violation_refs:** M-028
- **Acceptance_criteria:**
  1. IOSurface size validation moved from `mir_to_proto.rs` to placement validation
  2. Surface uniformity checks moved to placement validation
  3. Flat buffer layout constraints moved to placement validation
- **Agent_hints:** Identify all ANE-specific checks in mir_to_proto.rs. Move each to a validation function in placement_validate.rs.

### T-P7-08: Fix MILTile default engine assignment
- **Severity:** LOW
- **Depends_on:** None
- **Files:** `crates/ir/src/mir.rs`
- **Violation_refs:** N-009
- **Acceptance_criteria:**
  1. MILTile either added to CPU_ONLY_OPS or default engine assignment changed from PE
- **Agent_hints:** The legality_rewrite already decomposes Tile; this is a safety net fix.

### T-P7-09: Populate HAL sub-variant-specific hardware limits
- **Severity:** LOW
- **Depends_on:** None
- **Files:** `crates/ir/src/ane_hw_limits.rs`
- **Violation_refs:** F-HAL-01
- **Acceptance_criteria:**
  1. At least one sub-variant (e.g., H16s) has verified limits that differ from parent
  2. `verified: true` set for populated sub-variants
- **Agent_hints:** Requires hardware testing or binary research. Even partial data for one sub-variant would establish the pattern.

### T-P7-10: Add MIR-to-proto emission for high-priority ANEC ops
- **Severity:** LOW
- **Depends_on:** None
- **Files:** `crates/coreml-emit/src/mir_to_proto.rs`, `crates/ir/src/mir.rs`
- **Violation_refs:** F-OPS-01
- **Acceptance_criteria:**
  1. `anec_fused_conv_activate` and `anec_scaled_elementwise` have proto emission paths
  2. Ops removed from CPU_ONLY_OPS and placed on ANE when legal
- **Agent_hints:** Start with fused_conv_activate since it's the most commonly used. Study the binary's ANEC dialect for the proto format.
