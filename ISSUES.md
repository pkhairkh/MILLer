# ISSUES.md — MILLer Compatibility & Correctness Issue Tracker

> Active issue tracker for the MILLer compiler project.
> Resolved issues have been removed — see git history for the full audit trail.
> Issues are classified by domain, severity, and evidence strength.
> Each issue is independently resolvable by an AI coding agent with repository access.
> Last audit: 2026-05-06 (codebase verification against all tracked issues)

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique issue identifier (`V-NNN`, `M-NNN`, `F-NNN`, `N-NNN`) |
| `title` | Concise, GitHub-style issue title |
| `domain` | Primary concern area |
| `severity` | `critical` / `high` / `medium` / `low` |
| `class` | Violation classification from audit |
| `status` | `open` / `partial` / `deferred` |
| `labels` | GitHub-style labels for filtering |
| `forensic_ref` | Cross-reference to forensic constraint-summary section |
| `affects` | What is impacted (compilation, emission, runtime, documentation) |
| `reproduce` | Steps or conditions that trigger the issue |
| `fix_hint` | Suggested approach for resolution |
| `task_ref` | Reference to TASKS.md task |

---

## Resolved Issues (Removed)

The following issues were resolved across two audit rounds (2026-05-06):

**Round 1** (22 issues): V-006, V-011, V-014, V-023, V-026, V-037, V-051, V-055, V-056, F-CROSS-01, F-ARCH-01, M-006, M-009, M-012, M-015, N-002, N-003, N-006, N-007, N-008, N-010, N-011, N-012

**Round 2** (13 issues):
- **M-003** (T-P5-04): `compat_output_shape_fallible()` now handles all MirOp variants with explicit errors instead of silently returning empty vec; catch-all logs warnings
- **M-005** (T-P7-01): `slanc_scales.rs` removed entirely — deprecated stub-mimic pass that was never wired into pipeline
- **M-011** (T-P7-05): `validate_legality_status()` gate added to `RiskAnnotatePass`; compilation fails by default if any node has `LegalityStatus::Unknown`; opt-in `new_allow_unknown()` for development
- **M-013** (T-P7-02): CLI now calls `KnowledgeStore::load_seeds_from_directory()` during startup with default `knowledge/` directory fallback
- **M-014** (T-P5-04): MILConv now computes correct output shape `[batch, C_out, out_H, out_W]`; MILLinear documented as requiring weight metadata for output dim (Indeterminate error in fallible API)
- **M-016** (T-P5-09): Name-based dtype heuristic (`ends_with("_ids")`, `contains("mask")`) removed from `mil_lower.rs`; defaults to Fp16 with info log when no dtype_hint set
- **M-017** (T-P5-05): Renamed to `AneLegalityRewritePass` with `CompilationTarget` enum; `LegalityRewritePass` kept as backward-compat type alias; CPU target returns early
- **M-018** (T-P5-09): `compat_input_shape_explicit()` and `compat_output_shape_explicit()` accept `explicit_shape: Option<&[usize]>` to bypass name heuristics; legacy functions log deprecation warnings when heuristic fires
- **M-024** (T-P5-12): Renamed to `StateTopologyValidator` with backward-compat alias `StateTopologyPass`
- **M-025** (T-P7-03): Circular substitution chain now returns `anyhow::bail!()` hard error instead of logging warning and returning partial result
- **M-027** (T-P7-04): Unknown projections in `palettize_weights` now return hard error by default; `allow_unknown_projections: true` opt-in preserves backward compat
- **M-028** (T-P7-07): IOSurface size validation, surface uniformity, and flat buffer layout constraints extracted from `mir_to_proto.rs` into `placement_validate.rs::validate_surface_constraints()`
- **M-029** (T-P5-11): `apply_time_decay()` implements SPEC §553-554 linear decay (1% per 30 days); `KnowledgeStore::apply_confidence_decay()` applies to all entries
- **M-030** (T-P5-11): `KnowledgeStore::prune_below_threshold()` removes entries below confidence threshold, cleans up secondary indexes
- **M-033** (T-P5-04): `resolve_reshape_shape()` now accepts `batch_size: Option<usize>` parameter; when provided, uses known batch size instead of assuming batch=1; None fallback logs M-033 warning
- **N-004** (T-P6-03): All MirOp variants now have explicit match arms in placement_validate.rs; pooling, gather, LRN, constexpr_dequantize wired with constraint checks; 9 new N-004 tests
- **N-009** (T-P7-08): MILTile added to CPU_ONLY_OPS with `NoConverter` reason; safety net for when legality rewrite doesn't fire

---

## Partially Resolved Issues

### F-HAL-01: HAL sub-variants modeled but inherit parent limits

| Field | Value |
|-------|-------|
| `id` | F-HAL-01 |
| `title` | HAL sub-variants modeled but all inherit parent family limits with `verified: false` — no sub-variant-specific constraint differences |
| `domain` | hardware-modeling |
| `severity` | medium |
| `class` | LACUNA |
| `status` | partial |
| `labels` | `hal-variants`, `hardware-modeling`, `sub-variant`, `unverified-limits` |
| `forensic_ref` | §5.1 (HAL variant table), §8.1 (AneFamily coverage) |
| `affects` | Compilation — sub-variant-specific constraints may be missed; current limits are conservative defaults |
| `reproduce` | 1) Use `for_hal_sub_variant()` with a sub-variant; 2) Observe that limits are identical to parent family; 3) Warning logged that `verified: false` |
| `fix_hint` | Populate per-sub-variant limit overrides from binary research or hardware testing. Set `verified: true` once confirmed. |
| `task_ref` | T-P4-07 (structural scaffolding done; per-variant data needed) |

---

### F-OPS-01: ANEC operation stubs exist but lack emission mappings

| Field | Value |
|-------|-------|
| `id` | F-OPS-01 |
| `title` | 16 ANEC internal operations have CPU-only stubs (CpuOnlyReason::NoConverter) but no actual MIR-to-proto emission code |
| `domain` | operation-coverage |
| `severity` | medium |
| `class` | LACUNA |
| `status` | partial |
| `labels` | `op-coverage`, `unmapped-ops`, `anec-dialect`, `cpu-only-stub` |
| `forensic_ref` | §2.3 (ANEC dialect operation table) |
| `affects` | Coverage — ANEC-internal ops are safely marked CPU-only but cannot be emitted for ANE execution |
| `reproduce` | 1) Attempt to compile a model using `anec_fused_conv_activate`; 2) Observe CPU-only placement; 3) No proto emission path exists |
| `fix_hint` | Add MIR→proto emission mappings for high-priority ANEC ops (fused_conv_activate, scaled_elementwise). Lower-priority ops can remain as CPU-only stubs. |
| `task_ref` | T-P4-08 (safety stubs done; emission mappings needed) |

---

### M-019: Name heuristics deprecated but legacy fallbacks remain

| Field | Value |
|-------|-------|
| `id` | M-019 |
| `title` | Explicit shape API added (compat_input_shape_explicit / compat_output_shape_explicit) but legacy functions still use name-based heuristics as fallback when no explicit shape provided |
| `domain` | model-config |
| `severity` | low |
| `class` | BACKEND-COUPLING |
| `status` | partial |
| `labels` | `qwen3-default`, `name-heuristic`, `shape-inference`, `deprecated-api` |
| `forensic_ref` | N/A |
| `affects` | Shape inference — callers using legacy API may still get name-heuristic shapes |
| `reproduce` | 1) Call `compat_input_shape("input_ids", &[], 512)` without explicit_shape; 2) Observe deprecated heuristic firing |
| `fix_hint` | Migrate all callers to `compat_input_shape_explicit` / `compat_output_shape_explicit` with explicit shapes; then remove legacy functions. |
| `task_ref` | T-P7-06 |

---

## Deferred Issues (Long-Term / Out of Scope)

### M-020: Python Subprocess Bridge Produces Unverifiable Transformations

- **Severity:** MEDIUM | **Class:** DIALECT-MISBOUNDARY | **Confidence:** HIGH
- **Location:** `crates/bridge/src/subprocess.rs:67–177`
- **Status:** DEFERRED | **Remediation:** Long-term
- **Description:** Python bridge trusts `BridgeResult.status == "success"` as semantic legality without structural verification. Requires Python-side graph verification — significant infrastructure investment.

### M-032: Python MIL Emitter Performs Unverified Graph Construction

- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `python/mil_emitter.py` — all `build_*_program()` functions
- **Status:** DEFERRED | **Remediation:** Long-term
- **Description:** Python functions construct MIL graphs without verifying that the constructed graph matches the MIR specification. Requires Python-side schema validation.

### N-005: MLIR Placement Dialect Entirely Absent (Layer 2 of 5)

- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no placement dialect module
- **Status:** DEFERRED | **Remediation:** Long-term infrastructure
- **Description:** No region-based placement, no force-ane-placement, and no boundary ops exist. This is a major infrastructure project beyond the scope of incremental fixes.

### F-FW-01: Multi-ANE/firmware capabilities not modeled

| Field | Value |
|-------|-------|
| `id` | F-FW-01 |
| `title` | Binary reveals multi-ANE device enumeration, 4 firmware images, subType matching, program chaining — none modeled by MILLer |
| `domain` | hardware-modeling |
| `severity` | medium |
| `class` | LACUNA |
| `status` | deferred |
| `labels` | `multi-ane`, `firmware`, `chaining`, `not-modeled` |
| `forensic_ref` | §4.5 (firmware and multi-ANE), §7 (firmware paths, chaining API) |
| `affects` | Device management — multi-ANE systems and firmware loading not represented |
| `task_ref` | None (out of current scope, document for future) |

---

## Issue Statistics (Active Only)

| Domain | High | Medium | Low | Total |
|--------|------|--------|-----|-------|
| hardware-modeling | 0 | 2 | 0 | 2 |
| operation-coverage | 0 | 1 | 0 | 1 |
| model-config | 0 | 0 | 1 | 1 |
| **Total** | **0** | **3** | **1** | **4** |

### Status Breakdown

| Status | Count | Issues |
|--------|-------|--------|
| PARTIAL | 3 | F-HAL-01, F-OPS-01, M-019 |
| DEFERRED | 4 | M-020, M-032, N-005, F-FW-01 |
| **Total Active** | **7** | |

### Resolved Since Round 1: 35 issues total
Round 1 (22): V-006, V-011, V-014, V-023, V-026, V-037, V-051, V-055, V-056, F-CROSS-01, F-ARCH-01, M-006, M-009, M-012, M-015, N-002, N-003, N-006, N-007, N-008, N-010, N-011, N-012
Round 2 (13): M-003, M-005, M-011, M-013, M-014, M-016, M-017, M-018, M-024, M-025, M-027, M-028, M-029, M-030, M-033, N-004, N-009
