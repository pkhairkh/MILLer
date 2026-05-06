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
| `status` | `open` / `partial` |
| `labels` | GitHub-style labels for filtering |
| `forensic_ref` | Cross-reference to forensic constraint-summary section |
| `affects` | What is impacted (compilation, emission, runtime, documentation) |
| `reproduce` | Steps or conditions that trigger the issue |
| `fix_hint` | Suggested approach for resolution |
| `task_ref` | Reference to TASKS.md task |

---

## Resolved Issues (Removed)

The following issues were verified as resolved during the 2026-05-06 audit and removed from this tracker:

- **V-006** (T-P2-01): `validate_deconv_constraints()` wired into `MILConvTranspose` match arm in placement_validate.rs; A11Legacy restriction enforced
- **V-011** (T-P2-04): `cpu_only_ops_seed.json` expanded from ~84 to 176 entries, regenerated from Rust `CPU_ONLY_OPS` ground truth
- **V-014** (T-P2-05): `mir_to_compat.rs` returns `BridgeError::UnresolvedWeight` by default; zero-fill only with explicit `allow_missing_weights` opt-in
- **V-023** (T-P2-11): `ModelArchConfig` no longer implements `Default`; `architecture` and `max_seq_len` are required parameters
- **V-026** (T-P3-03): `LegalityStatus` enum (`Verified`/`Unverified`/`LikelyFallback`/`Unknown`) replaces hardcoded f32 stub fields; legacy format supported via `TryFrom` conversion
- **V-037** (T-P3-04): `validate_conv_channels()` exists and wired into `MILConv` match arm; `max_conv_channels: 32768` separate from `max_tensor_channels: 65536`
- **F-CROSS-01** (T-P3-09): `validate_cross_constraint_combinations()` enforces dilation+vector_palettize, aliasing+vector_palettize, shuffle+per_channel_palettize, palettize+large_stride
- **F-ARCH-01** (T-P3-10): `validate_architecture_gated_constraints()` rejects Softmax/LRN/InstanceNorm on A11Legacy/A12/A13
- **V-051** (T-P3-08): `validate_transpose_c_max()` wired into `MILTranspose` match arm
- **V-055** (T-P4-01): `LARGE_KERNEL_THRESHOLD` replaced by `AneHwLimits::large_kernel_mode_threshold` loaded per-revision
- **V-056** (T-P4-01): `MAX_POOLING_KERNEL_DIM` replaced by `AneHwLimits::max_pooling_kernel_dim` loaded per-revision
- **M-006** (T-P5-04): `shard_plan.rs` returns `bail!()` instead of silently defaulting to `[1,1,1,1]`
- **M-009** (T-P5-10): `MirOpCompat::Unsupported` now has `inputs: Vec<String>` field populated from `proto_input_refs()`, returning proper input names
- **M-012** (T-P5-11): SPEC.md updated to acknowledge JSON persistence is intentional; SQLite documented as future option
- **M-015** (T-P5-06): `validate_sdpa_constraints()` moved from `mil_lower.rs` to `placement_validate.rs`
- **N-002** (T-P6-01): 35+ hal_params added to `AneHwLimits` (kernel depth, padding limits, PE/NE limits, alignment, L2 cache, etc.)
- **N-003** (T-P6-02): `m1()` now inherits from `Self::a14()` instead of `Self::a17()`; `AneRevision::V17` maps to `AneFamily::A14`
- **N-006** (T-P6-05): `fusability.rs` implements `check_fusability()`, `classify_fusion_atom()`, `identify_fusion_groups()`
- **N-007** (T-P6-06): `l2_budget.rs` implements `check_l2_budget()`, `estimate_op_l2_footprint()`, `check_op_l2_fit()` using per-revision L2 cache sizes
- **N-008** (T-P6-04): `AneRevision::Vu1` added, maps to `AneFamily::A17`; dedicated `vu1()` factory in `ane_hw_limits.rs`
- **N-010** (T-P6-01): `ne_palette_lut_size_in_bytes` field on `AneHwLimits`; `validate_palette_lut_size()` method (A11-A14: 256, A15+: 512)
- **N-011** (T-P2-01): `validate_deconv_constraints()` wired into `MILConvTranspose` match arm
- **N-012** (T-P2-01): `validate_conv_dims()` and `validate_conv_channels()` wired into `MILConv` match arm

---

## Partially Resolved Issues

### F-HAL-01: HAL sub-variants modeled but inherit parent limits

| Field | Value |
|-------|-------|
| `id` | F-HAL-01 |
| `title` | HAL sub-variants (H13g, H14c, H14g, H15c, H15g, H16c, H16g, H16s, H17a) are modeled but all inherit parent family limits with `verified: false` — no sub-variant-specific constraint differences |
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

### M-011: AIR legality_confidence not automatically enforced

| Field | Value |
|-------|-------|
| `id` | M-011 |
| `title` | LegalityStatus enum exists and verify_strict() can reject Unknown nodes, but no automatic rejection gate in the standard compilation pipeline |
| `domain` | risk-assessment |
| `severity` | medium |
| `class` | PHANTOM-SEMANTIC |
| `status` | partial |
| `labels` | `risk-metrics`, `legality-enforcement`, `opt-in-only` |
| `forensic_ref` | SPEC §5.2 |
| `affects` | Compilation — nodes with `Unknown` legality pass through unless `verify_strict()` is explicitly called |
| `reproduce` | 1) Create an AirNode with `legality_status: LegalityStatus::Unknown`; 2) Run standard compilation pipeline; 3) Observe no rejection |
| `fix_hint` | Add an automatic legality gate in the compilation pipeline that rejects `Unknown` nodes by default (or requires opt-in to allow them). |
| `task_ref` | T-P5-03, T-P3-03 |

---

### M-019: Qwen3 name-based defaults persist in internal shape inference

| Field | Value |
|-------|-------|
| `id` | M-019 |
| `title` | Main entry point now requires architecture, but internal shape inference still uses `contains("input_ids")` / `contains("position")` name-based heuristics that default to Qwen3-specific patterns |
| `domain` | model-config |
| `severity` | medium |
| `class` | BACKEND-COUPLING |
| `status` | partial |
| `labels` | `qwen3-default`, `name-heuristic`, `shape-inference` |
| `forensic_ref` | N/A |
| `affects` | Shape inference — non-Qwen3 models may get incorrect shapes when node names happen to match Qwen3 patterns |
| `reproduce` | 1) Create a model with nodes named `foo_input_ids`; 2) Observe Qwen3-specific shape inference applied |
| `fix_hint` | Propagate `ModelArchitecture` through all shape inference paths; replace name-based heuristics with architecture-aware logic. |
| `task_ref` | T-P2-11 (main entry fixed; internal paths remain) |

---

### M-028: ANE constraints partially extracted from proto emission

| Field | Value |
|-------|-------|
| `id` | M-028 |
| `title` | ValidationPolicy system added for structured violation handling, but IOSurface size validation, surface uniformity, and flat buffer layout constraints remain in the proto emission layer |
| `domain` | emission-correctness |
| `severity` | medium |
| `class` | LAYER-LEAK |
| `status` | partial |
| `labels` | `ane-constraints`, `proto-emission`, `layer-separation` |
| `forensic_ref` | N/A |
| `affects` | Architecture — proto emission layer still contains some ANE-specific hardware constraints |
| `reproduce` | 1) Inspect `mir_to_proto.rs` lines 91-121, 491-663; 2) Observe IOSurface and layout constraints in emission code |
| `fix_hint` | Extract remaining ANE-specific validation (IOSurface size, surface uniformity, flat buffer layout) into the constraint validation layer. |
| `task_ref` | T-P5-07 |

---

### N-009: MILTile decomposed in legality_rewrite but default engine assignment unclear

| Field | Value |
|-------|-------|
| `id` | N-009 |
| `title` | MILTile is decomposed into ANE-legal broadcast Mul during legality rewrite, but the default engine assignment in mir.rs may still assign PE engine |
| `domain` | operation-coverage |
| `severity` | low |
| `class` | PHANTOM-SEMANTIC |
| `status` | partial |
| `labels` | `tile`, `engine-assignment`, `decomposition` |
| `forensic_ref` | N/A |
| `affects` | Compilation — if legality rewrite is skipped, Tile may be incorrectly placed on PE engine |
| `reproduce` | 1) Create a graph with MILTile; 2) Skip legality rewrite; 3) Observe PE engine assignment |
| `fix_hint` | Either add Tile to CPU_ONLY_OPS or fix default engine assignment to not assign PE engine. |
| `task_ref` | T-P5-07 |

---

## Open Issues — High Severity

### M-003: Shape Inference Returns Empty Vec for Unknown Operations

- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:401`; `crates/bridge/src/shape_inference.rs:528–531`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** `infer_shape()` returns `Ok(vec![])` as fallback; unknown shapes silently propagate instead of producing an error. This can lead to downstream passes operating on graphs with missing shape information, producing incorrect compilation results without any warning or error to the user.

---

### M-005: slanc_scales Pass Inserts Const+Mul with Uncomputed Scale Values

- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/passes/src/slanc_scales.rs:63–123`
- **Status:** OPEN | **Remediation:** T-P3-09
- **Description:** The pass inserts Mul ops with placeholder scale factors (always `computed_scales: false`) instead of computing actual scale values from weight metadata. The pass is deprecated and not wired into the compilation pipeline, but the code remains and could be accidentally invoked, silently corrupting the graph.

---

### M-013: Knowledge Seed Files Not Loaded by Any Runtime Crate

- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `knowledge/ane_op_family_matrix.json`; `knowledge/palettization_constraints_seed.json`
- **Status:** OPEN | **Remediation:** T-P2-04
- **Description:** Seed files are validated in tests but never loaded at runtime; all constraint data is hardcoded in Rust source files. The `KnowledgeStore::load_seeds_from_directory()` method exists but is not called by the CLI or compilation pipeline, making the seed files decorative rather than functional.

---

### M-014: MILLinear/MILConv Shape Inference Propagates Input Shape

- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:139, 376`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** For Linear and Conv operations, shape inference propagates the input shape instead of computing the correct output shape. The existing test `test_linear_propagates_input_shape` confirms this is the current behavior. This means shape-dependent passes may receive incorrect tensor dimensions.

---

### N-004: ValidateLayer Equivalent Only Partially Implemented

- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs`
- **Status:** OPEN | **Remediation:** T-P6-03
- **Description:** While placement_validate.rs now covers many constraints, it does not mirror the full set of 40+ ANEC ValidateLayer instantiations found in the binary. Gaps remain in constraint coverage — particularly for less common operation types and edge-case parameter combinations.

---

### N-005: MLIR Placement Dialect Entirely Absent (Layer 2 of 5)

- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no placement dialect module
- **Status:** OPEN | **Remediation:** Long-term infrastructure
- **Description:** No region-based placement, no force-ane-placement, and no boundary ops exist. The MLIR placement dialect would provide structured, composable placement decisions that can be verified and optimized independently from the lowering pipeline.

---

## Open Issues — Medium Severity

### M-016: Name-Based Dtype Heuristic in mil_lower.rs

- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:614`
- **Status:** OPEN | **Remediation:** T-P5-09
- **Description:** Uses `ends_with("_ids")` and `contains("mask")` for dtype inference instead of type signatures. While `dtype_hint` was added to `AirOp::Identity` as an override mechanism, the name-based heuristic remains as the fallback, making dtype inference fragile and dependent on naming conventions.

---

### M-017: LegalityRewritePass Is Entirely ANE-Specific

- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING / DIALECT-MISBOUNDARY | **Confidence:** HIGH
- **Location:** `crates/passes/src/legality_rewrite.rs:1–900+`
- **Status:** OPEN | **Remediation:** T-P5-05
- **Description:** The pass name "LegalityRewrite" is misleading — it is entirely ANE-specific with no target parameter or architecture abstraction. If MILLer ever supports multiple backends, this pass would need to be refactored into a target-parameterized framework.

---

### M-018: compat_input_shape Uses name.contains("input_ids") Heuristic

- **Severity:** MEDIUM | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:56–65`
- **Status:** OPEN | **Remediation:** T-P5-09
- **Description:** Implicit shape semantics are derived from node names (`contains("input_ids")`) instead of type signatures. While `max_seq_len` is now passed as a required parameter, the name-based heuristic still exists as a fallback, creating a fragile dependency on naming conventions.

---

### M-020: Python Subprocess Bridge Produces Unverifiable Transformations

- **Severity:** MEDIUM | **Class:** DIALECT-MISBOUNDARY | **Confidence:** HIGH
- **Location:** `crates/bridge/src/subprocess.rs:67–177`
- **Status:** OPEN | **Remediation:** Long-term
- **Description:** Python bridge trusts `BridgeResult.status == "success"` as semantic legality without structural verification. The bridge cannot verify that the transformation applied by the Python subprocess is semantically correct.

---

### M-024: StateTopologyPass Validates But Never Transforms

- **Severity:** MEDIUM | **Class:** CANONICALIZATION-MIXUP | **Confidence:** HIGH
- **Location:** `crates/passes/src/state_topology.rs:59–137`
- **Status:** OPEN | **Remediation:** T-P5-12
- **Description:** Pass validates state patterns and in strict mode returns errors, but it never transforms the graph — always returns `Ok(input)` unchanged. It is a validator masquerading as a transform pass. Should either be renamed to `StateTopologyValidator` or actually implement transformations.

---

### M-025: Circular Substitution Chain Produces Partial Resolution

- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/canonicalize.rs:109–135`
- **Status:** OPEN | **Remediation:** Code fix
- **Description:** When the 100-step limit for circular substitution chains is hit, the code logs a warning and returns whatever value it landed on. This should be a hard error — a partially resolved substitution could silently produce incorrect results.

---

### M-027: palettize_weights Silently Defaults Unknown Projections to mlp_bits

- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/palettize_weights.rs:176–185`
- **Status:** OPEN | **Remediation:** Code fix
- **Description:** Unknown projections silently default to `config.mlp_bits` with only a warning log. Non-standard architectures get wrong quantization bit-widths without any error, potentially degrading model accuracy.

---

### M-029: Confidence Decay Described in SPEC but Not Wired

- **Severity:** MEDIUM | **Class:** UNSUPPORTED-CLAIM | **Confidence:** HIGH
- **Location:** `SPEC.md:553–554`; `crates/knowledge/src/confidence.rs:13–24`
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes "1% per 30 days" linear decay; code has `update_confidence_bayesian()` for static Bayesian updates but no time-based decay. Neither the linear decay from the SPEC nor any operational decay mechanism is implemented.

---

### M-030: Knowledge Pruning Described in SPEC but Not Implemented

- **Severity:** MEDIUM | **Class:** UNSUPPORTED-CLAIM | **Confidence:** HIGH
- **Location:** `SPEC.md:531`; `crates/knowledge/src/` (all files)
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes a knowledge pruning mechanism for removing stale or low-confidence entries from the knowledge store. No implementation exists in any knowledge crate file.

---

### M-032: Python MIL Emitter Performs Unverified Graph Construction

- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `python/mil_emitter.py` — all `build_*_program()` functions
- **Status:** OPEN | **Remediation:** Long-term
- **Description:** Python functions construct MIL graphs without verifying that the constructed graph matches the MIR specification. The bridge has `verify` commands but they verify the emitted package, not the graph construction process itself.

---

### M-033: Multi-Zero Reshape Heuristic Assumes Batch=1

- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/bridge/src/mir_to_compat.rs:884–906`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** When a reshape has two or more zeros, the heuristic sets all but the last zero to 1, which assumes batch=1. This produces incorrect results for models with batch > 1.

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
| `forensic_ref` | §4.5 (firmware and multi-ANE), §7 (firmware paths, chaining API) |
| `affects` | Device management — multi-ANE systems and firmware loading not represented |
| `task_ref` | None (out of current scope, document for future) |

---

## Issue Statistics (Active Only)

| Domain | High | Medium | Low | Total |
|--------|------|--------|-----|-------|
| shape-inference | 2 | 0 | 0 | 2 |
| knowledge-store | 1 | 0 | 0 | 1 |
| stub-passes | 1 | 1 | 0 | 2 |
| constraint-enforcement | 0 | 0 | 1 | 1 |
| hardware-modeling | 2 | 1 | 0 | 3 |
| risk-assessment | 0 | 1 | 0 | 1 |
| backend-coupling | 0 | 3 | 0 | 3 |
| layer-separation | 0 | 1 | 0 | 1 |
| model-config | 0 | 1 | 0 | 1 |
| operation-coverage | 0 | 0 | 1 | 1 |
| spec-drift | 0 | 2 | 0 | 2 |
| **Total** | **6** | **11** | **2** | **19** |

### Status Breakdown

| Status | Count | Issues |
|--------|-------|--------|
| OPEN | 13 | M-003, M-005, M-013, M-014, M-016, M-017, M-018, M-020, M-024, M-025, M-027, M-029, M-030, M-032, M-033, N-004, N-005, F-FW-01 |
| PARTIAL | 6 | F-HAL-01, F-OPS-01, M-011, M-019, M-028, N-009 |
| **Total Active** | **19** | |

### Resolved Since Last Audit: 22 issues
V-006, V-011, V-014, V-023, V-026, V-037, V-051, V-055, V-056, F-CROSS-01, F-ARCH-01, M-006, M-009, M-012, M-015, N-002, N-003, N-006, N-007, N-008, N-010, N-011, N-012
