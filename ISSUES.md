# ISSUES.md — MILLer Compatibility & Correctness Issue Tracker

> Active issue tracker for the MILLer compiler project.
> Resolved issues have been removed — see git history for the full audit trail.
> Issues are classified by domain, severity, and evidence strength.
> Each issue is independently resolvable by an AI coding agent with repository access.

---

## Conventions

| Field | Meaning |
|-------|---------|
| `id` | Unique issue identifier (`V-NNN`, `M-NNN`, `F-NNN`, `N-NNN`) |
| `title` | Concise, GitHub-style issue title |
| `domain` | Primary concern area |
| `severity` | `critical` / `high` / `medium` / `low` |
| `class` | Violation classification from audit |
| `status` | `open` / `in-progress` |
| `labels` | GitHub-style labels for filtering |
| `forensic_ref` | Cross-reference to forensic constraint-summary section |
| `affects` | What is impacted (compilation, emission, runtime, documentation) |
| `reproduce` | Steps or conditions that trigger the issue |
| `fix_hint` | Suggested approach for resolution |
| `task_ref` | Reference to TASKS.md task |

---

## High Issues

### V-006: ConvTranspose placement skips all constraint checks

| Field | Value |
|-------|-------|
| `id` | V-006 |
| `title` | MILConvTranspose unconditionally allowed on ANE — 5 deconv constraints and A11Legacy restriction not enforced |
| `domain` | constraint-enforcement |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `silicon-gate`, `conv-transpose`, `unvalidated-constraints` |
| `forensic_ref` | §6.3 (deconv constraints verified in binary), §8.3 (validation gap) |
| `affects` | Compilation — dilated deconv, SOx!=2 deconv, A11Legacy deconv pass placement and fail at ANEC |
| `reproduce` | 1) Create MILConvTranspose with dilation > 1; 2) Run placement validation; 3) Observe `AneAllowed` instead of `CpuOnly` |
| `fix_hint` | Wire `validate_deconv_constraints()` into placement_validate.rs. Add A11Legacy family check from knowledge store. |
| `task_ref` | T-P2-01 |

---

### V-011: Knowledge seed missing 70+ CPU-only ops

| Field | Value |
|-------|-------|
| `id` | V-011 |
| `title` | cpu_only_ops_seed.json has ~84 entries vs >=154 in Rust code — 70+ ops missing from seed |
| `domain` | knowledge-store |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `knowledge-gap`, `cpu-only-ops`, `incomplete-seed` |
| `forensic_ref` | N/A |
| `affects` | Knowledge bootstrapping — incomplete CPU-only catalog |
| `reproduce` | 1) Count entries in cpu_only_ops_seed.json (~84); 2) Count CPU_ONLY_OPS in cpu_only_ops.rs (>=154); 3) Observe gap |
| `fix_hint` | Regenerate seed from the Rust code's CPU_ONLY_OPS set. Add all missing ops. |
| `task_ref` | T-P2-04 |

---

### V-014: Zero-filled weight emission on unresolvable MILConst

| Field | Value |
|-------|-------|
| `id` | V-014 |
| `title` | mir_op_to_compat silently emits zero bytes for unresolvable MILConst without error |
| `domain` | emission-correctness |
| `severity` | high |
| `class` | LACUNA |
| `status` | in-progress |
| `labels` | `silent-failure`, `weight-emission`, `zero-fill` |
| `forensic_ref` | N/A |
| `affects` | Emission — models with missing weights emit silently incorrect zero-filled data |
| `reproduce` | 1) Create MILConst with value_path that doesn't resolve; 2) Emit via mir_to_compat; 3) Observe zero-filled weight data without any error |
| `fix_hint` | Return `Err(BridgeError::UnresolvedWeight)` instead of zero-filling. Only `allow_missing_weights` should permit zero-fill. |
| `task_ref` | T-P2-05 |

---

### V-023: Qwen3 defaults applied universally

| Field | Value |
|-------|-------|
| `id` | V-023 |
| `title` | Three locations default to Qwen3-specific values when architecture is unspecified |
| `domain` | model-config |
| `severity` | high |
| `class` | ABERRANT |
| `status` | in-progress |
| `labels` | `qwen3-default`, `model-architecture`, `wrong-default` |
| `forensic_ref` | N/A |
| `affects` | Compilation — non-Qwen3 models get wrong vocab_size, weight patterns, and palettization |
| `reproduce` | 1) Compile a LLaMA model without specifying architecture; 2) Observe Qwen3 weight patterns applied |
| `fix_hint` | Remove Default impl for ModelArchConfig. Make architecture and max_seq_len required parameters. |
| `task_ref` | T-P2-11 |

---

## Medium Issues

### V-026: AIR risk fields are stub values

| Field | Value |
|-------|-------|
| `id` | V-026 |
| `title` | AIR legality_confidence/fallback_risk/drift_risk always hardcoded to ideal values (1.0/0.0/0.0) |
| `domain` | risk-assessment |
| `severity` | medium |
| `class` | STUB-MIMIC |
| `status` | in-progress |
| `labels` | `stub`, `risk-metrics`, `false-certainty` |
| `forensic_ref` | N/A |
| `affects` | Risk assessment — downstream code gets false certainty about ANE placement legality |
| `task_ref` | T-P3-03 |

---

### V-037: Conv channel limit not checked with tensor dims

| Field | Value |
|-------|-------|
| `id` | V-037 |
| `title` | validate_tensor_dims() checks 65536 channel limit but convs need 32768 — separate method easy to miss |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `conv-channels`, `api-design`, `validation-gap` |
| `forensic_ref` | N/A |
| `affects` | Validation — convs with 32K-64K channels pass general validation but fail at ANEC |
| `task_ref` | T-P3-04 |

---

### F-CROSS-01: Cross-constraint combinations not validated

| Field | Value |
|-------|-------|
| `id` | F-CROSS-01 |
| `title` | Binary enforces constraint combinations that MILLer doesn't check (dilation+vector_palettize, aliasing+vector_palettize, shuffle+per-channel_palettize, palettize+large_kernel_stride) |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `cross-constraint`, `binary-verified`, `validation-gap` |
| `forensic_ref` | §6.7 (vector palettization combos), §6.11 (dilation combos) |
| `affects` | Compilation — ops with invalid constraint combinations pass placement but fail at ANEC |
| `task_ref` | T-P3-09 |

---

### F-ARCH-01: Architecture-gated constraints not per-family validated

| Field | Value |
|-------|-------|
| `id` | F-ARCH-01 |
| `title` | Binary contains per-family rejection strings (Softmax on old HW, LRN, depth-axis broadcast, A14 resize) that MILLer doesn't enforce |
| `domain` | constraint-enforcement |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `architecture-gated`, `binary-verified`, `per-family` |
| `forensic_ref` | §6.10 (architecture-gated constraint strings) |
| `affects` | Compilation — ops that are architecture-restricted pass placement on wrong family |
| `task_ref` | T-P3-10 |

---

### F-HAL-01: 9 HAL sub-variants and 7 non-Hxx targets not modeled

| Field | Value |
|-------|-------|
| `id` | F-HAL-01 |
| `title` | Binary has 24 HAL variants (H11–H18 with sub-variants, M9/M11/M12, T0/T1, U1/U2) but MILLer only models 8 base families |
| `domain` | hardware-modeling |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `hal-variants`, `hardware-modeling`, `sub-variant` |
| `forensic_ref` | §5.1 (HAL variant table), §8.1 (AneFamily coverage) |
| `affects` | Compilation — sub-variant-specific constraints may be missed; Mac chip targets not modeled |
| `task_ref` | T-P4-07 |

---

### F-OPS-01: 12 ANEC operations have no MILLer MirOp mapping

| Field | Value |
|-------|-------|
| `id` | F-OPS-01 |
| `title` | Binary exposes 12 genuinely unmapped ANEC operations (broadcast, scaled_elementwise, global_arg_min_max, etc.) |
| `domain` | operation-coverage |
| `severity` | medium |
| `class` | LACUNA |
| `status` | open |
| `labels` | `op-coverage`, `unmapped-ops`, `anec-dialect` |
| `forensic_ref` | §2.3 (ANEC dialect operation table) |
| `affects` | Coverage — models using these operations cannot be compiled through MILLer |
| `task_ref` | T-P4-08 |

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

## Low Issues

### V-051: ne_transpose_c_max never validated

| Field | Value |
|-------|-------|
| `id` | V-051 |
| `title` | AneHwLimits has ne_transpose_c_max field but no validation method exists |
| `domain` | constraint-enforcement |
| `severity` | low |
| `class` | LACUNA |
| `status` | open |
| `labels` | `transpose`, `unvalidated-limit` |
| `task_ref` | T-P3-08 |

---

### V-055: LARGE_KERNEL_THRESHOLD hardcoded

| Field | Value |
|-------|-------|
| `id` | V-055 |
| `title` | LARGE_KERNEL_THRESHOLD=16 hardcoded instead of loaded from knowledge store |
| `domain` | hardcoded-constant |
| `severity` | low |
| `class` | UNVERIFIED |
| `status` | open |
| `labels` | `hardcoded`, `knowledge-store-gap` |
| `task_ref` | T-P4-01 |

---

### V-056: MAX_POOLING_KERNEL_DIM hardcoded

| Field | Value |
|-------|-------|
| `id` | V-056 |
| `title` | MAX_POOLING_KERNEL_DIM=27 hardcoded instead of loaded from knowledge store |
| `domain` | hardcoded-constant |
| `severity` | low |
| `class` | UNVERIFIED |
| `status` | open |
| `labels` | `hardcoded`, `knowledge-store-gap` |
| `task_ref` | T-P4-01 |

---

## MLIR-Method Violations (M-prefix) — OPEN

### M-003: Shape Inference Returns Empty Vec for Unknown Operations
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:401`; `crates/bridge/src/shape_inference.rs:528–531`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** `infer_shape()` returns `Ok(vec![])` as fallback; unknown shapes silently propagate.

### M-005: slanc_scales Pass Inserts Const+Mul with Uncomputed Scale Values
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/passes/src/slanc_scales.rs:63–123`
- **Status:** OPEN | **Remediation:** T-P3-09
- **Description:** Inserts Mul ops with uninitialized scale factors, silently corrupting the graph.

### M-006: shard_plan.rs Falls Back to Dimension=1 with Only Warning
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/shard_plan.rs:239–247`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** Falls back to [1,1,1,1] when shape unavailable, producing wrong PIR specs.

### M-009: MirOpCompat::Unsupported Invisible to Weight Materialization
- **Severity:** HIGH | **Class:** STUB-MIMIC | **Confidence:** HIGH
- **Location:** `crates/coreml-proto/src/lib.rs:1043–1051, 1337`
- **Status:** OPEN | **Remediation:** T-P5-10
- **Description:** Unsupported ops return `input_names(): vec![]`, invisible to weight materialization.

### M-011: AIR legality_confidence Field Not Enforced
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/ir/src/air.rs:890`; SPEC §5.2
- **Status:** OPEN | **Remediation:** T-P5-03, T-P3-03
- **Description:** legality_confidence defaults to 0.0 with no rejection gate; SPEC claims "hard invariant."

### M-012: SPEC Claims SQLite Knowledge Store; Implementation Is JSON
- **Severity:** HIGH | **Class:** DOC-CODE-DRIFT | **Confidence:** HIGH
- **Location:** `SPEC.md:304, 516–520`; `crates/knowledge/src/store.rs:8–10`
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes SQLite; implementation uses JSON with linear-scan queries.

### M-013: Knowledge Seed Files Not Loaded by Any Runtime Crate
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `knowledge/ane_op_family_matrix.json`; `knowledge/palettization_constraints_seed.json`
- **Status:** OPEN | **Remediation:** T-P2-04
- **Description:** Seed files validated in tests but never loaded at runtime; data hardcoded in Rust.

### M-014: MILLinear/MILConv Shape Inference Propagates Input Shape
- **Severity:** HIGH | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:139, 376`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** Propagates input shape instead of computing correct output shape.

### M-015: ANE Constraints Leak into AIR→MIR Lowering
- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:406–443`
- **Status:** OPEN | **Remediation:** T-P5-06
- **Description:** `validate_sdpa_constraints()` enforces ANE-specific constraints during AIR→MIR lowering.

### M-016: Name-Based Dtype Heuristic in mil_lower.rs
- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING | **Confidence:** HIGH
- **Location:** `crates/passes/src/mil_lower.rs:585–598`
- **Status:** OPEN | **Remediation:** T-P5-09
- **Description:** Uses `ends_with("_ids")`, `contains("mask")` for dtype inference instead of type signatures.

### M-017: LegalityRewritePass Is Entirely ANE-Specific
- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING / DIALECT-MISBOUNDARY | **Confidence:** HIGH
- **Location:** `crates/passes/src/legality_rewrite.rs:1–900+`
- **Status:** OPEN | **Remediation:** T-P5-05
- **Description:** Pass name "LegalityRewrite" is misleading — entirely ANE-specific with no target parameter.

### M-018: compat_input_shape Uses name.contains("input_ids") Heuristic
- **Severity:** MEDIUM | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:56–65`
- **Status:** OPEN | **Remediation:** T-P5-09
- **Description:** Implicit semantics derived from node names instead of type signatures.

### M-019: Hardcoded Qwen3-0.6B Defaults in Multiple Locations
- **Severity:** MEDIUM | **Class:** BACKEND-COUPLING | **Confidence:** HIGH
- **Location:** `crates/bridge/src/shape_inference.rs:75–81, 569–580`; `crates/bridge/src/mir_to_compat.rs:169–175`
- **Status:** OPEN | **Remediation:** T-P2-11
- **Description:** Three locations silently default to Qwen3-0.6B parameters.

### M-020: Python Subprocess Bridge Produces Unverifiable Transformations
- **Severity:** MEDIUM | **Class:** DIALECT-MISBOUNDARY | **Confidence:** HIGH
- **Location:** `crates/bridge/src/subprocess.rs:67–177`
- **Status:** OPEN | **Remediation:** Long-term
- **Description:** Python bridge trusts `BridgeResult.status == "success"` as semantic legality without structural verification.

### M-024: StateTopologyPass Validates But Never Transforms
- **Severity:** MEDIUM | **Class:** CANONICALIZATION-MIXUP | **Confidence:** HIGH
- **Location:** `crates/passes/src/state_topology.rs:59–137`
- **Status:** OPEN | **Remediation:** T-P5-12
- **Description:** Pass validates state patterns but always returns `Ok(input)` unchanged — validator masquerading as transform pass.

### M-025: Circular Substitution Chain Produces Partial Resolution
- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/canonicalize.rs:109–135`
- **Status:** OPEN | **Remediation:** Code fix
- **Description:** 100-step limit for circular chains logs warning and returns whatever value it landed on — should be hard error.

### M-027: palettize_weights Silently Defaults Unknown Projections to mlp_bits
- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/passes/src/palettize_weights.rs:199–208`
- **Status:** OPEN | **Remediation:** Code fix
- **Description:** Non-standard architectures silently get wrong quantization bit-widths.

### M-028: ANE-Specific Constraints in Proto Emission Layer
- **Severity:** MEDIUM | **Class:** LAYER-LEAK | **Confidence:** HIGH
- **Location:** `crates/coreml-emit/src/mir_to_proto.rs:91–121, 491–663`
- **Status:** OPEN | **Remediation:** T-P5-07
- **Description:** ANE-specific hardware constraints baked into proto emission layer.

### M-029: Confidence Decay Described in SPEC but Not Wired
- **Severity:** MEDIUM | **Class:** UNSUPPORTED-CLAIM | **Confidence:** HIGH
- **Location:** `SPEC.md:553–554`; `crates/knowledge/src/confidence.rs:13–24`
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes "1% per 30 days" linear decay; code uses exponential decay; neither is operational.

### M-030: Knowledge Pruning Described in SPEC but Not Implemented
- **Severity:** MEDIUM | **Class:** UNSUPPORTED-CLAIM | **Confidence:** HIGH
- **Location:** `SPEC.md:531`; `crates/knowledge/src/` (all files)
- **Status:** OPEN | **Remediation:** T-P5-11
- **Description:** SPEC describes pruning mechanism; no implementation exists.

### M-032: Python MIL Emitter Performs Unverified Graph Construction
- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `python/mil_emitter.py` — all `build_*_program()` functions
- **Status:** OPEN | **Remediation:** Long-term
- **Description:** Python functions construct MIL graphs without verifying constructed graph matches MIR specification.

### M-033: Multi-Zero Reshape Heuristic Assumes Batch=1
- **Severity:** MEDIUM | **Class:** UNVERIFIED-INVARIANT | **Confidence:** HIGH
- **Location:** `crates/bridge/src/mir_to_compat.rs:884–906`
- **Status:** OPEN | **Remediation:** T-P5-04
- **Description:** Sets all but the last zero to `1` (assuming batch), incorrect for batch > 1.

---

## Binary-Research Forensic Findings (N-prefix) — OPEN

### N-002: 35+ ANEC hal_params Not Modeled — 70% of Hardware Constraints Missing
- **Severity:** CRITICAL | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_hw_limits.rs:10–30`
- **Status:** OPEN | **Remediation:** T-P6-01
- **Description:** Only 15 of 50+ hal_params modeled. Missing: kernel depth, padding limits, PE/NE limits, alignment.

### N-003: M1 Hardware Limits Inherit from A17 Despite A14 Family
- **Severity:** HIGH | **Class:** ABERRANT | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_hw_limits.rs:133–135`
- **Status:** OPEN | **Remediation:** T-P6-02
- **Description:** m1() uses ..Self::a17() but M1 is A14-family; may allow oversized tensors.

### N-004: ValidateLayer Equivalent Entirely Absent (Layer 1 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs` (entire file)
- **Status:** OPEN | **Remediation:** T-P6-03
- **Description:** 40+ ANEC ValidateLayer instantiations have no MILLer equivalent.

### N-005: MLIR Placement Dialect Entirely Absent (Layer 2 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no placement dialect module
- **Status:** OPEN | **Remediation:** Long-term infrastructure
- **Description:** No region-based placement, no force-ane-placement, no boundary ops.

### N-006: Fusability Checks Entirely Absent (Layer 4 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no fusability module
- **Status:** OPEN | **Remediation:** T-P6-05
- **Description:** No IsFusable checks; ops may pass placement but fail to fuse into engine layers.

### N-007: Memory Pressure / L2 Legalization Absent (Layer 5 of 5)
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** Entire MILLer codebase — no L2/memory module
- **Status:** OPEN | **Remediation:** T-P6-06
- **Description:** No L2 budget modeling; individually legal ops may collectively exceed limits.

### N-008: Missing AneRevision::Vu1 for uANE
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** MEDIUM
- **Location:** `crates/ir/src/ane_target.rs:200–213`
- **Status:** OPEN | **Remediation:** T-P6-04
- **Description:** uANE variant not modeled; potential constraint differences unknown.

### N-009: MILTile Assigned PE Engine but Tile Has No ANEC Converter
- **Severity:** HIGH | **Class:** PHANTOM-SEMANTIC | **Confidence:** HIGH
- **Location:** `crates/ir/src/mir.rs:1181`
- **Status:** OPEN | **Remediation:** T-P5-07
- **Description:** Tile assigned PE engine but not in CPU_ONLY_OPS; legality_rewrite decomposes it.

### N-010: ne_palette_lut_size_in_bytes Not Modeled
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/ir/src/ane_layout.rs` (entire file)
- **Status:** OPEN | **Remediation:** T-P6-01
- **Description:** No LUT size overflow detection; large palettes may exceed hardware limit.

### N-011: Deconv Validator Exists but Not Wired; Missing 12+ Binary Constraints
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs:589`; `op_constraints.rs:340–396`
- **Status:** OPEN | **Remediation:** T-P2-01
- **Description:** validate_deconv_constraints() checks 5 constraints; binary documents 12+ additional.

### N-012: Conv Kernel Constraints Not Wired into Placement
- **Severity:** HIGH | **Class:** LACUNA | **Confidence:** HIGH
- **Location:** `crates/passes/src/placement_validate.rs` — no MILConv match arm
- **Status:** OPEN | **Remediation:** T-P2-01
- **Description:** validate_conv_constraints() exists but is not called from placement validation for MILConv.

---

## Issue Statistics (Active Only)

| Domain | High | Medium | Low | Total |
|--------|------|--------|-----|-------|
| constraint-enforcement | 1 | 3 | 1 | 5 |
| knowledge-store | 1 | 0 | 0 | 1 |
| emission-correctness | 1 | 0 | 0 | 1 |
| model-config | 1 | 1 | 0 | 2 |
| risk-assessment | 0 | 1 | 0 | 1 |
| hardware-modeling | 0 | 2 | 0 | 2 |
| operation-coverage | 0 | 1 | 0 | 1 |
| hardcoded-constant | 0 | 0 | 2 | 2 |
| MLIR-method (various) | 5 | 9 | 0 | 14 |
| Forensic (various) | 1 | 0 | 0 | 1 |
| **Total** | **10** | **17** | **3** | **30** |
