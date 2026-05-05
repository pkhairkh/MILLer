# MLIR-Method Violation Audit Report

## I. EXECUTIVE ABSTRACT

**Repository audited:** [MILLer](https://github.com/pkhairkh/MILLer) — an open-source compiler targeting Apple's Neural Engine (ANE) via Core ML, implemented in Rust with a Python bridge.

**Reference used:** `MLIR.pdf` (MLIR: Multi-Level Intermediate Representation, LLVM Dev Mtg 2020, Mehdi Amini and River Riddle).

**Audit method:** This audit applies MLIR's compiler-design discipline — dialect separation, verifier obligation, progressive lowering with legality checks, pass-pipeline discipline, target-specific isolation, and diagnostic correctness — as a conceptual lens. It does not require that MILLer literally use the MLIR framework or APIs.

**High-level findings:** MILLer implements a four-level IR stack (SIR → AIR → MIR → PIR) with explicit lowering passes and a knowledge-driven constraint system. The architecture is ambitious and many structural ideas align with MLIR-method principles. However, the audit reveals systematic gaps between the stated architecture and its implementation: verifier gates are absent or permissive across most IR layers, ANE-specific constraints leak into target-independent passes, lowering paths proceed without exhaustive legality contracts, and several passes present themselves as functional compiler logic while actually being stubs or no-ops. The Python bridge introduces an unverifiable transformation boundary, and the SPEC document makes architectural claims not backed by implementation.

**Findings by severity and class:**

| Severity   | Count | Key concern |
|-----------|-------|-------------|
| CRITICAL   | 2     | Silent miscompilation via zero-filled "production" models; MatMul dimension mismatch treated as warning |
| HIGH       | 12    | Missing verifiers, phantom semantics, stub mimics, backend leakage into lowering |
| MEDIUM     | 19    | Misleading boundaries, incomplete pass contracts, doc-code drift, permissive fallbacks |
| LOW        | 11    | Naming, minor structure issues, stale documentation, local cleanup |

| Class                     | Count |
|--------------------------|-------|
| UNVERIFIED-INVARIANT     | 12    |
| PHANTOM-SEMANTIC         | 9     |
| BACKEND-COUPLING         | 8     |
| STUB-MIMIC               | 6     |
| DOC-CODE-DRIFT           | 5     |
| LOWERING-GAP             | 4     |
| DIALECT-MISBOUNDARY      | 3     |
| LAYER-LEAK               | 3     |
| CANONICALIZATION-MIXUP   | 2     |
| UNSUPPORTED-CLAIM        | 5     |

---

## II. MLIR-METHOD AUDIT TABLES

### Table II-A: IR Layer Ownership Expectations

| IR Layer | Stated Ownership | Expected (MLIR method) | Actual | Gap |
|----------|-----------------|----------------------|--------|-----|
| SIR | "Semantic/Task IR — all 167 MIL ops" | Owns operations, types, and invariants for the semantic level | Enum of 167 variants with no verifier, no type constraints on results, no invariant checks | No local invariant enforcement |
| AIR | "ANE-Legal IR — after legality verification" | Owns legality-verified operations with enforced invariants | Enum of 167 variants; `legality_confidence` field defaults to 0.0 with no rejection gate; `staticization_decisions` always empty | Phantom fields, no verification boundary |
| MIR | "MIL-Emission IR — 1:1 mapping to Core ML MIL Builder" | Owns emission-ready operations with type/dtype/shape invariants | Enum of 167 variants; ANE engine assignment hardcoded in `base_engine()`; no verifier that shape/dtype is known before emission | Backend coupling in IR definition |
| PIR | "Package/Deployment IR" | Owns deployment packaging with complete metadata | Struct-based with derived shapes; shape fallback produces `[1,1,1,1]` on missing info | Missing shape invariant enforcement |

### Table II-B: Verifier and Legality Obligations

| Obligation | MLIR-method expectation | MILLer status | Gap |
|-----------|----------------------|---------------|-----|
| Operation-level verifier | Every operation has a `verify()` that checks local invariants | No `verify()` method on SirOp, AirOp, MirOp enums | All invariants are convention-only |
| Type system enforcement | Result types are declared and checked | Result types are not modeled in SIR/AIR; shapes are optional in MIR | No type discipline |
| Shape inference verification | Shapes must be known or explicitly `dynamic` before lowering | `infer_shape()` returns `Ok(vec![])` for unknown ops; `compat_output_shape()` returns `vec![]` for unhandled variants | Unknown shapes silently propagate |
| Dtype legality checking | Dtype legality must be verified before emission | `is_dtype_ane_legal()` returns `Ok(())` for constrained dtypes with deferred "caller must also check" comments | Deferred validation silently skipped |
| Cross-type compatibility | Cross-dtype operations must be explicitly legal | `validate_cross_type_compatibility()` returns `Ok(())` for all cases | No actual validation |

### Table II-C: Pass Responsibility Separation

| Pass | Stated Responsibility | Actual Behavior | Gap |
|------|----------------------|-----------------|-----|
| `CanonicalizePass` | SIR→SIR canonicalization | Resolves substitution chains with 100-step limit and partial resolution; circular chains produce a warning, not an error | Canonicalization silently produces potentially incorrect results |
| `LegalityRewritePass` | SIR→AIR legality rewriting | Entirely ANE-specific; Select→arithmetic decomposition, Tile→Reshape+Mul are ANE-only patterns | Not a general legalizer; name is misleading |
| `MilLowerPass` | AIR→MIR lowering | Mixed with `validate_sdpa_constraints()` (ANE legality check); `infer_shape()` has catch-all `Ok(vec![])` | Legality checks mixed with lowering |
| `StaticizePass` | SIR→SIR staticization | Pure pass-through (`Ok(input)`), deprecated but still present with 1500+ line test suite | Stub mimic consuming pipeline trust |
| `StateTopologyPass` | State topology validation | Returns `Ok(input)` unchanged; "flags" are log::warn/info only | Validator masquerading as transform pass |

### Table II-D: Canonicalization vs. Lowering Boundaries

| Concern | MLIR-method expectation | MILLer status |
|---------|----------------------|---------------|
| Canonicalization within same IR level | Simplifies IR without changing abstraction level | `CanonicalizePass` is the only intra-level pass; it does not clearly separate from lowering concerns |
| Lowering as explicit dialect conversion | Each conversion pattern must be declared; unconverted ops cause failure | AIR→MIR lowering proceeds without exhaustive coverage; new AirOp variants silently produce empty shapes |
| Legality before lowering | Legality is checked in a separate pass; lowering is pure 1:1 mapping | `MilLowerPass` calls `validate_sdpa_constraints()` during lowering; legality and lowering are interleaved |
| Post-lowering verification | Lowered IR is verified against target dialect constraints | No post-MIR verification exists; emission proceeds directly |

### Table II-E: Target-Specific Isolation Expectations

| Concern | MLIR-method expectation | MILLer status |
|---------|----------------------|---------------|
| Target-independent IR layers | No backend-specific attributes, constraints, or types | SIR contains `palette_bits` (ANE palettization); MIR contains `kernel_scale`, `kernel_zero_point`, `kernel_palettized_lut` (ANEC attributes) |
| Engine assignment | Determined by a target-specific placement pass, not hardcoded in IR | `MirOp::base_engine()` hardcodes ANE engine assignment (NE, PE, TransposeEngine, CPU-only) inside the MIR definition |
| Legality rewrite | Parameterized by target; different targets produce different legalizations | `LegalityRewritePass` is entirely ANE-specific with no target parameter |
| Opset version | Derived from target deployment specification | Hardcoded as `"iOS18"` in `DEFAULT_OPSET_VERSION` and in 9 locations in `role_mir.rs` |

### Table II-F: Diagnostic and Failure-Mode Expectations

| Concern | MLIR-method expectation | MILLer status |
|---------|----------------------|---------------|
| Semantic errors must be hard failures | MatMul dimension mismatch, invalid dtype, missing shapes → `Err` | MatMul mismatch produces `eprintln!("[WARN]")` and continues; `shard_plan.rs` falls back to `[1,1,1,1]` shapes with `log::warn!` |
| Unknown shapes must be errors or explicitly dynamic | Return `Err` or `Shape::Dynamic` | `infer_shape()` returns `Ok(vec![])`; `compat_output_shape()` returns `vec![]` meaning "unknown" |
| Missing weights must be errors | Emission must fail if weights are unavailable | `allow_missing_weights=true` with `EmptyWeightResolver` produces zero-filled models; even with a real resolver, missing weights silently become zero-filled Fp16 scalars |
| Python bridge must preserve semantics | Bridge output must be verified against MIR input | Python subprocess success (`status == "success"`) treated as semantic legality; no structural verification of output graph |

---

## III. VIOLATION CATALOGUE

### CRITICAL

#### M-001: Zero-Filled "Production" Models via `allow_missing_weights=true`

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/proto_direct.rs:164–189` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | `emit_mir_graph_proto_direct()` uses `EmptyWeightResolver` (returns `None` for all lookups) with `allow_missing_weights=true`, producing a model where all weights are zero-filled `[1]`-shaped Fp16 scalars. Even `emit_mir_graph_proto_direct_with_resolver()` (the production path) still passes `allow_missing_weights=true` when a real resolver is provided, meaning partially-resolved weights silently become zero-filled placeholders. |
| **Violated principle** | Progressive lowering must preserve meaning; emission success must not be treated as semantic legality |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | CRITICAL |
| **Reference** | Table II-F |

#### M-002: MatMul Inner Dimension Mismatch Treated as Warning

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/mil_lower.rs:92–98` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | When MatMul inner dimensions don't match (lhs_cols ≠ rhs_rows), the code only prints `eprintln!("[WARN]")` and proceeds to produce a result with incorrect output dimensions. A MatMul with mismatched inner dimensions is a hard semantic error that produces garbage. The lowering should `bail!()` instead of continuing. |
| **Violated principle** | Semantic errors must be hard failures; verifier must reject malformed IR |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | CRITICAL |
| **Reference** | Table II-B, Table II-F |

### HIGH

#### M-003: Shape Inference Returns Empty Vec for Unknown Operations

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/mil_lower.rs:401`; `crates/bridge/src/shape_inference.rs:528–531` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | `infer_shape()` returns `Ok(vec![])` as a "conservative fallback" for any AIR op whose shapes are unknown. `compat_output_shape()` has a catch-all `_ => vec![]` for unhandled MirOp variants. Empty shapes propagate through the graph and downstream treats them as "unknown" rather than as errors. There is no exhaustive coverage check ensuring all MirOp variants produce a non-empty shape. |
| **Violated principle** | Progressive lowering must preserve meaning; unknown shapes must be errors or explicitly dynamic |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-B |

#### M-004: `StaticizePass` Is a Pure Pass-Through (Stub Mimic)

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/staticize.rs:52–63` |
| **Class** | STUB-MIMIC |
| **Description** | `StaticizePass::run()` returns `Ok(input)` unchanged. The module documentation states it was "a pure pass-through that consumed a pipeline step while doing nothing, wasting developer trust." Although deprecated, the module still exists with a 1500+ line test suite that verifies the pass-through behavior. |
| **Violated principle** | Pass-pipeline discipline; a pass must either transform or validate, not silently pass through |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-C |

#### M-005: `slanc_scales` Pass Inserts Const+Mul with Uncomputed Scale Values

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/slanc_scales.rs:63–123` |
| **Class** | STUB-MIMIC |
| **Description** | The normalization stabilization pass inserts `Const` + `Mul` ops before RMSNorm nodes, but the actual scale values are not computed — they are marked for "later weight-dependent computation" that never happens within this pass. If no downstream resolver handles these, the inserted Mul ops multiply by uninitialized/zero scale factors, silently corrupting the graph. |
| **Violated principle** | Passes must not produce semantically invalid IR; meaning must be preserved |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-C |

#### M-006: `shard_plan.rs` Falls Back to Dimension=1 with Only a Warning

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/shard_plan.rs:239–247` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | `derive_primary_shapes()` falls back to `batch=1, seq=1, embed=1, vocab=1` when no shape information is available, with only `log::warn!`. These incorrect shapes propagate into PIR `TensorSpec` and `Handoff` structures, producing deployment packages with wrong input/output specifications. |
| **Violated principle** | Missing metadata must be errors, not permissive defaults |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-F |

#### M-007: `dtype_constraints.rs` Returns `Ok(())` for Constrained Dtypes with Deferred Checks

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/dtype_constraints.rs:127–161` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | `is_dtype_ane_legal()` returns `Ok(())` for Int4, UInt4, UInt16, and Bool with comments like "constrained: caller must also check interleave==8" or "caller must also validate op context." These are deferred validations that can be silently skipped by callers who don't perform the additional checks. There is no enforcement mechanism. |
| **Violated principle** | Verifier responsibility; validation must be complete, not deferred to callers |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-B |

#### M-008: `validate_cross_type_compatibility` Is Effectively a No-Op

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/dtype_constraints.rs:440–470` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | The function documents that ANEC rejects BF16/F16 cross-type operations, but the implementation only logs a warning for FP16→FP32 and returns `Ok(())` for all cases. No cross-type operation is ever rejected. |
| **Violated principle** | Verifier must reject illegal operations |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-B |

#### M-009: `MirOpCompat::Unsupported` Is Invisible to Weight Materialization

| Field | Value |
|-------|-------|
| **Location** | `crates/coreml-proto/src/lib.rs:1043–1051, 1337` |
| **Class** | STUB-MIMIC |
| **Description** | `MirOpCompat::Unsupported` stores op metadata but its `input_names()` returns `vec![]`, making these ops invisible to the weight materialization pass. The emission layer correctly rejects Unsupported ops, but the compat layer accepts them silently. If any code path inspects inputs/outputs before emission, Unsupported ops will appear to have no inputs, potentially causing missing weight materialization. |
| **Violated principle** | Dialect must own its operation semantics; unsupported ops must be hard errors, not invisible |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-A |

#### M-010: `kv_cache_rewrite` Generates ANE-Illegal Ops

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/kv_cache_rewrite.rs:1–313` |
| **Class** | STUB-MIMIC |
| **Description** | The module generates `SirOp::Where` ops which are ANE-illegal (no ANE converter). It is deprecated and gated behind `#[cfg(feature = "deprecated-kv-cache-rewrite")]`, but the code still exists and could be accidentally enabled. The SPEC does not document this as a known hazard. |
| **Violated principle** | Passes must not produce target-illegal operations |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-C |

#### M-011: AIR `legality_confidence` Field Not Enforced

| Field | Value |
|-------|-------|
| **Location** | `crates/ir/src/air.rs:890`; SPEC §5.2 |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | `AirNode.legality_confidence` defaults to 0.0 if no knowledge is available, and no downstream pass rejects or flags a 0.0 confidence. The SPEC says this should be a "hard invariant" of AIR, but there is no validation layer enforcing it. Similarly, `AirGraph.staticization_decisions` is always empty because StaticizePass was never implemented. |
| **Violated principle** | Dialect must enforce its invariants; phantom fields violate the dialect contract |
| **Evidence basis** | Source + SPEC |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-A |

#### M-012: SPEC Claims SQLite Knowledge Store; Implementation Is JSON

| Field | Value |
|-------|-------|
| **Location** | `SPEC.md:304, 516–520`; `crates/knowledge/src/store.rs:8–10` |
| **Class** | DOC-CODE-DRIFT |
| **Description** | The SPEC claims "SQLite as the persistence backend" and describes tables, composite indexes, and SQL dumps. The implementation uses JSON files with linear-scan queries. The `docs/knowledge_schema.md` is honest about this, but the SPEC itself is not. |
| **Violated principle** | Documentation must accurately describe implementation |
| **Evidence basis** | Source + Docs + SPEC |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-F |

#### M-013: Knowledge Seed Files Not Loaded by Any Runtime Crate

| Field | Value |
|-------|-------|
| **Location** | `knowledge/ane_op_family_matrix.json`; `knowledge/palettization_constraints_seed.json`; `docs/knowledge_schema.md:141–142` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | Two seed files (`ane_op_family_matrix.json`, `palettization_constraints_seed.json`) are validated in tests but never loaded into any knowledge store or queried by any compiler pass at runtime. The data they contain is instead hardcoded in Rust. Updating the seed files will not change compiler behavior. The `Kir` enum references these as `KnowledgeType` variants, creating the impression they are operational. |
| **Violated principle** | Declarative constraints must be the source of truth, not documentation |
| **Evidence basis** | Source + Docs + Tests |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-B |

#### M-014: `MILLinear`/`MILConv` Shape Inference Propagates Input Shape (Wrong)

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/shape_inference.rs:139, 376` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | Shape inference for `MILLinear` simply propagates the input shape instead of computing `[batch, out_dim]`. `MILConv` has the same issue. These produce incorrect shapes when the MIR node shape is empty, potentially causing downstream reshape failures. |
| **Violated principle** | Progressive lowering must preserve meaning; shape inference must be correct or explicitly fail |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | HIGH |
| **Reference** | Table II-B |

### MEDIUM

#### M-015: ANE Constraints Leak into AIR→MIR Lowering

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/mil_lower.rs:406–443` |
| **Class** | BACKEND-COUPLING |
| **Description** | `validate_sdpa_constraints()` enforces ANE-specific constraints (operand rank ≤ 4) during AIR→MIR lowering. This conflates the lowering step with a legality/placement check. If a different backend were targeted, these constraints would be incorrect. |
| **Violated principle** | Target-specific isolation; lowering must be target-independent |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-E |

#### M-016: Name-Based Dtype Heuristic in mil_lower.rs

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/mil_lower.rs:585–598` |
| **Class** | BACKEND-COUPLING |
| **Description** | When `precision_override` is `None`, the lowering pass uses a name-based heuristic (`ends_with("_ids")`, `contains("mask")`) to determine dtype. This is implicit semantics — operation meaning is derived from node naming conventions, not from type signatures. |
| **Violated principle** | Operations must carry semantics in their type signature, not in naming conventions |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-E |

#### M-017: `LegalityRewritePass` Is Entirely ANE-Specific

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/legality_rewrite.rs:1–900+` |
| **Class** | BACKEND-COUPLING / DIALECT-MISBOUNDARY |
| **Description** | The Select/Where decomposition to arithmetic, the Tile decomposition to Reshape+broadcast Mul, and the knowledge-query-driven legality scoring are all ANE-specific. The pass name "LegalityRewrite" is misleading — it should be "AneLegalRewrite" or accept a target parameter. |
| **Violated principle** | Target-specific isolation; dialect boundaries must be honest |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-E |

#### M-018: `compat_input_shape` Uses `name.contains("input_ids")` Heuristic

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/shape_inference.rs:56–65` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | When the MIR node shape is empty, `compat_input_shape` falls back to `name.contains("input_ids")` to produce `[1, max_seq_len]` and `[1]` for everything else. This is implicit semantics derived from node names. |
| **Violated principle** | Operation semantics must not depend on naming conventions |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-B |

#### M-019: Hardcoded Qwen3-0.6B Defaults in Multiple Locations

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/shape_inference.rs:75–81, 569–580`; `crates/bridge/src/mir_to_compat.rs:169–175` |
| **Class** | BACKEND-COUPLING |
| **Description** | Three locations silently default to Qwen3-0.6B parameters (max_seq_len=32768). `build_input_alias_map()` defaults to `ModelArchitecture::Qwen3` when `architecture` is `None`. These are correctness hazards for non-Qwen3 models. |
| **Violated principle** | Target assumptions must be explicit and parameterized |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-E |

#### M-020: Python Subprocess Bridge Produces Unverifiable Transformations

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/subprocess.rs:67–177` |
| **Class** | DIALECT-MISBOUNDARY |
| **Description** | The Python bridge shells out to a Python script and trusts `BridgeResult.status == "success"` as semantic legality. Success only means the Python process exited with code 0 — not that the emitted model faithfully represents the MIR. There is no structural verification of the output graph against the input. |
| **Violated principle** | Emission success ≠ semantic legality; conversion contracts must be verified |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-F |

#### M-021: `mir_to_proto` Falls Back to Float16/Empty Shape for Missing I/O Descriptors

| Field | Value |
|-------|-------|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:410–434` |
| **Class** | LOWERING-GAP |
| **Description** | When input/output descriptors don't contain a matching entry, the code falls back to `shape: vec![]` and `dtype: CoreMlDataType::Float16`. This is a silent permissive fallback — Core ML may or may not infer correctly, and if it doesn't, the model silently produces wrong results. |
| **Violated principle** | Lowering must not silently guess missing information |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-D |

#### M-022: Weight Dtype Always Assumed Fp16 During Materialization

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/mir_to_compat.rs:252–253` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | When materializing `Const` ops for referenced weights, the code always sets `dtype: MilDtypeCompat::Fp16` regardless of the actual weight data type. The `WeightData` struct carries no dtype field. |
| **Violated principle** | Type mutation must be verified; assuming Fp16 for all weights is unverified type mutation |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-B |

#### M-023: `MILConvTranspose` Placement Skips Existing Deconv Validator

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/placement_validate.rs:589` |
| **Class** | LOWERING-GAP |
| **Description** | `validate_deconv_constraints()` exists and validates 5 ANE-specific deconv constraints, but the placement validator for `MILConvTranspose` returns `AneAllowed` unconditionally. The comment says "deconvolution constraints now enforced" but no constraints are actually checked. |
| **Violated principle** | Legality checks must be wired; placement must verify before allowing |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-B |

#### M-024: `StateTopologyPass` Validates But Never Transforms

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/state_topology.rs:59–137` |
| **Class** | CANONICALIZATION-MIXUP |
| **Description** | The pass validates state read/write patterns but always returns `Ok(input)` unchanged. "Flags" are `log::warn!` / `log::info!` in non-strict mode. The pass is a validator masquerading as a transform pass. |
| **Violated principle** | Pass-pipeline discipline; validators should not be scheduled as transforms |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-C |

#### M-025: Circular Substitution Chain Produces Partial Resolution

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/canonicalize.rs:109–135` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | `resolve_subst_chain()` hits a 100-step limit for circular chains and logs a warning, then returns whatever value it landed on. A circular substitution chain indicates a malformed graph and should be a hard error. |
| **Violated principle** | Malformed IR must be rejected, not partially resolved |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-C |

#### M-026: `static_tables.rs` Depends on Node Insertion Order

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/static_tables.rs:100–106` |
| **Class** | LOWERING-GAP |
| **Description** | The static tables pass prepends Const nodes and relies on LegalityRewritePass processing nodes in order. If canonicalize or any future pass reorders nodes, this silently breaks. |
| **Violated principle** | Pass-pipeline discipline; correctness must not depend on hidden ordering |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-C |

#### M-027: `palettize_weights.rs` Silently Defaults Unknown Projections to mlp_bits

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/palettize_weights.rs:199–208` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | When a LinearProjection's name doesn't match any known attention/MLP pattern, the pass defaults to `mlp_bits` with a warning. For non-standard architectures, this silently assigns wrong quantization bit-widths. |
| **Violated principle** | Missing information must be errors, not silent defaults |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-B |

#### M-028: ANE-Specific Constraints in Proto Emission Layer

| Field | Value |
|-------|-------|
| **Location** | `crates/coreml-emit/src/mir_to_proto.rs:91–121, 491–663` |
| **Class** | LAYER-LEAK |
| **Description** | The proto emission layer includes validation gates that reject ANE-illegal ops (Fill, Select, Where), validate IOSurface sizes, surface uniformity, and flat buffer layout. These are ANE-specific hardware constraints (referencing "Orion" issue numbers) baked into the proto emission layer. |
| **Violated principle** | Target-specific validation belongs in target-specific boundary, not emission |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-E |

#### M-029: Confidence Decay Described in SPEC but Not Wired Into Production

| Field | Value |
|-------|-------|
| **Location** | `SPEC.md:553–554`; `crates/knowledge/src/confidence.rs:13–24` |
| **Class** | UNSUPPORTED-CLAIM |
| **Description** | The SPEC describes "1% per 30 days" linear decay; the implementation uses exponential decay and is test-only. Neither form is wired into the knowledge update pipeline. |
| **Violated principle** | Claimed behavior must be implemented or explicitly marked as future |
| **Evidence basis** | Source + SPEC |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-F |

#### M-030: Knowledge Pruning Described in SPEC but Not Implemented

| Field | Value |
|-------|-------|
| **Location** | `SPEC.md:531`; `crates/knowledge/src/` (all files) |
| **Class** | UNSUPPORTED-CLAIM |
| **Description** | The SPEC describes a pruning mechanism (evidence_count=1, age>90 days, confidence<0.3). No pruning logic exists anywhere in the knowledge crate. |
| **Violated principle** | Claimed behavior must be implemented or explicitly marked as future |
| **Evidence basis** | Source + SPEC |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-F |

#### M-031: StaticizePass Still Referenced in `docs/ir_reference.md`

| Field | Value |
|-------|-------|
| **Location** | `docs/ir_reference.md:66–67` |
| **Class** | DOC-CODE-DRIFT |
| **Description** | The IR reference doc lists StaticizePass in the pipeline ordering, but the pass was deprecated and removed (T-107). |
| **Violated principle** | Documentation must accurately describe the pipeline |
| **Evidence basis** | Docs |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-C |

#### M-032: Python MIL Emitter Performs Unverified Graph Construction

| Field | Value |
|-------|-------|
| **Location** | `python/mil_emitter.py` — all `build_*_program()` functions |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | The Python `build_*_program()` functions construct MIL graphs using coremltools `mb.*` calls without verifying the constructed graph matches the MIR specification. The only "verification" is that coremltools doesn't crash during `ct.convert()`. LUT emission bitwidth is not wired, and the stateless KV-cache variant remains available despite being documented as not the intended path. |
| **Violated principle** | Bridge must preserve semantics; success at the Python layer is not semantic legality |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-F |

#### M-033: Multi-Zero Reshape Heuristic Assumes Batch=1

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/mir_to_compat.rs:884–906` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | When a reshape target has two or more zero placeholders, the heuristic sets all but the last zero to `1` (assuming batch dimension). This is incorrect for batch > 1, which is valid for inference. |
| **Violated principle** | Shape inference must be correct or explicitly fail |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | MEDIUM |
| **Reference** | Table II-B |

### LOW

#### M-034: `op_constraints.rs` Validates ANEC-Specific Attribute Shapes

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/op_constraints.rs:943–1100+` |
| **Class** | BACKEND-COUPLING |
| **Description** | `validate_anec_attribute_shapes()` validates conv/pool/deconv attribute shapes against ANEC compiler expectations. This is ANE-backend-specific validation in the general passes crate. |
| **Violated principle** | Target-specific validation belongs in target-specific modules |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-E |

#### M-035: `role_mir.rs` Hardcodes "iOS18" Opset Version

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/role_mir.rs:218, 315, 397, 509, 600, 683, 747, 787, 847` |
| **Class** | BACKEND-COUPLING |
| **Description** | Every `MirGraph` produced by `RoleMirBuilder::build_mir()` hardcodes `opset_version: "iOS18"`. Should be configurable or derived from target deployment spec. |
| **Violated principle** | Target assumptions must be parameterized |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-E |

#### M-036: `risk_annotate.rs` Uses "unknown" Catch-All

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/risk_annotate.rs:115–116` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | The `_ => "unknown"` catch-all for AIR op patterns means any new or unrecognized variant gets default low risk scores, bypassing risk annotation. |
| **Violated principle** | New operations should be flagged, not silently accepted |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-B |

#### M-037: `precision_policy.rs` Uses `Misc_{VariantName}` Catch-All

| Field | Value |
|-------|-------|
| **Location** | `crates/passes/src/precision_policy.rs:339–343` |
| **Class** | UNVERIFIED-INVARIANT |
| **Description** | New SirOp variants bypass precision hazard checking via `"Misc_*"` patterns that the knowledge store is unlikely to have entries for. |
| **Violated principle** | New operations should be flagged for review |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-B |

#### M-038: `coreml-ffi` Crate Is Entirely Stub

| Field | Value |
|-------|-------|
| **Location** | `crates/coreml-ffi/src/api.rs`, `capi.rs`, `model.rs` |
| **Class** | STUB-MIMIC |
| **Description** | The FFI crate returns `"unknown"` for version, `Err("Not implemented")` for compile_model, and `Ok(Self { handle: None })` for load. It cannot validate any emitted model. |
| **Violated principle** | Stub code must not present itself as functional |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-C |

#### M-039: `validate_proto_direct_package` Only Checks Filesystem Structure

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/proto_direct.rs:200–304` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | Package validation checks that files exist and are non-empty, but does not validate protobuf content, weight format, or I/O shape/dtype consistency. |
| **Violated principle** | Validation must be semantic, not just structural |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-F |

#### M-040: `MILIdentity` Hardcodes Fp16 Before Override

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/mir_to_compat.rs:1264–1268` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | `mir_op_to_compat()` for MILIdentity creates `Identity { dtype: Fp16 }` with a comment "will be overridden by mir_node_to_compat". If `mir_op_to_compat()` is called directly, the Fp16 default is used without correction. |
| **Violated principle** | Type must be derived from IR, not hardcoded and patched |
| **Evidence basis** | Source |
| **Confidence** | MEDIUM |
| **Severity** | LOW |
| **Reference** | Table II-B |

#### M-041: `"__placeholder__"` Magic String in Shape Inference

| Field | Value |
|-------|-------|
| **Location** | `crates/bridge/src/shape_inference.rs:427` |
| **Class** | BACKEND-COUPLING |
| **Description** | `MILIdentity { x, .. } if x.0 == "__placeholder__"` special-cases a magic string to produce `[1, max_seq_len]`. This is implicit semantics via node naming. |
| **Violated principle** | Operation semantics must not depend on naming conventions |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-B |

#### M-042: Python `bridge.py` Reports Median as `mean_ms`

| Field | Value |
|-------|-------|
| **Location** | `python/bridge.py:497` |
| **Class** | PHANTOM-SEMANTIC |
| **Description** | The `mean_ms` field is computed as the median, not the actual mean. `std_dev_ms` is hardcoded to `0.0`. The field names promise statistical properties the data does not have. |
| **Violated principle** | Field names must accurately represent their content |
| **Evidence basis** | Source |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-F |

#### M-043: Cross-Validated Confidence Value Differs from SPEC

| Field | Value |
|-------|-------|
| **Location** | `SPEC.md:545`; `crates/knowledge/src/update.rs:275` |
| **Class** | DOC-CODE-DRIFT |
| **Description** | The SPEC claims CrossValidated initial confidence is 0.85; the implementation computes 0.6. |
| **Violated principle** | Documentation must match implementation |
| **Evidence basis** | Source + SPEC |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-F |

#### M-044: README Claims 1270 Tests; CHANGELOG Claims 1651

| Field | Value |
|-------|-------|
| **Location** | `README.md:18`; `CHANGELOG.md:5` |
| **Class** | DOC-CODE-DRIFT |
| **Description** | The README was not updated after subsequent sprint cycles added ~380 tests. |
| **Violated principle** | Documentation must be kept current |
| **Evidence basis** | Docs |
| **Confidence** | HIGH |
| **Severity** | LOW |
| **Reference** | Table II-F |

---

## IV. ARCHITECTURAL MISCONCEPTIONS

The following recurring misconception patterns are evidenced by the repository:

### 1. Treating Naming Conventions as Dialect Boundaries

MILLer's SIR, AIR, and MIR are separate Rust enums with separate node ID types, but there are no verifier boundaries between them. The "boundary" between SIR and AIR is enforced only by the pass pipeline order — no structural invariant prevents AIR ops from appearing in an SIR graph or vice versa. The dialects share types (MilDtype), and operations in one layer can reference operations in another via node ID strings. This is a naming convention, not a dialect boundary.

**Evidence:** M-011, M-017, M-020.

### 2. Treating Pass Order as a Verifier Substitute

Multiple findings show that correctness depends on specific pass ordering without that dependency being declared or enforced. `static_tables.rs` prepends Const nodes and relies on `LegalityRewritePass` processing them in order. The `CanonicalizePass` must run before `LegalityRewritePass` for substitution resolution to work. No pass declares its dependencies.

**Evidence:** M-026, M-025.

### 3. Treating Emission Success as Semantic Legality

The Python bridge treats `BridgeResult.status == "success"` as proof that the output faithfully represents the input MIR. The proto emission layer produces a file and declares success even when shapes are empty and dtypes are defaulted. The `validate_proto_direct_package` function checks only filesystem structure. At no point in the pipeline is there a semantic verification step that confirms the emitted model faithfully represents the compiler's IR.

**Evidence:** M-001, M-020, M-039.

### 4. Mixing Operation Semantics with Target Descriptor Layout

SIR operations carry ANE-specific attributes (`palette_bits` on `LinearProjection` and `Const`). MIR operations carry ANEC attributes (`kernel_scale`, `kernel_zero_point`, `kernel_palettized_lut` on `MILConv`). The `MirOp::base_engine()` method hardcodes ANE engine assignments directly in the IR definition. These are target-specific details that belong in a target-specific mapping layer, not in the target-independent IR.

**Evidence:** M-015, M-016, M-017, M-028, M-034, M-035.

### 5. Using Documentation as Implementation Evidence

The SPEC describes SQLite storage, confidence decay, knowledge pruning, and staticization decisions that are not implemented. The `docs/ir_reference.md` describes a pipeline with StaticizePass that no longer runs. The `Kir` enum references knowledge types (`PalettizationConstraints`, `AneOpFamilyMatrix`) that are not loaded at runtime. The documentation creates the appearance of features that do not exist.

**Evidence:** M-012, M-013, M-029, M-030, M-031, M-043, M-044.

### 6. Allowing Fallback Paths to Mask Missing Lowering

The pipeline has multiple fallback paths: `infer_shape()` returns `Ok(vec![])` for unknown ops, `compat_output_shape()` returns `vec![]` for unhandled MirOp variants, `is_dtype_ane_legal()` returns `Ok(())` for constrained dtypes with deferred checks, `shard_plan.rs` defaults to `[1,1,1,1]` shapes, `mir_to_proto.rs` defaults to Float16 for missing I/O descriptors, and `palettize_weights.rs` defaults to `mlp_bits` for unknown projections. Each fallback silently produces metadata that is probably wrong, and the pipeline continues as if it were correct.

**Evidence:** M-003, M-006, M-007, M-014, M-021, M-027, M-033.

### 7. Using Ad-Hoc Shape/Type Checks Instead of Centralized Verifier Logic

Shape inference is scattered across `mil_lower.rs::infer_shape()`, `bridge/shape_inference.rs::compat_output_shape()`, `bridge/mir_to_compat.rs::resolve_reshape_shape()`, and the Python `mil_emitter.py`. Dtype legality is scattered across `dtype_constraints.rs::is_dtype_ane_legal()`, `mil_lower.rs` name-based heuristics, and `mir_to_compat.rs` Fp16 defaults. There is no single verifier that checks all invariants before a lowering or emission step.

**Evidence:** M-003, M-014, M-016, M-018, M-022, M-033.

---

## V. REMEDIATION ROADMAP

### Phase 1: Critical and High-Severity Fixes

1. **Make `allow_missing_weights=false` the default when a real resolver is provided.** Change `emit_mir_graph_proto_direct_with_resolver()` to require explicit opt-in for missing weights. The zero-fill path should only be available through a separate `emit_for_testing()` function.

2. **Convert MatMul dimension mismatch from warning to hard error.** Replace `eprintln!("[WARN]")` with `anyhow::bail!()` in `mil_lower.rs:92–98`. Add a test that verifies the error is produced.

3. **Add per-IR-layer verifiers.** Implement `SirGraph::verify()`, `AirGraph::verify()`, `MirGraph::verify()` methods that check local invariants (node reference integrity, required fields, dtype legality, shape non-emptiness where required). Call these after each pass.

4. **Remove or clearly quarantine `StaticizePass`.** Delete the module or gate it behind a `#[cfg(test)]` attribute. Remove the 1500+ line test suite that verifies pass-through behavior. Update `docs/ir_reference.md`.

5. **Replace empty-shape fallbacks with explicit errors or `Dynamic` markers.** `infer_shape()` and `compat_output_shape()` should return `Err` for unhandled variants or return a `Shape::Dynamic` type that downstream must explicitly handle.

6. **Wire `validate_deconv_constraints()` into the ConvTranspose placement check.** The validator exists; it just needs to be called.

7. **Remove `legality_confidence` default-to-zero or add a rejection gate.** Either remove the field (since it is not enforced) or add a post-AIR verification step that rejects nodes with `legality_confidence == 0.0` in strict mode.

### Phase 2: Boundary and Separation Fixes

8. **Rename `LegalityRewritePass` to `AneLegalityRewritePass`.** Or add a target parameter so the same pass framework can produce different legalizations for different backends.

9. **Move ANE-specific validation out of `mil_lower.rs`.** `validate_sdpa_constraints()` belongs in `placement_validate.rs` or a new `ane_legality_check.rs`. The lowering pass should be a pure AIR→MIR mapping.

10. **Move ANE engine assignment out of `MirOp::base_engine()`.** Create a separate `ane_placement.rs` module that maps MirOp variants to engines, parameterized by AneFamily. The MIR definition should be target-independent.

11. **Remove ANE-specific attributes from SIR and MIR IR definitions.** `palette_bits`, `kernel_scale`, `kernel_zero_point`, and `kernel_palettized_lut` should be added during target-specific lowering, not carried in the target-independent IR.

12. **Add dtype field to `WeightData`.** Weight materialization should not assume Fp16.

13. **Fix `MILLinear` and `MILConv` shape inference.** Compute the correct output shape from weight dimensions, or return an error.

14. **Replace name-based heuristics with explicit shape/dtype annotations.** The `name.contains("input_ids")` heuristic in `compat_input_shape()` and the `ends_with("_ids")` heuristic in `mil_lower.rs` should be replaced with dtype/shape fields carried from SIR through AIR to MIR.

### Phase 3: Completeness and Diagnostic Fixes

15. **Add post-lowering verification.** After AIR→MIR lowering, run a verification pass that checks every MIR node has a known shape and dtype. Before proto emission, run a verification pass that checks every node is legal for the target.

16. **Add semantic verification to the Python bridge.** After `build_*_program()`, validate the constructed program's op set against the ANE-legal op set before passing to `ct.convert()`.

17. **Convert `log::warn!` on invariant violations to `anyhow::bail!()`.** The `shard_plan.rs` dimension fallback, the canonicalize circular chain, and the palettize unknown projection default should all be hard errors.

18. **Update SPEC.md** to reflect JSON file-backed store (not SQLite), mark confidence decay and pruning as "not yet implemented," and update the CrossValidated confidence value from 0.85 to 0.6.

19. **Wire `ane_op_family_matrix.json` and `palettization_constraints_seed.json` into the knowledge store loader**, or remove the `KnowledgeType` enum variants and mark them as aspirational in the schema documentation.

20. **Add regression tests** for each HIGH/CRITICAL violation: zero-filled model detection, MatMul mismatch rejection, empty-shape rejection, ConvTranspose deconv constraint enforcement, and `allow_missing_weights` default behavior.

---

## VI. AUDIT NOTE

- `MLIR.pdf` was used as a local conceptual reference for compiler-design discipline. It was not used as a framework requirement.
- Local notes are stored in `references/`, including `mlir-method-summary.md` and `MLIR_full.txt`.
- `references/` and `MLIR.pdf` are excluded from the repository by `.gitignore`.
- Only `MLIRVIOLATIONS.md` and the updated `.gitignore` are intended for commit.
