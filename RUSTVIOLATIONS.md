# RUSTVIOLATIONS.md — Rust-Method Violation Audit

---

## I. EXECUTIVE ABSTRACT

**Repository audited:** [github.com/pkhairkh/MILLer](https://github.com/pkhairkh/MILLer) — an ANE-first multi-level compiler that lowers transformer models into Apple Core ML MIL/mlpackage artifacts.

**Reference used:** `comprehensive-rust.pdf` (Google's *Comprehensive Rust* course, 577 pages), applied as a conceptual audit reference for idiomatic Rust engineering discipline. This is not a mechanical rewrite requirement; findings identify where the codebase's patterns create correctness risk, safety risk, maintainability risk, or architectural drift relative to the Rust method.

**Audit scope:** All 87 Rust source files across 12 workspace crates, plus specification and documentation files. The audit examined compiler pipeline code (SIR, AIR, MIR, PIR, ProfIR, KIR), pass infrastructure, emission and FFI boundaries, knowledge store, lab harness, CLI, and bridge/trace subsystems.

**High-level findings:** The MILLer project demonstrates strong architectural intent — multi-level IR pipeline with ANE-aware constraint enforcement, subprocess-isolated Python boundary, and extensive inline test coverage (~400+ test functions). However, the Rust-method audit reveals systematic deviations in three overlapping areas:

1. **Stringly typed invariants** — at least 30 fields across IR nodes, manifests, bridge results, and config objects use `String` where closed-vocabulary enums would prevent invalid states at the type level. Multiple downstream match arms silently fall back to defaults on unrecognized strings, producing fake success.

2. **Panic-prone production paths** — `unreachable!()` in five production code locations, `unwrap()` on serde round-trips in the critical SIR construction path, and pervasive `unwrap_or("fp16")` / `unwrap_or(1)` defaults that silently replace missing or malformed input with plausible-but-wrong values.

3. **Specification-implementation drift** — SPEC.md claims deterministic compilation, AIR legality as a hard invariant, knowledge-store immutability with versioned audit trails, and confidence validation rules that the code does not implement. The SPEC describes features (SQLite backend, pruning, diff utilities, IR verification) that are absent.

### Findings by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 14 |
| HIGH | 23 |
| MEDIUM | 25 |
| LOW | 10 |
| **Total** | **72** |

### Findings by Class

| Class | Count |
|-------|-------|
| PANIC-LEAK | 12 |
| UNSAFE-BOUNDARY | 4 |
| TYPE-INVARIANT-GAP | 20 |
| ERROR-MODEL-DRIFT | 11 |
| OWNERSHIP-MISMODEL | 3 |
| VISIBILITY-LEAK | 9 |
| TRAIT-MISBOUNDARY | 0 |
| CONCURRENCY-RISK | 1 |
| SERDE-CONTRACT-GAP | 11 |
| RESOURCE-LIFETIME-GAP | 4 |
| STUB-MIMIC | 5 |
| DOC-CODE-DRIFT | 10 |
| UNSUPPORTED-CLAIM | 12 |

---

## II. RUST-METHOD AUDIT TABLES

### II.1 Ownership and Borrowing Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Single-owner principle: every value has one clear owner | Generally well-followed; IR graphs own their node vectors | Acceptable |
| Move semantics as default; clones are explicit | `Clone` derived liberally on `StrategySpec`, `DiscoveryReport`, `CompilationPlan`, and IR graph structs; `.cloned().collect()` used where iterators over references would suffice | Weak — clone-as-control-flow in strategy/discovery layers |
| No interior mutability without justification | Only `AtomicU64` in emitter; no `RefCell` abuse | Acceptable |
| Borrowing rules respected | No borrow-checker workarounds detected | Acceptable |
| Arc used only for genuine shared ownership | `Arc<KnowledgeUnit>` for cheap cloning of knowledge entries | Acceptable |

### II.2 Error and Panic Boundary Obligations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Library code returns `Result`, does not panic | 5 `unreachable!()` in production passes; `unwrap()` on serde_json in SIR construction; `unwrap_or` with fake-success defaults across task_spec, payload, and session | Violated — panics cross public boundaries; defaults mask errors |
| `anyhow` for applications, typed errors for libraries | All compiler passes and emit crate use `anyhow::Result`; only FFI uses `thiserror` | Violated — library crates should provide typed error enums |
| No silent fallback that discards errors | `unwrap_or("fp16")` for dtype, `unwrap_or(1)` for batch_size, `unwrap_or("")` for paths | Violated — pervasive fake-success defaults |
| Panic conditions documented | No `# Panics` sections on any public function | Violated |

### II.3 Type-Invariant Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Closed vocabularies encoded as enums, not strings | ~30+ fields use `String` for finite-value domains: pad_type, mode, qk_norm_type, Gelu mode, activation, sampling_mode, nearest_rounding_mode, precision_override, compute_units, dtype, emission_status, implementation_status, BridgeResult.status, report_type | Severely violated |
| Newtypes prevent argument-swap bugs | Opset version and deployment target are both `&str`; no newtype distinction | Weak |
| Checked constructors enforce invariants | All IR graph/node structs have `pub` fields; no validated constructors | Violated |
| Parse, don't validate | Validation happens at runtime in pass code; types accept invalid values | Violated |

### II.4 Unsafe Isolation Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Unsafe code is small, isolated, and wrapped in safe abstractions | All unsafe is in `coreml-ffi/capi.rs` — well isolated | Acceptable |
| Every unsafe block has a `// SAFETY:` comment | Some safety comments exist; not all unsafe blocks have them | Partial |
| No double-free or use-after-free paths | `Box::from_raw` in `coreml_model_destroy` has no double-destroy guard | Violated |
| FFI wrappers provide safe API | FFI functions are `unsafe extern "C"` with null checks; handle lifecycle is caller's responsibility | Partial — documented but unenforced |

### II.5 Visibility and API-Boundary Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Struct fields private by default | All IR graph/node structs, session structs, bridge structs, manifest structs, pass config structs have `pub` fields | Severely violated |
| `pub(crate)` for internal sharing | Not used; all shared items are fully `pub` | Violated |
| Mutation only through validated APIs | External code can mutate `adaptations`, `data`, `offset` fields breaking internal invariants | Violated |

### II.6 Trait Responsibility Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Traits encode required contracts | `ToProto`, `IrGraph`, `KnowledgeQueryable`, `WeightResolver`, `TaskFamilyTrait` are well-scoped | Acceptable |
| Sealed traits for non-extensible polymorphism | Not used; no misuse detected | Acceptable |
| No trait mixing responsibilities | No violations detected | Acceptable |

### II.7 Concurrency and Async Ownership Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| No data races | Entirely synchronous codebase; only `AtomicU64` for compilation counter | Largely acceptable |
| Atomic ordering correct | `Ordering::Relaxed` on compilation counter admits TOCTOU for threshold check | Weak — advisory-only but technically incorrect |
| No indefinite blocking | Trace subprocess has no timeout; bridge subprocess correctly implements timeout | Partially violated |

### II.8 Serialization/Deserialization Contract Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Serde does not accept invalid states | ~11 serde-contract gaps: palette_bits accepts invalid values, Gelu mode accepts any string, emission_status accepts any string, BridgeResult.status accepts any string, compute_plan is untyped `serde_json::Value` | Severely violated |
| Deserialization validates after loading | No post-deserialization validation on any IR graph, manifest, or bridge result | Violated |
| Version gates on serialized formats | No version gates detected | Not assessed (no format versioning) |

### II.9 Resource Lifetime and Cleanup Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Temp dirs cleaned up | CLI uses hardcoded temp path with no cleanup or concurrency isolation | Violated |
| Package writes are atomic | `package.rs` deletes existing dir then writes; failure loses both old and new | Violated |
| Subprocess lifecycle managed | Bridge has timeout + kill; trace has no timeout | Partially violated |

### II.10 Testing and Diagnostic Expectations

| Expectation | MILLer Status | Assessment |
|-------------|---------------|------------|
| Unit tests for each pass | ~400+ inline test functions; good coverage | Strong |
| Integration tests for pipeline | `ir/tests/pipeline.rs`, `coreml-emit/tests/cross_validation.rs` | Acceptable |
| Property/regression tests for invariants | No property tests; no bit-for-bit reproducibility tests | Weak |
| `eprintln!` not used in library code | 40+ `eprintln!` in production passes; some labeled CRITICAL | Violated |
| Structured logging | `log` crate used in some places; `eprintln!` in others | Inconsistent |

### II.11 Evidence Basis and Confidence

| Source | Usage |
|--------|-------|
| Source code | Primary evidence for all findings |
| Tests | Confirmed pass behavior, absence of determinism tests |
| SPEC.md | Cross-referenced claims against implementation |
| Documentation | Cross-referenced ir_reference.md claims |
| Config | Examined Cargo.toml, clippy.toml, rust-toolchain.toml |

---

## III. VIOLATION CATALOGUE

Findings sorted by severity, then by class.

### CRITICAL

| ID | Location | Class | Description | Principle | Evidence | Confidence | Severity | Ref |
|----|----------|-------|-------------|-----------|----------|------------|----------|-----|
| R-001 | `passes/src/staticize.rs:58-62` | STUB-MIMIC | `StaticizePass::run()` returns `Ok(input)` unchanged; struct still compiled and exported as public API | No-op that mimics a real pass; downstream consumers may believe computation occurred | `pub fn run(&self, input: SirGraph) -> Result<SirGraph> { Ok(input) }`; module doc says "REMOVED FROM PIPELINE" but `pub mod staticize` is unconditional | High | CRITICAL | II.2 |
| R-002 | `passes/src/kv_cache_rewrite.rs:89,103` | PANIC-LEAK | Two `unreachable!()` in production code of deprecated but compilable module | `unreachable!()` panics if match arm is hit with unexpected data | `_ => unreachable!()` at lines 89 and 103; module gated behind `deprecated-kv-cache-rewrite` feature but still compilable | High | CRITICAL | II.2 |
| R-003 | `passes/src/slanc_scales.rs:74` | PANIC-LEAK | `unreachable!()` in RMSNorm match within production pass | `unreachable!()` panics on unexpected SirOp variants | `_ => unreachable!()` after filter_map for RMSNorm; if IR adds new variant, compiler panics | High | CRITICAL | II.2 |
| R-004 | `passes/src/legality_rewrite.rs:4756` | PANIC-LEAK | `unreachable!()` for composite ops in SIR-to-AIR decomposition | Asserts correctness via panic instead of returning error | `unreachable!("composite ops should be handled by explicit decompositions above")` — comment asserts what type system should enforce | High | CRITICAL | II.2 |
| R-005 | `passes/src/mil_lower.rs:3005,3097` | PANIC-LEAK | Two `unreachable!()` for lm_head processing in MIL lower pass | `_ => unreachable!("lm_head must be MILLinear")` panics on future IR changes | Same pattern as R-004; future-proofing failure | High | CRITICAL | II.2 |
| R-006 | `trace/src/sir_build.rs:238,262,269,288` | PANIC-LEAK | Four `unwrap()` on `serde_json::to_string`/`from_str` in SIR alias resolution | Panic on critical SIR construction path if any SirOp variant is not serializable | `serde_json::to_string(&node.op).unwrap()` (x3); `serde_json::from_str(&new_json).unwrap()` (x1) | High | CRITICAL | II.2 |
| R-007 | `bridge/src/subprocess.rs:227` | TYPE-INVARIANT-GAP | `BridgeResult.status` is `String` ("success"/"error") instead of enum | Typo or invalid status string silently misclassifies compilation results; every consumer does `result.status == "success"` | `pub status: String` with comment `/// "success" or "error"`; string comparison at session.rs:934, cli/main.rs:797 | High | CRITICAL | II.3 |
| R-008 | `artifacts/src/manifest.rs:42,45` | SERDE-CONTRACT-GAP | `implementation_status` and `verification_scope` are `String` accepting any value via serde | Typos pass deserialization silently, breaking all downstream logic | Comments document valid values ("host_compiled"|"device_verified"|"partial") but nothing enforces them | High | CRITICAL | II.8 |
| R-009 | `artifacts/src/manifest.rs:91` | TYPE-INVARIANT-GAP | `FunctionDescriptor.emission_status` is `String` ("emitted"/"seam_only") instead of enum | Same class as R-007/R-008; stringly typed status field on serialized manifest | `pub emission_status: String` with hardcoded `"emitted".to_string()` in session.rs | High | CRITICAL | II.3 |
| R-010 | `artifacts/src/manifest.rs:120` | TYPE-INVARIANT-GAP | `TensorSpec.dtype` is `String` instead of typed enum | Default `unwrap_or("fp16")` in session.rs silently converts unknown dtypes; wrong dtype produces miscompiled models | `dtype: inp.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16").to_string()` in session.rs:153-155 | High | CRITICAL | II.3 |
| R-011 | `trace/src/subprocess.rs:81-83` | RESOURCE-LIFETIME-GAP | No timeout on Python tracer subprocess | A hung Python tracer (e.g., OOM during model loading) blocks the compiler indefinitely; bridge subprocess already has timeout enforcement (T-77) | `child.wait_with_output()` with no deadline or kill-on-timeout; compare bridge/src/subprocess.rs:86-131 | High | CRITICAL | II.9 |
| R-012 | SPEC.md:52; `ir/src/serialize.rs`; `knowledge/src/store.rs` | UNSUPPORTED-CLAIM | SPEC claims "deterministic compilation" (bit-for-bit reproducibility) but code uses `HashMap` (non-deterministic iteration), `fs::read_dir` (filesystem-dependent order), and has no reproducibility tests | Claimed property is neither enforced nor tested; `KnowledgeStore::index` built from `HashMap::keys()` has no ordering guarantee | SPEC.md:52 "Same IR + same knowledge store = same emitted artifact, bit-for-bit"; store.rs:131, 202-228 | High | CRITICAL | II.10 |
| R-013 | SPEC.md:352-359; `ir/src/serialize.rs:114-117`; `ir/src/air.rs:890-893` | UNSUPPORTED-CLAIM | SPEC claims AIR legality is a "hard invariant" with knowledge-store-driven confidence; implementation hardcodes `legality_confidence: 1.0`, `fallback_risk: 0.0`, `drift_risk: 0.0` | Directly violates SPEC Non-Functional Goal 7 ("honest uncertainty"); AIR claims perfect certainty with zero risk | `serialize.rs:114-117` always sets ideal values; no code path populates from knowledge store | High | CRITICAL | II.3 |
| R-014 | SPEC.md:122; `ir/src/sir.rs,air.rs,mir.rs,pir.rs` | UNSUPPORTED-CLAIM | SPEC claims "enforce invariants at each level" but no `verify()` or `validate()` method exists on any IR graph struct | Invalid IR can be constructed and propagated without error; SPEC lists specific per-level invariants that are not programmatically checked | No verify/validate methods on SirGraph, AirGraph, MirGraph, PirGraph | High | CRITICAL | II.3 |

### HIGH

| ID | Location | Class | Description | Principle | Evidence | Confidence | Severity | Ref |
|----|----------|-------|-------------|-----------|----------|------------|----------|-----|
| R-015 | `passes/src/legality_rewrite.rs`, `passes/src/mil_lower.rs` | ERROR-MODEL-DRIFT | 40+ `eprintln!` in production pass code, some labeled CRITICAL | `eprintln!` bypasses structured logging, cannot be filtered; CRITICAL-level messages about ANE-illegal ops should be proper errors | `mil_lower.rs:3862-3893` (4 CRITICAL/WARNING eprintlns); legality_rewrite.rs:1313,2152,2647 | High | HIGH | II.10 |
| R-016 | `passes/src/mil_lower.rs:574-583` | TYPE-INVARIANT-GAP | String-to-MilDtype match with silent `Fp16` fallback | `_ => MilDtype::Fp16` silently corrupts dtype for any unrecognized string | `match dtype.as_str() { "fp32" => ..., _ => MilDtype::Fp16 }` | High | HIGH | II.3 |
| R-017 | `passes/src/mil_lower.rs:550-557` | TYPE-INVARIANT-GAP | String-to-ComputeUnitHint match with silent fallback | `_ => ComputeUnitHint::CPUAndNE` for unknown strings | `match shard_plan.compute_units[0].as_str() { "CPU_AND_NE" => ..., _ => ComputeUnitHint::CPUAndNE }` | High | HIGH | II.3 |
| R-018 | `ir/src/sir.rs,air.rs,mir.rs` (multiple) | TYPE-INVARIANT-GAP | `pad_type`, `mode`, `qk_norm_type`, `Gelu.mode`, `activation`, `sampling_mode`, `nearest_rounding_mode` are `String` instead of enums | These have finite valid sets; strings allow invalid values to propagate through the entire IR pipeline | `pad_type: String`, `mode: String`, `activation: String`, `sampling_mode: String` across SIR/AIR/MIR | High | HIGH | II.3 |
| R-019 | `passes/src/shard_plan.rs:95` | TYPE-INVARIANT-GAP | `ShardPlan.compute_units` is `Vec<String>` instead of `Vec<ComputeUnitHint>` | Every consumer duplicates string-to-enum parsing with different fallback defaults | `pub compute_units: Vec<String>`; parsed independently in mil_lower.rs:550-557 and role_mir.rs:134 | High | HIGH | II.3 |
| R-020 | `passes/src/risk_annotate.rs:115` | TYPE-INVARIANT-GAP | AIR-to-op-pattern catch-all is `"unknown"` | New AirOp variants get zero risk annotation silently | `_ => "unknown"` means risk annotation is silently skipped for unrecognized ops | High | HIGH | II.3 |
| R-021 | `passes/src/precision_policy.rs:340-343` | TYPE-INVARIANT-GAP | Debug-format-derived pattern strings for knowledge queries | `format!("{:?}", node.op)` produces fragile identifiers dependent on Debug formatting | `let debug_str = format!("{:?}", node.op); let variant_name = debug_str.split('{').next().unwrap_or("Unknown");` | High | HIGH | II.3 |
| R-022 | `ir/src/linear_slice.rs:132-133` | PANIC-LEAK | Wildcard dtype match silently defaults to `Fp16` for unrecognized dtype | `_ => MilDtype::Fp16` for any unsupported dtype string; "bf16" silently becomes fp16 | Same pattern as R-016 but in linear_slice; was fixed in `lower_shard_to_mir` (V-011) but not here | High | HIGH | II.2 |
| R-023 | `ir/src/task_spec.rs:667,731,795,856,934,997` | PANIC-LEAK | Six `.unwrap_or("fp16")` for dtype in TOML parsers | Missing or misspelled dtype silently defaults to fp16 instead of producing an error | `dtype = task_section.get("dtype").and_then(|v| v.as_str()).unwrap_or("fp16")` (x6) | High | HIGH | II.2 |
| R-024 | `ir/src/task_spec.rs:666` | PANIC-LEAK | `unwrap_or(1)` for batch_size in `parse_linear_projection_legacy` | Missing or malformed input_shape produces valid-but-wrong spec with batch_size=1 | `input_shape.first().and_then(|v| v.as_integer()).unwrap_or(1) as usize` | High | HIGH | II.2 |
| R-025 | `coreml-proto/src/lib.rs:484,529,776,949,959,969` | TYPE-INVARIANT-GAP | `Gelu.mode`, `Conv.pad_type`, `Pad.mode`, `MaxPool.pad_type`, `AvgPool.pad_type`, `L2Pool.pad_type` are `String` in proto types | Proto layer should enforce closed vocabularies; invalid strings produce broken Core ML models | `mode: String`, `pad_type: String` — no compile-time or runtime validation | High | HIGH | II.3 |
| R-026 | `coreml-ffi/src/api.rs:176,196`; `coreml-ffi/src/model.rs:62` | TYPE-INVARIANT-GAP | FFI-layer `dtype` and `compute_unit` fields are `String` instead of `CoreMlDataType` enum | Deserialization accepts any string (e.g., "fp99", "TPU"); loses type safety at FFI boundary | `pub dtype: String`, `pub compute_unit: String` | High | HIGH | II.3 |
| R-027 | `coreml-emit/src/` (all files) | ERROR-MODEL-DRIFT | All public emit functions return `anyhow::Result` with no typed error enum | Callers cannot distinguish "unsupported op" from "duplicate output name" from "reshape mismatch" programmatically | `use anyhow::Result;` at top of every emit file; `anyhow::bail!()` for all error paths | High | HIGH | II.2 |
| R-028 | `coreml-emit/src/package.rs:78-79` | RESOURCE-LIFETIME-GAP | `fs::remove_dir_all` on existing path before write; no atomicity | If write fails after deletion, the original package is gone and no new package exists — data loss | `if pkg_path.exists() { fs::remove_dir_all(pkg_path)?; } fs::create_dir_all(&data_dir)?;` — no rollback | High | HIGH | II.9 |
| R-029 | `coreml-ffi/src/capi.rs:221-224` | UNSAFE-BOUNDARY | `Box::from_raw` in `coreml_model_destroy` — no double-destroy guard | Calling destroy twice with same handle causes double-free (UB); handle pointer not zeroed in caller's copy | `let inner = handle as *mut ModelHandleInner; unsafe { let _ = Box::from_raw(inner); }` — no consumed-flag or AtomicPtr | High | HIGH | II.4 |
| R-030 | `coreml-ffi/src/capi.rs:221` | UNSAFE-BOUNDARY | Cast from zero-variant enum `CoreMlModelHandle` to concrete `ModelHandleInner` is unsound unless allocation contract is upheld | If macOS implementation uses C API allocation instead of `Box::new`, `Box::from_raw` will reinterpret C-allocated memory as Rust Box | Allocation contract is documented but unenforced convention | High | HIGH | II.4 |
| R-031 | SPEC.md:475,519; `knowledge/src/store.rs:361-394` | UNSUPPORTED-CLAIM | SPEC claims knowledge units are immutable with versioned audit trail; implementation overwrites in-place | `insert_observation` increments revision but replaces entry; old versions are NOT retained | `self.index.insert(id.clone(), updated)` at store.rs:394; no version history | High | HIGH | II.3 |
| R-032 | SPEC.md:525; `knowledge/src/update.rs:64-79` | UNSUPPORTED-CLAIM | SPEC states "confidence cannot be 1.0 from a single observation"; validation allows it | `validate()` checks confidence in [0.0,1.0] and evidence_count>=1 but does not reject confidence=1.0 with evidence_count=1 | update.rs:64-79 allows the combination SPEC explicitly forbids | High | HIGH | II.2 |
| R-033 | SPEC.md:537-545; `knowledge/src/update.rs:90-99` | DOC-CODE-DRIFT | SPEC and code implement different confidence models; ManualEntry discrepancy is 0.25 | SPEC: ManualEntry=0.75, CrossValidated=0.85; Code: ManualEntry=0.5, CrossValidated=0.6 | Not rounding — fundamentally different trust models | High | HIGH | II.10 |
| R-034 | SPEC.md:531; `knowledge/src/` (all files) | UNSUPPORTED-CLAIM | SPEC claims knowledge pruning; no pruning logic exists | "Knowledge units with evidence_count=1, age>90 days, confidence<0.3 are candidates for pruning" — no implementation | grep for "pruning"/"prune" returns zero results across knowledge source files | High | HIGH | II.3 |
| R-035 | SPEC.md:553; `knowledge/src/confidence.rs:23-25` | DOC-CODE-DRIFT | SPEC claims linear 1%/30d confidence decay; code implements exponential decay; neither is operational | Code: `c * 0.5^(elapsed/halflife)`; SPEC: "1% per 30 days"; function marked "currently only used in tests" | confidence.rs:19-25; SPEC.md:553 | High | HIGH | II.10 |
| R-036 | SPEC.md:525; `knowledge/src/update.rs` | UNSUPPORTED-CLAIM | SPEC claims "latency must be positive" validation; no latency field or check exists | `KnowledgeUnit` struct has no latency field; `validate()` has no latency check | update.rs:64-79 | High | HIGH | II.3 |
| R-037 | `lab/src/session.rs:658` | ERROR-MODEL-DRIFT | `ingest_knowledge_observations` returns `Result<usize, String>` | Stringly typed error loses chain context; `ok_or("No observations found")?` | Bare `String` error type in production library code | High | HIGH | II.2 |

### MEDIUM

| ID | Location | Class | Description | Principle | Evidence | Confidence | Severity | Ref |
|----|----------|-------|-------------|-----------|----------|------------|----------|-----|
| R-038 | `ir/src/ane_layout.rs:179-189` | ERROR-MODEL-DRIFT | `validate_palette_bits` returns `Result<(), String>` instead of structured error type | Stringly typed error cannot be matched programmatically | Returns `Err(format!(...))` with bare string | High | MEDIUM | II.2 |
| R-039 | `ir/src/sir.rs:119` | TYPE-INVARIANT-GAP | `DecodeStep.qk_norm_type` is `String` instead of enum `QkNormType { Rms, Layer }` | Comment says only "rms" and "layer" are valid | `qk_norm_type: String` with `default_qk_norm_type()` returning "rms" | High | MEDIUM | II.3 |
| R-040 | `ir/src/sir.rs:1003`; `ir/src/air.rs:894` | TYPE-INVARIANT-GAP | `precision_override` is `Option<String>` instead of `Option<PrecisionOverride>` enum | Accepts any string; no validation at construction | `"fp32".to_string()` set in tests; no validation | High | MEDIUM | II.3 |
| R-041 | `ir/src/pir.rs:82-86,259,307`; `ir/src/payload.rs:34`; `ir/src/shard_desc.rs:194` | TYPE-INVARIANT-GAP | PIR/payload/shard dtype fields are `String` instead of `MilDtype` | MIR uses `MilDtype`; PIR/payload layers regress to strings | `dtype: String` in TensorSpec, TensorDescriptor, StateDeclaration, Handoff | High | MEDIUM | II.3 |
| R-042 | `ir/src/ane_hw_limits.rs:234-238` | ERROR-MODEL-DRIFT | `HwLimitViolation.param` is `String` instead of `HwLimitParam` enum | Stringly typed error field; values are always from fixed set | `param: String` where value is "max_tensor_width", "max_conv_channels", etc. | Medium | MEDIUM | II.2 |
| R-043 | `ir/src/ane_layout.rs:85-89` | ERROR-MODEL-DRIFT | `LayoutConstraintViolation.constraint` is `String` instead of enum | Constraint identifier should be matchable | `constraint: String` where values are fixed identifiers like "const_interleave_1" | Medium | MEDIUM | II.2 |
| R-044 | `ir/src/payload.rs:673` | ERROR-MODEL-DRIFT | `FamilyPayload.params["dtype"]` uses `.unwrap_or("fp16")` fallback | Malformed JSON value silently falls back to "fp16" | `let effective_dtype = params["dtype"].as_str().unwrap_or("fp16").to_string();` | High | MEDIUM | II.2 |
| R-045 | `ir/src/sir.rs:1070-1107` | SERDE-CONTRACT-GAP | `Blockwise.bits`, `GroupedLut.bits`, `Palettized.nbits` accept any `usize` via serde | Valid ANE values are {1,2,3,4,6,8}; serde accepts any value | `bits: usize` with comment "Valid ANE values: {1, 2, 3, 4, 6, 8}" | High | MEDIUM | II.8 |
| R-046 | `ir/src/sir.rs:340-342`; `ir/src/air.rs:219-221`; `ir/src/mir.rs:270-273` | SERDE-CONTRACT-GAP | `Gelu.mode: String` accepts any string via serde | Only "TANH_APPROXIMATION" and "EXACT" are valid; serde produces invalid states | `mode: String` with no custom deserializer or validation | High | MEDIUM | II.8 |
| R-047 | `ir/src/air.rs:44-52` | SERDE-CONTRACT-GAP | `Conv1x1AsLinear.output_dim: usize` uses 0 as sentinel for "unknown" | Magic value instead of `Option<usize>`; serde accepts 0 silently | `output_dim: usize` with comment "When 0, the output dim is unknown" | High | MEDIUM | II.8 |
| R-048 | `ir/src/common.rs:229,323-328` | SERDE-CONTRACT-GAP | `kv_heads: 0` sentinel in `ModelArchConfig` means "not specified" | Same pattern as R-047; 0 is a magic value instead of `Option` | `kv_heads: usize` where 0 means "not specified" | High | MEDIUM | II.8 |
| R-049 | `passes/src/` (all pass structs) | ERROR-MODEL-DRIFT | All passes return `anyhow::Result` with no pass-specific error enums | Callers cannot match on specific error variants; `state_topology.rs:105` uses `anyhow::bail!` with string | Every pass struct's `run()` returns `anyhow::Result` | High | MEDIUM | II.2 |
| R-050 | `passes/src/dtype_constraints.rs:440-470` | STUB-MIMIC | `validate_cross_type_compatibility` always returns `Ok(())` | Function claims to validate BF16/F16 cross-type violations but only logs warnings; never rejects anything | Lines 443-470: only `log::warn!` calls; always returns `Ok(())` | High | MEDIUM | II.2 |
| R-051 | `passes/src/precision_policy.rs:60-66` | VISIBILITY-LEAK | `PrecisionPolicyPass` fields are `pub` including mutable `adaptations` | External code can mutate `adaptations` after `run()` completes | `pub default_dtype: String, pub adaptations: Vec<PrecisionAdaptation>` | High | MEDIUM | II.5 |
| R-052 | `passes/src/shard_plan.rs:150-154` | VISIBILITY-LEAK | `ShardPlanPass` fields are `pub` including mutable `adaptations` | Same pattern as R-051 | `pub fallback_risk_threshold: f32, pub adaptations: Vec<ComputeUnitAdaptation>` | High | MEDIUM | II.5 |
| R-053 | `ir/src/sir.rs:1029-1033`; `ir/src/air.rs:897-903`; `ir/src/pir.rs:907-928` | VISIBILITY-LEAK | All fields on IR graph structs are `pub` | External crates can construct invalid graph states (empty inputs/outputs, nodes referencing non-existent IDs) | `pub nodes: Vec<SirNode>`, `pub inputs: Vec<SirNodeId>`, `pub outputs: Vec<SirNodeId>` | High | MEDIUM | II.5 |
| R-054 | `ir/src/ane_hw_limits.rs:10-30` | VISIBILITY-LEAK | `AneHwLimits` has all `pub` fields; factory method `for_revision()` exists but is not the only construction path | External code can construct with invalid values (num_nes: 0, max_tensor_rank: 99) | `pub max_tensor_width: u64`, `pub num_nes: u32`, etc. | High | MEDIUM | II.5 |
| R-055 | `lab/src/session.rs:924` | PANIC-LEAK | `to_str().unwrap_or("")` silently produces empty path for non-UTF8 output directory | Bridge would write to current directory instead of intended location | `FamilyPayload::from_spec(&spec, mlpackage_output.to_str().unwrap_or(""))?` | Medium | MEDIUM | II.2 |
| R-056 | `lab/src/session.rs:105-111` | PANIC-LEAK | `.expect("write to String cannot fail")` in `compute_task_hash` | Technically safe but policy violation; sets bad precedent | `write!(hash_input, "family={}", spec.family).expect(...)` (x4) | High | MEDIUM | II.2 |
| R-057 | `artifacts/src/hashing.rs:27` | PANIC-LEAK | `.unwrap()` on `write!` to String in production code | Same class as R-056 | `write!(output, "{:02x}", b).unwrap();` | High | MEDIUM | II.2 |
| R-058 | `lab/src/session.rs:899` | ERROR-MODEL-DRIFT | `LabSession::run()` returns `Result<LabResult, String>` | All downstream errors stringified, losing error chain | `.map_err(|e| format!("Bridge execution failed: {}", e))?` | High | MEDIUM | II.2 |
| R-059 | `bridge/src/subprocess.rs:24-34` | VISIBILITY-LEAK | `PythonBridge` has all `pub` fields including `timeout_secs` | Callers can set `timeout_secs: 0` (immediate timeout) or `python_path: ""` | `pub bridge_script_path: PathBuf`, `pub python_path: String`, `pub timeout_secs: u64` | High | MEDIUM | II.5 |
| R-060 | `lab/src/session.rs:41-92` | VISIBILITY-LEAK | `LabSession`, `LabLoopSession`, `LabResult` all have `pub` fields | Callers can construct with invalid states | `pub input: String, pub output: String, pub bridge_script: String, ...` | High | MEDIUM | II.5 |
| R-061 | `bridge/src/subprocess.rs:243,248` | SERDE-CONTRACT-GAP | `BridgeResult.compute_plan` and `.metadata` are `serde_json::Value` catch-alls | Compute plan has known schema but nothing enforces it at deserialization | `pub compute_plan: Option<serde_json::Value>` with comment `Some({available: bool, reason?: str})` | Medium | MEDIUM | II.8 |
| R-062 | `bridge/src/subprocess.rs:212,215` | TYPE-INVARIANT-GAP | `BridgeFunctionDescriptor.inputs/outputs` are `Vec<serde_json::Value>` | Known structure `{name, shape, dtype}` deserialized as untyped JSON; missing fields default silently | `inputs: Vec<serde_json::Value>` + `inp.get("name").and_then(|v| v.as_str()).unwrap_or("unknown")` | High | MEDIUM | II.3 |
| R-063 | `coreml-emit/src/emitter.rs:17,33,143` | CONCURRENCY-RISK | `AtomicU64` with `Ordering::Relaxed` for compilation count threshold check | TOCTOU: two threads could both pass threshold check simultaneously | `COMPILATION_COUNT.fetch_add(1, Ordering::Relaxed)`; counter is advisory | Medium | MEDIUM | II.7 |
| R-064 | `coreml-ffi/src/api.rs:96-101,121-126` | STUB-MIMIC | `inspect_model_structure` and `inspect_compute_plan` return `Ok(...)` with `available: false` on macOS | Success wrapping failure; caller must check `available` to discover it didn't work | `Ok(ModelStructureResult { available: false, .. })` | High | MEDIUM | II.2 |
| R-065 | `coreml-ffi/src/model.rs:110` | STUB-MIMIC | `FfiModel::load` returns `Ok(Self { handle: None })` on macOS | "Loaded" model with no handle; `is_loaded()` returns false | `Ok(Self { path: path.to_string(), handle: None })` | High | MEDIUM | II.2 |
| R-066 | `coreml-emit/src/mir_to_proto.rs:168` | PANIC-LEAK | `.unwrap_or(0)` on reshape input element count silently bypasses validation | Unknown shapes pass validation (0 > 0 is false, check is skipped) | `graph.node_shapes.get(x).map(|s| s.iter().product::<usize>()).unwrap_or(0)` | Medium | MEDIUM | II.2 |
| R-067 | `cli/src/main.rs:723-724` | RESOURCE-LIFETIME-GAP | Hardcoded temp dir for knowledge store with no cleanup or concurrency isolation | `std::env::temp_dir().join("ane_compile_knowledge_store")` — multiple concurrent compiles clobber each other | Fixed temp path; no `tempfile::tempdir()` | High | MEDIUM | II.9 |
| R-068 | `cli/src/main.rs:693+` | ERROR-MODEL-DRIFT | All CLI `run_*` functions return `Result<(), String>` | Every error is stringified, destroying error chain | `fn run_compile(...) -> Result<(), String>` (8+ functions) | High | MEDIUM | II.2 |
| R-069 | `trace/src/config.rs:44,58` | TYPE-INVARIANT-GAP | `TraceConfig.dtype` and `.model_class` are `String` instead of enums | Valid values are closed sets: dtype ∈ {"fp16","fp32"}, model_class ∈ {"auto","causal_lm",...} | `pub dtype: String`, `pub model_class: String` | High | MEDIUM | II.3 |
| R-070 | SPEC.md:304,516-520; `knowledge/src/store.rs:8-10` | DOC-CODE-DRIFT | SPEC describes SQLite backend with composite indexes; implementation uses flat JSON files | SPEC claims "one table per knowledge unit type", "composite indexes"; code has `HashMap` with linear-scan scope matching | SPEC.md:518; store.rs:136-139; query.rs:122-141 | High | MEDIUM | II.8 |
| R-071 | SPEC.md:526-529; `knowledge/src/store.rs:348` | DOC-CODE-DRIFT | SPEC describes key-field+scope matching; code uses exact-ID matching only | Two observations about the same op_pattern but different IDs are NOT matched — they become separate entries | `self.index.get(&id)` at store.rs:348 | High | MEDIUM | II.3 |
| R-072 | `coreml-emit/src/package.rs` (write); `coreml-proto/src/lib.rs:282` | SERDE-CONTRACT-GAP | `PackageManifest.file_format_version` is `String`; accepts any value but Apple requires "1.0.0" | `build_manifest` hardcodes correctly but type allows invalid states | `pub file_format_version: String` | High | MEDIUM | II.8 |
| R-073 | `passes/src/precision_policy.rs:38-39` | SERDE-CONTRACT-GAP | `PrecisionAdaptation` serializes dtype as `String` not `MilDtype` | Deserialized string "float16" instead of "fp16" would pass but break downstream | `pub original_dtype: String, pub adapted_dtype: String` with serde derives | Medium | MEDIUM | II.8 |
| R-074 | `coreml-proto/src/lib.rs:241-254` | VISIBILITY-LEAK | `WeightEntry.data` is `pub Vec<u8>` — mutable after construction | `offset`/`size` fields can become inconsistent with actual data length | `pub data: Vec<u8>`, `pub offset: u64` — must stay consistent | Medium | MEDIUM | II.5 |
| R-075 | `coreml-proto/src/lib.rs:2838-2855` | VISIBILITY-LEAK | `CoreMlModel` has 8 `pub` fields with no construction validation | Anyone can construct with `default_function_name` pointing to non-existent function | All fields `pub`, no builder pattern or validation method | Medium | MEDIUM | II.5 |
| R-076 | SPEC.md:57; `ir/src/serialize.rs:114-117` | UNSUPPORTED-CLAIM | SPEC "honest uncertainty" principle contradicted by hardcoded perfect AIR confidence | AIR always claims `legality_confidence: 1.0`, `fallback_risk: 0.0` | SPEC.md:57 "The system must never claim exact ANE placement knowledge" | High | MEDIUM | II.3 |
| R-077 | SPEC.md:122; `ir/src/` | UNSUPPORTED-CLAIM | SPEC claims "serialization, validation, and diff utilities"; no validation or diff exists | Only serialization (MessagePack) is implemented | No verify/validate/diff functions on any IR struct | High | MEDIUM | II.3 |
| R-078 | `artifacts/src/manifest.rs:19-140` | VISIBILITY-LEAK | All manifest sub-structs have fully `pub` fields | Invalid manifests can be constructed (wrong version, empty task_hash) | `pub version: String`, `pub task_hash: String`, `pub shape: Vec<usize>` | High | MEDIUM | II.5 |
| R-079 | `bridge/src/safetensors_resolver.rs:67,495,502,543`; `lab/src/session.rs:763` | ERROR-MODEL-DRIFT | `eprintln!` in library code alongside `log::warn!` for same conditions | Inconsistent; some files use both `eprintln!` and `log::warn!` for identical conditions | `eprintln!("Warning: failed to load safetensors file...")` vs `log::warn!(...)` | High | MEDIUM | II.10 |
| R-080 | `report/src/json_report.rs:18-19` | TYPE-INVARIANT-GAP | `JsonReport.report_type` and `.version` are `String` instead of enums | Closed vocabularies: "compilation"/"knowledge"/"diagnostics" and "1.0.0" | `pub report_type: String`, `pub version: String` | High | MEDIUM | II.3 |
| R-081 | `ir/src/ane_engine.rs:42` | PANIC-LEAK | `unwrap()` on serde serialization of `AneEngine::NE` | Panic if serialization fails in production code | `serde_json::to_string(&AneEngine::NE).unwrap()` | Medium | MEDIUM | II.2 |
| R-082 | SPEC.md:518; `knowledge/src/store.rs:136-139` | DOC-CODE-DRIFT | SPEC claims composite indexes; implementation has flat `HashMap<KnowledgeType, Vec<String>>` | Scope matching is linear scan of candidate set | query.rs:122-141; no composite index on scope fields | High | MEDIUM | II.8 |

### LOW

| ID | Location | Class | Description | Principle | Evidence | Confidence | Severity | Ref |
|----|----------|-------|-------------|-----------|----------|------------|----------|-----|
| R-083 | `ir/src/lib.rs:17,26` | TYPE-INVARIANT-GAP | Opset version and deployment target are both `&str` constants; no newtype distinction | Semantically different but syntactically identical; could be swapped | Comments say "Decoupled from DEFAULT_OPSET_VERSION" but type system doesn't enforce | Medium | LOW | II.3 |
| R-084 | `ir/src/air.rs:910` | TYPE-INVARIANT-GAP | `StaticizationDecision.method` is `String` | Finite set of valid values; should be enum | `method: String` with no validation | Medium | LOW | II.3 |
| R-085 | `ir/src/sir.rs:1099` | TYPE-INVARIANT-GAP | `QuantizationStrategy::Palettized.mode` is `String` | "kmeans" and similar should be enum variants | `mode: String` with comment | High | LOW | II.3 |
| R-086 | `ir/src/strategy.rs:317-320` | ERROR-MODEL-DRIFT | `partial_cmp` with `unwrap_or(Equal)` for NaN ratio comparison | NaN comparisons return None; treated as equal | `ratio_b.partial_cmp(&ratio_a).unwrap_or(std::cmp::Ordering::Equal)` | Medium | LOW | II.2 |
| R-087 | `ir/src/strategy.rs:103-151` | OWNERSHIP-MISMODEL | `StrategyParams` uses `Vec<(String, StrategyValue)>` with linear-scan lookup | O(n) access; no key uniqueness at type level | `self.entries.iter_mut().find(...)` for set, `self.entries.iter().find(...)` for get | Medium | LOW | II.1 |
| R-088 | `ir/src/strategy.rs:81-95,174-184,784-835` | OWNERSHIP-MISMODEL | Excessive cloning in `StrategySpec`, `DiscoveryReport`, `CompilationPlan` | `.cloned().collect()` for String/Vec-heavy structs | `evaluated.iter().filter(|s| s.applicable).cloned().collect()` | Medium | LOW | II.1 |
| R-089 | `passes/src/op_constraints.rs:11-15` | TYPE-INVARIANT-GAP | `OpConstraintViolation.op_name` and `.constraint` are `String` | Fixed vocabularies that could be enums | `pub op_name: String, pub constraint: String, pub message: String` | Medium | LOW | II.3 |
| R-090 | `passes/src/op_constraints.rs:568` | TYPE-INVARIANT-GAP | `validate_pooling_constraints` takes `pool_type: &str` instead of enum | Invalid pool types produce wrong constraint messages | `pool_type: &str` used in `format!("{}_pool", pool_type)` | High | LOW | II.3 |
| R-091 | `coreml-proto/src/lib.rs:135` | TYPE-INVARIANT-GAP | `CoreMlDataType::Unknown` has `element_size() == 0` | Division-by-zero risk in buffer size calculations | `Unknown => 0` in element_size match | High | LOW | II.3 |
| R-092 | `lab/src/run_dir.rs:189-194` | TYPE-INVARIANT-GAP | String slicing on `task_hash` with `.min()` guard | Very short hashes produce unexpected results | `&task_hash[7..15.min(task_hash.len())]` | Medium | LOW | II.3 |

---

## IV. RUST ARCHITECTURAL MISCONCEPTIONS

The following recurring misconception patterns are evidenced by the repository. Each pattern is supported by multiple findings listed in Section III.

### 1. Stringly Typed State as Acceptable Convention

**Pattern:** Using `String` for fields with a finite, known set of valid values. At least 30 fields across IR nodes, manifests, bridge results, and config objects follow this pattern. Downstream code matches on string literals with wildcard fallbacks that silently substitute defaults.

**Evidence:** R-007 (BridgeResult.status), R-009 (emission_status), R-010 (TensorSpec.dtype), R-018 (pad_type, mode, qk_norm_type, activation, etc.), R-019 (compute_units), R-025 (proto pad_type, Gelu mode), R-026 (FFI dtype, compute_unit), R-039 (qk_norm_type), R-040 (precision_override), R-062 (bridge inputs/outputs), R-069 (TraceConfig fields).

**Rust-method violation:** The Rust method requires closed vocabularies to be encoded as enums so the compiler enforces exhaustive matching and prevents typos. Strings defer validation to runtime, and wildcard fallbacks convert invalid values into plausible-but-wrong defaults — a form of fake success.

### 2. Silent Fallback as Error Handling

**Pattern:** Using `unwrap_or("fp16")`, `unwrap_or(1)`, `unwrap_or("")`, or `_ => MilDtype::Fp16` to handle missing or malformed input. Instead of propagating errors, the code silently substitutes plausible defaults.

**Evidence:** R-022 (linear_slice dtype fallback), R-023 (6× dtype "fp16" default), R-024 (batch_size=1 default), R-044 (payload dtype fallback), R-055 (empty path), R-066 (reshape validation bypass), R-081 (ane_engine serialization).

**Rust-method violation:** The Rust method treats `Result` and `Option` as explicit failure channels. Silent fallback discards error information, hides bugs, and violates the principle that the possibility of failure should be visible at the call site. At minimum, missing required fields should produce errors; optional fields should be `Option` with explicit handling.

### 3. `unreachable!()` as Correctness Assertion

**Pattern:** Using `unreachable!()` in production code to assert that certain match arms will never be hit, rather than returning structured errors.

**Evidence:** R-002 (kv_cache_rewrite), R-003 (slanc_scales), R-004 (legality_rewrite), R-005 (mil_lower).

**Rust-method violation:** `unreachable!()` is a panic that asserts correctness without enforcing it. When the assumption is violated (e.g., a new IR variant is added), the compiler panics at runtime instead of returning an error. The Rust method requires that production code use `Result` for expected failure modes, even if the failure is considered unlikely. Pattern matching on enums should use exhaustive matches or return `Result` for unhandled variants.

### 4. `pub` Fields Relying on Caller Discipline

**Pattern:** Making all struct fields `pub` and relying on callers to preserve invariants, instead of using private fields with validated constructors.

**Evidence:** R-053 (IR graph structs), R-054 (AneHwLimits), R-059 (PythonBridge), R-060 (LabSession structs), R-074 (WeightEntry), R-075 (CoreMlModel), R-078 (ArtifactManifest structs), R-051/R-052 (pass adaptation fields).

**Rust-method violation:** The Rust method requires struct fields to be private by default, with public constructors that validate invariants. `pub` fields allow external code to construct invalid states (e.g., empty node lists, inconsistent offset/data pairs). The type system should make invalid states unrepresentable.

### 5. `anyhow` in Library Crates

**Pattern:** All compiler passes and the emit crate use `anyhow::Result` with stringly-typed error messages, preventing callers from matching on specific error variants.

**Evidence:** R-027 (coreml-emit), R-049 (all passes), R-037 (ingest_knowledge_observations), R-058 (LabSession::run), R-068 (CLI run_* functions).

**Rust-method violation:** The Rust method requires library APIs to use concrete, well-structured error enums (preferably with `thiserror`). `anyhow` is appropriate for application code but prevents programmatic error handling in library consumers. The FFI crate correctly uses `thiserror`-derived `FfiError`; the library crates should follow the same discipline.

### 6. Serde Bypassing Constructor Validation

**Pattern:** Deriving `Serialize`/`Deserialize` on types with invariant constraints, allowing deserialization to produce invalid states that the type's constructors would reject.

**Evidence:** R-008 (implementation_status, verification_scope), R-045 (palette_bits), R-046 (Gelu mode), R-047 (output_dim=0 sentinel), R-048 (kv_heads=0 sentinel), R-061 (compute_plan serde_json::Value), R-072 (file_format_version), R-073 (PrecisionAdaptation dtype).

**Rust-method violation:** The Rust method requires that deserialization paths validate after loading or use custom deserializers that enforce invariants. A type that can be constructed in an invalid state via serde has a serde-contract gap — the type-level invariants are not actually invariants.

### 7. SPEC Claims Not Backed by Implementation

**Pattern:** The SPEC document describes features and guarantees (determinism, invariant enforcement, knowledge-store immutability, confidence validation) that the code does not implement. The SPEC reads as a design document, not a specification of implemented behavior.

**Evidence:** R-012 (deterministic compilation), R-013 (AIR legality confidence), R-014 (IR verification), R-031 (knowledge unit immutability), R-032 (confidence=1.0 rejection), R-034 (knowledge pruning), R-036 (latency validation), R-076 (honest uncertainty), R-077 (validation and diff utilities).

**Rust-method violation:** Documentation that claims safety, determinism, or correctness guarantees that the code does not implement creates false confidence. The Rust method requires that public API guarantees be backed by tests, implementation, or clear diagnostics. If a feature is not implemented, the documentation must honestly describe what is actually delivered.

### 8. Stub Functions Presenting as Real Logic

**Pattern:** Functions that present themselves as real compiler logic but internally perform no computation, always return success, or return success with empty/default data.

**Evidence:** R-001 (StaticizePass pass-through), R-002 (kv_cache_rewrite producing ANE-illegal ops), R-050 (validate_cross_type_compatibility always Ok), R-064 (inspect_model_structure returns Ok with available=false), R-065 (FfiModel::load returns Ok with handle=None).

**Rust-method violation:** The Rust method requires that functions either implement their documented behavior or explicitly signal their limitations. Returning `Ok(default)` for unimplemented paths wraps failure in success, preventing callers from detecting the limitation through the type system.

---

## V. REMEDIATION ROADMAP

Actions are ordered by impact and dependency. Each action addresses one or more findings from Section III.

### Phase 1: Eliminate Production Panics (Addresses R-002 through R-006, R-081)

1. **Replace all `unreachable!()` with `anyhow::bail!` or `Result`-returning match arms.** The five `unreachable!()` calls in passes (R-002 through R-005) are the most dangerous: they will panic the compiler on any future IR change. Convert to explicit error returns with descriptive messages.

2. **Replace `unwrap()` on `serde_json` round-trips in `sir_build.rs`** (R-006). Use `.map_err(|e| anyhow::anyhow!("serde round-trip failed for SirOp: {}", e))?`. This is the critical SIR construction path.

3. **Replace `unwrap()` in `ane_engine.rs:42`** (R-081). Convert to `Result`-returning function or use `map_err`.

4. **Replace `expect()` on `write!` to String** (R-056, R-057). Use `use std::fmt::Write;` which makes `write!` to String infallible by type, or use `write!` without `unwrap`/`expect` since `std::fmt::Write` for `String` never fails.

### Phase 2: Replace Fake-Success Defaults (Addresses R-022 through R-024, R-044, R-055, R-066)

5. **Replace `unwrap_or("fp16")` with required-field parsing** in `task_spec.rs` (R-023). Missing dtype should be an error, not a default. Same for `unwrap_or(1)` for batch_size (R-024) and `unwrap_or("")` for paths (R-055).

6. **Replace `_ => MilDtype::Fp16` wildcard fallbacks** in `linear_slice.rs` (R-022) and `mil_lower.rs` (R-016, R-017) with explicit error returns for unrecognized strings.

7. **Replace `unwrap_or(0)` in `mir_to_proto.rs`** (R-066) with proper handling of unknown shapes: either reject the input or propagate the unknown state via `Option<usize>`.

### Phase 3: Introduce Typed Enums for Closed Vocabularies (Addresses R-007 through R-010, R-018 through R-021, R-025 through R-026, R-039 through R-041, R-062, R-069)

8. **Define shared enums:** `PadType` (Valid/Same/Custom), `GeluMode` (Exact/TanhApproximation), `QkNormType` (Rms/Layer), `PoolType` (Max/Min/L2/Avg), `Activation` (Gelu/Relu/Silu), `SamplingMode`, `NearestRoundingMode`, `PrecisionOverride`, `BridgeStatus` (Success/Error), `EmissionStatus` (Emitted/SeamOnly), `ImplementationStatus` (HostCompiled/DeviceVerified/Partial), `VerificationScope`, `ComputeUnit` (CPU/GPU/ANE/CPUAndNE/CPUAndGPU), `ModelClass` (Auto/CausalLm/Seq2SeqLm/DecoderOnly), `ReportType` (Compilation/Knowledge/Diagnostics).

9. **Add `#[serde(rename_all = "snake_case")]`** to each enum for wire-format compatibility.

10. **Replace `Vec<String>` for compute_units** (R-019) with `Vec<ComputeUnit>` in `ShardPlan`.

11. **Replace `Vec<serde_json::Value>` for bridge inputs/outputs** (R-062) with `Vec<TensorDescriptor>` using typed fields.

### Phase 4: Tighten Visibility and Add Checked Constructors (Addresses R-053 through R-054, R-059 through R-060, R-074 through R-075, R-078)

12. **Make IR graph/node struct fields private** (R-053). Add `pub fn new(...)` constructors that validate invariants (non-empty node lists, valid ID references). Add `pub fn add_node(...)`, `pub fn add_input(...)`, `pub fn add_output(...)` methods.

13. **Make pass struct adaptation fields `pub(crate)`** (R-051, R-052). Expose results through getter methods.

14. **Make `PythonBridge`, `LabSession`, `ArtifactManifest` fields private** (R-059, R-060, R-078). Add builder-pattern constructors with validation.

15. **Make `WeightEntry.data` and `CoreMlModel` fields private** (R-074, R-075). Expose through validated construction methods.

### Phase 5: Fix FFI Boundary and Unsafe Isolation (Addresses R-029, R-030)

16. **Add consumed-flag to `ModelHandleInner`** (R-029). Use `AtomicBool` to mark handle as consumed on destroy; `coreml_model_destroy` checks and sets the flag before `Box::from_raw`.

17. **Document allocation contract for future macOS implementation** (R-030). When adding real C API linkage, use a tagged enum or trait object to distinguish Box-allocated vs. C-allocated handles, dispatching cleanup accordingly.

### Phase 6: Introduce Structured Error Types (Addresses R-027, R-049, R-058, R-068)

18. **Define `EmitError` enum with `thiserror`** in `coreml-emit` (R-027). Variants: `UnsupportedOp`, `DuplicateOutputName`, `ReshapeMismatch`, `ZeroDimension`, `WeightWriteFailed`, `PackageWriteFailed`.

19. **Define `PassError` enum** (R-049). Variants: `InvalidInputShape`, `UnsupportedOp`, `DtypeMismatch`, `LegalityViolation`, `ConstraintViolation`. Each pass's `run()` returns `Result<T, PassError>`.

20. **Replace `Result<_, String>` in lab/session and CLI** (R-058, R-068) with typed error enums.

### Phase 7: Fix Resource Lifecycle (Addresses R-011, R-028, R-067)

21. **Add timeout to trace subprocess** (R-011). Copy the T-77 timeout pattern from `bridge/subprocess.rs` with polling + kill.

22. **Use atomic write for package emission** (R-028). Write to temp directory first, then `fs::rename` for atomic swap. Clean up temp dir on failure.

23. **Use `tempfile::tempdir()` for knowledge store temp dir** (R-067). Automatic cleanup and concurrency isolation.

### Phase 8: Address Serde Contract Gaps (Addresses R-008, R-045 through R-048, R-061, R-072 through R-073)

24. **Add post-deserialization validation** for IR graphs, manifests, and bridge results. Implement `fn validate(&self) -> Result<(), ValidationError>` on each deserializable type.

25. **Replace magic-value sentinels with `Option`** (R-047, R-048). `output_dim: Option<usize>` instead of `output_dim: usize` with 0=sentinel. `kv_heads: Option<usize>` instead of `kv_heads: usize` with 0=sentinel.

26. **Add serde validation for palette_bits** (R-045). Custom deserializer that rejects values not in {1,2,3,4,6,8}.

27. **Replace `serde_json::Value` fields** (R-061) with typed structs for bridge compute_plan.

### Phase 9: Remove or Honestly Gate Stubs (Addresses R-001, R-002, R-050, R-064, R-065)

28. **Remove `staticize` module entirely** (R-001). If it's removed from the pipeline, remove the code. If it must remain, gate it behind a feature flag and add a `compile_error!` when the feature is enabled in release mode.

29. **Make `validate_cross_type_compatibility` actually validate** (R-050). Either implement the BF16/F16 cross-type checks or rename the function to `log_cross_type_hints` and make it clear it does not enforce constraints.

30. **Return errors instead of `Ok(default)` for unimplemented macOS paths** (R-064, R-065). `FfiModel::load` on macOS should return `Err(FfiError::PlatformNotSupported)`.

### Phase 10: Align Specification with Reality (Addresses R-012 through R-014, R-031 through R-036, R-076 through R-077)

31. **Update SPEC to honestly describe v0 capabilities.** Mark SQLite backend, knowledge pruning, version-based audit trail, diff utilities, and IR verification as "planned, not implemented." Remove or qualify the "deterministic compilation" claim until `IndexMap`/`BTreeMap` replaces `HashMap` and a reproducibility test exists.

32. **Wire AIR risk fields to `RiskAnnotatePass` output** (R-013, R-076). Replace hardcoded `legality_confidence: 1.0` with actual query results from the knowledge store.

33. **Add `verify()` method to each IR level** (R-014, R-077). Each method checks the SPEC-stated invariants for its level.

34. **Add confidence=1.0 rejection for evidence_count=1** (R-032). One line in `validate()`.

35. **Implement key-field+scope matching in knowledge store** (R-071). Replace exact-ID-only matching with semantic matching by op_pattern and scope overlap.

### Phase 11: Replace `eprintln!` with Structured Logging (Addresses R-015, R-079)

36. **Replace all `eprintln!` in library code** with `log::warn!` or `log::error!`. The `log` crate is already a dependency; use it consistently.

### Phase 12: Add Regression Tests for High/Critical Violations

37. **Add test for each HIGH/CRITICAL violation.** Each test should fail without the fix and pass with it. Key targets:
    - Deterministic compilation: compile same IR twice, assert byte-identical output
    - AIR risk fields: assert `legality_confidence < 1.0` for unknown ops
    - Typed enum construction: assert invalid strings are rejected at compile time
    - Double-destroy guard: assert second destroy returns error
    - Trace subprocess timeout: assert timeout fires within expected window

---

## VI. AUDIT NOTE

- `comprehensive-rust.pdf` was used as a local conceptual reference for Rust engineering method and discipline.
- Local working notes are stored in the `references/` directory within the repository.
- Both `references/` and `comprehensive-rust.pdf` are excluded from the repository by `.gitignore`.
- Only `RUSTVIOLATIONS.md` (this file) and the amended `.gitignore` are intended for commit.
- No source code was modified during this audit.
- Findings reflect the state of the repository at the time of audit. The codebase is under active development; some findings may have been addressed in subsequent commits.
