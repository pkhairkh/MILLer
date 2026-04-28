# ANE-First Multi-Level Compiler and Empirical Core ML Lab

## Software Specification

**Codename**: `MILLer`
**Version**: 0.1-draft
**Date**: 2026-04-21

---

# 1. Executive Summary

## What Is Being Built

A narrow ANE-first compiler pipeline that lowers a constrained multi-level IR into Core ML MIL / mlpackage artifacts, coupled with an empirical execution lab that generates synthetic and real MIL tasks, runs them through Core ML on actual Apple devices, and accumulates backend knowledge from observed compile, load, and run behavior. The system includes a first-class knowledge adaptation layer that feeds empirical observations back into compiler decisions.

The system produces three categories of output: (1) Core ML mlpackage artifacts structured for ANE-likely execution, including shard-aware and state-aware deployment forms; (2) structured profiling data from real device execution capturing latency, throughput, drift, fallback behavior, and per-device fingerprints; (3) a versioned, queryable backend knowledge store that encodes what the system has learned about ANE legality, survival, precision hazards, and shard topology.

## What It Is Not

It is not a general ML compiler. It is not an ONNX replacement or a broad LLVM analogue. It is not a generic multi-backend serving platform. It is not a pseudo-research framework that never touches devices. It does not claim to control Apple's internal ANE scheduling, and it does not pretend to know the exact compute-unit assignment algorithm. It does not support arbitrary dynamic control flow inside Core ML, and it does not attempt to build a CPU inside Core ML.

## Why Qwen3 Shard-Aware Deployment Matters

The `pkhairkh/qwen3-coreml-palettized` project demonstrates that a 1.7B-parameter transformer can survive ANE constraints by decomposing into five coordinated Core ML MLPrograms: `io_model` (embedding + LM head on CPU+GPU), three decoder shards (left/mid/right on CPU+NE), and a `sampler_model` (CPU+GPU). Each decoder shard owns its own KV state, uses a reverse ring-buffer cache, and applies mixed-precision grouped LUT palettization (4/6/8-bit varying by layer depth and weight type). This is not a packaging artifact. It is evidence of a backend-constrained execution form: shard-aware, state-aware, package-aware, and ANE-survivable. The near-perfect perplexity parity (delta < 0.6 on wikitext-103) proves this decomposition can preserve quality. Our system must generalize from this evidence rather than flatten it away.

## Why Synthetic + Non-Synthetic MIL Adaptation Is Central

The `pkhairkh/ANE-sha256d` project proves that non-native semantics can be re-expressed into ANE-compatible graph form. SHA-256d, a pure bit-twiddling algorithm, runs on the ANE via fp16 Boolean emulation (AND=mul, XOR=abs(a-b), OR=max) and 1x1-conv permutation matrices for bit rotations. This teaches us that hostile operator families can produce real backend knowledge: which graph motifs survive ANE placement, which force fallback, and what the latency cliff looks like when you push a matrix engine into element-wise Boolean work. But synthetic stress tests alone are insufficient. Real models like Qwen3 expose state-topology constraints, palettization sensitivity, and cross-shard handoff costs that synthetic microkernels cannot reproduce. The system must ingest both and treat each as a different evidence class with different confidence properties.

## Why Core ML Tools Must Be Treated as a Real Boundary, Not Ignored

Core ML Tools 8.x exposes a specific public API surface: converters, model APIs, MIL Builder, MIL input types, MIL ops, graph passes, optimizers, and palettization utilities. ML programs are emitted as `.mlpackage` (not `.mlmodel`). Typed execution means intermediate tensor precision is explicit in the model and respected as a minimum by the runtime, but actual compute-unit partitioning varies by hardware and software generation. Stateful models require iOS 18+/macOS 15+. Palettization supports 1/2/3/4/6/8-bit LUT with per-grouped-channel granularity, but the system must not assume built-in palettization flows are sufficient for ANE quality retention. These are the real tool boundaries. The system must interface with them honestly rather than reimagine a fake standalone universe, and it must not let coremltools implicitly become the compiler.

---

# 2. Goals

## Functional Goals

1. **Emit MIL/mlpackage artifacts** that are structured for ANE-likely execution, including shard-aware and state-aware deployment forms, using Core ML Tools as the emission backend.
2. **Support a constrained op set** sufficient for transformer decode workloads: linear/matmul, 1x1-conv-as-linear, add/mul/reshape/transpose/split/concat, RMSNorm/LayerNorm, RoPE/table-based positional transforms, masking, softmax, scalar-LUT/grouped palettized projection, state read/write, stateful decode-step blocks, sampler path, and selected synthetic microkernels.
3. **Implement a multi-level IR stack** with semantic/task IR, ANE-legal IR, MIL-emission IR, and package/deployment IR, each with defined invariants and metadata.
4. **Implement a partitioning and shard planner** that generalizes from the Qwen3 left/mid/right pattern, supports entry/interior/exit shard semantics, and captures shard templates as reusable knowledge.
5. **Implement a precision/palette policy engine** that assigns bitwidth and palettization strategy per-layer and per-weight-type, informed by empirical results rather than blindly trusting built-in flows.
6. **Build an execution lab** that generates synthetic and real MIL tasks, runs them through Core ML on actual Apple devices, and captures latency, throughput, numerical drift, fallback suspicion, and per-device behavioral fingerprints.
7. **Build a backend knowledge store** that accumulates structured empirical compiler knowledge: legality rules, graph motif catalogs, fallback-risk signatures, per-op/per-pattern survival matrices, shard-template catalogs, precision/palette hazard tables, and device behavior fingerprints.
8. **Provide CLI/workflow tooling** for compilation, profiling, knowledge queries, and report generation.

## Non-Functional Goals

1. **Deterministic compilation**: Same IR + same knowledge store = same emitted artifact, bit-for-bit.
2. **Reproducible profiling**: Same task + same device + same OS version = comparable results within measurement noise.
3. **Rust-owning architecture**: All compiler logic, IR, passes, planning, knowledge storage, and orchestration in Rust. Python/coremltools used only at the narrow emission boundary.
4. **Narrow Rust/Python boundary**: The interface between Rust and Python must be serialized, subprocess-isolated, and auditable. Python must not swallow core compiler logic.
5. **Offline-first knowledge**: The knowledge store must be locally queryable without network access. No cloud dependency for compiler decisions.
6. **Testable passes**: Each compiler pass must be unit-testable in isolation with deterministic inputs and outputs.
7. **Honest uncertainty**: The system must never claim exact ANE placement knowledge. Fallback suspicion is a first-class output, not a failure mode.

## Anti-Goals

1. **General ML compiler**: No support for arbitrary model architectures, training graphs, or non-ANE backends.
2. **ONNX replacement**: No cross-framework IR interchange format.
3. **LLVM equivalence**: No claim of full compiler infrastructure generality.
4. **Generic multi-backend serving**: No GPU/CPU-only deployment paths as primary targets.
5. **Arbitrary dynamic control flow in v1**: No while loops, data-dependent branching, or recursive patterns inside Core ML.
6. **Fake ANE placement certainty**: No claims about Apple's internal scheduling algorithm.
7. **Python as the compiler**: No migration of IR, pass, or planning logic into Python.
8. **Blind trust in coremltools optimization**: No assumption that default palettization, quantization, or graph passes produce ANE-optimal results.
9. **"Future all-model support"**: No vague extensibility promises.

---

# 3. Core Design Thesis

**ANE-first emission is a compilation problem, not a conversion problem.**

Converting a PyTorch model to Core ML via `ct.convert()` and hoping the ANE picks it up is not compilation. It is translation without planning. The ANE has specific constraints: limited operator support, fp16 precision, state-capacity boundaries, shape restrictions, and undocumented partitioning heuristics. Emitting a graph that is syntactically valid in MIL does not guarantee ANE placement. The gap between "emits without error" and "runs on the ANE" is the space where compilation must operate.

**Emitability is not enough.**

A model that loads successfully in Core ML may execute partially on CPU. This is fallback, and it is invisible without profiling. A graph that is ANE-placed on M2 may fall back on M1. A palettization that preserves perplexity at 4-bit on layers 0-11 may destroy attention quality on layers 24-27. The system must measure, not assume. Empirical profiling is not optional validation; it is the primary source of correctness knowledge for a backend whose behavior is not fully documented.

**Core ML Tools constraints matter because they define the real emission surface.**

Core ML Tools exposes MIL Builder for programmatic graph construction, typed execution for precision control, StateType for stateful models (iOS 18+), and palettization APIs with specific bitwidth support (1/2/3/4/6/8-bit). These are not implementation details to abstract away. They are the material constraints of the target. The system must target them directly, not through a hypothetical clean IR that ignores package format, opset versioning, and compute-unit hints.

**Typed execution and stateful-model behavior matter because they change the deployment contract.**

ML programs have explicitly typed intermediates. The runtime respects these as minimum precision. This means fp16-typed graphs may execute in fp32 on CPU, altering numerical behavior across compute-unit boundaries. Stateful models (iOS 18+) change the deployment topology: KV cache becomes persistent state rather than input/output copying. The system must reason about these differences, not pretend they do not exist.

**Knowledge must be accumulated from real MIL/runtime behavior because the ANE is a black box.**

Apple does not document the ANE's operator placement algorithm. The only way to learn what survives is to emit MIL, run it, and observe. Synthetic programs teach us about operator-level survival. Real models teach us about topology-level survival. Both are necessary, and both have different confidence properties. The knowledge store must be structured, versioned, and honest about what it knows versus what it suspects.

**The Qwen3 shard layout is evidence of a backend-constrained execution form, not a one-off hack.**

The five-package decomposition (io_model + 3 decoder shards + sampler) exists because a single monolithic Qwen3-1.7B graph cannot survive ANE placement intact. The shard boundaries are not arbitrary: each shard must fit within ANE state and constant budgets, each owns its own KV state to avoid cross-package state coupling, and the io_model/sampler are on CPU+GPU because embedding, logit projection, and sampling contain operations that are ANE-hostile or ANE-suboptimal. The mixed-precision palettization (4/6/8-bit varying by layer and weight type) is not random: deeper layers and attention projections get higher precision because sensitivity analysis showed they need it. This entire form is a response to real backend constraints. The system must capture it as a reusable shard template, generalize the entry/interior/exit semantics, and allow new partitionings to be explored and validated against the same constraints.

---

# 4. System Overview

## 4.1 Task/Model Frontend

**Responsibility**: Ingest task descriptions and model definitions into semantic/task IR. Parse Qwen3-like deployment manifests, synthetic task specifications, and imported MIL dumps.

**Inputs**: Task spec files (TOML/JSON), model configuration files, imported MIL text dumps, manually authored benchmark graph descriptions.

**Outputs**: Semantic/task IR instances ready for canonicalization.

**Implementation**: Rust.

**Main risks**: Overly broad input schema; importing MIL dumps requires a MIL text parser that must track coremltools dialect evolution.

## 4.2 Multi-Level IR Core

**Responsibility**: Define and manage the IR stack: semantic/task IR, ANE-legal IR, MIL-emission IR, package/deployment IR. Enforce invariants at each level. Provide serialization, validation, and diff utilities.

**Inputs**: IR instances from the frontend or from pass outputs.

**Outputs**: Validated IR instances at each level, serialized to a stable binary format (MessagePack or bincode).

**Implementation**: Rust.

**Main risks**: IR level proliferation if boundaries are unclear; invariant enforcement must not become a performance bottleneck.

## 4.X Family Registry Architecture

**Principle**: A task family is a first-class, self-contained unit of compilation surface. Adding a new family must not require architectural changes, new sprints, or modifications to the CLI dispatch code. The system is generic by design — each family is an instance of the same pattern, not a special case.

### Family Contract

Every task family must provide:

1. **TaskOp variant** — A `TaskOp::NewFamily { ... }` enum variant carrying the family's specific parameters (dimensions, dtypes, etc.)
2. **TaskOp method implementations** — Match arms in each of the generic methods: `family_id()`, `bridge_command()`, `identity_string()`, `primary_dims()`, `op_type_str()`, `is_sharded()`, `family_params()`, `input_tensor_shape()`, `output_tensor_shape()`, `input_tensor_name()`, `input_tensor_dtype()`
3. **TOML parser** — A `parse_new_family()` function and an entry in the `FAMILY_PARSERS` registry
4. **TaskFamilyTrait implementation** — A `NewFamilyFamily` struct implementing the `TaskFamilyTrait` (generation, serialization, determinism), plus a `TaskFamilyId::NewFamily` variant
5. **Baseline computation** — A `compute_new_family()` method on `BaselineComputer` providing deterministic FP32 reference computation
6. **Python MIL emitter** — A `build_new_family_program()` and `emit_new_family()` function in `mil_emitter.py`, plus an entry in `bridge.py`'s command dispatch

### What Does NOT Need to Change

When adding a new family, the following files and subsystems require **zero changes**:

- `crates/cli/src/main.rs` — Generic dispatch via `TaskOp` methods and `FamilyPayload` eliminates all per-family match blocks
- `crates/ir/src/linear_slice.rs` (payload) — `FamilyPayload` is family-agnostic
- `python/bridge.py` — Auto-flattens `FamilyPayload.params` into top-level keys for backward compatibility with emitters
- `crates/ir/src/linear_slice.rs` (SIR/MIR) — `sir_from_linear_projection` and `lower_linear_projection_to_mir` use `primary_dims()` generically
- Knowledge store schema — Family-agnostic query interface
- Artifact manifest construction — Uses `primary_dims()` generically
- Task hash computation — Uses `identity_string()` generically

### Generic Payload Model

All families use a single `FamilyPayload` struct for bridge communication:

```json
{
  "bridge_version": 1,
  "command": "emit_new_family",
  "task_name": "...",
  "family": "NewFamily",
  "params": { "field1": ..., "field2": ... },
  "opset_version": "iOS18",
  "compute_units": "CPU_AND_NE",
  "output_path": "...",
  "seed": 42,
  "functions": [...]
}
```

The `params` field carries all family-specific parameters. The Python bridge automatically flattens `params` into the top-level command dict before passing to emitter functions, maintaining backward compatibility with existing per-family emitter implementations.

### Currently Registered Families

| Family | TaskOp Variant | Bridge Command | TOML Section | Status |
|--------|---------------|----------------|-------------|--------|
| LinearProjection | `TaskOp::LinearProjection` | `emit_linear_projection` | `linear_projection` | Active |
| ShardedLinearPipeline | `TaskOp::ShardedLinearPipeline` | `emit_linear_projection` | `sharded_linear_pipeline` | Active |
| LutProjection | `TaskOp::LutProjection` | `emit_lut_projection` | `lut_projection` | Active |
| DecodeStep | `TaskOp::DecodeStep` | `emit_stateful_decode_step` | `decode_step` | Active |
| ShardedDecodeStep | `TaskOp::ShardedDecodeStep` | `emit_shard_decode_step` | `sharded_decode_step` | Active |
| MlpBlock | `TaskOp::MlpBlock` | `emit_mlp_block` | `mlp_block` | Active |
| Attention | `TaskOp::Attention` | `emit_attention` | `attention` | Active |

### Remaining Per-Family Dispatch Points

Two code paths still require per-variant match and cannot be fully generic:

1. **Shard pipeline spec construction** — `ShardedLinearPipeline` and `ShardedDecodeStep` require different `ShardPipelineSpec` factory methods with variant-specific fields. This match is in `run_compile_sharded` and `run_compile_full_sharded`.

2. **Baseline computation** — `BaselineComputer` has per-family compute methods (`compute_linear_projection`, `compute_lut_projection`, etc.) with genuinely different mathematical logic. This match is in `run_lab` and `run_lab_loop`.

## 4.3 Legality/Staticization Engine

**Responsibility**: Determine whether a given semantic/task IR operation or pattern is legal for ANE emission. Rewrite illegal patterns into legal equivalents using known transformations. Staticize dynamic constructs where possible (e.g., replace dynamic shapes with fixed sequence lengths, replace runtimecomputed indices with static tables).

**Inputs**: Semantic/task IR.

**Outputs**: ANE-legal IR with all operations within the supported op set and all dynamic constructs resolved or rewritten.

**Implementation**: Rust.

**Main risks**: Incomplete legality model leading to false positives (claiming legal when ANE will reject); over-aggressive staticization removing necessary flexibility.

## 4.4 Partitioning/Shard Planner

**Responsibility**: Decompose an ANE-legal IR graph into one or more shards (packages) based on state budget, constant budget, operator compatibility, and performance heuristics. Assign entry/interior/exit roles. Determine io-model and sampler boundaries. Plan state ownership across shards.

**Inputs**: ANE-legal IR, shard templates from the knowledge store, device capability hints.

**Outputs**: Shard plan (which operations go in which package, state ownership map, inter-shard handoff protocol), package/deployment IR skeleton.

**Implementation**: Rust.

**Main risks**: Hardcoding the left/mid/right pattern as the only option; producing shard plans that compile but perform poorly due to handoff overhead.

## 4.5 Precision/Palette Policy Engine

**Responsibility**: Assign per-operation, per-weight precision and palettization strategy. Decide bitwidth (1/2/3/4/6/8), granularity (per-tensor vs per-grouped-channel), group size, and LUT type. Apply knowledge about precision hazards (e.g., attention projections needing higher bitwidth in deeper layers).

**Inputs**: ANE-legal IR or package/deployment IR, precision hazard table from knowledge store, quality targets.

**Outputs**: Precision-annotated IR with palettization assignments for each weight tensor.

**Implementation**: Rust.

**Main risks**: Ad hoc policy rules that are not grounded in empirical data; blindly applying uniform bitwidth across all layers.

## 4.6 MIL Emission Bridge

**Responsibility**: Lower MIL-emission IR into actual MIL Builder calls via the Python bridge. Construct MIL programs using `mb.program()`, `mb.<op>()`, and related APIs. Handle op naming, constant materialization, and graph structure.

**Inputs**: MIL-emission IR (Rust-internal representation of the target MIL graph).

**Outputs**: Serialized MIL program specification passed to the Python bridge for coremltools consumption.

**Implementation**: Rust (IR to intermediate representation) + Python (actual `mb.*` calls via coremltools).

**Main risks**: The Rust-side IR must be a complete and exact representation of what the Python side will construct; any mismatch causes emission failures.

## 4.7 Core ML Tools Integration Layer

**Responsibility**: Execute `ct.convert()` or direct MIL-to-mlprogram conversion, apply compute-unit hints, set opset version, configure typed execution precision, handle stateful model registration, and invoke palettization/optimization passes.

**Inputs**: MIL program from the emission bridge, compilation configuration (compute units, opset, precision).

**Outputs**: `MLModel` objects ready for packaging.

**Implementation**: Python (coremltools).

**Main risks**: coremltools graph passes silently altering the emitted graph; version skew between coremltools versions producing different results.

## 4.8 Core ML Packaging Layer

**Responsibility**: Save MLModel objects as `.mlpackage` bundles. Generate package manifests, shard manifests, and reproducibility hashes. Organize multi-package deployments (io_model + decoder shards + sampler).

**Inputs**: `MLModel` objects, shard plan metadata.

**Outputs**: `.mlpackage` directories on disk, `manifest.json` files, shard manifest files.

**Implementation**: Python (coremltools `model.save()`) + Rust (manifest generation, hashing).

**Main risks**: Package format changes across coremltools versions; incomplete manifest metadata making reproduction difficult.

## 4.9 Execution Lab

**Responsibility**: Generate synthetic and real MIL tasks, run them through Core ML on actual Apple devices, capture compile/load/run outcomes, measure latency/throughput, detect numerical drift, and flag fallback suspicion.

**Inputs**: Task specifications, `.mlpackage` artifacts, device configuration.

**Outputs**: Run traces, drift reports, fallback suspicion reports, device fingerprints.

**Implementation**: Rust (orchestration, task generation, result ingestion) + Python (Core ML runtime via `MLModel.predict()`).

**Main risks**: Profiling noise on shared devices; inability to strongly detect fallback (Core ML does not expose compute-unit assignment per-op at runtime).

## 4.10 Run Ingestion + Drift Analysis

**Responsibility**: Process raw run traces into structured observations. Compute numerical drift metrics (cosine distance, max absolute error, relative error). Detect latency cliffs. Correlate observations with device metadata and IR structure. Feed results into the knowledge store.

**Inputs**: Raw run traces from the execution lab.

**Outputs**: Structured observations (drift measurements, latency observations, fallback suspicions), knowledge store update entries.

**Implementation**: Rust.

**Main risks**: Noisy measurements producing false knowledge; drift attribution (is the drift from fp16, from palettization, or from fallback?).

## 4.11 Backend Knowledge Store

**Responsibility**: Store, version, and query structured empirical compiler knowledge. Support legality rules, graph motif catalogs, survival matrices, shard templates, precision/palette hazard tables, device fingerprints, and confidence-scored observations.

**Inputs**: Structured observations from run ingestion, manually authored knowledge entries.

**Outputs**: Query results used by the legality engine, shard planner, precision engine, and risk annotator.

**Implementation**: Rust (storage engine, query engine) with SQLite as the persistence backend.

**Main risks**: Knowledge contamination from noisy observations; conflicting observations from different devices; knowledge becoming stale as OS versions change.

## 4.12 Reporting Layer

**Responsibility**: Generate human-readable reports from compilation runs, profiling sessions, drift analyses, and knowledge queries. Output Markdown and structured JSON.

**Inputs**: IR dumps, pass reports, run traces, knowledge store query results.

**Outputs**: Markdown reports, JSON artifacts.

**Implementation**: Rust.

**Main risks**: Report format drift; information overload without actionable structure.

---

# 5. IR Stack

## 5.1 Semantic/Task IR (`SIR`)

**Purpose**: Represent the user's intent at the highest level of abstraction. Capture what the task is, not how it maps to hardware. This is the entry point for all compilation.

**Invariants**:
- All operations are from the supported semantic op set (linear, attention, norm, rope, mask, softmax, state-read, state-write, decode-block, sampler).
- Shapes are symbolic but must resolve to concrete dimensions before lowering.
- No ANE-specific constraints are assumed or enforced.

**Required metadata**:
- Task origin (synthetic, real-model, mil-import, manual).
- Model identifier and version if applicable.
- Target quality/performance contract (if specified).

**Example node types**:
- `LinearProjection(input, weight, bias?)` — semantic linear operation, no commitment to matmul vs 1x1-conv.
- `AttentionBlock(q, k, v, mask, rope?)` — multi-head attention, no commitment to fused vs decomposed form.
- `RMSNorm(input, weight, epsilon)` — normalization.
- `RoPETransform(input, tables)` — rotary positional encoding.
- `StateRead(state_id, offset, shape)` — read from persistent state.
- `StateWrite(state_id, offset, value)` — write to persistent state.
- `DecodeStep(token, state_map)` — one step of autoregressive decode.
- `Sampler(logits, temperature, top_p, rep_penalty)` — next-token sampling.

**What must already be resolved**: Nothing hardware-specific. This level is pure semantics.

## 5.2 ANE-Legal IR (`AIR`)

**Purpose**: Represent the graph after it has been verified legal for ANE emission and all dynamic constructs have been staticized. This is the level at which legality is a hard invariant.

**Invariants**:
- All operations are from the ANE-supported op set (the constrained subset listed in the scope rules).
- All shapes are concrete (no symbolic dimensions).
- All dynamic index computations have been replaced with static tables or constant tensors.
- All control flow has been eliminated (no loops, no data-dependent branching).
- Every operation has an associated legality confidence score from the knowledge store.

**Required metadata**:
- Staticization decisions (what was dynamic, what it became).
- Legality confidence per operation.
- Source mapping back to SIR nodes.

**Example node types**:
- `MatMul(a, b, output_shape)` — concrete-shaped matrix multiply.
- `Conv1x1AsLinear(input, weight, pad_type)` — 1x1 convolution used as linear projection.
- `ElementWise(op, inputs...)` — add, mul, abs, maximum, minimum.
- `Reshape(input, target_shape)` — static reshape.
- `Transpose(input, perm)` — static permutation.
- `Split(input, axis, num_splits)` — static split.
- `Concat(inputs, axis)` — static concatenation.
- `Softmax(input, axis)` — softmax on a static axis.
- `StaticLUTProjection(input, indices, lut, group_size)` — grouped scalar-LUT palettized projection.
- `StateReadFixed(state_id, shape)` — read from state with fixed shape.
- `StateWriteFixed(state_id, value)` — write to state with fixed shape.

**What must already be resolved**: Dynamic shapes, legality of each operation, staticization of all runtime-computed values.

## 5.3 MIL-Emission IR (`MIR`)

**Purpose**: Represent the graph in a form that is a one-to-one mapping to Core ML MIL Builder calls. Each MIR node corresponds to exactly one `mb.<op>()` call or a small fixed sequence of calls.

**Invariants**:
- Every node maps to a specific MIL op with specific named parameters.
- All types are explicit (fp16, fp32, int32, etc.).
- All constants are materialized as named const nodes with concrete values.
- The graph structure matches what the Python emission bridge will construct.
- State inputs/outputs are explicitly typed as StateType-compatible.

**Required metadata**:
- MIL op name and named parameters for each node.
- Type annotation for every edge (tensor dtype and shape).
- Compute-unit hint per operation (if applicable).
- Opset version requirement.
- Mapping back to AIR nodes.

**Example node types**:
- `MILConst(name, value, dtype)` — `mb.const(val=..., name=...)`.
- `MILMatMul(name, x, y)` — `mb.matmul(x=..., y=..., name=...)`.
- `MILConv(name, x, weight, pad_type, groups)` — `mb.conv(x=..., weight=..., pad_type=..., groups=..., name=...)`.
- `MILAdd(name, x, y)` — `mb.add(x=..., y=..., name=...)`.
- `MILMul(name, x, y)` — `mb.mul(x=..., y=..., name=...)`.
- `MILAbs(name, x)` — `mb.abs(x=..., name=...)`.
- `MILReshape(name, x, shape)` — `mb.reshape(x=..., shape=..., name=...)`.
- `MILTranspose(name, x, perm)` — `mb.transpose(x=..., perm=..., name=...)`.
- `MILSplit(name, x, axis, num_splits)` — `mb.split(x=..., axis=..., num_splits=..., name=...)`.
- `MILConcat(name, values, axis)` — `mb.concat(values=..., axis=..., name=...)`.
- `MILSoftmax(name, x, axis)` — `mb.softmax(x=..., axis=..., name=...)`.
- `MILStateWrite(name, state_ref, value)` — state write via MIL.

**What must already be resolved**: ANE legality, precision assignment, shard assignment, all MIL-level details.

## 5.4 Package/Deployment IR (`PIR`)

**Purpose**: Represent the full deployment artifact: which MIL programs go in which packages, how they connect, what state each package owns, and what the inter-package handoff protocol is.

**Invariants**:
- Every package has a defined role (io, decoder-shard, sampler).
- Every decoder shard has a defined entry/interior/exit semantic.
- State ownership is exclusive (no shared mutable state across packages).
- Inter-package data flow is via tensor handoff (no cross-package state references).
- Every package has a compute-unit assignment (CPU_AND_NE, CPU_AND_GPU).

**Required metadata**:
- Package list with roles and compute-unit assignments.
- State ownership map (which package owns which state tensors).
- Inter-package handoff specification (tensor names, shapes, dtypes).
- Shard template identifier (if derived from a known template).
- Palettization manifest (which weights at which bitwidth, granularity, group size).
- Context window contract (fixed sequence length).
- Opset version and minimum deployment target.

**Example node types**:
- `Package(name, role, compute_units, mil_program_ref)` — a single mlpackage.
- `StateDeclaration(state_id, shape, dtype, owner_package)` — a state tensor owned by a package.
- `Handoff(from_package, to_package, tensor_name, shape, dtype)` — inter-package data transfer.
- `ShardTemplate(template_id, partition_spec, entry_role, exit_role)` — reusable shard pattern.

**What must already be resolved**: Shard plan, state topology, palettization assignments, compute-unit assignments.

## 5.5 Profiling/Task IR (`ProfIR`)

**Purpose**: Represent a profiling task: what to compile, what to run, what to measure, and what baseline to compare against.

**Invariants**:
- Every task has a unique identifier and a reproducibility hash.
- Every task specifies the device class and OS version requirement.
- Baseline specification is mandatory (reference output or reference model).

**Required metadata**:
- Task family (linear/projection, attention, decode-step, etc.).
- Input specification (shapes, dtypes, value ranges).
- Expected output specification (or reference computation method).
- Measurement specification (latency, throughput, drift, fallback check).
- Device requirements.
- Repetition count and statistical method.

**Example node types**:
- `ProfileTask(task_id, family, mil_package_ref, inputs, baseline, metrics)`.
- `BaselineReference(method, reference_data_or_computation)`.
- `DeviceRequirement(device_class, os_version_range, compute_units)`.

**What must already be resolved**: The MIL artifact to profile, the measurement methodology, the baseline computation.

## 5.6 Backend-Knowledge Representation IR (`KIR`)

**Purpose**: Represent individual knowledge units in a structured, queryable, and versioned form. This is the schema of the knowledge store.

**Invariants**:
- Every knowledge unit has a confidence score in [0.0, 1.0].
- Every knowledge unit has an evidence source (synthetic-run, real-run, manual, compile-failure, etc.).
- Every knowledge unit has a timestamp and version.
- Knowledge units are immutable once committed (updates create new versions).

**Required metadata**:
- Knowledge type (legality-rule, motif-catalog, survival-matrix, shard-template, precision-hazard, device-fingerprint, fallback-signature).
- Confidence score.
- Evidence source and count (how many observations support this).
- Applicability scope (device classes, OS versions, opset versions).
- Conflict resolution priority.

**Example node types**:
- `LegalityRule(op_pattern, ane_legal: bool, confidence, scope)`.
- `SurvivalMatrixEntry(op, device_class, os_version, ane_placed: bool, confidence)`.
- `ShardTemplateKnowledge(template_id, known_good: bool, quality_delta, confidence)`.
- `PrecisionHazard(op, bitwidth, quality_impact, confidence)`.
- `FallbackSignature(graph_motif, fallback_risk: f64, evidence_count)`.
- `DeviceFingerprint(device_class, os_version, behavior_hash, anomalies)`.

**What must already be resolved**: The observation must be structured and validated before insertion.

---

# 6. Knowledge Adaptation Model

## 6.1 Knowledge Units

The knowledge store contains the following unit types:

| Unit Type | Key Fields | Example |
|-----------|-----------|---------|
| **LegalityRule** | `op_pattern`, `ane_legal`, `confidence`, `scope` | `MatMul(shape=(1,4096,4096))` is ANE-legal on M2 with confidence 0.95 |
| **MotifCatalog** | `motif_hash`, `motif_structure`, `survival_rate`, `evidence_count` | A linear-then-relu-then-reshape motif survives ANE 87% of the time across 30 runs |
| **SurvivalMatrixEntry** | `op_or_pattern`, `device_class`, `os_version`, `ane_placed`, `confidence` | `mb.softmax(axis=-1)` is ANE-placed on M2/iOS18 with confidence 0.9, but falls back on M1/iOS17 |
| **ShardTemplateKnowledge** | `template_id`, `known_good`, `quality_delta`, `latency_profile`, `confidence` | The Qwen3 left/mid/right three-shard template produces <0.6 PPL delta with confidence 0.92 |
| **PrecisionHazard** | `op`, `weight_type`, `bitwidth`, `granularity`, `quality_impact`, `confidence` | W_Q at 4-bit in layers 24-27 causes >2% quality loss (confidence 0.85) |
| **FallbackSignature** | `graph_motif_hash`, `fallback_risk`, `evidence_count`, `scope` | Dynamic reshape followed by gather has 73% fallback risk |
| **DeviceFingerprint** | `device_class`, `os_version`, `opset_version`, `behavior_hash`, `anomalies` | M2 Pro on macOS 15.3 shows different partitioning than M2 on macOS 15.3 for attention blocks |
| **StateTopologyOutcome** | `state_config_hash`, `package_count`, `ane_survival`, `latency`, `confidence` | Per-shard KV state with reverse ring-buffer survives ANE at batch=1 with <2ms overhead |
| **SyntheticTransferAnnotation** | `synthetic_pattern`, `real_model_pattern`, `transferability_score` | Boolean emulation microkernel results transfer to attention-mask patterns with score 0.6 |

## 6.2 Storage Model

- **Backend**: SQLite with one table per knowledge unit type.
- **Schema**: Each table has `id`, `version`, `timestamp`, `confidence`, `evidence_source`, `evidence_count`, `scope_device_classes`, `scope_os_versions`, `scope_opset_versions`, `conflict_priority`, plus type-specific fields.
- **Indexing**: Composite indexes on `(type_specific_key, scope_device_classes, scope_os_versions)` for fast lookup during compilation.
- **Immutability**: Knowledge units are append-only. Updates insert a new version with incremented version number. Old versions remain queryable for audit.
- **Snapshots**: The full knowledge store can be exported as a snapshot file (JSON + SQLite dump) for reproducibility.

## 6.3 Update Pipeline

1. **Observation ingestion**: A structured observation arrives from run ingestion or manual entry. It contains: observation type, observed values, device metadata, IR context hash, task identifier.
2. **Validation**: The observation is checked for structural validity and basic sanity (e.g., confidence cannot be 1.0 from a single observation, latency must be positive).
3. **Matching**: The system checks whether an existing knowledge unit matches the observation's key fields and scope.
4. **Update or Insert**:
   - If a matching unit exists: compute new confidence using Bayesian update (prior confidence + new evidence). If the observation conflicts, flag for review rather than silently overwriting.
   - If no matching unit: insert a new unit with initial confidence = `base_confidence(evidence_source)` (e.g., 0.3 for a single synthetic run, 0.5 for a single real-model run, 0.7 for a manual entry from a trusted source).
5. **Conflict detection**: If a new observation contradicts an existing unit with confidence > 0.8, create a conflict entry requiring manual resolution. Do not auto-resolve high-confidence conflicts.
6. **Pruning**: Knowledge units with evidence_count = 1 and age > 90 days and confidence < 0.3 are candidates for pruning. Pruning requires explicit approval.

## 6.4 Confidence Model

Confidence is a float in [0.0, 1.0] computed as follows:

- **Base confidence by evidence source**:
  - Single synthetic run: 0.2
  - Multiple synthetic runs (n >= 5): 0.4
  - Single real-model run: 0.35
  - Multiple real-model runs (n >= 5): 0.6
  - Compile failure (deterministic): 0.7
  - Load failure (deterministic): 0.8
  - Manual authoritative entry: 0.75
  - Cross-validated (synthetic + real agreement): 0.85

- **Confidence update rule**: Given existing confidence `c_old` and new evidence with source weight `w`:
  ```
  c_new = c_old + w * (1.0 - c_old) * agreement_factor
  ```
  Where `agreement_factor` = +1 if the new evidence agrees, -0.5 if it disagrees (asymmetric: disagreement reduces confidence less than agreement increases it, because a single negative observation should not override accumulated positive evidence without more weight).

- **Confidence decay**: Confidence decays by 1% per 30 days for observations not re-confirmed. Re-confirmation resets the decay clock.

## 6.5 Conflict Resolution

When two knowledge units with overlapping scope contradict each other:

1. **Priority by evidence count**: Higher evidence_count wins if the confidence difference is < 0.1.
2. **Priority by evidence source**: Real-model observations override synthetic observations at the same confidence level.
3. **Priority by recency**: If evidence count and source are equivalent, the more recent observation wins.
4. **Scope narrowing**: If the conflict appears device-specific, narrow the scope rather than choosing a winner. E.g., "survives on M2, falls back on M1" is more useful than "survives" or "falls back" globally.
5. **Manual override**: A human can force-resolve a conflict with a manual entry at confidence 0.9.

## 6.6 Synthetic-to-Real Transfer Strategy

Synthetic microkernels test isolated operator behavior. Real models test integrated topology behavior. The transfer strategy:

1. **Operator-level transfer**: Synthetic survival results for individual ops (e.g., "mb.softmax with axis=-1 survives ANE on M2") transfer directly to real-model compilation as initial legality hints with reduced confidence (multiply by 0.7).
2. **Pattern-level transfer**: Synthetic motif results (e.g., "linear-relu-reshape sequence") transfer to real-model subgraphs that match the same motif hash, with confidence scaled by pattern similarity (0.5-0.8).
3. **Topology-level non-transfer**: Synthetic results about graph size, state budget, or shard topology do NOT transfer. These must be learned from real models directly.
4. **Transfer annotations**: Every synthetic-derived knowledge unit used in a real-model compilation decision is tagged with `transfer_source=synthetic` and `transfer_confidence_adjustment`. This allows the system to track when synthetic assumptions were wrong.

## 6.7 When Knowledge Is Trusted

Knowledge is trusted (used as a hard constraint) when:
- Confidence >= 0.85 AND evidence_count >= 10 AND evidence from at least 2 independent sources.
- The observation is a deterministic failure (compile failure, load failure) — these are always trusted.

## 6.8 When Knowledge Is Advisory Only

Knowledge is advisory (used as a hint, not a constraint) when:
- Confidence < 0.85.
- Evidence comes from a single source or a single device class.
- The scope is narrow (only one OS version tested).
- The observation is about runtime performance (latency, throughput) — these are always advisory because they are device- and load-dependent.

---

# 7. Input Classes

## 7.1 Synthetic Microtasks

**Purpose**: Isolate and test individual ANE-legal operations or small operation sequences. Measure per-op survival, latency, and precision behavior on specific devices.

**What it teaches**: Operator-level legality, per-op latency distribution, fp16 precision bounds for simple computations, baseline compute-unit assignment per op.

**Risks of overfitting**: Isolated ops may survive ANE in microtask context but fail in larger graphs due to state budget or constant volume. Latency of isolated ops does not predict latency in integrated graphs due to scheduling and memory effects.

## 7.2 Synthetic Stateful Decode Tasks

**Purpose**: Test state read/write patterns, KV-cache topology, and stateful decode-step blocks in isolation. Verify that state persists correctly across predict() calls.

**What it teaches**: State serialization correctness, state budget limits, per-shard state overhead, decode-step latency, state-aware partitioning constraints.

**Risks of overfitting**: Synthetic state patterns may not match real KV-cache behavior (e.g., reverse ring-buffer vs. naive append). State budget that works for a 4-layer test may not scale to 28 layers.

## 7.3 Projection/LUT Tasks

**Purpose**: Test grouped scalar-LUT palettized projections at various bitwidths, group sizes, and granularities. Measure quality impact and ANE survival of palettized vs. dense projections.

**What it teaches**: Per-bitwidth quality loss, per-weight-type sensitivity, ANE compatibility of constexpr_lut operations, optimal group sizes for ANE.

**Risks of overfitting**: Quality impact on random-weight projections may not match quality impact on trained-model projections. The "optimal" group size from synthetic tests may not be optimal when combined with other graph structure.

## 7.4 Pathological Legality Probes

**Purpose**: Deliberately construct MIL graphs that are on the boundary of ANE legality: operations with unusual shapes, mixed precision, dynamic-looking but staticized patterns, and operator combinations that might trigger fallback.

**What it teaches**: ANE legality boundary, fallback-triggering graph motifs, shape constraints, precision interaction effects.

**Risks of overfitting**: Pathological cases may not represent real workload patterns. Over-indexing on edge cases can lead to overly conservative legality rules that reject valid graphs.

## 7.5 Imported MIL Dumps from Existing Projects

**Purpose**: Ingest MIL text dumps from real projects (e.g., the Qwen3 MIL dumps in `artifacts/coreml/mil/`) as both compilation targets and profiling inputs. These represent real ANE-surviving graphs.

**What it teaches**: Real graph structure that survives ANE, actual operator usage patterns, real constant/state volumes, proven shard boundaries.

**Risks of overfitting**: A single project's MIL structure may encode project-specific decisions rather than general ANE constraints. The system must not assume that what Qwen3 does is the only way.

## 7.6 Emitted MIL from New Compilation Runs

**Purpose**: The system's own compilation output, fed back into the lab for validation. Every emitted artifact should be profileable.

**What it teaches**: Whether the compiler's decisions actually produce ANE-placed, low-drift, performant artifacts. Compilation regression detection.

**Risks of overfitting**: None significant. This is self-validation, not external data.

## 7.7 Manually Authored Benchmark Graphs

**Purpose**: Expert-written MIL graphs designed to test specific hypotheses about ANE behavior. These are the equivalent of unit tests for the knowledge store.

**What it teaches**: Targeted knowledge about specific ANE behaviors, regression tests for known issues, validation of compiler assumptions.

**Risks of overfitting**: Expert bias — the benchmarks test what the expert thinks matters, which may miss unexpected behaviors.

## 7.8 Real Shard-Based Deployment Artifacts

**Purpose**: Complete deployment packages (io_model + decoder shards + sampler) from real projects. These are end-to-end validation targets.

**What it teaches**: End-to-end latency, cross-shard handoff overhead, real quality metrics (perplexity, accuracy), real state management behavior.

**Risks of overfitting**: A single deployment's characteristics may not generalize. The system must treat specific artifact behavior as evidence, not universal truth.

---

# 8. Compiler Pipeline

## 8.1 Canonicalization

**Input**: Semantic/Task IR (SIR).

**Output**: Canonical SIR with normalized operation ordering, deduplicated constants, and consistent naming.

**Deterministic transformation rule**:
- Sort commutative binary operations into a canonical operand order (lexicographic by name).
- Merge identical constant definitions into single references.
- Normalize naming to `op_type_index` format.
- Flatten nested identity operations (e.g., reshape to same shape).

**Rejection conditions**: None at this stage. If SIR is structurally valid, canonicalization always succeeds.

**Knowledge influence**: None. This pass is purely syntactic.

## 8.2 Staticization

**Input**: Canonical SIR.

**Output**: SIR with all dynamic constructs resolved to static equivalents. Symbolic shapes become concrete. Runtime-computed indices become precomputed tables.

**Deterministic transformation rule**:
- Replace all symbolic dimensions with concrete values from the deployment configuration (e.g., SEQ_LEN=4096, BATCH_SIZE=1).
- Replace runtime positional index computations with static lookup tables (e.g., RoPE tables, causal mask tables).
- Replace dynamic cache append/shift with fixed-shape state read/write (e.g., reverse ring-buffer pattern).
- Replace dynamic slice indices with static ranges.

**Rejection conditions**: Reject if a dynamic construct cannot be resolved to a static equivalent within the supported op set. E.g., data-dependent branching, truly variable-length sequences.

**Knowledge influence**: StateTopologyOutcome knowledge informs which staticization patterns are known to survive ANE (e.g., reverse ring-buffer vs. naive append).

## 8.3 State-Topology Resolution

**Input**: Staticized SIR.

**Output**: SIR with explicit state declarations, state ownership assignments, and state access patterns.

**Deterministic transformation rule**:
- Identify all state-like patterns (KV cache, running statistics, accumulated context).
- Assign each state a unique identifier and concrete shape.
- Determine state ownership: states that are accessed by operations in a single future shard are owned by that shard; states accessed across shard boundaries require explicit handoff.
- Replace implicit state access with explicit StateRead/StateWrite operations.
- For decode-step workloads: determine whether to use iOS 18+ StateType or simulate state via input/output tensors.

**Rejection conditions**: Reject if state ownership cannot be resolved without cross-shard mutable state sharing (which is unsupported).

**Knowledge influence**: StateTopologyOutcome and ShardTemplateKnowledge inform which state topologies are known to work.

## 8.4 Shard/Partition Planning

**Input**: SIR with state topology resolved.

**Output**: Shard plan (list of shards with assigned operations, roles, state ownership) and PIR skeleton.

**Deterministic transformation rule**:
- Compute per-operation cost estimates (FLOPs, parameter count, state size).
- Check against ANE budget thresholds (from knowledge store: SurvivalMatrixEntry, DeviceFingerprint).
- If the total graph exceeds a single shard's budget, partition into multiple shards using the following heuristics:
  - Group operations by layer boundary (decoder layers are natural partition points).
  - Assign entry role to the first shard (owns initial embedding/transform).
  - Assign exit role to the last shard (owns final norm and output projection).
  - Assign interior role to middle shards.
  - Ensure each shard's state ownership is self-contained.
- Check the shard plan against known-good shard templates from the knowledge store.
- If no template matches, create a new partition and flag it for empirical validation.

**Rejection conditions**: Reject if no valid partition exists within budget constraints and the supported op set.

**Knowledge influence**: ShardTemplateKnowledge, SurvivalMatrixEntry, DeviceFingerprint. This pass is heavily knowledge-driven.

## 8.5 Precision/Palette Assignment

**Input**: Shard plan + SIR with state topology.

**Output**: SIR with per-weight precision and palettization annotations.

**Deterministic transformation rule**:
- For each weight tensor, query the PrecisionHazard table for the given op type, weight type, and device class.
- Assign bitwidth based on the highest-confidence non-hazardous configuration.
- Default to 4-bit for early layers, 6-bit for middle layers, 8-bit for late/sensitive layers (generalized from Qwen3 evidence).
- Override defaults with knowledge-store entries where available.
- Assign granularity: `per_grouped_channel` with group_size=128 for projection weights (matching proven Qwen3 configuration).
- Assign LUT type: scalar (cluster_dim=1) by default; vector palettization only where explicitly supported by knowledge.

**Rejection conditions**: Reject if no safe bitwidth exists for a critical weight (confidence > 0.85 that all tested bitwidths cause unacceptable quality loss).

**Knowledge influence**: PrecisionHazard, SurvivalMatrixEntry (palettized ops must survive ANE).

## 8.6 Legality Rewrite

**Input**: Precision-annotated SIR with shard plan.

**Output**: ANE-Legal IR (AIR). All operations rewritten into ANE-legal form.

**Deterministic transformation rule**:
- For each SIR operation, check legality against the knowledge store (LegalityRule, SurvivalMatrixEntry).
- If legal, lower to the corresponding AIR node type.
- If illegal but a known rewrite exists (e.g., gather -> 1x1-conv permutation, dynamic reshape -> static reshape), apply the rewrite.
- If illegal with no known rewrite, reject.
- Annotate each AIR node with legality confidence from the knowledge store.

**Rejection conditions**: Reject if any operation is illegal and no known rewrite can make it legal.

**Knowledge influence**: LegalityRule, SurvivalMatrixEntry, FallbackSignature, MotifCatalog. This pass is the primary consumer of legality knowledge.

## 8.7 Risk Annotation

**Input**: AIR.

**Output**: AIR with fallback risk and drift risk annotations on every node and edge.

**Deterministic transformation rule**:
- For each AIR node, compute fallback risk as the maximum FallbackSignature confidence that matches the node's graph context.
- For each AIR edge, compute drift risk based on precision transition (e.g., fp16 output feeding into an op that may execute on CPU in fp32).
- Annotate the entire graph with aggregate risk scores.
- Flag any subgraph with fallback risk > 0.5 for empirical validation.

**Rejection conditions**: Reject if aggregate fallback risk > 0.8 (the graph is unlikely to survive ANE). Produce a diagnostic report instead.

**Knowledge influence**: FallbackSignature, DeviceFingerprint, SurvivalMatrixEntry.

## 8.8 MIL Lowering

**Input**: Risk-annotated AIR + shard plan + precision annotations.

**Output**: MIL-Emission IR (MIR). One MIR instance per shard.

**Deterministic transformation rule**:
- For each AIR node, emit the corresponding MIR node(s) with exact MIL Builder call specifications.
- Materialize all constants as MILConst nodes with concrete values.
- Assign names following `shard_op_index` convention.
- Insert type annotations (fp16 for all ANE-targeted tensors).
- Insert compute-unit hints per the shard plan.
- Emit state declarations as StateType-compatible MIL structures.
- For palettized weights: emit index tensors and LUT tensors as separate MILConst nodes with the appropriate constexpr_lut structure.

**Rejection conditions**: Reject if any AIR node cannot be mapped to a valid MIL op sequence.

**Knowledge influence**: None at this stage. MIR is a faithful lowering of AIR.

## 8.9 Core ML Tools Integration/Emission

**Input**: MIR instances (one per shard).

**Output**: Core ML MLModel objects (one per shard).

**Deterministic transformation rule**:
- Serialize MIR to the Rust/Python boundary format (JSON command file).
- Invoke the Python bridge subprocess.
- Python constructs MIL programs using `mb.program()` and `mb.<op>()` calls per the MIR specification.
- Python calls `ct.convert()` with `convert_to="mlprogram"`, specified opset, compute precision, and compute-unit hints.
- Python applies post-conversion palettization via `coremltools.optimize.coreml.palettize_weights()` per the precision annotations.
- Python returns the serialized mlpackage path or an error.

**Rejection conditions**: Reject on any coremltools exception, conversion failure, or output that does not match the MIR specification.

**Knowledge influence**: None directly. But the knowledge store should be updated if emission reveals that coremltools graph passes altered the expected structure.

## 8.10 Package Emission

**Input**: MLModel objects + PIR metadata.

**Output**: `.mlpackage` directories, manifests, reproducibility hashes.

**Deterministic transformation rule**:
- Save each MLModel to its designated `.mlpackage` path.
- Generate `manifest.json` for each package with: model name, role, compute units, opset version, state declarations, input/output specifications.
- Generate shard manifest listing all packages, their roles, and handoff specifications.
- Compute SHA-256 hashes of all package contents for reproducibility.
- Write PIR to disk alongside packages.

**Rejection conditions**: Reject if any package save fails or if hash computation detects inconsistency.

**Knowledge influence**: None. This pass is deterministic packaging.

---

# 9. Partitioning and Shard Model

## 9.1 Entry/Interior/Exit Semantics

**Entry shard**: The first shard in the decode chain. Responsible for receiving the token embedding (or raw token IDs if embedding is included), applying the initial transformer layers, and producing the hidden state for the next shard. Entry shards may include embedding lookup if the io-model is merged rather than separate. Entry shards have no upstream handoff (they receive io-model output or raw input).

**Interior shard**: A middle shard that receives hidden state from an upstream shard, applies its transformer layers, and passes hidden state to a downstream shard. Interior shards own their own KV state for their layer range. They have both upstream and downstream handoffs.

**Exit shard**: The final decoder shard. Receives hidden state from an upstream shard, applies its transformer layers including the final norm (RMSNorm), and produces the output hidden state for logit projection. Exit shards may include final-norm responsibility. They have upstream handoff but no downstream handoff (they feed io-model or sampler).

## 9.2 IO-Model Role

The io-model handles embedding lookup and logit projection. These operations are placed on CPU+GPU because:
- Embedding lookup involves gather operations that may not survive ANE.
- Logit projection operates on a large vocabulary dimension that may exceed ANE budget.
- The io-model is called once per decode step at the beginning (embedding) and once at the end (logit projection), so its latency is not in the critical path of the repeated shard chain.

The io-model may also handle OmniQuant-style blockwise weight-only quantization for the embedding/LM-head, which uses a different quantization scheme than the grouped LUT palettization used in the decoder shards.

## 9.3 Sampler Role

The sampler model handles next-token selection: temperature scaling, min-p pruning, repetition penalty, and noise injection. It runs on CPU+GPU. It is a dedicated Core ML MLProgram, not host-side post-processing. This ensures the entire decode loop can run on-device without returning to the host application for sampling logic.

## 9.4 State Ownership

Each decoder shard owns its own KV state tensors for its layer range. State is never shared across shards. When a shard completes, it writes its updated KV state internally (via StateType on iOS 18+ or via input/output tensor passing on earlier versions). The hidden state (the activation flowing between shards) is the only inter-shard data transfer.

State ownership rules:
- KV state for layers [N, M] is owned exclusively by the shard covering those layers.
- The io-model owns the embedding table state (if stateful).
- The sampler owns no persistent state (it is stateless per invocation).

## 9.5 When Sharding Is Required

Sharding is required when:
- The total parameter count (including palettized indices and LUTs) exceeds the ANE constant budget for a single model. The exact budget is device-dependent and must be determined empirically; the knowledge store tracks observed limits.
- The total state size (KV cache across all layers) exceeds the ANE state budget for a single model.
- The graph contains operations that must run on different compute units (e.g., some ops require CPU+GPU and would force fallback for the entire model if monolithic).

## 9.6 When Sharding Is Optional

Sharding is optional when:
- The model fits within a single shard's budget but partitioning might improve latency (e.g., parallel shard execution on different compute units).
- The model is small enough to be monolithic but the user wants to isolate specific components for independent update or replacement.

## 9.7 Shard Template Representation

Shard templates are stored as KIR units of type `ShardTemplateKnowledge`:

```
ShardTemplateKnowledge {
  template_id: "qwen3-three-shard-v1",
  partition_spec: [
    { role: entry, layers: [0, 10], compute_units: CPU_AND_NE },
    { role: interior, layers: [11, 19], compute_units: CPU_AND_NE },
    { role: exit, layers: [20, 27], compute_units: CPU_AND_NE },
  ],
  io_model: { compute_units: CPU_AND_GPU },
  sampler: { compute_units: CPU_AND_GPU },
  state_config: per_shard_kv_reverse_ring_buffer,
  context_length: 4096,
  known_good: true,
  quality_delta: { perplexity_delta: -0.57, confidence: 0.92 },
  latency_profile: { ... },
  evidence_count: 15,
  confidence: 0.92,
  scope: { device_classes: ["M2", "M2_Pro", "M3"], os_versions: ["macOS_15"] }
}
```

## 9.8 Known-Good Shard Pattern Capture

When a shard-based deployment artifact is validated (e.g., Qwen3 achieves perplexity parity), the system:
1. Extracts the shard plan (partition boundaries, roles, state topology, compute-unit assignments).
2. Generalizes layer-specific numbers into parameterized templates (e.g., "N layers split into K shards with entry/interior/exit roles").
3. Stores the template as a ShardTemplateKnowledge unit with the observed quality and latency data.
4. Tags the template with the evidence source and scope.

## 9.9 New Partitioning Exploration and Validation

When no known template matches the current compilation target:
1. The shard planner generates a candidate partition based on cost estimates and budget thresholds.
2. The candidate is flagged as `validation_required: true`.
3. The execution lab runs the candidate through compile/load/run checks.
4. If the candidate survives and meets quality targets, it is captured as a new shard template.
5. If the candidate fails, the failure is recorded as a knowledge unit (what partition boundary caused failure, what the fallback or quality impact was).
6. The planner may iterate with adjusted partition boundaries before producing a validated template.

---

# 10. Rust / Python Boundary

## 10.1 What Is in Rust

- All IR definitions (SIR, AIR, MIR, PIR, ProfIR, KIR).
- All compiler passes (canonicalization, staticization, state-topology resolution, shard planning, precision/palette assignment, legality rewrite, risk annotation, MIL lowering).
- All knowledge store logic (storage, querying, confidence updates, conflict resolution, pruning).
- All execution lab orchestration (task generation, run scheduling, result ingestion).
- All drift analysis logic.
- All artifact management (manifest generation, hashing, reproducibility).
- All CLI/workflow tooling.
- All report generation.

## 10.2 What Stays in Python/coremltools

- MIL Builder calls (`mb.program()`, `mb.<op>()`).
- `ct.convert()` invocation with all parameters.
- `coremltools.optimize.coreml.palettize_weights()` invocation.
- `MLModel.save()` for `.mlpackage` serialization.
- `MLModel.predict()` for runtime profiling.
- `MLModel.make_state()` for stateful model testing.
- `MLComputePlan` queries for compute-device usage inspection.
- Any coremltools-specific input type construction (`ct.TensorType`, `ct.StateType`, `ct.RangeDim`).

## 10.3 Boundary Format

The Rust and Python sides communicate via a **JSON command file** and a **JSON result file**. The Python bridge runs as a subprocess.

**Command file** (`emission_command.json`):
```json
{
  "command": "emit_mlprogram",
  "mir_spec": { ... },
  "config": {
    "opset_version": "iOS18",
    "compute_precision": "FLOAT16",
    "compute_units": "CPU_AND_NE",
    "optimization_hints": { "reshapeFrequency": "Infrequent" },
    "palettization": [
      {
        "weight_name": "W_Q_layer_0",
        "mode": "kmeans",
        "nbits": 4,
        "granularity": "per_grouped_channel",
        "group_size": 128,
        "channel_axis": 1
      }
    ]
  },
  "output_path": "/path/to/output.mlpackage"
}
```

**Result file** (`emission_result.json`):
```json
{
  "status": "success" | "error",
  "output_path": "/path/to/output.mlpackage",
  "error_message": null | "...",
  "coremltools_version": "8.1",
  "emission_hash": "sha256:...",
  "warnings": []
}
```

## 10.4 Serialization Between Both Sides

- **Rust → Python**: JSON command file written to a temporary directory. Rust spawns the Python subprocess with the command file path as an argument.
- **Python → Rust**: JSON result file written to the same temporary directory. Rust reads the result file after the subprocess completes.
- **Binary data** (weight tensors, LUT values): Written as `.npy` or raw binary files in the temporary directory. Referenced by path in the JSON command.
- **All data is self-contained**: The Python subprocess receives everything it needs via the command file and associated data files. It does not read from the Rust process's memory or share state.

## 10.5 Subprocess Strategy

- The Python bridge runs as a **separate subprocess** (not FFI, not embedded).
- Rust spawns `python3 bridge.py <command_file_path> <result_file_path>`.
- The subprocess has a timeout (default: 300 seconds for compilation, 60 seconds for prediction).
- If the subprocess times out or crashes, Rust treats it as an emission failure and records the error.
- The Python subprocess is stateless between invocations. No persistent Python process.

Rationale: Subprocess isolation prevents Python crashes from corrupting Rust state, ensures clean resource cleanup, and makes the boundary auditable. FFI (PyO3) is rejected because it creates a tight coupling that makes Python errors leak into Rust memory safety and makes the boundary harder to audit.

## 10.6 Determinism Preservation

- The MIR specification is fully deterministic given the same AIR input and knowledge store state.
- The Python bridge constructs MIL programs in a deterministic order (topological sort of MIR nodes).
- The command file includes an `emission_seed` that the Python bridge uses for any randomized operations (e.g., k-means initialization in palettization).
- The result file includes an `emission_hash` computed over the output `.mlpackage` contents, enabling bit-for-bit reproducibility checks.

## 10.7 Preventing Python from Swallowing Core Compiler Logic

- **No IR logic in Python**: Python never reads, modifies, or reasons about IR. It only executes the MIR specification.
- **No pass logic in Python**: Python never applies compiler passes. Graph passes that coremltools runs internally are treated as opaque backend behavior to be observed, not controlled.
- **No knowledge logic in Python**: Python never reads or writes the knowledge store.
- **No planning logic in Python**: Python never makes partitioning, precision, or legality decisions.
- **Audit rule**: If any Python file grows beyond 500 lines of logic (excluding data definition), it is a red flag that compiler logic is leaking into Python.

---

# 11. Core ML Tools Integration Strategy

## 11.1 What the System Uses

### MIL Builder
- Used to construct MIL programs programmatically via `mb.program()` decorator and `mb.<op>()` calls.
- The Rust side produces a complete MIR specification; the Python bridge translates it into Builder calls.
- No improvisation: every `mb.<op>()` call is specified in the MIR. The Python bridge does not add, remove, or reorder operations.

### MIL Input Types
- `ct.TensorType` for specifying input shapes and dtypes.
- `ct.StateType` for stateful model inputs (iOS 18+).
- `ct.RangeDim` for flexible batch dimensions (where applicable).
- All input type specifications are generated from the PIR by the Rust side and passed to Python in the command file.

### MIL Ops
- The system targets the `MIL::Core` dialect exclusively.
- Supported ops: `mb.const`, `mb.matmul`, `mb.conv`, `mb.add`, `mb.mul`, `mb.sub`, `mb.abs`, `mb.maximum`, `mb.minimum`, `mb.reshape`, `mb.transpose`, `mb.split`, `mb.concat`, `mb.softmax`, `mb.reduce_sum`, `mb.reduce_max`, `mb.slice_by_size`, `mb.cast`, `mb.squeeze`, `mb.expand_dims`, `mb.gather` (only where ANE-survival is confirmed), `mb.constexpr_lut_to_sparse` (for palettized weights).
- Any MIL op not in this list requires explicit approval via a knowledge-store entry confirming ANE survival.

### Graph Pass Interfaces
- The system does NOT directly invoke coremltools graph passes.
- `ct.convert()` runs internal graph passes automatically. The system treats these as opaque backend behavior.
- The `pass_pipeline` parameter may be used to disable specific passes that are known to interfere with ANE placement (tracked in the knowledge store).
- Any pass-pipeline configuration must be driven by knowledge-store entries, not hard-coded.

### Model/Package APIs
- `MLModel.save()` for writing `.mlpackage` to disk.
- `MLModel.predict()` for runtime profiling in the execution lab.
- `MLModel.make_state()` for stateful model testing.
- `MLComputePlan` for pre-computing compute-device assignment per op (where supported).
- `MLModel.get_spec()` for inspecting the emitted protobuf structure.

### Typed Execution Awareness
- All emitted MLPrograms are typed (explicit intermediate tensor types).
- The system sets `compute_precision=ct.precision.FLOAT16` for ANE-targeted packages.
- The system sets `compute_precision=ct.precision.FLOAT32` only for CPU-only reference models used in drift analysis.
- The system is aware that fp16-typed intermediates may execute in fp32 on CPU fallback, and annotates drift-risk accordingly.

### Stateful Model Support
- For iOS 18+ targets: `ct.StateType` is used to declare persistent KV state.
- For pre-iOS 18 targets: state is simulated via explicit input/output tensors (the older pattern).
- The system tracks which state model is in use and adjusts the PIR accordingly.

### Conversion-to-MLProgram Behavior
- The system always uses `convert_to="mlprogram"` (never `convert_to="neuralnetwork"`).
- The system always sets `minimum_deployment_target` explicitly (iOS 18 for stateful models, iOS 16 for non-stateful).
- The system never relies on default conversion behavior; all parameters are explicit.

### Package Save/Load Behavior
- All artifacts are saved as `.mlpackage` (directory format with separate weights).
- `.mlmodel` (single-file format) is never used.
- Loading for profiling uses `MLModel(path, compute_units=...)` with explicit compute-unit specification.

## 11.2 What the System Will NOT Delegate to coremltools

1. **Compiler passes**: The system does not delegate canonicalization, staticization, legality analysis, shard planning, precision assignment, or risk annotation to coremltools. These are Rust-owned compiler logic.
2. **Legality decisions**: The system does not assume that successful `ct.convert()` means ANE placement. Legality is determined by the knowledge store, not by conversion success.
3. **Palettization policy**: The system does not blindly apply `palettize_weights()` with default configuration. Bitwidth, granularity, and group-size decisions are made by the Rust-side precision engine and passed as explicit parameters.
4. **Graph optimization**: The system does not rely on coremltools graph passes to optimize for ANE. It produces MIR that is already in ANE-friendly form before emission.
5. **Partitioning**: The system does not ask coremltools to partition the graph. Shard boundaries are determined by the Rust-side shard planner.
6. **State management**: The system does not delegate state topology decisions to coremltools. State ownership and handoff are planned in Rust.

## 11.3 Clear Distinction

**"We use coremltools as a backend interface"**: coremltools is the emission target and runtime environment. We use its APIs to construct, serialize, and execute MIL programs. We respect its format requirements, opset versioning, and typed execution semantics.

**"coremltools becomes the compiler"**: This is what we prevent. If coremltools' internal graph passes, default palettization, or automatic partitioning are making decisions that we should be making, we have failed. The Rust side must own all compilation decisions; the Python side must only execute them.

---

# 12. Execution Lab

## 12.1 Task Generation

Tasks are generated from ProfIR specifications. Each task specifies:
- The MIL package to run (path to `.mlpackage`).
- Input data specification (shapes, dtypes, value ranges, or explicit numpy arrays).
- Baseline computation method (reference model, numpy computation, or fp32 CPU run).
- Metrics to capture (latency, throughput, numerical drift, fallback suspicion).
- Device requirements (compute units, minimum OS version).
- Repetition count and warmup count.

Task generation is deterministic: the same task spec produces the same input data (using seeded RNG) and the same measurement configuration.

## 12.2 Baseline Generation

Every profiling task has a baseline. Baseline options:
- **FP32 CPU reference**: Run the same model with `compute_units=CPU_ONLY` and `compute_precision=FLOAT32`. This is the gold standard for drift measurement.
- **Numpy computation**: For synthetic microtasks, compute the expected output directly in numpy (fp64).
- **Reference artifact**: For real-model tasks, use a known-good artifact (e.g., the original Qwen3 Core ML packages) as the baseline.

Baselines are generated before the profiling run and stored alongside the task spec.

## 12.3 Device Metadata Capture

Before each profiling session, capture:
- Device model identifier (e.g., "MacBookPro18,3").
- Chip identifier (e.g., "Apple M2 Pro").
- OS version (e.g., "macOS 15.3").
- Core ML version.
- Available compute devices (via `MLComputeDevice.get_all_compute_devices()`).
- Available memory.
- Thermal state (if accessible).

This metadata is stored with every run trace and used to scope knowledge-store entries.

## 12.4 Run Harness

1. Load the `.mlpackage` with specified `compute_units`.
2. If stateful, call `make_state()`.
3. Run warmup predictions (default: 5 iterations) to prime caches.
4. Run measured predictions (default: 20 iterations).
5. For each iteration:
   - Record wall-clock time before and after `predict()`.
   - Record `last_predict_duration_in_nano_seconds` from the MLModel object.
   - Record the output tensors.
6. Compute aggregate latency statistics (median, p90, p99, min, max).
7. Compute drift against baseline.

## 12.5 Result Schema

```json
{
  "task_id": "...",
  "run_id": "...",
  "timestamp": "2026-04-21T12:00:00Z",
  "device_metadata": { ... },
  "latency": {
    "median_ns": 1234567,
    "p90_ns": 1345678,
    "p99_ns": 1456789,
    "min_ns": 1100000,
    "max_ns": 2000000,
    "iterations": 20
  },
  "drift": {
    "cosine_distance": 0.0001,
    "max_absolute_error": 0.002,
    "mean_absolute_error": 0.0005,
    "relative_error_p99": 0.001
  },
  "fallback_suspicion": {
    "risk_score": 0.3,
    "evidence": ["latency_higher_than_expected", "ne_not_in_compute_plan"]
  },
  "output_snapshot_path": "/path/to/outputs.npy",
  "reproducibility_hash": "sha256:..."
}
```

## 12.6 Drift Metrics

- **Cosine distance**: `1 - cos_sim(output, baseline)`. Measures directional drift. Sensitive to scale-invariant errors.
- **Max absolute error**: `max(|output - baseline|)`. Measures worst-case single-element drift.
- **Mean absolute error**: `mean(|output - baseline|)`. Measures average drift magnitude.
- **Relative error p99**: 99th percentile of `|output - baseline| / max(|baseline|, epsilon)`. Measures worst-case proportional drift.

Drift is considered **acceptable** if:
- Cosine distance < 0.001.
- Max absolute error < 0.01 (for fp16 outputs).
- Relative error p99 < 0.01.

Drift is considered **concerning** if any metric exceeds 2x the acceptable threshold. Concerning drift triggers a knowledge-store update and a fallback suspicion check.

## 12.7 Fallback Suspicion Logic

Fallback suspicion is assessed via multiple signals:

1. **Latency anomaly**: If the measured latency is > 2x the expected ANE latency for the given FLOP count and device, suspect fallback.
2. **Compute plan inspection**: If `MLComputePlan` is available and shows any operation assigned to CPU or GPU when ANE was expected, confirm fallback.
3. **Precision consistency check**: Run the same model with `CPU_ONLY` and compare outputs. If the ANE-targeted run matches CPU_ONLY output more closely than expected (i.e., the "ANE" output is actually CPU output), suspect fallback.
4. **Cross-device comparison**: If the same model runs significantly faster on a device known to have better ANE support, the slower device may be falling back.

Fallback suspicion score is a float in [0.0, 1.0]:
- 0.0: No evidence of fallback.
- 0.3: Latency anomaly only.
- 0.6: Latency anomaly + precision consistency evidence.
- 0.9: Compute plan confirms fallback.

## 12.8 Task Families

### Linear/Projection
- Single matmul or 1x1-conv operation at various sizes.
- Tests: (1, 4096) x (4096, 4096), (1, 4096) x (4096, 11008), etc.
- Measures: ANE survival, latency scaling, fp16 precision.

### Grouped Scalar-LUT Projection
- Palettized linear projection at 2/3/4/6/8-bit with group_size=32/64/128/256.
- Measures: ANE survival of constexpr_lut ops, quality vs. bitwidth, latency overhead vs. dense.

### MLP Block
- gate_proj + up_proj + SiLU gating + down_proj.
- Measures: Integrated subgraph survival, fused operation behavior, precision interaction.

### Attention Microblock
- Q/K/V projections + RoPE + causal mask + softmax + value combination + output projection.
- Measures: Attention graph survival, RoPE table compatibility, mask handling, softmax ANE placement.

### Decode-Step with State
- Full decode step with state read, attention computation, FFN, state write.
- Measures: Stateful model correctness, state persistence, per-step latency, state overhead.

### Shape-Hostility Probes
- Graphs with unusual shapes: tall-skinny matrices, high-dimensional tensors, batch dimensions, etc.
- Measures: ANE shape constraints, which shapes trigger fallback.

### Operator-Remap Probes
- Non-native operations re-expressed as ANE-compatible patterns (inspired by ANE-sha256d).
- Boolean emulation via fp16 (AND=mul, XOR=abs(a-b), OR=max).
- 1x1-conv permutation for bit rotation.
- Measures: Which remapping patterns survive ANE, latency of emulated vs. native ops, precision of emulated Boolean logic.

### Shard-Survival Probes
- Multi-package deployments at various shard counts and boundaries.
- Measures: Which shard boundaries survive, inter-shard handoff latency, cross-shard state consistency.

---

# 13. Artifact Model

| Artifact | Format | Purpose |
|----------|--------|---------|
| SIR dump | JSON (MessagePack binary) | Debug, audit, reproducibility |
| AIR dump | JSON (MessagePack binary) | Debug, legality verification |
| MIR dump | JSON (MessagePack binary) | Debug, emission verification |
| PIR dump | JSON (MessagePack binary) | Debug, deployment audit |
| Pass report | Markdown + JSON | Compilation trace, decision log |
| MIL dump | Text (MIL text format) | Core ML Tools compatibility check |
| mlpackage | Directory (Apple format) | Deployment artifact |
| Package manifest | JSON | Metadata for each mlpackage |
| Shard manifest | JSON | Cross-package deployment specification |
| Task spec | TOML/JSON | Profiling task definition |
| Run trace | JSON + numpy arrays | Raw profiling results |
| Drift report | Markdown + JSON | Numerical drift analysis |
| Fallback suspicion report | Markdown + JSON | Fallback risk assessment |
| Backend knowledge snapshot | SQLite dump + JSON | Knowledge store export |
| Reproducibility hash | Text (SHA-256) | Bit-for-bit reproducibility verification |

All artifacts are stored under the project's output directory structure and are named with a combination of compilation-run-id and artifact type.

---

# 14. Repository Layout

```
MILLer/
├── SPEC.md                          # This specification
├── Cargo.toml                       # Workspace root
├── Cargo.lock
├── rustfmt.toml
├── clippy.toml
│
├── crates/
│   ├── ir/                          # IR definitions (SIR, AIR, MIR, PIR, ProfIR, KIR)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── sir.rs               # Semantic/Task IR
│   │       ├── air.rs               # ANE-Legal IR
│   │       ├── mir.rs               # MIL-Emission IR
│   │       ├── pir.rs               # Package/Deployment IR
│   │       ├── prof_ir.rs           # Profiling/Task IR
│   │       ├── kir.rs               # Backend-Knowledge Representation IR
│   │       └── serialize.rs         # Serialization utilities
│   │
│   ├── passes/                      # Compiler passes
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── canonicalize.rs
│   │       ├── staticize.rs
│   │       │                         # state_topology.rs removed (Sprint 58):
│   │       │                         #   State topology validation is planned
│   │       │                         #   future work for KV-cache state
│   │       │                         #   ownership analysis
│   │       ├── shard_plan.rs
│   │       ├── precision_policy.rs
│   │       ├── legality_rewrite.rs
│   │       ├── risk_annotate.rs
│   │       └── mil_lower.rs
│   │
│   ├── knowledge/                   # Backend knowledge store
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── store.rs             # SQLite-backed storage
│   │       ├── query.rs             # Query engine
│   │       ├── update.rs            # Update pipeline
│   │       ├── confidence.rs        # Confidence model
│   │       ├── conflict.rs          # Conflict resolution
│   │       ├── transfer.rs          # Synthetic-to-real transfer
│   │       └── snapshot.rs          # Export/import
│   │
│   ├── lab/                         # Execution lab
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── task_gen.rs          # Task generation
│   │       ├── baseline.rs          # Baseline generation
│   │       ├── device_meta.rs       # Device metadata capture
│   │       ├── harness.rs           # Run harness orchestration
│   │       ├── drift.rs             # Drift analysis
│   │       ├── fallback.rs          # Fallback suspicion logic
│   │       └── families/            # Task family definitions
│   │           ├── mod.rs
│   │           ├── linear.rs
│   │           ├── lut_projection.rs
│   │           ├── mlp_block.rs
│   │           ├── attention.rs
│   │           ├── decode_step.rs
│   │           ├── shape_hostile.rs
│   │           ├── op_remap.rs
│   │           └── shard_survival.rs
│   │
│   ├── bridge/                      # Rust/Python bridge
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── command.rs           # Command file generation
│   │       ├── subprocess.rs        # Python subprocess management
│   │       └── result.rs            # Result file parsing
│   │
│   ├── artifacts/                   # Artifact management
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── manifest.rs          # Manifest generation
│   │       ├── hashing.rs           # Reproducibility hashes
│   │       └── packaging.rs         # Package organization
│   │
│   ├── report/                      # Reporting layer
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── markdown.rs          # Markdown report generation
│   │       └── json_report.rs       # JSON report generation
│   │
│   └── cli/                         # CLI tool
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── compile.rs           # compile subcommand
│           ├── profile.rs           # profile subcommand
│           ├── query.rs             # knowledge query subcommand
│           ├── report.rs            # report subcommand
│           └── import.rs            # MIL import subcommand
│
├── python/
│   ├── bridge.py                    # Python bridge subprocess entry point
│   ├── mil_emitter.py              # MIL Builder construction from MIR spec
│   ├── converter.py                # ct.convert() invocation
│   ├── palettize.py                # Palettization invocation
│   ├── profiler.py                 # Runtime profiling via MLModel.predict()
│   ├── compute_plan.py             # Compute plan inspection
│   └── requirements.txt            # Python dependencies (coremltools, numpy)
│
├── benchmarks/
│   ├── synthetic/                   # Synthetic task specifications
│   │   ├── linear_projection.toml
│   │   ├── lut_projection.toml
│   │   ├── mlp_block.toml
│   │   ├── attention_micro.toml
│   │   ├── decode_step.toml
│   │   ├── shape_hostile.toml
│   │   ├── op_remap_sha256.toml
│   │   └── shard_survival.toml
│   ├── real/                        # Real model task specifications
│   │   └── qwen3_1.7b/
│   │       ├── task.toml
│   │       └── baseline_config.toml
│   └── mil_imports/                 # Imported MIL dump specifications
│       └── qwen3/
│           ├── import.toml
│           └── mil/                 # Symlink or copy of MIL text dumps
│
├── artifacts/                       # Generated artifacts (gitignored)
│   ├── compile/                     # Compilation outputs
│   │   └── <run-id>/
│   │       ├── sir.json
│   │       ├── air.json
│   │       ├── mir.json
│   │       ├── pir.json
│   │       ├── pass_report.md
│   │       ├── mil/
│   │       └── mlpackage/
│   ├── profile/                     # Profiling outputs
│   │   └── <run-id>/
│   │       ├── run_trace.json
│   │       ├── drift_report.md
│   │       ├── fallback_report.md
│   │       └── outputs/
│   └── knowledge/                   # Knowledge store snapshots
│       └── snapshot_<version>.json
│
├── knowledge/                       # Seed knowledge entries
│   ├── legality_seed.json           # Initial legality rules from Qwen3 evidence
│   ├── shard_template_seed.json     # Qwen3 three-shard template
│   └── precision_hazard_seed.json   # Initial precision hazards from Qwen3 evidence
│
├── docs/                            # Documentation (Markdown only)
│   ├── architecture.md
│   ├── ir_reference.md
│   ├── knowledge_schema.md
│   ├── bridge_protocol.md
│   └── profiling_methodology.md
│
├── .gitignore
└── README.md
```

---

# 15. Phased Build Plan

## Phase 0: Skeleton + Spec + Repo Layout

**Deliverables**:
- This specification document (`SPEC.md`).
- Complete repository layout with all crates, Python modules, benchmark directories, and documentation stubs.
- `Cargo.toml` workspace with all crate dependencies.
- Python `requirements.txt` with `coremltools>=8.0` and `numpy`.
- CI skeleton (build, test, clippy).
- Seed knowledge files from Qwen3 and ANE-sha256d evidence.

**Acceptance criteria**:
- `cargo build` succeeds for all crates (with stub implementations).
- `cargo test` succeeds for all crates (with no-op tests).
- Python bridge runs and reports version info.

**Explicit non-goals**:
- No compiler passes implemented.
- No knowledge store logic.
- No profiling capability.
- No real compilation or execution.

## Phase 1: IR + Legality Core

**Deliverables**:
- Complete SIR, AIR, MIR, PIR, ProfIR, KIR type definitions in `crates/ir/`.
- Serialization/deserialization for all IR types (MessagePack).
- Canonicalization pass (fully functional).
- Staticization pass (fully functional for supported patterns).
- Legality rewrite pass (with initial rule set derived from Qwen3 evidence).
- Knowledge store schema and SQLite backend in `crates/knowledge/`.
- Seed knowledge loaded from `knowledge/` files.
- Query API for knowledge store (legality rules, survival matrices).

**Acceptance criteria**:
- SIR can be constructed, serialized, deserialized, and validated.
- Canonicalization and staticization produce deterministic output for given input.
- Legality rewrite accepts known-legal Qwen3 patterns and rejects known-illegal patterns.
- Knowledge store can be queried and returns correct seed knowledge.
- Unit test coverage > 80% for all new code.

**Explicit non-goals**:
- No shard planning.
- No precision policy.
- No MIL emission.
- No profiling.
- No real Core ML interaction.

## Phase 2: Qwen-Like Shard-Aware Compilation Skeleton

**Deliverables**:
- State-topology resolution pass.
- Shard/partition planning pass (supports the Qwen3 three-shard template as initial pattern).
- Entry/interior/exit shard semantics implemented.
- PIR generation from shard plans.
- Risk annotation pass (initial version using knowledge store).
- CLI `compile` subcommand that accepts a task spec and produces a shard plan + PIR.

**Acceptance criteria**:
- Given a Qwen3-like task spec, the compiler produces a three-shard plan matching the left/mid/right pattern.
- State ownership is correctly assigned per shard.
- The shard plan passes risk annotation with fallback risk < 0.5 for known-good configurations.
- CLI produces human-readable pass reports.

**Explicit non-goals**:
- No MIL emission (this phase produces plans, not packages).
- No precision policy (uniform 6-bit placeholder).
- No profiling.
- No exploration of alternative shard patterns.

## Phase 3: Core ML Tools Bridge + MIL Emission

**Deliverables**:
- MIL lowering pass (AIR → MIR).
- Python bridge subprocess (`bridge.py`, `mil_emitter.py`, `converter.py`).
- Rust/Python command/result file protocol.
- `ct.convert()` integration with all required parameters.
- `.mlpackage` save via `MLModel.save()`.
- Package manifest generation.
- Reproducibility hashing.
- CLI `compile` subcommand extended to produce actual `.mlpackage` artifacts.

**Acceptance criteria**:
- Given a Qwen3-like task spec, the compiler produces valid `.mlpackage` artifacts for each shard.
- Each `.mlpackage` loads successfully in Core ML (on a macOS 15+ device).
- Emission is deterministic: same input + same knowledge = same output hash.
- Python bridge crashes are handled gracefully with clear error messages.
- Latency of emission is reasonable (< 60 seconds for a single shard).

**Explicit non-goals**:
- No palettization (emits dense weights only).
- No profiling.
- No knowledge updates from compilation results.

## Phase 4: Profiling Lab v0

**Deliverables**:
- Task generation from ProfIR specifications.
- Baseline generation (FP32 CPU reference).
- Device metadata capture.
- Run harness (warmup, measured runs, latency capture).
- Drift analysis (cosine distance, max/mean absolute error, relative error p99).
- Fallback suspicion logic (latency anomaly + compute plan inspection).
- Result schema and storage.
- CLI `profile` subcommand.
- At least 3 task families implemented: linear/projection, attention microblock, decode-step with state.

**Acceptance criteria**:
- Can profile a `.mlpackage` on a real Apple device and produce a complete run trace.
- Drift metrics are computed correctly against FP32 CPU baseline.
- Fallback suspicion is reported when latency anomaly is detected.
- Results are stored and queryable.
- Profiling is reproducible (same task + same device + same conditions = comparable results).

**Explicit non-goals**:
- No knowledge store updates from profiling results (that is Phase 5).
- No palettization profiling.
- No cross-device comparison automation.

## Phase 5: Backend Knowledge Adaptation v0

**Deliverables**:
- Run ingestion pipeline: profiling results → structured observations → knowledge store updates.
- Confidence model implementation.
- Conflict detection (flag contradictory observations).
- Synthetic-to-real transfer annotations.
- Knowledge store query API used by compiler passes.
- CLI `query` subcommand for knowledge inspection.
- Knowledge store snapshot export/import.

**Acceptance criteria**:
- Profiling results from Phase 4 are ingested and produce knowledge store updates.
- Confidence scores are computed correctly per the confidence model.
- Conflicting observations are detected and flagged for review.
- Compiler passes (legality, shard planning) can query the knowledge store and receive relevant entries.
- Knowledge store can be exported and re-imported without data loss.

**Explicit non-goals**:
- No automated conflict resolution (manual review only).
- No drift detection across knowledge versions.
- No confidence decay (all entries are fresh).

## Phase 6: Precision/Palette + Drift Intelligence

**Deliverables**:
- Precision/palette policy engine.
- Per-weight bitwidth assignment using knowledge store.
- Python bridge palettization integration (`palettize.py`).
- Palettization profiling task family.
- Precision hazard knowledge accumulation.
- Drift regression detection (if a knowledge-store entry's observed drift increases over time).
- CLI `compile` subcommand extended with palettization.

**Acceptance criteria**:
- Compiler can produce palettized `.mlpackage` artifacts with per-weight bitwidth assignments.
- Palettized artifacts load and run on real devices.
- Drift is measured for palettized vs. dense vs. FP32 baseline.
- Precision hazards discovered from profiling are recorded in the knowledge store.
- Compiler uses precision hazard knowledge to avoid known-bad bitwidth assignments.

**Explicit non-goals**:
- No vector palettization (cluster_dim > 1).
- No joint compression (prune + palettize).
- No training-aware palettization (post-training only).

## Phase 7: Generalized Partition Exploration

**Deliverables**:
- Alternative shard template generation (beyond left/mid/right).
- Automated partition exploration: given a model, try multiple partitionings and validate each.
- Shard-survival profiling task family.
- New shard template capture from validated explorations.
- Partition quality comparison reports.
- Support for models that are not strict Qwen3 derivatives.

**Acceptance criteria**:
- Compiler can propose and validate alternative shard patterns.
- At least one non-Qwen3 model is compiled and profiled successfully.
- New shard templates are captured as knowledge and reusable.
- Partition exploration produces a ranked list of candidates with quality/latency tradeoffs.

**Explicit non-goals**:
- No automated search over the full partition space (guided exploration only).
- No multi-model serving or batching.
- No support for models outside the constrained op set.

---

# 16. Acceptance Criteria

## v0 Criteria (End of Phase 4)

1. **Deterministic output**: Given the same task spec, same knowledge store state, and same seed, the compiler produces bit-for-bit identical `.mlpackage` artifacts.
2. **Shard-aware emission**: The compiler can produce multi-package deployments with correct inter-package handoff specifications and state ownership.
3. **Real device profiling**: The execution lab can profile `.mlpackage` artifacts on actual Apple devices and produce complete run traces with latency, drift, and fallback suspicion metrics.
4. **No fake ANE placement claims**: The system never claims to know exactly which operations the ANE will execute. Fallback suspicion is reported as a risk score, not a certainty.
5. **Useful fallback suspicion**: When fallback is suspected, the system provides actionable evidence (latency comparison, compute plan inspection result) rather than a binary yes/no.

## v1 Criteria (End of Phase 7)

1. **All v0 criteria still pass.**
2. **Synthetic + real MIL ingestion**: The system can import and reason over both synthetic MIL programs and real MIL dumps from existing projects.
3. **Numerical drift analysis**: The system can detect, quantify, and attribute numerical drift across palettization configurations, precision settings, and device classes.
4. **Knowledge-store updates from runs**: Every profiling run produces knowledge-store updates. The knowledge store reflects accumulated empirical observations.
5. **Precision/palette policy grounded in evidence**: Bitwidth and palettization assignments are driven by knowledge-store entries, not hardcoded defaults.
6. **Generalized shard exploration**: The system can propose, validate, and capture new shard templates beyond the initial Qwen3 three-shard pattern.
7. **Reproducibility**: Any compilation or profiling run can be reproduced from the task spec, knowledge store snapshot, and seed.
8. **No undocumented coremltools dependency**: Every coremltools API used is explicitly documented in the spec. No implicit reliance on coremltools internal behavior.

---

# 17. Risks

## R1: Overgeneralization

**Risk**: The system drifts from its narrow ANE-first focus into a general ML compiler, accumulating scope and complexity without depth.

**Mitigation**: The constrained op set is a hard boundary. Any operation not in the scope rules requires an explicit spec amendment. The anti-goals list is enforced by code review. The CLI rejects tasks that require unsupported operations.

## R2: Python Bridge Swallowing the Compiler

**Risk**: Compiler logic gradually migrates into Python because it is "easier" to implement near coremltools. The Rust side becomes a thin wrapper.

**Mitigation**: The 500-line audit rule (Section 10.7). The subprocess boundary is non-negotiable. Any logic that reasons about IR, makes compilation decisions, or manages knowledge must be in Rust. Python is only allowed to execute MIR specifications.

## R3: Synthetic Benchmark Overfitting

**Risk**: The knowledge store becomes dominated by synthetic microtask observations that do not transfer to real models. Compiler decisions based on synthetic data fail on real deployments.

**Mitigation**: Synthetic-to-real transfer annotations (Section 6.6). Confidence scaling for synthetic observations (lower base confidence). Mandatory real-model validation before any knowledge unit is promoted to "trusted" status. The system tracks which decisions were based on synthetic evidence and flags them.

## R4: False Knowledge from Noisy Profiling

**Risk**: Noisy device measurements produce contradictory observations. The knowledge store fills with low-confidence entries that cannot be resolved. Worse, a few noisy observations with high apparent confidence produce wrong rules.

**Mitigation**: The confidence model requires multiple independent observations before reaching high confidence (Section 6.4). Single observations start at low confidence (0.2-0.35). Conflict detection flags contradictions. High-confidence conflicts require manual resolution, not automatic averaging.

## R5: Inability to Infer Fallback Strongly Enough

**Risk**: Fallback detection relies on indirect signals (latency, compute plan). None of these are perfectly reliable. The system may report low fallback risk when fallback is actually occurring, or high fallback risk when the model is running correctly on ANE.

**Mitigation**: The system never claims fallback certainty. Fallback suspicion is a risk score with attached evidence, not a binary classification. Multiple independent signals (latency, compute plan, precision consistency) are combined. When signals disagree, the risk score reflects the ambiguity.

## R6: Shard Logic Becoming Hardcoded Folklore

**Risk**: The Qwen3 left/mid/right pattern becomes the only shard pattern anyone uses, not because it is optimal but because it is the only one that works. New models are forced into this pattern even when it is suboptimal.

**Mitigation**: Shard templates are parameterized, not hardcoded. Phase 7 explicitly builds partition exploration. The shard planner is designed to generate candidates, not just look up the Qwen3 template. New shard patterns must be empirically validated, but the system must make validation possible.

## R7: Palette/Precision Rules Becoming Ad Hoc

**Risk**: Precision assignments evolve through ad hoc tweaking ("increase bitwidth here because quality dropped") rather than systematic knowledge accumulation. The precision engine becomes a collection of special cases.

**Mitigation**: All precision decisions must be traceable to knowledge-store entries. Every precision assignment in the output includes the knowledge unit ID that informed it. Ad hoc overrides are allowed but must be flagged as `source=manual_override` with a lower confidence ceiling.

## R8: Accidental Overdelegation to coremltools Graph Behavior

**Risk**: coremltools' internal graph passes silently alter the emitted graph in ways that affect ANE placement, precision, or numerical behavior. The system does not notice because it treats coremltools as a black box.

**Mitigation**: The execution lab profiles every emitted artifact. If coremltools graph passes alter ANE placement, it shows up as unexpected fallback suspicion or latency changes. The system compares the actual MIL output (via `get_spec()`) against the expected MIR structure. Divergences are flagged as knowledge-store entries.

---

# 18. Final Recommendation

## Build First

1. **The IR core and legality engine** (Phase 1). Without a solid IR foundation and a working legality engine, nothing else matters. This is the load-bearing wall.
2. **The Qwen3 shard-aware skeleton** (Phase 2). This proves the system can reason about the most important real-world evidence. If it cannot reproduce the Qwen3 decomposition, the system is wrong at a fundamental level.
3. **The Core ML Tools bridge** (Phase 3). This is where the system touches reality. Emitting a valid `.mlpackage` that loads on a real device is the first moment of truth.

## Postpone

1. **Generalized partition exploration** (Phase 7). This is valuable but depends on a working profiling lab and knowledge store. Premature exploration without empirical validation is speculation.
2. **Vector palettization and joint compression**. These are optimization surfaces that require deeper empirical grounding. Postpone until the basic scalar-LUT palettization is validated.
3. **Automated conflict resolution**. Manual review is sufficient for v1. Automated resolution risks silently resolving conflicts in the wrong direction.

## Reject Outright

1. **Generic multi-backend support**. This system is ANE-first. CPU and GPU paths exist only as fallback targets and reference baselines. Building a multi-backend compiler is a different project.
2. **Training-aware palettization integration**. Post-training palettization is the realistic starting point. Training-aware methods (DKM, SKM) require a training infrastructure that is orthogonal to this system's purpose.
3. **Arbitrary dynamic control flow inside Core ML**. This is a fundamental scope violation. The ANE does not support it, and pretending otherwise produces systems that compile but do not run.
4. **Claims of exact ANE placement knowledge**. Apple does not document the ANE scheduling algorithm. Any system that claims to know exactly which ops run on the ANE is lying or relying on undocumented behavior that will break.
5. **Cloud-dependent compilation or knowledge**. This system must work offline. Any design that requires network access for compilation decisions or knowledge queries is rejected.
