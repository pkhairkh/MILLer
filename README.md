# MILLer

An **ANE-first** multi-level compiler and empirical Core ML lab. Emits MIL/mlpackage artifacts from Rust IR, accumulates backend knowledge from real Core ML behavior, and progressively replaces Python/coremltools boundary ownership with native Rust proto-direct serialization and FFI.

## What This Project Does

`MILLer` is a research compiler that targets the Apple Neural Engine (ANE) through Core ML's ML Program format. It provides:

- **Multi-level IR** — SIR (semantic), AIR (analytic), MIR (machine), PIR (partition) — with lowering passes between each level
- **Knowledge-driven compilation** — stored empirical evidence (precision hazards, ANE fallback risk, compute plan observations) materially changes compiler behavior on active paths
- **Shard-aware compilation** — multi-shard pipeline decomposition (Entry/Interior/Exit, QKV/Attention/OutputProjection) with role-specific op structure, not just dimension changes
- **Proto-direct emission** — Rust-native protobuf serialization via `prost` produces valid `.mlpackage` artifacts without the Python bridge, including true cross-function weight sharing that coremltools 9.0 cannot do
- **Compute plan offline verification** — structural proofs of op-to-device placement that can be verified on any platform, even without Apple hardware
- **Host-side evidence loop** — task generation, compilation, baseline computation, drift detection, and knowledge store persistence in a single `lab-loop` command

## Project Status

The project compiles and passes **440 tests** on Linux x86_64. End-to-end validation of emitted models (loading in Core ML runtime, predict() output, actual ANE placement) requires Apple hardware with macOS.

| Verification Level | Status |
|---|---|
| Compiles (`cargo build`) | Yes — zero warnings |
| Unit tests (`cargo test`) | Yes — 440 passing |
| Python bridge produces `.mlpackage` | Yes — via coremltools 9.0 on Linux |
| Proto-direct emission produces `.mlpackage` | Yes — via `prost` protobuf on any platform |
| Apple device/runtime verified | No — requires macOS with Core ML runtime |

## Architecture

```
TOML task spec
    │
    ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   SIR    │──▶│   AIR    │──▶│   MIR    │──▶│  PIR     │
│(semantic)│   │(analytic)│   │(machine) │   │(partition)│
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌──────────┐  ┌──────────┐  ┌──────────────┐
            │ Python   │  │ Proto-   │  │ FFI          │
            │ Bridge   │  │ Direct   │  │ (C API)      │
            │(coreml-  │  │(prost +  │  │(CoreML.frame-│
            │ tools)   │  │ weight   │  │ work, macOS) │
            │          │  │ sharing) │  │              │
            └──────────┘  └──────────┘  └──────────────┘
                    │              │              │
                    ▼              ▼              ▼
               .mlpackage     .mlpackage     On-device
               (verified)    (smaller)      prediction
```

### Workspace Crates

| Crate | Purpose |
|---|---|
| `ane-ir` | SIR, AIR, MIR, PIR types; task specs; shard pipeline definitions; `ShardOpProfile` for role-specific op structure |
| `ane-passes` | Compilation passes: Canonicalize, Staticize, PrecisionPolicy, LegalityRewrite, RiskAnnotate, ShardPlan, MilLower, **RoleMirBuilder** |
| `ane-knowledge` | File-backed knowledge store with query/update/conflict resolution; **compute plan offline verification** (`ComputePlanVerifier`) |
| `ane-lab` | Task generation (5 families), baseline computation, drift detection, fallback suspicion, lab run harness |
| `ane-bridge` | Python bridge subprocess + **proto-direct emission** (Rust-only path via `mir_to_compat`) |
| `ane-coreml-proto` | Core ML protobuf definitions compiled via `prost-build`; bidirectional type conversion |
| `ane-coreml-emit` | Direct mlpackage emission: `WeightBinBuilder` (with deduplication metrics), `MlPackageWriter`, `ProtoEmitter` |
| `ane-coreml-ffi` | C API FFI skeleton: `extern "C"` functions, `coreml_validate_proto_package()`, cross-platform validation |
| `ane-artifacts` | Manifest generation, content hashing, deterministic packaging |
| `ane-report` | JSON and Markdown report generation |
| `ane-trace` | **HuggingFace Transformers model tracing** via `torch.fx`; ANE-faithful SIR construction; versioned compilation with per-family constraint enforcement; model architecture registry (GPT-2, LLaMA/Qwen, BERT, Phi) |
| `ane-cli` | CLI entry point: `compile`, `compile-full`, `compile-sharded`, `compile-full-sharded`, `lab`, `lab-loop`, `generate-tasks`, `profile`, `report`, `package`, **`trace-compile`** |

### Task Families

Five task families are active with dedicated emission paths:

| Family | CLI Flag | MIL Program | Key Ops |
|---|---|---|---|
| Linear Projection | `--family linear` | `mb.linear` | Linear + bias |
| LUT Projection | `--family lut` | `mb.gather` | Grouped scalar-LUT palettized pattern |
| Decode Step | `--family decode` | `mb.linear` + `mb.scaled_dot_product_attention` + `mb.read_state`/`mb.coreml_update_state` | QKV → Attention → Output (stateful) |
| MLP Block | `--family mlp` | `mb.linear` + `mb.gelu(mode="TANH_APPROXIMATION")` | Up-proj → GELU → Down-proj |
| Attention | `--family attn` | `mb.linear` + `mb.scaled_dot_product_attention` | QKV → Multi-head attention → Output |
| Shape Hostile | `--family shape` | `mb.linear` | Odd/prime/mismatched dimension stress testing |
| Op Remap | `--family remap` | `mb.linear` | Formulation equivalence testing |
| Shard Survival | `--family survival` | `mb.linear` | Shard boundary survival testing |

## Model Tracing and Versioned Compilation

The `ane-trace` crate extends MILLer to trace HuggingFace Transformers models and compile them into ANE-faithful computation graphs with per-family constraint enforcement. The `trace-compile` CLI command provides the end-to-end workflow:

```
HuggingFace Model (PyTorch)
    │
    ▼
torch.fx symbolic trace (Python subprocess)
    │
    ▼ JSON TracedGraph
    │
    ▼
ane-trace: SIR construction + ANE-faithful decomposition
    │
    ▼
VersionedCompiler: per-family constraint validation
    │
    ▼
Standard MILLer Pipeline: SIR → AIR → MIR → Core ML emission
```

### Supported Model Architectures

| Architecture | Model Type | Key Decomposition |
|---|---|---|
| GPT-2 family | Causal LM | QKV projection → SDPA → output projection |
| LLaMA / Qwen family | RoPE-based | RoPE tables + grouped-query attention |
| BERT family | Encoder-only | Bidirectional attention + pooler |
| Phi family | Small form factor | Parallel attention + MLP |

Custom architectures can be registered via `ModelRegistry::register()`.

### ANE Version-Aware Compilation

The `VersionedCompiler` enforces constraints specific to each ANE family:

| ANE Family | Devices | SDPA Support | Key Constraints |
|---|---|---|---|
| A11 Legacy | A11, A12, A13 | No | Limited op set, strict alignment |
| A14 | A14, M1 | Partial | FP16 only, 8-channel alignment |
| A15 | A15, M2 | Yes | FP16, improved fusion |
| A16 | A16, M3, M4 | Yes | Mixed-precision, enhanced fusion |
| A18 | A17 Pro, M4+ | Yes | Full mixed-precision, expanded op set |

The `--target-family` flag selects the constraint profile. The `--ane-only` flag rejects any op that would fall back to CPU.

### Usage

```bash
# Trace and compile a HuggingFace model for A16 (default, reliable SDPA)
ane-cli trace-compile --model "bert-base-uncased" --output artifacts/bert

# Trace with specific ANE family and custom input shapes
ane-cli trace-compile \
  --model "gpt2" \
  --target-family A15 \
  --batch-size 1 \
  --seq-len 64 \
  --output artifacts/gpt2

# Enforce ANE-only compilation (reject CPU-fallback ops)
ane-cli trace-compile \
  --model "meta-llama/Llama-2-7b-hf" \
  --target-family A16 \
  --ane-only \
  --output artifacts/llama

# Load a pre-traced graph (skip Python tracing)
ane-cli trace-compile --model traced_graph.json --output artifacts/pretraced
```

## Key Capabilities

### 1. Role-Specific Sharding (Sprint 43 + Sprint 44)

Shard roles produce **genuinely different op structures end-to-end**, not just dimension changes. The Rust `RoleMirBuilder` (Sprint 43) and Python bridge emitters (Sprint 44) now both produce role-specific ops:

| Role | Op Profile | Ops Produced |
|---|---|---|
| Entry | `EntryLinear` | Const → Linear → **Reshape** |
| Interior | `InteriorLinear` | Const → Linear → **GELU** |
| Exit | `ExitLinear` | Const → Linear → **LayerNorm** |
| QKV Projection | `QkvProjection` | Const → Linear → **Split** |
| Attention | `AttentionComputation` | **ReadState** → **SDPA** → **UpdateState** |
| Output Projection | `OutputProjection` | Const → Linear → **LayerNorm** |
| IO Embedding | `IoEmbedding` | Const → **Gather** (CPU+GPU) |
| Sampler | `SamplerTopk` | **Topk** → **Softmax** (CPU+GPU) |

The `RoleMirBuilder` produces these graphs from `ShardSpec` + `ShardOpProfile`, and the `op_type_signature()` function proves they differ structurally. Before Sprint 43, all decoder shards produced the same `[Const, Linear]` structure.

### 2. Compute Plan Offline Verification

The `ComputePlanVerifier` in `ane-knowledge` proves structural properties of compute plans without Apple hardware:

- **Structural proof** — `ComputePlanProof` captures op-to-device placement in a deterministic, hashable form; tamper detection via SHA-256 hash integrity checks
- **Knowledge cross-reference** — placements are checked against known op-to-device mappings (e.g., `mb.linear` → NeuralEngine, `mb.embedding` → CPU) with conflict detection
- **Invariant checking** — verifies op count consistency, ANE count consistency, duplicate detection, and valid device classes
- **Synthetic prediction** — `predict_proof()` generates a predicted compute plan from op lists, enabling predict-then-verify roundtrips on any platform

### 3. True Weight Sharing via Proto-Direct Emission

coremltools 9.0's `add_function()` + standard conversion **duplicates** constants across function boundaries — each function gets its own copy in `weight.bin`. coremltools does provide cross-function deduplication via the `save_multifunction()` API (which internally assigns shared `weight_id` values), but the direct `add_function()` + `ct.convert()` path does not perform this deduplication. The proto-direct emission path (`ane-coreml-emit`) stores shared weights **once** and both functions reference the same offset, and additionally offers opt-in content-hash deduplication for differently-named weights with identical data — producing smaller mlpackages than either coremltools path.

The `WeightBinBuilder` tracks deduplication metrics:
- `deduplicated_count` — number of deduplication events
- `deduplicated_bytes` — total bytes saved
- Shape/dtype mismatch rejection — prevents silent corruption

### 4. Knowledge-Driven Compilation

Two passes materially adapt from stored empirical knowledge:

| Pass | Knowledge | Effect |
|---|---|---|
| `PrecisionPolicyPass` | Precision hazard (e.g., fp16 causes quality degradation) | Overrides dtype to fp32 |
| `ShardPlanPass` | ANE fallback risk (e.g., op falls back to CPU) | Overrides compute units to CPU+GPU |

Adaptation provenance propagates through SIR → AIR → MIR → bridge payload, and is recorded in the compile manifest.

### 5. Unified Verification Harness (Sprint 40)

A four-dimension verification harness checks emitted mlpackage artifacts against compiler intent:

| Dimension | Method (macOS) | Method (Linux) |
|---|---|---|
| Op graph fidelity | MLModelStructure | Spec-based extraction (coremltools) |
| Compute-unit placement | MLComputePlan | Unavailable |
| State conformance | MLModelStructure | Spec-based StateType + op detection |
| Multi-function conformance | MLModelStructure | Spec-based function counting |

The `verify` bridge command produces structured JSON artifacts with an overall weighted score.

## Quick Start

### Prerequisites

- **Rust** 1.75+ (`rustup` recommended)
- **protoc** 3.x (`apt install protobuf-compiler` on Ubuntu, `brew install protobuf` on macOS)
- **Python 3.9+** with `coremltools` 9.0+ (for Python bridge emission; not required for proto-direct)

### Build

```bash
cargo build --workspace
cargo test --workspace    # 440 tests
```

### Generate Tasks and Compile

```bash
# Generate deterministic task specs
cargo run -p ane-cli -- generate-tasks --family linear --output artifacts/gen_linear --seed 42
cargo run -p ane-cli -- generate-tasks --family decode --output artifacts/gen_decode --seed 42
cargo run -p ane-cli -- generate-tasks --family mlp   --output artifacts/gen_mlp    --seed 42
cargo run -p ane-cli -- generate-tasks --family attn  --output artifacts/gen_attn   --seed 42
cargo run -p ane-cli -- generate-tasks --family lut   --output artifacts/gen_lut    --seed 42

# Compile a single task (fast path)
cargo run -p ane-cli -- compile \
  --input artifacts/gen_linear/LinearProjection/linear_64x32_b1_fp16.toml \
  --output artifacts/compile/run \
  --bridge python/bridge.py

# Compile through the full pass pipeline (knowledge-informed)
cargo run -p ane-cli -- compile-full \
  --input artifacts/gen_linear/LinearProjection/linear_64x32_b1_fp16.toml \
  --output artifacts/compile_full/run \
  --bridge python/bridge.py \
  --knowledge knowledge
```

### Sharded Compilation

```bash
# 3-shard Entry/Interior/Exit compile
cargo run -p ane-cli -- compile-sharded \
  --input benchmarks/synthetic/sharded_linear_pipeline.toml \
  --output artifacts/sharded/run \
  --bridge python/bridge.py \
  --knowledge knowledge

# Full pass pipeline per shard (knowledge-informed per shard)
cargo run -p ane-cli -- compile-full-sharded \
  --input benchmarks/synthetic/sharded_linear_pipeline.toml \
  --output artifacts/sharded_full/run \
  --bridge python/bridge.py \
  --knowledge knowledge
```

### Host-Side Evidence Loop

```bash
# Complete loop: task → compile → baseline → drift → knowledge store persistence
cargo run -p ane-cli -- lab-loop \
  --input artifacts/gen_linear/LinearProjection/linear_64x32_b1_fp16.toml \
  --output artifacts/lab_loop/run \
  --bridge python/bridge.py \
  --knowledge knowledge
```

### Python Bridge (standalone)

```bash
# Test the bridge directly without Rust
python3 python/bridge.py <(echo '{"command":"emit_linear_projection","task_name":"test","input_dim":64,"output_dim":32,"batch_size":1,"dtype":"fp16","opset_version":"iOS18","compute_units":"CPU_AND_NE","output_path":"/tmp/ane_test","seed":42}') /tmp/result.json
cat /tmp/result.json
```

### Verify an Emitted Model

```bash
# Verify a compiled mlpackage against compiler intent (Sprint 46)
cargo run -p ane-cli -- verify \
  --mlpackage artifacts/compile/run/linear_64x32_b1_fp16.mlpackage \
  --output artifacts/verify/result.json \
  --bridge python/bridge.py
```

### Smoke Test

```bash
./scripts/smoke_test.sh
```

The smoke test verifies the narrow compile path and honestly reports limitations (e.g., missing coremltools, no Apple hardware for predict()).

## Verification Honesty

This project is explicit about what it can and cannot verify:

| Claim | Verified On | Not Verified On |
|---|---|---|
| Rust code compiles and tests pass | Linux x86_64 | — |
| Python bridge emits valid `.mlpackage` | Linux with coremltools | — |
| Proto-direct emission produces `.mlpackage` | Any platform | — |
| Weight sharing reduces `weight.bin` size | Linux (dedup metrics) | — |
| Role-specific MIR graphs differ structurally | Linux (unit tests) | — |
| Compute plan proofs verify structurally | Any platform | — |
| Op graph fidelity (MIR vs emitted ops) | Linux (spec-based) | macOS gives MLModelStructure fidelity |
| State conformance (read/write ops) | Linux (spec-based) | macOS gives MLModelStructure fidelity |
| Multi-function conformance | Linux (spec-based) | macOS gives MLModelStructure fidelity |
| Model loads in Core ML runtime | — | Requires macOS |
| `predict()` produces correct output | — | Requires macOS |
| Actual ANE placement matches predicted | — | Requires macOS + MLComputePlan |
| Compute plan harvest from real device | — | Requires macOS + MLComputePlan |

See [STATUS.md](STATUS.md) for exhaustive verification status of every component.

## Directory Layout

```
MILLer/
├── crates/
│   ├── ir/              # SIR, AIR, MIR, PIR types; task specs; shard pipeline definitions
│   ├── passes/          # Compilation passes including RoleMirBuilder
│   ├── knowledge/       # Knowledge store; compute plan offline verification
│   ├── lab/             # Task generation, baseline, drift, fallback detection
│   ├── bridge/          # Python bridge subprocess + proto-direct emission
│   ├── coreml-proto/    # Core ML protobuf definitions (prost-build)
│   ├── coreml-emit/     # Direct mlpackage emission with weight sharing
│   ├── coreml-ffi/      # C API FFI skeleton for Core ML framework
│   ├── artifacts/       # Manifest generation, hashing, packaging
│   ├── report/          # JSON and Markdown reporting
│   ├── trace/           # HuggingFace model tracing and ANE-faithful SIR construction
│   └── cli/             # CLI entry point (ane-cli) including trace-compile
├── python/
│   ├── bridge.py        # Bridge dispatch (15 commands including verify)
│   ├── mil_emitter.py   # MIL program construction (10 emission paths)
│   ├── trace_model.py   # torch.fx model tracing for HuggingFace Transformers
│   ├── converter.py     # MIL → MLModel conversion
│   ├── compute_plan.py  # Compute plan inspection (macOS only)
│   ├── model_structure.py # MLModelStructure inspection + MIR comparison
│   ├── verify.py        # Unified verification harness (4 dimensions)
│   ├── profiler.py      # Model profiling (macOS only)
│   └── palettize.py     # Palettization support
├── knowledge/           # Seed knowledge entries (legality, precision, shard templates)
├── benchmarks/          # Synthetic task specifications
├── scripts/
│   └── smoke_test.sh    # Honest smoke test with limitation reporting
├── docs/                # Architecture, IR reference, bridge protocol, knowledge schema
├── SPEC.md              # Full project specification
├── STATUS.md            # Exhaustive verification status
└── TASKS.md             # Sprint tracker with task-level detail
```

## Key Design Decisions

1. **Rust-heavy, Python at boundary only** — The compiler is Rust-first. Python/coremltools is used only for the emission boundary. Proto-direct emission (Sprint 41) progressively eliminates even this dependency. Model tracing (`ane-trace`) uses a Python subprocess for `torch.fx` symbolic tracing, but the traced graph is immediately handed back to Rust via JSON for all compilation and constraint enforcement.

2. **ANE-faithful, not just ANE-compatible** — The `ane-trace` crate enforces ANE constraints *during* lowering, not after. Operations that would fall back to CPU at runtime are detected and either decomposed into ANE-native sequences or flagged as violations. The `VersionedCompiler` applies per-family constraint profiles (A11 through A18) so the compiled graph is faithful to the target hardware generation.

3. **Truth over claims** — Every feature honestly reports its verification scope. The `ArtifactManifest` includes `implementation_status`, `verification_scope`, and `environment_limitations` fields. The knowledge store rejects observations with zero evidence.

4. **Knowledge must materially change behavior** — "Self-learning compiler" claims are justified only where stored evidence changes pass pipeline output on an active path. The `PrecisionPolicyPass` and `ShardPlanPass` are the two proven knowledge-affecting passes.

5. **Sharding is structurally real** — Shard roles produce different op structures, not just dimension changes. `ShardOpProfile` determines the MIR op sequence; `RoleMirBuilder` constructs the graph; `op_type_signature()` proves structural divergence.

6. **Proto-direct weight sharing solves a real coremltools gap** — coremltools 9.0 duplicates constants per function boundary. Rust-controlled `weight.bin` layout stores shared weights once, producing smaller packages with deduplication metrics to prove it.

## Residuals

On-device profiling, fallback detection with real timing, numerical drift measurement with actual model output, and compute plan harvesting from real hardware all require Apple hardware and are **not exercised** on the active path in this host. The system is honest about what it cannot verify without a real device. Baseline reference computation and drift detection infrastructure are implemented and will produce real metrics when run on Apple hardware. See [STATUS.md](STATUS.md) for complete details.
