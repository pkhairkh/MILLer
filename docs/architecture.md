# Architecture

See [SPEC.md](../SPEC.md) for the full specification.

## Current State

One end-to-end path: TOML task spec → Rust IR → JSON bridge → Python MIL emission → .mlpackage → manifest + knowledge update.

The Python bridge now wires converter.py, palettize.py, and compute_plan.py as proper bridge commands, not standalone scripts.

See [CHANGELOG.md](../CHANGELOG.md) for implementation history and verification status.

### Key Files

| File | Role |
|------|------|
| `crates/ir/src/task_spec.rs` | TOML task spec parsing |
| `crates/ir/src/linear_slice.rs` | SIR→MIR lowering, bridge payload, function descriptors |
| `crates/ir/src/pir.rs` | Package IR with `FunctionEntry` (multifunction seam) |
| `crates/ir/src/ane_target.rs` | `AneFamily`, `AneRevision`, `AneTarget` types with chip-to-family mapping |
| `crates/ir/src/ane_engine.rs` | `AneEngine` enum (NE/PE/TransposeEngine) with per-MirOp engine assignment |
| `crates/ir/src/ane_hw_limits.rs` | `AneHwLimits` per-revision hardware limit enforcement (has `verified: bool` field indicating whether limits have been confirmed on real hardware) |
| `crates/ir/src/ane_layout.rs` | `AneInterleave`/`AneLayout` types for interleave and ChannelLast constraints |
| `crates/bridge/src/subprocess.rs` | Rust→Python subprocess bridge, `BridgeResult` with full field capture |
| `crates/cli/src/main.rs` | CLI compile, report, trace-compile commands, deterministic task hashing, spec-driven manifest |
| `crates/trace/src/graph.rs` | `TracedGraph`, `TracedNode`, `TracedOp`, `TensorShape` data structures |
| `crates/trace/src/discovery.rs` | Strategy discovery for traced models — bridges strategy framework with tracing pipeline |
| `crates/trace/src/config.rs` | `TraceConfig`, `TraceTarget`, `InputShape` — tracing configuration |
| `crates/trace/src/sir_build.rs` | `build_sir_from_trace()` — ANE-faithful SIR construction from traced graphs |
| `crates/trace/src/versioned.rs` | `VersionedCompiler`, `AnceFaithfulnessReport` — version-aware constraint validation |
| `crates/trace/src/subprocess.rs` | `trace_model()` — Python subprocess launcher for torch.fx tracing |
| `crates/artifacts/src/hashing.rs` | SHA-256 content hashing (file and byte), `hash_file` now implemented |
| `crates/artifacts/src/packaging.rs` | Artifact packaging — deterministic zip archives of compile output |
| `crates/report/src/markdown.rs` | Markdown report generation — compilation, knowledge, diagnostics reports |
| `crates/report/src/json_report.rs` | JSON report generation — structured machine-readable reports |
| `crates/passes/src/*.rs` | Pass pipeline (8 passes): 3 have real transformation logic (LegalityRewrite, MilLower, RiskAnnotate), 5 are pass-throughs for the current linear projection slice. Wired into `compile-full` CLI subcommand. |
| `python/bridge.py` | Thin dispatch (subprocess entry point) — 15 commands including verify and validate_proto_direct |
| `python/trace_model.py` | torch.fx symbolic tracing for HuggingFace Transformers models |
| `python/mil_emitter.py` | MIL program construction, mlpackage save, compute plan info |
| `python/converter.py` | Encapsulates ct.convert() for MIL→MLModel conversion; wired through bridge |
| `python/palettize.py` | Post-training palettization; wired through bridge; fixed for coremltools 9.0 |
| `python/compute_plan.py` | MLComputePlan inspection; wired through bridge; real implementation |
| `python/profiler.py` | Profiling logic; wired through bridge.py's `profile` command via `profile_model()` (requires Apple hardware) |

### Data Flow Integrity

The Rust `BridgeResult` struct now captures all fields returned by the Python bridge:
- `status`, `error_message`, `output_path`, `coremltools_version` (original)
- `content_hash` — SHA-256 of mlpackage directory (was dropped)
- `package_files` — file inventory with paths and sizes (was dropped)
- `compute_plan` — compute plan availability (was dropped)
- `function_descriptors` — per-function I/O specs (was dropped)

The manifest builder now derives function descriptors from the bridge result
rather than hardcoding dimensions. A deterministic task hash is computed from
spec parameters and included in both manifest and knowledge update outputs.

### Python Module Decomposition

The Python emission layer is now decomposed into focused modules:
1. `mil_emitter.build_linear_projection_program()` — constructs MIL Program object
2. `converter.convert_milprogram()` — converts MIL Program → MLModel via ct.convert()
3. `mil_emitter.save_mlpackage()` — saves MLModel, computes hash, inventories files
4. `palettize.apply_palettization()` — applies post-training palettization
5. `compute_plan.inspect_compute_plan()` — inspects compute plan via MLComputePlan

Bridge commands compose these modules:
- `emit_linear_projection`: build → convert → save
- `emit_mlprogram`: same composition (will diverge for other program types)
- `convert`: build → convert with explicit precision/opset control
- `palettize`: load existing → palettize → save
- `compute_plan`: load → inspect compute plan

### Report and Packaging Flow

The CLI `report` subcommand generates human-readable and machine-readable reports from compilation artifacts:

```
compile output dir (manifest.json + knowledge/update_*.json)
  → CLI report command
  → MarkdownReporter or JsonReporter
  → .md or .json report file(s)
```

The `Packager` produces deterministic zip archives of the compile output directory:

```
compile output dir (mlpackage + manifest + MIR dump + knowledge)
  → Packager::package()
  → {model_id}.zip
```

Both the reporter and packager operate on the same data produced by the compile command. No `todo!()` stubs remain in the core compiler pipeline. The CAPI layer (`coreml-ffi`) contains stub implementations that return honest errors for unsupported operations on macOS.

### Bridge Error Types

The bridge (`ane-bridge`) and emission (`ane-coreml-emit`) layers expose typed error enums for programmatic error matching:

- `BridgeError::UnresolvedWeight { path }` — weight not found in resolver (T-P2-05)
- `EmissionError::MissingIODescriptor { kind, name, function }` — I/O shape/dtype unknown (T-P2-10)
- `EmissionError::UndersizedIOSurface`, `NonUniformSurface`, `InvalidFlatBufferLayout` — ANE constraint violations (T-P3-01)

These replace `anyhow::bail!` calls that produced opaque error messages, enabling callers to match on specific error kinds.

### ValidationPolicy

The emission layer uses `ValidationPolicy` (T-P3-01) to control ANE constraint enforcement:
- `ValidationPolicy::strict()` (default) — violations are hard errors
- `ValidationPolicy::warn_only()` — violations produce warnings, emission continues

### Architecture-Aware Bridge

The bridge layer now requires explicit architecture and max_seq_len parameters (T-P2-11):
- `mir_graph_to_compat_with_arch(graph, resolver, &architecture, max_seq_len, allow_missing)` — preferred API
- `mir_graph_to_compat()` and `mir_graph_to_compat_with_allow_missing()` — deprecated, default to Qwen3 with warning
- CLI `trace-compile` accepts `--architecture` and `--max-seq-len` flags
