# Bridge Protocol

## Structure

- `python/bridge.py` — thin subprocess entry point: reads command JSON, dispatches, writes result JSON
- `python/mil_emitter.py` — real emission, inspection, program construction, save, and compute plan logic
- `python/converter.py` — encapsulates `ct.convert()` for MIL program → MLModel conversion
- `python/palettize.py` — post-training palettization via coremltools optimize APIs
- `python/compute_plan.py` — compute plan inspection via MLComputePlan
- `python/profiler.py` — profiling logic (wired through bridge via `profile` command; requires Apple hardware for predict())

## Rust/Python Boundary

This section explicitly documents what Python owns and what Rust owns.
This boundary must not silently shift; any change requires a doc update.

### Python Owns

1. **MIL graph construction** via `coremltools.converters.mil.Builder` (`mb.*` calls).
   Python is the only place where coremltools MIL Builder is invoked.
   No Rust code calls coremltools directly.

2. **ct.convert() invocation** via `converter.convert_milprogram()`.
   Python handles all conversion settings (opset, compute_precision, compute_units).
   Rust provides the specification; Python executes the conversion.

3. **mlpackage save** via `mil_emitter.save_mlpackage()`.
   Python calls `mlmodel.save()` to write the .mlpackage directory.
   Python also computes the content hash and file inventory.

4. **Palettization** via `palettize.apply_palettization()`.
   Python uses coremltools optimize APIs for post-training palettization.
   Rust decides what to palettize and passes specs via the bridge payload.

5. **Compute plan inspection** via `compute_plan.inspect_compute_plan()`.
   Python calls `MLComputePlan.load_from_path()` on Apple hardware.
   On non-Apple platforms, it reports unavailability with a reason string.

6. **Bridge dispatch** via `bridge.py`.
   Python reads command JSON, dispatches to the correct handler,
   and writes result JSON. No compiler decisions are made in bridge.py.

7. **On-device profiling** via `profiler.profile_model()`.
   Python executes `model.predict()` on Apple hardware and captures
   timing statistics. Rust processes the results but never runs predict().
   On non-Apple platforms, the profile command returns an honest error.

### Rust Owns

1. **All compiler decisions**: task selection, IR lowering, pass pipeline,
   shard planning, precision policy, legality checking.
   Python never decides what to compile or how to structure the IR.

2. **Bridge payload construction**: Rust serializes the task specification
   into JSON for Python to consume. Rust defines the schema and version.

3. **Result ingestion**: Rust deserializes the Python bridge result
   (`BridgeResult`) and produces manifests, knowledge updates, and reports.

4. **Artifact management**: hashing, packaging, manifest generation,
   and knowledge update production. All in Rust.

5. **CLI orchestration**: the entire compile pipeline is driven by
   the Rust CLI. Python is called as a subprocess at one specific point.

6. **Knowledge store**: storage, querying, confidence management.
   Python never reads or writes the knowledge store.

7. **Lab run orchestration**: the `lab` CLI command drives compile + inspect
   + structured run record generation. Python is only called for specific
   operations (emission, inspection, profiling).

8. **Fallback suspicion assessment**: Rust owns the weak, honest suspicion
   model that processes timing data and device metadata. Python provides
   raw timing data; Rust interprets it conservatively.

### What Must NOT Migrate to Python

- IR definitions and transformations
- Pass pipeline logic
- Shard planning or partitioning decisions
- Precision/palettization policy decisions (only execution goes to Python)
- Knowledge store access
- Manifest or report generation

### Bridge Contract

The bridge is a JSON-in/JSON-out subprocess protocol. The contract is:
1. Rust writes a command JSON to a temp file.
2. Rust invokes `python3 bridge.py <command_file> <result_file>`.
3. Python reads the command, checks the `bridge_version` field, executes it, writes the result JSON.
4. Rust reads the result JSON and continues.

All communication is serialized. No shared memory, no network, no IPC beyond
file I/O. The Python process is short-lived (one invocation per compile step).

### Bridge Versioning

Every bridge payload includes a `bridge_version` field (currently `1`).
Python checks this field on receipt and rejects payloads with incompatible versions
with a clear error message. This prevents silent misinterpretation when the Rust
and Python sides are built from different commits.

The current bridge version is defined in `crates/ir/src/linear_slice.rs` as `BRIDGE_VERSION`
and checked in `python/bridge.py` as `EXPECTED_BRIDGE_VERSION`.

## Commands

### `emit_lut_projection`

Build a dedicated LUT (Look-Up Table) projection MIL program, convert via converter.py, and save as mlpackage.

This is the dedicated emission path for LUT projection tasks. Unlike `emit_linear_projection`,
this path constructs a gather-based program that models the `constexpr_lut`-to-`gather` pattern used in
ANE palettized inference, rather than the matmul+add pattern of linear projection.

Payload:
```json
{
  "bridge_version": 1,
  "command": "emit_lut_projection",
  "task_name": "lut_proj_4bit",
  "family": "LutProjection",
  "vocab_size": 32000,
  "embed_dim": 512,
  "num_groups": 64,
  "lut_bitwidth": 4,
  "batch_size": 1,
  "dtype": "fp16",
  "opset_version": "iOS18",
  "compute_units": "CPU_AND_NE",
  "output_path": "/path/to/output",
  "seed": 42,
  "functions": [{"name": "main", "inputs": [...], "outputs": [...], "stateful": false}]
}
```

Result: same structure as emit_linear_projection, with `metadata.emission_path: "lut_projection"`.

**Limitations:**
- The gather-based program models a simplified LUT pattern; true grouped-palette semantics
  (per-group independent LUTs with per-group index tensors) are approximated by offset indexing.
- Precision override (dtype adaptation from knowledge) is not yet wired into the LUT payload path.
  This is a known gap; LUT tasks always use the spec's default dtype.
- End-to-end validation of the LUT emission output requires Apple hardware with Core ML runtime.
- The `lut_bitwidth` field is carried in the payload but does not yet influence the coremltools
  conversion parameters; full bitwidth-specific emission behavior requires deeper integration with
  the palettization pipeline.

### `emit_linear_projection`

Build a linear projection MIL program, convert via converter.py, and save as mlpackage.

Payload:
```json
{
  "command": "emit_linear_projection",
  "task_name": "linear_proj_slice",
  "input_dim": 64, "output_dim": 32, "batch_size": 1, "dtype": "fp16",
  "opset_version": "iOS18", "compute_units": "CPU_AND_NE",
  "output_path": "/path/to/output", "seed": 42,
  "functions": [{"name": "main", "inputs": [...], "outputs": [...], "stateful": false}]
}
```

Result:
```json
{
  "status": "success",
  "error_message": null,
  "output_path": "/path/to/output/task.mlpackage",
  "coremltools_version": "9.0",
  "content_hash": "sha256:<hex>",
  "package_files": [{"path": "Manifest.json", "size_bytes": 123}],
  "compute_plan": {"available": false, "reason": "Apple runtime not available on this platform"},
  "function_descriptors": [{"name": "main", "inputs": [...], "outputs": [...], "stateful": false}],
  "metadata": {...}
}
```

### `emit_mlprogram`

Same as emit_linear_projection but uses the explicit build → convert → save pipeline.
Functionally equivalent for linear projection; will diverge when more program types are supported.

Payload and result: same as emit_linear_projection.

### `convert`

Build a fresh MIL program from the spec and convert with specified settings (opset, precision).
Uses converter.py for the conversion step.

Payload:
```json
{
  "command": "convert",
  "task_name": "my_task",
  "input_dim": 64, "output_dim": 32, "batch_size": 1, "dtype": "fp16",
  "opset_version": "iOS18",
  "compute_precision": "FLOAT32",
  "compute_units": "CPU_AND_NE",
  "output_path": "/path/to/output",
  "seed": 42
}
```

Result: same structure as emit_linear_projection, with metadata including `compute_precision`.

### `palettize`

Apply post-training palettization to an existing mlpackage.

Payload:
```json
{
  "command": "palettize",
  "mlpackage_path": "/path/to/source.mlpackage",
  "palettization_specs": [
    {
      "weight_name": "weight",
      "mode": "kmeans",
      "nbits": 4,
      "granularity": "per_grouped_channel",
      "group_size": 32,
      "channel_axis": 1
    }
  ],
  "output_path": "/path/to/output"
}
```

Result: standard result structure with `metadata.palettization_applied` summarizing the specs applied.

### `compute_plan`

Inspect the compute plan for an mlpackage.

Payload:
```json
{
  "command": "compute_plan",
  "mlpackage_path": "/path/to/model.mlpackage",
  "compute_units": "CPU_AND_NE"
}
```

Result:
```json
{
  "status": "success",
  "compute_plan": {
    "available": false,
    "reason": "Apple Core ML runtime not available on this platform",
    "operations": [],
    "total_operations": 0
  },
  "metadata": {"mlpackage_path": "...", "compute_units": "..."}
}
```

On Apple hardware, `compute_plan.available` will be `true` and `operations` will contain per-op device assignments.

### `inspect_mlpackage`

Inspect mlpackage structure and contents.

Payload:
```json
{
  "command": "inspect_mlpackage",
  "mlpackage_path": "/path/to/model.mlpackage"
}
```

### `host_inspect`

Host-side inspection of mlpackage artifacts. This performs honest host-side
inspection: checks what can be determined without executing the model on a
device runtime. It NEVER infers ANE behavior or compute unit placement from
host-only evidence.

Payload:
```json
{
  "command": "host_inspect",
  "bridge_version": 1,
  "mlpackage_path": "/path/to/model.mlpackage",
  "compute_units": "CPU_AND_NE"
}
```

Result:
```json
{
  "status": "success",
  "package_present": true,
  "manifest_readable": true,
  "manifest_contents": {...},
  "model_loadable": false,
  "model_load_failure_reason": "Apple Core ML runtime not available",
  "function_count": 1,
  "input_specs": [{"name": "x", "shape": [1, 64], "dtype": "fp16"}],
  "output_specs": [{"name": "output", "shape": [1, 32], "dtype": "fp16"}],
  "compute_plan_available": false,
  "file_inventory": [{"path": "Manifest.json", "size_bytes": 123}],
  "total_size_bytes": 4567,
  "warnings": ["Host-side inspection only — no ANE placement or runtime behavior is implied"]
}
```

### `profile`

Profile an mlpackage with timing. Requires Apple hardware for `predict()`.
On non-Apple platforms, returns an honest error.

Payload:
```json
{
  "command": "profile",
  "bridge_version": 1,
  "mlpackage_path": "/path/to/model.mlpackage",
  "compute_units": "CPU_AND_NE",
  "warmup_iterations": 5,
  "measured_iterations": 20,
  "seed": 42
}
```

Result (on Apple hardware):
```json
{
  "status": "success",
  "output_path": null,
  "coremltools_version": "9.0",
  "metadata": {
    "timing": {
      "warmup_iterations": 5,
      "measured_iterations": 20,
      "p50_ms": 0.123,
      "p90_ms": 0.156,
      "p99_ms": 0.189,
      "min_ms": 0.098,
      "max_ms": 0.234,
      "median_ms": 0.128,
      "std_dev_ms": null,
      "compute_units": "CPU_AND_NE",
      "scope_note": "Device execution with CPU_AND_NE hint. Compute unit assignment is not guaranteed."
    }
  }
}
```

Result (on non-Apple hardware):
```json
{
  "status": "error",
  "error_message": "Profiling requires Apple hardware with Core ML runtime. Error: ..."
}
```

## All result fields

Every bridge command result contains these fields (matching Rust BridgeResult):

| Field | Type | Description |
|-------|------|-------------|
| status | string | "success" or "error" |
| error_message | string or null | Error description if status == "error" |
| output_path | string or null | Path to saved .mlpackage |
| coremltools_version | string or null | coremltools version (e.g., "9.0") |
| content_hash | string or null | "sha256:<hex>" hash of mlpackage directory |
| package_files | array | File inventory with {path, size_bytes} |
| compute_plan | object or null | Compute plan info |
| function_descriptors | array | Per-function I/O specs |
| metadata | object | Additional metadata |

## Multifunction Seam

The `functions` field in the payload is the integration point for multifunction mlpackage emission.
Multifunction emission IS now supported — the `emit_multifunction` and
`emit_multifunction_shared_weights` bridge commands construct one MIL program
per function and use `prog.add_function()` to merge multiple named functions
into a single mlpackage. The Rust `BridgeResult` captures `function_descriptors`
from the Python result, and the CLI manifest builder uses these descriptors
rather than hardcoding dimension values.

## Task Identity

The CLI computes a deterministic task hash from spec parameters before invoking the bridge.
This hash is included in the manifest (`task_hash`) and knowledge update, enabling artifact
identity verification independent of the compilation output.

## Internal Architecture

The Python emission layer is now decomposed into:
1. `build_linear_projection_program()` — constructs MIL Program object for linear projection
2. `build_lut_projection_program()` — constructs MIL Program object for LUT projection (gather-based)
3. `converter.convert_milprogram()` — converts MIL Program → MLModel
4. `save_mlpackage()` — saves MLModel, computes hash, inventories files

`emit_linear_projection()` and `emit_mlprogram()` compose these steps for linear projection.
`emit_lut_projection()` composes build + convert + save for LUT projection.
`handle_convert()` composes build + convert with explicit precision/opset control.
`handle_palettize()` loads an existing mlpackage, applies palettize.py, saves result.
`handle_compute_plan()` calls compute_plan.py for MLComputePlan inspection.
`handle_host_inspect()` performs honest host-side inspection (package presence, model load, compute plan check).
`handle_profile()` runs warmup + measured iterations on Apple hardware, captures timing statistics.

## Limitations

- `predict()` / profiling requires Apple hardware runtime (wired through bridge, honest error on non-Apple)
- Compute plan inspection reports unavailable on non-Apple platforms (real code, no data)
- Palettization produces real output but quality validation requires on-device predict()
- Multifunction emission is now supported via `emit_multifunction` and `emit_multifunction_shared_weights` commands
- `allow_missing_weights` now defaults to `false` for production paths (T-P2-09). When a real `WeightResolver` is provided, missing weights produce hard errors instead of silently zero-filling. This prevents the emission of models with zero-filled garbage weights.
- `mir_graph_to_compat_with_arch()` now requires explicit `architecture` and `max_seq_len` parameters (T-P2-11). The deprecated `mir_graph_to_compat()` and `mir_graph_to_compat_with_allow_missing()` functions default to Qwen3 architecture with a warning — callers should migrate to the explicit API.
- `ValidationPolicy` (T-P3-01) controls whether ANE constraint violations (undersized IOSurface, non-uniform surfaces, invalid flat buffer layout) produce errors (strict mode, default) or warnings (warn-only mode).
- `verify_emission_semantics()` (T-P5-12) performs semantic verification of emitted mlpackages: checks I/O names, weight files, dtype mismatches, and placeholder names.
- Profile timing reports use `median_ms` and `std_dev_ms: null` (T-P4-05). Previously `mean_ms` reported the median and `std_dev_ms` was always 0.0.
- Fallback suspicion is deliberately weak — no hard placement claims without device-backed evidence
- LUT projection emission is v0: gather pattern is simplified; grouped-palette semantics are approximated by offset indexing; precision override not yet wired; lut_bitwidth does not yet influence conversion parameters
