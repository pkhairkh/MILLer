# Profiling Methodology

See [SPEC.md](../SPEC.md) section 12 (Execution Lab).

## Lab Run Schema

Every lab run produces a structured `LabRun` record (defined in `crates/lab/src/harness.rs`).
The schema version is `1.0.0`.

### Verification Scope

Lab runs are classified by verification scope, which determines what
class of evidence the run provides:

| Scope | Meaning | Timing | Fallback |
|-------|---------|--------|----------|
| `host_only_inspection` | Only host-side operations: compile, file checks, metadata | None | Unavailable |
| `host_runtime_execution` | Model executed on host CPU/GPU (no Apple hardware) | Real but not ANE timing | Unavailable |
| `device_backed_execution` | Model executed on Apple hardware with Core ML runtime | Real device timing | Assessed (weak) |

This classification is structural — the `VerificationScope` enum in Rust
prevents host-only runs from being misread as device-backed runs.

### Host-Side Inspection

Host-side inspection (`HostInspector` in `crates/lab/src/host_inspect.rs`)
checks what can honestly be determined without executing the model on a
device runtime:

1. **Package presence**: Does the .mlpackage directory exist?
2. **Manifest readability**: Can the mlpackage Manifest.json be read?
3. **Model loadability**: Can coremltools load the model? (May fail on non-Apple)
4. **I/O spec extraction**: What are the input/output tensor shapes and dtypes?
5. **Compute plan availability**: Is compute plan inspection available?

Host-side inspection NEVER infers:
- ANE placement or compute unit assignment
- Runtime performance characteristics
- Numerical accuracy or drift

## Device Metadata

Device metadata (`DeviceMetadata` in `crates/lab/src/device_meta.rs`)
distinguishes host-only from device-backed environments structurally
via the `MetadataSource` enum:

- `HostOnly`: No Apple hardware available. Device-specific fields are None.
- `DeviceBacked`: Running on Apple hardware. Device fields are populated.

On non-Apple platforms, `DeviceMetadata::device_backed()` returns host-only
metadata with an honest explanation.

## Timing

Timing results (`TimingResult` in `crates/lab/src/harness.rs`) include:
- p50, p90, p99, min, max, mean, std_dev (milliseconds)
- Number of warmup and measured iterations
- Compute units requested
- **Scope note**: Explicit statement of what was measured

Every timing result includes a `scope_note` that honestly qualifies the
measurement. For example: "Device execution with CPU_AND_NE hint.
Compute unit assignment is not guaranteed — Core ML may fall back."

## Fallback Suspicion

Fallback suspicion (`FallbackDetector` in `crates/lab/src/fallback.rs`)
is a deliberately weak and honest assessment model. It does NOT make
hard placement claims. The suspicion levels are:

| Level | Meaning |
|-------|---------|
| `unavailable` | Cannot assess fallback — no device-backed execution or no timing data |
| `low_confidence_suspicion` | Some weak evidence suggests possible fallback, but not conclusive |
| `no_conclusion` | No strong evidence of fallback found, but absence of evidence ≠ evidence of absence |

Each suspicion assessment includes:
- An explanation string
- A list of evidence items, each with kind, description, and strength (0.0–1.0)
- No single evidence item exceeds strength 0.4 (weak signals only)

## Baseline Computation

Baseline computation (`BaselineComputer` in `crates/lab/src/baseline.rs`)
produces a deterministic FP32 reference output for the canonical task.

For synthetic linear projection tasks, the baseline is computed by:
1. Materializing weight and bias tensors from a deterministic seed (LCG PRNG)
2. Computing `y = x @ W + b` in pure FP32 arithmetic
3. Recording the reference output tensor and all computation parameters

Key properties:
- **Deterministic**: Same task spec + seed always produces the same baseline
- **Reproducible**: Baseline identity is linked to the task_hash
- **Host-side only**: No ANE, no Core ML, no GPU involved
- **Versioned**: BaselineResult schema version 1.0.0

The baseline artifact (`baseline.json`) is written in every lab run directory
and is available for drift comparison when actual model output can be obtained.

## Drift Detection

Drift detection (`DriftDetector` in `crates/lab/src/drift.rs`) computes
numerical metrics between the FP32 baseline reference and actual model output.

### Metrics

| Metric | Description |
|--------|-------------|
| `max_absolute_error` | Maximum element-wise absolute difference |
| `mean_absolute_error` | Mean element-wise absolute difference |
| `rmse` | Root mean squared error |
| `cosine_distance` | 1 - cosine_similarity (directional similarity) |
| `relative_error_p99` | 99th percentile of relative error |

### Computation Status

Drift reports include a `DriftComputationStatus` that MUST be checked before
interpreting numeric metrics:

| Status | Meaning |
|--------|---------|
| `computed` | Metrics were computed successfully from baseline and actual output |
| `unavailable` | Actual model output could not be obtained (requires Apple hardware for predict()) |
| `length_mismatch` | Baseline and actual tensor dimensions differ |
| `empty_input` | Empty tensors provided — no comparison possible |

When status is not `computed`, all numeric fields are NaN and MUST NOT be
treated as meaningful measurements.

### Current Limitation

On non-Apple hardware, drift computation always reports `unavailable` because
obtaining actual model output requires `predict()` which requires the Core ML
runtime. When Apple hardware is available, the `DriftDetector::detect()`
method will compute real FP32-vs-FP16 drift metrics.

### Drift in Knowledge Artifacts

Drift evidence is included in knowledge update artifacts (version 3) as:
- A `PrecisionHazard` observation with scope/confidence/evidence_source fields
- A `baseline_provenance` section linking the baseline to the task identity
- A `drift_evidence` section with computation status and metrics

## Lab Run Directory

The canonical directory structure produced by each lab run is defined in
`crates/lab/src/run_dir.rs`.

## Task Generation

Task generation (`TaskGenerator` in `crates/lab/src/task_gen.rs`) produces
deterministic task specifications that can be compiled by the active compile
path. Currently only the linear family (`LinearFamily` in `crates/lab/src/families/linear.rs`)
is implemented; all other families remain open.

### Generated Task Provenance

When a lab run uses a generated task (rather than a hand-authored TOML), the
`LabRun` record includes a `generator_provenance` field (`GeneratorProvenance`
in `crates/lab/src/harness.rs`) that records:

- `generator_version` — the version of the generator that produced the task
- `family` — the task family (e.g., "LinearProjection")
- `seed` — the random seed used for deterministic generation
- `task_name` — the name of the generated task within the set

This provenance is attached via the `--generated-from` CLI flag on the `lab`
command. When no `--generated-from` is provided, the provenance field is `None`,
and no false provenance is implied.

### Generating Tasks

```bash
ane-cli generate-tasks --family linear --output <dir> --seed 42
```

This creates a directory structure:
```
<dir>/
  generated_tasks.json       — Manifest of all generated tasks
  LinearProjection/
    linear_64x32_b1_fp16.toml
    linear_128x64_b1_fp16.toml
    linear_256x128_b1_fp16.toml
```

Each TOML file can be fed directly into `compile`, `compile-full`, or `lab`.

## Knowledge Consumption on Active Paths

Knowledge is no longer confined to the `compile-full` pass pipeline path.

### Compile Fast Path

The `compile` command now accepts `--knowledge <dir>` to load the knowledge
store. While the fast-path compile does not drive the pass pipeline, it
records whether knowledge was consulted in the manifest:

- `knowledge_consulted` — true if knowledge was successfully loaded
- `knowledge_seed_count` — number of seed entries in the store
- `knowledge_observation_count` — number of observation entries
- `knowledge_path` — "fast_path_compile" to distinguish from pass-pipeline knowledge use

When `--knowledge` is not provided, the manifest has no knowledge-related fields,
and behavior is identical to previous versions.

### Knowledge Influence Test

A test in `crates/passes/src/legality_rewrite.rs` proves that knowledge
changes pass outputs: when a known-legal seed is present, the AIR nodes
get higher `legality_confidence` than when no knowledge is available.
This demonstrates that the knowledge system is not inert.
