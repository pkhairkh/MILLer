#!/usr/bin/env python3
"""MILLer — Python Bridge

Thin subprocess entry point. Reads a command JSON, dispatches to
the real emission/conversion/palettization/profiling logic,
writes result JSON.

Rust/Python boundary for this module:
  - bridge.py does NOT make compiler decisions.
  - bridge.py does NOT construct MIL graphs (that is mil_emitter.py).
  - bridge.py does NOT convert programs (that is converter.py).
  - bridge.py only dispatches and marshals JSON I/O.
  - All compiler logic lives in Rust; Python is the emission boundary.

Usage: python3 bridge.py <command_file> <result_file>

Commands:
  emit_linear_projection  — Build MIL program, convert, save mlpackage
  emit_lut_projection     — Build LUT gather-based MIL program, convert, save mlpackage
  emit_decode_step        — Build stateful decode-step with real KV-cache state (iOS 18+) — now the default (Sprint 40)
  emit_stateless_decode_step — Build stateless decode-step with deterministic mb.const KV cache (for single-step testing)
  emit_stateful_decode_step — Build stateful decode-step with real KV-cache state (iOS 18+)
  emit_shard_decode_step — Build shard-role-aware decode-step (Entry/Interior/Exit produce different dims)
  emit_palettized_linear_projection — Emit linear projection then apply real coremltools palettization
  emit_mlp_block          — Build MLP block (up-proj + activation + down-proj) MIL program, convert, save mlpackage
  emit_attention          — Build attention (QKV + scaled dot-prod + out-proj) MIL program, convert, save mlpackage
  emit_mlprogram          — Same as emit_linear_projection (explicit mlprogram path)
  emit_multifunction      — Build multi-function (embedding+decode_step) MIL program, convert, save mlpackage (Sprint 39)
  emit_multifunction_shared_weights — Build multi-function with shared weights between functions (Sprint 42)
  validate_multifunction  — Validate a multi-function mlpackage has expected functions (Sprint 39)
  convert                 — Re-convert an existing MIL program with different settings
  palettize               — Apply palettization to an existing mlpackage
  compute_plan            — Inspect compute plan for an mlpackage
  compute_plan_harvest    — Harvest per-op placement + cost data, produce observations (Sprint 35)
  profile                 — Profile an mlpackage (requires Apple hardware)
  inspect_mlpackage       — Inspect mlpackage structure and contents
  host_inspect            — Host-side inspection of mlpackage artifacts
  model_structure         — Structural introspection via MLModelStructure (Sprint 34)
  validate_proto_direct   — Validate a proto-direct emitted mlpackage against coremltools (Sprint 41)
  verify                  — Unified verification harness: op fidelity + placement + state + multifunction (Sprint 40)
"""

import json
import os
import sys

os.environ.setdefault("COREMLTOOLS_DISABLE_TELEMETRY", "1")

# Import shared constants and helpers (W-17, W-18, W-19, W-26 fixes)
from common import _error_result

# Import the real emission logic
from mil_emitter import (
    emit_attention,
    emit_decode_step,
    emit_linear_projection,
    emit_lut_projection,
    emit_mlp_block,
    emit_mlprogram,
    emit_multifunction,
    emit_multifunction_shared_weights,
    emit_palettized_linear_projection,
    emit_shard_decode_step,
    emit_stateful_decode_step,
    emit_stateless_decode_step,
    inspect_mlpackage,
    validate_multifunction_package,
)

# Import structural verification (Sprint 34)
from model_structure import (
    fallback_file_structure,
    inspect_model_structure,
    inspect_model_structure_with_mir_comparison,
)

# Import unified verification harness (Sprint 40)
from verify import save_verification_result, verify_model


def handle_convert(command: dict) -> dict:
    """Handle the 'convert' command: build a fresh MIL program and convert with specified settings.

    The convert command rebuilds a linear projection MIL program from the
    specification in the payload and converts it using converter.py with the
    requested precision/opset settings. This is not a "re-convert existing
    mlpackage" operation (ct.convert from a loaded spec is unreliable across
    coremltools versions); instead it builds a fresh program with the same
    dimensions.

    For re-converting an existing mlpackage, use emit_mlprogram with different
    compute_precision.

    Payload fields:
        task_name: str — task name
        input_dim: int — input dimension
        output_dim: int — output dimension
        batch_size: int — batch size (default 1)
        dtype: str — "fp16" or "fp32" (default "fp16")
        opset_version: str — target opset (default "iOS18")
        compute_precision: str — "FLOAT16" or "FLOAT32" (default "FLOAT16")
        compute_units: str — compute unit hint (default "CPU_AND_NE")
        output_path: str — where to save the converted mlpackage
        seed: int — random seed (default 42)
    """
    try:
        import coremltools as ct
        from converter import convert_milprogram
        from mil_emitter import build_linear_projection_program, save_mlpackage
    except ImportError as e:
        return _error_result(f"Required module not available: {e}")

    output_path = command.get("output_path", "")
    if not output_path:
        return _error_result("output_path is required for convert command")

    try:
        # Step 1: Build the MIL program from the specification
        prog, prog_meta = build_linear_projection_program(command)

        # Step 2: Convert using converter.py with the requested settings
        opset_version = command.get("opset_version", "iOS18")
        compute_precision = command.get("compute_precision", "FLOAT16")
        compute_units = command.get("compute_units", "CPU_AND_NE")

        mlmodel = convert_milprogram(
            prog,
            opset_version=opset_version,
            compute_precision=compute_precision,
            compute_units=compute_units,
        )

        # Step 3: Save
        task_name = command.get("task_name", "linear_projection")
        from pathlib import Path
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_path = out_dir / f"{task_name}.mlpackage"

        save_info = save_mlpackage(mlmodel, str(mlpackage_path))

        return {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": None,
            "function_descriptors": [],
            "metadata": {
                "opset_version": opset_version,
                "compute_precision": compute_precision,
                "compute_units": compute_units,
                **prog_meta,
            },
        }

    except Exception as e:
        return _error_result(f"Convert command failed: {e}")


def handle_palettize(command: dict) -> dict:
    """Handle the 'palettize' command: apply palettization to an mlpackage.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to palettize
        palettization_specs: list — per-weight palettization specifications
            Each spec: {weight_name, mode, nbits, granularity, group_size, channel_axis}
        output_path: str — where to save the palettized mlpackage
        compute_units: str — compute unit hint (default "CPU_AND_NE")
    """
    try:
        import coremltools as ct
        from mil_emitter import save_mlpackage
        from palettize import apply_palettization
    except ImportError as e:
        return _error_result(f"Required module not available: {e}")

    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for palettize command")

    palettization_specs = command.get("palettization_specs", [])
    if not palettization_specs:
        return _error_result("palettization_specs is required for palettize command")

    output_path = command.get("output_path", "")
    if not output_path:
        return _error_result("output_path is required for palettize command")

    try:
        # Load the model (skip_model_load=True avoids macOS SIGABRT
        # from C++ model compilation for palettization use case)
        try:
            model = ct.models.MLModel(mlpackage_path, skip_model_load=True)
        except TypeError:
            model = ct.models.MLModel(mlpackage_path)

        # Apply palettization
        palettized_model = apply_palettization(model, palettization_specs)

        # Save to new path
        from pathlib import Path
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_name = Path(mlpackage_path).name
        target_path = out_dir / mlpackage_name

        save_info = save_mlpackage(palettized_model, str(target_path))

        # Summarize what was applied
        spec_summary = []
        for spec in palettization_specs:
            spec_summary.append({
                "weight_name": spec.get("weight_name", "unknown"),
                "nbits": spec.get("nbits", 4),
                "mode": spec.get("mode", "kmeans"),
                "granularity": spec.get("granularity", "per_grouped_channel"),
            })

        return {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": None,
            "function_descriptors": [],
            "metadata": {
                "source_mlpackage": mlpackage_path,
                "palettization_applied": spec_summary,
            },
        }

    except Exception as e:
        return _error_result(f"Palettize command failed: {e}")


def handle_compute_plan(command: dict) -> dict:
    """Handle the 'compute_plan' command: inspect compute plan for an mlpackage.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to inspect
        compute_units: str — compute units for planning (default "CPU_AND_NE")
    """
    try:
        from compute_plan import inspect_compute_plan
    except ImportError as e:
        return _error_result(f"compute_plan module not available: {e}")

    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for compute_plan command")

    compute_units = command.get("compute_units", "CPU_AND_NE")

    try:
        result = inspect_compute_plan(mlpackage_path, compute_units)
        return {
            "status": "success",
            "error_message": None,
            "output_path": None,
            "coremltools_version": None,
            "content_hash": None,
            "package_files": [],
            "compute_plan": result,
            "function_descriptors": [],
            "metadata": {
                "mlpackage_path": mlpackage_path,
                "compute_units": compute_units,
            },
        }
    except Exception as e:
        return _error_result(f"Compute plan command failed: {e}")


def handle_host_inspect(command: dict) -> dict:
    """Handle the 'host_inspect' command: host-side inspection of mlpackage artifacts.

    This performs honest host-side inspection: checks what can be determined
    without executing the model on a device runtime. It NEVER infers ANE
    behavior or compute unit placement from host-only evidence.

    Refactored (W-21 fix): delegates structural inspection to model_structure.py
    and compute plan checking to compute_plan.py, instead of reimplementing
    their logic inline.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to inspect
        compute_units: str — compute units hint for compute plan check (default "CPU_AND_NE")

    Returns:
        Structured inspection result with:
          - package_present: bool
          - manifest_readable: bool
          - manifest_contents: dict or None
          - model_loadable: bool
          - model_load_failure_reason: str or None
          - function_count: int or None
          - input_specs: list of {name, shape, dtype}
          - output_specs: list of {name, shape, dtype}
          - compute_plan_available: bool
          - file_inventory: list of {path, size_bytes}
          - total_size_bytes: int
          - warnings: list of str
    """
    from pathlib import Path

    mlpackage_path = command.get("mlpackage_path", "")
    compute_units = command.get("compute_units", "CPU_AND_NE")
    warnings = []

    # Step 1: Check package presence
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for host_inspect command")

    pkg_path = Path(mlpackage_path)
    package_present = pkg_path.exists() and pkg_path.is_dir()

    if not package_present:
        return {
            "status": "success",
            "error_message": None,
            "package_present": False,
            "manifest_readable": False,
            "manifest_contents": None,
            "model_loadable": False,
            "model_load_failure_reason": "mlpackage directory does not exist",
            "function_count": None,
            "input_specs": [],
            "output_specs": [],
            "compute_plan_available": False,
            "file_inventory": [],
            "total_size_bytes": 0,
            "warnings": ["mlpackage directory not found"],
        }

    # Step 2: Delegate structural inspection to model_structure.py (W-21 fix)
    # Previously this reimplemented ~100 lines of model loading and I/O spec
    # extraction. Now we delegate to the dedicated module.
    structure_result = inspect_model_structure(mlpackage_path)

    # Extract structural info from model_structure result
    model_loadable = structure_result.get("available", False)
    model_load_failure_reason = None if model_loadable else structure_result.get("reason", "unknown")
    input_specs = []
    output_specs = []
    function_count = None

    if model_loadable:
        for func in structure_result.get("functions", []):
            for inp in func.get("inputs", []):
                input_specs.append({
                    "name": inp.get("name", "unknown"),
                    "shape": inp.get("shape", []),
                    "dtype": str(inp.get("dtype", "unknown")),
                })
            for outp in func.get("outputs", []):
                output_specs.append({
                    "name": outp.get("name", "unknown"),
                    "shape": outp.get("shape", []),
                    "dtype": str(outp.get("dtype", "unknown")),
                })
        function_count = len(structure_result.get("functions", [])) or 1
    else:
        warnings.append(f"Structural inspection unavailable: {model_load_failure_reason}")

    # Step 3: Read Manifest.json (lightweight, no delegation needed)
    manifest_path = pkg_path / "Manifest.json"
    manifest_readable = False
    manifest_contents = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest_contents = json.load(f)
            manifest_readable = True
        except Exception as e:
            warnings.append(f"Failed to read Manifest.json: {e}")

    # Step 4: Delegate compute plan check to compute_plan.py
    compute_plan_available = False
    try:
        from compute_plan import inspect_compute_plan
        plan_result = inspect_compute_plan(str(pkg_path), compute_units)
        compute_plan_available = plan_result.get("available", False)
        if not compute_plan_available:
            reason = plan_result.get("reason", "unknown")
            warnings.append(f"Compute plan not available: {reason}")
    except ImportError:
        warnings.append("compute_plan module not available")
    except Exception as e:
        warnings.append(f"Compute plan check failed: {e}")

    # Step 5: File inventory (delegate to fallback_file_structure if available)
    file_inventory = []
    total_size_bytes = 0
    fallback = fallback_file_structure(mlpackage_path) if not model_loadable else None
    if fallback and fallback.get("available", False):
        file_inventory = fallback.get("file_inventory", [])
        total_size_bytes = sum(f.get("size_bytes", 0) for f in file_inventory)
    else:
        for root, dirs, files in os.walk(pkg_path):
            for f in files:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, pkg_path)
                sz = os.path.getsize(fp)
                total_size_bytes += sz
                file_inventory.append({"path": rel, "size_bytes": sz})

    # Important: do not infer ANE behavior from host-only inspection
    warnings.append("Host-side inspection only — no ANE placement or runtime behavior is implied")

    return {
        "status": "success",
        "error_message": None,
        "package_present": package_present,
        "manifest_readable": manifest_readable,
        "manifest_contents": manifest_contents,
        "model_loadable": model_loadable,
        "model_load_failure_reason": model_load_failure_reason,
        "function_count": function_count,
        "input_specs": input_specs,
        "output_specs": output_specs,
        "compute_plan_available": compute_plan_available,
        "file_inventory": file_inventory,
        "total_size_bytes": total_size_bytes,
        "warnings": warnings,
    }


def handle_profile(command: dict) -> dict:
    """Handle the 'profile' command: profile an mlpackage with timing.

    This requires Apple hardware with Core ML runtime for predict().
    On non-Apple platforms, it returns an honest error.

    Delegates input generation to profiler.generate_inputs() (W-22 fix)
    and profiling to profiler.profile_model().

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to profile
        compute_units: str — compute units for execution (default "CPU_AND_NE")
        warmup_iterations: int — warmup iterations (default 5)
        measured_iterations: int — measured iterations (default 20)
        seed: int — random seed for input generation (default 42)

    Returns:
        Standard result structure with timing metadata, or error if
        profiling is not available.
    """
    try:
        import coremltools as ct
        from profiler import generate_inputs, profile_model
    except ImportError as e:
        return _error_result(f"Required module not available for profiling: {e}")

    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for profile command")

    compute_units_str = command.get("compute_units", "CPU_AND_NE")
    warmup_iterations = command.get("warmup_iterations", 5)
    measured_iterations = command.get("measured_iterations", 20)
    seed = command.get("seed", 42)

    try:
        # Delegate input generation to profiler.generate_inputs() (W-22 fix)
        # Previously this duplicated the input-generation logic inline.
        inputs = generate_inputs(mlpackage_path, compute_units_str, seed)

        # Delegate profiling to profiler.profile_model()
        profile_result = profile_model(
            mlpackage_path=mlpackage_path,
            inputs=inputs,
            compute_units=compute_units_str,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
        )

        # Convert profiler result format to bridge timing format
        latency = profile_result.get("latency", {})
        n = latency.get("iterations", measured_iterations)
        median_ns = latency.get("median_ns", 0)
        p90_ns = latency.get("p90_ns", 0)
        p99_ns = latency.get("p99_ns", 0)
        min_ns = latency.get("min_ns", 0)
        max_ns = latency.get("max_ns", 0)

        # profile_model returns ns values; convert to ms
        timing_result = {
            "warmup_iterations": warmup_iterations,
            "measured_iterations": n,
            "p50_ms": median_ns / 1_000_000.0,
            "p90_ms": p90_ns / 1_000_000.0,
            "p99_ms": p99_ns / 1_000_000.0,
            "min_ms": min_ns / 1_000_000.0,
            "max_ms": max_ns / 1_000_000.0,
            # T-P4-05: Renamed from mean_ms to median_ms. The profiler returns
            # median_ns, not mean_ns. The previous key name was misleading.
            "median_ms": median_ns / 1_000_000.0,
            # T-P4-05: std_dev_ms set to None since the profiler does not compute
            # standard deviation. Previously was hardcoded to 0.0 which was misleading.
            "std_dev_ms": None,
            "compute_units": compute_units_str,
            "scope_note": (
                f"Device execution with {compute_units_str} hint. "
                "Compute unit assignment is not guaranteed — Core ML may fall back. "
                "Use compute plan for per-op device assignment."
            ),
        }

        # Also try compute plan
        compute_plan_info = None
        try:
            from compute_plan import inspect_compute_plan
            compute_plan_info = inspect_compute_plan(mlpackage_path, compute_units_str)
        except Exception:
            pass

        return {
            "status": "success",
            "error_message": None,
            "output_path": None,
            "coremltools_version": ct.__version__,
            "content_hash": None,
            "package_files": [],
            "compute_plan": compute_plan_info,
            "function_descriptors": [],
            "metadata": {
                "timing": timing_result,
                "mlpackage_path": mlpackage_path,
            },
        }

    except Exception as e:
        error_str = str(e)
        if "libcoremlpython" in error_str or "CoreML" in error_str:
            return _error_result(
                "Profiling requires Apple hardware with Core ML runtime. "
                f"Error: {error_str}"
            )
        return _error_result(f"Profile command failed: {e}")


def handle_model_structure(command: dict) -> dict:
    """Handle the 'model_structure' command: structural introspection via MLModelStructure.

    This is the Sprint 34 structural verification command. It replaces
    weak file-existence checks with real structural introspection using
    MLModelStructure.load_from_path().

    On Apple hardware, this provides per-op structural verification including
    op names, input/output signatures, state declarations, and function
    structure. On non-Apple platforms, it gracefully reports unavailability.

    If mir_ops is provided in the payload, it also performs MIR-vs-structure
    comparison, computing op fidelity scores and reporting missing/extra ops.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to inspect
        mir_ops: list (optional) — list of MIR op dicts with "op_type" keys
        include_fallback: bool (optional) — whether to include fallback file
            check when MLModelStructure is unavailable (default: True)

    Returns:
        Structured result with:
          - available: bool — whether MLModelStructure inspection succeeded
          - functions: list — function descriptors
          - operations: list — operation descriptors
          - state_declarations: list — state descriptors
          - mir_comparison: dict or None — MIR-vs-structure comparison result
          - mir_comparison_possible: bool
          - fallback: dict or None — fallback file check result (if requested)
    """
    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for model_structure command")

    mir_ops = command.get("mir_ops", None)
    include_fallback = command.get("include_fallback", True)

    # Perform structural inspection + optional MIR comparison
    result = inspect_model_structure_with_mir_comparison(mlpackage_path, mir_ops)

    # If MLModelStructure is unavailable and fallback is requested,
    # also run the fallback file check
    fallback = None
    if not result.get("available", False) and include_fallback:
        fallback = fallback_file_structure(mlpackage_path)

    return {
        "status": "success",
        "error_message": None,
        **result,
        "fallback": fallback,
    }


def handle_compute_plan_harvest(command: dict) -> dict:
    """Handle the 'compute_plan_harvest' command: harvest per-op placement and cost data.

    This is the Sprint 35 compute plan harvesting command. It extracts per-op
    device placement and estimated cost from MLComputePlan, converts the data
    into knowledge store observation format, and optionally persists both as
    JSON artifacts.

    On non-Apple platforms, the harvesting path reports unavailable gracefully.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to harvest
        compute_units: str — compute units for planning (default "CPU_AND_NE")
        output_path: str (optional) — directory to persist artifact JSON files

    Returns:
        Dict with:
          - harvest: dict — the full harvest_compute_plan() result
          - observations: list — knowledge store observations from harvest_to_observations()
          - artifact_paths: dict or None — paths to persisted artifacts (if output_path given)
    """
    from pathlib import Path

    try:
        from compute_plan import harvest_compute_plan, harvest_to_observations
    except ImportError as e:
        return _error_result(f"compute_plan module not available: {e}")

    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for compute_plan_harvest command")

    compute_units = command.get("compute_units", "CPU_AND_NE")
    output_path = command.get("output_path", None)

    try:
        # Step 1: Harvest compute plan data
        harvest_result = harvest_compute_plan(mlpackage_path, compute_units)

        # Step 2: Convert to observations
        observations = harvest_to_observations(harvest_result)

        # Step 3: Optionally persist artifacts
        artifact_paths = None
        if output_path:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)

            harvest_artifact_path = out_dir / "compute_plan_harvest.json"
            with open(str(harvest_artifact_path), "w") as f:
                json.dump(harvest_result, f, indent=2)

            observations_artifact_path = out_dir / "compute_plan_observations.json"
            with open(str(observations_artifact_path), "w") as f:
                json.dump(observations, f, indent=2)

            artifact_paths = {
                "harvest": str(harvest_artifact_path),
                "observations": str(observations_artifact_path),
            }

        return {
            "status": "success",
            "error_message": None,
            "harvest": harvest_result,
            "observations": observations,
            "artifact_paths": artifact_paths,
            "metadata": {
                "mlpackage_path": mlpackage_path,
                "compute_units": compute_units,
            },
        }

    except Exception as e:
        return _error_result(f"Compute plan harvest command failed: {e}")


def handle_validate_proto_direct(command: dict) -> dict:
    """Handle the 'validate_proto_direct' command: validate a proto-direct emitted mlpackage.

    This is the Sprint 41 validation command. It takes a proto-direct emitted
    mlpackage (produced by ane-coreml-emit) and validates it against coremltools,
    checking:
    1. The mlpackage directory structure is correct
    2. Manifest.json is present and valid
    3. The model.mlmodel protobuf file can be parsed
    4. The model can be loaded via ct.models.MLModel (on macOS)
    5. Weight.bin structure matches protobuf references
    6. Function count and op count match expectations

    Additionally, if a coremltools-emitted reference mlpackage is provided,
    it compares the two for structural equivalence.

    Payload fields:
        mlpackage_path: str — path to the proto-direct emitted .mlpackage
        reference_mlpackage_path: str (optional) — path to coremltools-emitted .mlpackage for comparison
        expected_function_count: int (optional) — expected number of functions
        expected_weight_bin_size: int (optional) — expected weight.bin size in bytes

    Returns:
        Validation result with:
        - valid: bool — whether the proto-direct mlpackage is structurally valid
        - manifest_valid: bool
        - model_loadable: bool (only on macOS)
        - function_count: int or None
        - weight_bin_size: int
        - comparison: dict or None — comparison with reference mlpackage
    """
    from pathlib import Path

    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for validate_proto_direct command")

    reference_path = command.get("reference_mlpackage_path", None)
    expected_function_count = command.get("expected_function_count", None)
    expected_weight_bin_size = command.get("expected_weight_bin_size", None)

    pkg_path = Path(mlpackage_path)
    validation_errors = []
    validation_warnings = []

    # Step 1: Check directory structure
    if not pkg_path.exists() or not pkg_path.is_dir():
        return {
            "status": "success",
            "valid": False,
            "validation_errors": ["mlpackage directory does not exist"],
            "validation_warnings": [],
            "manifest_valid": False,
            "model_loadable": False,
            "function_count": None,
            "weight_bin_size": None,
        }

    # Check required subdirectories
    # T-P1-03: The proto-direct emitter writes model.mlmodel to Data/com.apple.CoreML/
    # (see crates/coreml-emit/src/package.rs line 339), not Model/com.apple.CoreML/.
    # The previous path check was incorrect, causing false validation failures
    # for proto-direct emitted packages.
    data_dir = pkg_path / "Data" / "com.apple.CoreML"
    weights_dir = pkg_path / "Data" / "com.apple.CoreML" / "weights"

    if not data_dir.exists():
        validation_errors.append("Data/com.apple.CoreML/ directory missing")
    if not weights_dir.exists():
        validation_errors.append("Data/com.apple.CoreML/weights/ directory missing")

    # Step 2: Validate Manifest.json
    manifest_path = pkg_path / "Manifest.json"
    manifest_valid = False
    manifest_contents = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest_contents = json.load(f)
            manifest_valid = True

            # Check for proto-direct emission metadata
            metadata = manifest_contents.get("metadata", {})
            user_defined = metadata.get("userDefined", {}) or metadata.get("user_defined", {})
            if "com.apple.coreml.mlemission" in user_defined:
                emission_source = user_defined["com.apple.coreml.mlemission"]
                if "proto-direct" not in emission_source:
                    validation_warnings.append(
                        f"Manifest emission source is '{emission_source}', expected 'proto-direct'"
                    )
            else:
                validation_warnings.append("No emission source metadata in manifest")
        except Exception as e:
            validation_errors.append(f"Failed to parse Manifest.json: {e}")
    else:
        validation_errors.append("Manifest.json missing")

    # Step 3: Check model.mlmodel (protobuf)
    # T-P1-03: Use data_dir instead of model_dir — proto-direct emitter writes to Data/
    mlmodel_path = data_dir / "model.mlmodel"
    if not mlmodel_path.exists():
        validation_errors.append("model.mlmodel file missing")
    else:
        mlmodel_size = mlmodel_path.stat().st_size
        if mlmodel_size == 0:
            validation_errors.append("model.mlmodel is empty")

    # Step 4: Check weight.bin
    weight_bin_path = weights_dir / "weight.bin"
    weight_bin_size = None
    if weight_bin_path.exists():
        weight_bin_size = weight_bin_path.stat().st_size
        if weight_bin_size == 0:
            validation_warnings.append("weight.bin is empty")

        if expected_weight_bin_size is not None and weight_bin_size != expected_weight_bin_size:
            validation_errors.append(
                f"weight.bin size mismatch: got {weight_bin_size}, "
                f"expected {expected_weight_bin_size}"
            )
    else:
        validation_warnings.append("weight.bin file missing (may be empty model)")

    # Step 5: Try to load model via coremltools
    model_loadable = False
    function_count = None
    try:
        import coremltools as ct
        try:
            # skip_model_load=True avoids macOS SIGABRT from C++ compilation
            try:
                model = ct.models.MLModel(str(pkg_path), skip_model_load=True)
            except TypeError:
                model = ct.models.MLModel(str(pkg_path))
            model_loadable = True

            # Extract function count
            try:
                spec = model.get_spec()
                if hasattr(spec, 'functions') and spec.functions:
                    function_count = len(spec.functions)
                else:
                    function_count = 1
            except Exception:
                function_count = 1

        except Exception as e:
            error_str = str(e)
            if "libcoremlpython" in error_str or "CoreML" in error_str:
                validation_warnings.append(
                    "Core ML runtime not available — model load cannot be verified"
                )
            else:
                validation_errors.append(f"Model load failed: {error_str}")
    except ImportError:
        validation_warnings.append("coremltools not available — model validation limited")

    # Check function count
    if expected_function_count is not None and function_count is not None:
        if function_count != expected_function_count:
            validation_errors.append(
                f"Function count mismatch: got {function_count}, "
                f"expected {expected_function_count}"
            )

    # Step 6: Compare with reference mlpackage (if provided)
    comparison = None
    if reference_path:
        ref_path = Path(reference_path)
        if ref_path.exists() and ref_path.is_dir():
            comparison = _compare_mlpackages(str(pkg_path), str(ref_path))
        else:
            validation_warnings.append(
                f"Reference mlpackage not found at {reference_path}"
            )

    valid = len(validation_errors) == 0

    return {
        "status": "success",
        "valid": valid,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "manifest_valid": manifest_valid,
        "model_loadable": model_loadable,
        "function_count": function_count,
        "weight_bin_size": weight_bin_size,
        "comparison": comparison,
    }


def handle_verify(command: dict) -> dict:
    """Handle the 'verify' command: unified verification of an mlpackage.

    This is the Sprint 40 unified verification harness command. It performs
    four verification dimensions against an emitted mlpackage:

    1. Op graph fidelity — compare emitted model ops against intended MIR ops
    2. Compute-unit placement — ANE placement rate from MLComputePlan (macOS)
    3. State conformance — verify state declarations and read/write ops
    4. Multi-function conformance — verify function count and names

    On non-Apple platforms, verification is partial but still provides useful
    structural checks via spec-based extraction.

    If output_path is provided, the full verification result is persisted as
    a JSON artifact file.

    Payload fields:
        mlpackage_path: str — path to the .mlpackage to verify
        mir_ops: list (optional) — list of MIR op dicts with "op_type" keys
            for op fidelity comparison
        expected_function_names: list of str (optional) — expected function
            names for multi-function conformance
        expected_state_names: list of str (optional) — expected state input
            names for state conformance
        compute_units: str — compute units for placement inspection
            (default "CPU_AND_NE")
        output_path: str (optional) — directory to persist verification
            artifact JSON files

    Returns:
        Structured verification result with:
          - op_fidelity: dict — op fidelity verification result
          - placement: dict — compute-unit placement result
          - state_conformance: dict — state model conformance result
          - multifunction_conformance: dict — multi-function conformance
          - overall_score: float — weighted score across dimensions
          - artifact_paths: dict or None — paths to persisted artifacts
    """
    mlpackage_path = command.get("mlpackage_path", "")
    if not mlpackage_path:
        return _error_result("mlpackage_path is required for verify command")

    mir_ops = command.get("mir_ops", None)
    expected_function_names = command.get("expected_function_names", None)
    expected_state_names = command.get("expected_state_names", None)
    compute_units = command.get("compute_units", "CPU_AND_NE")
    output_path = command.get("output_path", None)

    try:
        # Run unified verification
        result = verify_model(
            mlpackage_path=mlpackage_path,
            mir_ops=mir_ops,
            expected_function_names=expected_function_names,
            expected_state_names=expected_state_names,
            compute_units=compute_units,
        )

        # Optionally persist artifacts
        artifact_paths = None
        if output_path:
            artifact_paths = save_verification_result(
                result, output_path, artifact_name="verification_result"
            )

        return {
            "status": "success",
            "error_message": None,
            "output_path": None,
            "coremltools_version": None,
            "content_hash": None,
            "package_files": [],
            "compute_plan": None,
            "function_descriptors": [],
            "metadata": result.to_dict(),
            "artifact_paths": artifact_paths,
        }

    except Exception as e:
        return _error_result(f"Verification harness failed: {e}")


def _compare_mlpackages(proto_path: str, reference_path: str) -> dict:
    """Compare a proto-direct mlpackage with a coremltools reference mlpackage.

    Compares:
    - Directory structure
    - File sizes (especially weight.bin)
    - Model structure (if loadable)

    Returns a comparison result dict.
    """
    from pathlib import Path

    proto = Path(proto_path)
    reference = Path(reference_path)

    # Compare file structure
    proto_files = {}
    for root, dirs, files in os.walk(proto):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, proto)
            proto_files[rel] = os.path.getsize(fp)

    ref_files = {}
    for root, dirs, files in os.walk(reference):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, reference)
            ref_files[rel] = os.path.getsize(fp)

    # Check for files in reference but not in proto
    missing_from_proto = set(ref_files.keys()) - set(proto_files.keys())
    extra_in_proto = set(proto_files.keys()) - set(ref_files.keys())

    # Compare weight.bin sizes
    proto_weight_size = proto_files.get("Data/com.apple.CoreML/weights/weight.bin", 0)
    ref_weight_size = ref_files.get("Data/com.apple.CoreML/weights/weight.bin", 0)

    proto_is_smaller = proto_weight_size < ref_weight_size if proto_weight_size and ref_weight_size else None
    size_difference = ref_weight_size - proto_weight_size if proto_weight_size and ref_weight_size else None

    return {
        "proto_file_count": len(proto_files),
        "reference_file_count": len(ref_files),
        "missing_from_proto": list(missing_from_proto),
        "extra_in_proto": list(extra_in_proto),
        "proto_weight_bin_size": proto_weight_size,
        "reference_weight_bin_size": ref_weight_size,
        "proto_is_smaller": proto_is_smaller,
        "size_difference_bytes": size_difference,
    }


# _error_result is now imported from common.py (W-18 fix).
# The local definition has been removed.


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <command_file> <result_file>", file=sys.stderr)
        sys.exit(1)

    command_file = sys.argv[1]
    result_file = sys.argv[2]

    try:
        with open(command_file, "r") as f:
            command = json.load(f)
    except Exception as e:
        _write_result(result_file, _error_result(f"Failed to read command file: {e}"))
        sys.exit(1)

    cmd_type = command.get("command", "")

    # Bridge version check: reject incompatible payload versions
    payload_version = command.get("bridge_version", 0)
    EXPECTED_BRIDGE_VERSION = 1
    if payload_version != EXPECTED_BRIDGE_VERSION:
        _write_result(result_file, _error_result(
            f"Bridge version mismatch: payload has {payload_version}, bridge expects {EXPECTED_BRIDGE_VERSION}. "
            f"Please rebuild the CLI or update the Python bridge."
        ))
        return

    # Support the generic FamilyPayload format: if the payload contains a
    # "params" dict, flatten it into the top-level command dict so that
    # existing emitter functions see the fields they expect at top level.
    # This allows the Rust side to use a single generic FamilyPayload
    # struct while keeping the Python emitter functions unchanged.
    if "params" in command and isinstance(command["params"], dict):
        # Merge params into command, but don't overwrite existing top-level keys
        for key, value in command["params"].items():
            if key not in command:
                command[key] = value

    if cmd_type == "emit_linear_projection":
        result = emit_linear_projection(command)
    elif cmd_type == "emit_lut_projection":
        result = emit_lut_projection(command)
    elif cmd_type == "emit_decode_step":
        # W-20 fix: emit_decode_step now routes to the stateful path by default
        # (matching Sprint 40's intent). The function itself delegates to
        # emit_stateful_decode_step, so we call it directly.
        result = emit_decode_step(command)
    elif cmd_type == "emit_stateless_decode_step":
        result = emit_stateless_decode_step(command)
    elif cmd_type == "emit_stateful_decode_step":
        result = emit_stateful_decode_step(command)
    elif cmd_type == "emit_shard_decode_step":
        result = emit_shard_decode_step(command)
    elif cmd_type == "emit_palettized_linear_projection":
        result = emit_palettized_linear_projection(command)
    elif cmd_type == "emit_mlp_block":
        result = emit_mlp_block(command)
    elif cmd_type == "emit_attention":
        result = emit_attention(command)
    elif cmd_type == "emit_mlprogram":
        result = emit_mlprogram(command)
    elif cmd_type == "emit_multifunction":
        result = emit_multifunction(command)
    elif cmd_type == "emit_multifunction_shared_weights":
        result = emit_multifunction_shared_weights(command)
    elif cmd_type == "validate_multifunction":
        mlpackage_path = command.get("mlpackage_path", "")
        result = validate_multifunction_package(mlpackage_path)
    elif cmd_type == "convert":
        result = handle_convert(command)
    elif cmd_type == "palettize":
        result = handle_palettize(command)
    elif cmd_type == "compute_plan":
        result = handle_compute_plan(command)
    elif cmd_type == "compute_plan_harvest":
        result = handle_compute_plan_harvest(command)
    elif cmd_type == "inspect_mlpackage":
        result = inspect_mlpackage(command.get("mlpackage_path", ""))
    elif cmd_type == "host_inspect":
        result = handle_host_inspect(command)
    elif cmd_type == "model_structure":
        result = handle_model_structure(command)
    elif cmd_type == "verify":
        result = handle_verify(command)
    elif cmd_type == "validate_proto_direct":
        result = handle_validate_proto_direct(command)
    elif cmd_type == "profile":
        result = handle_profile(command)
    else:
        result = _error_result(f"Unknown command: {cmd_type}")

    _write_result(result_file, result)


def _write_result(path: str, result: dict):
    try:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"Failed to write result file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
