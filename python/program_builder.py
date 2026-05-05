"""Shared program building utilities for MIL program construction.

Deduplicates the common boilerplate that was repeated across 9 build_*_program()
functions and 12 emit_*() functions in mil_emitter.py (C-13 fix):

  - opset_map: was duplicated 9× → now centralized in resolve_opset_target()
  - np.random.seed/save/restore: was duplicated 9× → now in rng_seed_context()
  - dtype resolution: was duplicated 9× → now in resolve_dtypes()
  - emit_*() pattern: was duplicated 12× → now in emit_program()
  - shard role→op mapping: duplicated from Rust RoleMirBuilder → now in
    SHARD_ROLE_OP_MAP with source-of-truth comment (W-27 fix)

W-20 fix: emit_decode_step now routes to the stateful path by default.
W-27 fix: SHARD_ROLE_OP_MAP is documented as derived from Rust source of truth.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import _error_result, COMPUTE_MAP, _ensure_coremltools


# ---------------------------------------------------------------------------
# Opset resolution (eliminates 9× opset_map duplication — C-13)
# ---------------------------------------------------------------------------

def resolve_opset_target(ct, opset_version: str):
    """Resolve opset version string to coremltools deployment target.

    Every build_*_program() function previously had its own copy of:
        opset_map = {
            "iOS16": ct.target.iOS16,
            "iOS17": ct.target.iOS17,
            "iOS18": ct.target.iOS18,
        }
        target_os = opset_map.get(opset_version, ct.target.iOS18)

    This function replaces all 9 copies.
    """
    opset_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    return opset_map.get(opset_version, ct.target.iOS18)


# ---------------------------------------------------------------------------
# Dtype resolution (eliminates repeated inline dtype logic — C-13)
# ---------------------------------------------------------------------------

def resolve_dtypes(dtype_str: str, types):
    """Resolve dtype string to (np_dtype, mil_dtype) pair.

    Most build_*_program() functions had one of two inline patterns:
        np_dtype, _ = _resolve_dtype(dtype_str)     # linear, lut, decode_step
        mil_dtype = types.fp16 if dtype_str == "fp16" else types.fp32
    or:
        np_dtype = np.float16 if dtype_str == "fp16" else np.float32   # stateful, shard, mlp, attention, multifunction
        mil_dtype = types.fp16 if dtype_str == "fp16" else types.fp32

    This function provides both resolutions in a single call.
    """
    np_dtype = np.float16 if dtype_str == "fp16" else np.float32
    mil_dtype = types.fp16 if dtype_str == "fp16" else types.fp32
    return np_dtype, mil_dtype


# ---------------------------------------------------------------------------
# RNG seed context manager (eliminates 9× rng_state duplication — C-13)
# ---------------------------------------------------------------------------

@contextmanager
def rng_seed_context(seed: int):
    """Context manager that saves RNG state, seeds, and restores on exit.

    Every build_*_program() function previously had:
        rng_state = np.random.get_state()
        np.random.seed(seed)
        ...  # program construction
        np.random.set_state(rng_state)

    This context manager replaces all 9 copies. Usage:
        with rng_seed_context(seed):
            ...  # program construction
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(rng_state)


# ---------------------------------------------------------------------------
# Shard role → op mapping (W-27 fix)
# ---------------------------------------------------------------------------

# SOURCE OF TRUTH: crates/passes/src/role_mir.rs — RoleMirBuilder
#
# This mapping MUST be kept in sync with the Rust implementation's
# ShardOpProfile assignments. If you update this table, you MUST also
# update the corresponding Rust RoleMirBuilder code.
#
# Rust role → op mapping (from role_mir.rs):
#   Entry    → Reshape  (handoff preparation for next shard)
#   Interior → GELU     (MLP-like activation after attention)
#   Exit     → LayerNorm (normalization before IO model)
SHARD_ROLE_OP_MAP = {
    "Entry": "reshape",
    "Interior": "gelu",
    "Exit": "layernorm",
}


# ---------------------------------------------------------------------------
# Common emit pattern (eliminates 12× emit_*() duplication — C-13)
# ---------------------------------------------------------------------------

def emit_program(
    command: dict,
    build_fn: Callable,
    resolve_descriptors_fn: Callable,
    default_task_name: str,
    use_stateful_pipeline: bool = False,
    extra_result_fields: Optional[Dict] = None,
    error_prefix: str = "Program emission failed",
) -> dict:
    """Common emit pattern: build → convert → resolve descriptors → save → compute plan.

    This replaces the repeated try/except → import → build → convert →
    resolve descriptors → save → compute plan → return pattern that was
    duplicated across 12 emit_*() functions.

    Args:
        command: The command dict from the bridge.
        build_fn: callable(command) -> (program, metadata) that builds the
            MIL program.
        resolve_descriptors_fn: callable(command, dtype_str) -> list that
            resolves function descriptors for the result payload.
        default_task_name: Default task name for the mlpackage.
        use_stateful_pipeline: Whether to use make_stateful_pass_pipeline()
            for conversion (required for programs with mb.coreml_update_state).
        extra_result_fields: Optional dict of additional fields to merge
            into the result dict.
        error_prefix: Prefix for error messages.

    Returns:
        Result dict with status, output_path, content_hash, etc.
    """
    try:
        ct = _ensure_coremltools()

        from converter import convert_milprogram
        if use_stateful_pipeline:
            from converter import make_stateful_pass_pipeline

        output_path = command.get("output_path", "")
        if not output_path:
            return _error_result("No output_path specified")

        compute_units_str = command.get("compute_units", "CPU_AND_NE")
        dtype_str = command.get("dtype", "fp16")

        # Step 1: Build the MIL program
        prog, prog_meta = build_fn(command)

        # Step 2: Convert
        precision_str = "FLOAT16" if dtype_str == "fp16" else "FLOAT32"
        convert_kwargs = dict(
            opset_version=command.get("opset_version", "iOS18"),
            compute_precision=precision_str,
            compute_units=compute_units_str,
        )
        if use_stateful_pipeline:
            convert_kwargs["pass_pipeline"] = make_stateful_pass_pipeline()
        mlmodel = convert_milprogram(prog, **convert_kwargs)

        # Step 3: Resolve function descriptors
        function_descriptors = resolve_descriptors_fn(command, dtype_str)

        # Step 4: Save
        # Lazy import to avoid circular dependency at module load time.
        # mil_emitter imports program_builder, so program_builder cannot
        # import mil_emitter at the top level.
        from mil_emitter import save_mlpackage, compute_plan_info

        task_name = command.get("task_name", default_task_name)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_path = output_dir / f"{task_name}.mlpackage"

        save_info = save_mlpackage(mlmodel, str(mlpackage_path))

        # Step 5: Compute plan (best-effort)
        compute_plan = compute_plan_info(str(mlpackage_path), compute_units_str)

        result = {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": compute_plan,
            "function_descriptors": function_descriptors,
            "metadata": prog_meta,
        }

        if extra_result_fields:
            result.update(extra_result_fields)

        return result

    except Exception as e:
        return _error_result(f"{error_prefix}: {e}")
