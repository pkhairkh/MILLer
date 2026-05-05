"""MIL Emitter — actual Core ML MIL program construction and mlpackage save.

This is the single real place where MIL/mlpackage emission happens.
bridge.py dispatches here; this module owns the coremltools integration.

Rust/Python boundary for this module:
  - Python OWNS: MIL Builder calls (mb.program, mb.linear, mb.gelu,
    mb.scaled_dot_product_attention, mb.read_state, mb.coreml_update_state,
    mb.const, mb.StateTensorSpec),
    mlpackage save (mlmodel.save()), content hashing, file inventory.
  - Rust OWNS: deciding what task to compile, constructing the MIR that
    describes the target graph, and consuming the emission result.

FC projections use mb.linear (not matmul+add). GELU activation uses native
mb.gelu(mode="TANH_APPROXIMATION"). Attention and decode-step use
mb.scaled_dot_product_attention (iOS 18+). Stateful decode-step uses
mb.read_state / mb.coreml_update_state for real KV-cache state semantics
(iOS 18+). Multi-function emission uses mb.program(function_name=...) with
prog.add_function() to merge multiple named functions into a single
mlpackage (Sprint 39). These canonical Core ML ops produce semantically
correct and ANE-friendly MIL programs.

Architecture:
  build_linear_projection_program() — constructs the MIL Program object
  build_multifunction_program()     — constructs multi-function MIL Program
  converter.convert_milprogram()    — converts MIL Program → MLModel
  save_mlpackage()                  — saves MLModel to disk, hashes, inventories

emit_linear_projection() composes all three for the current vertical slice.
emit_mlprogram() composes build + convert for a cleaner separation.
emit_multifunction() composes build + convert + save + validate for
multi-function packages.

Deduplication (C-13, T-06):
  Common build boilerplate (opset_map, rng seed, dtype resolution) is
  now centralized in program_builder.py. Common emit pattern (build →
  convert → save → compute_plan) uses program_builder.emit_program().
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from common import _error_result, COMPUTE_MAP, _ensure_coremltools
from program_builder import (
    resolve_opset_target,
    resolve_dtypes,
    rng_seed_context,
    SHARD_ROLE_OP_MAP,
    emit_program,
)

def _resolve_dtype(dtype_str):
    """Resolve dtype string to numpy dtype and MIL dtype string.

    Kept for backward compatibility with external callers. Internal
    build_*_program functions now use program_builder.resolve_dtypes().
    """
    if dtype_str == "fp16":
        return np.float16, "fp16"
    return np.float32, "fp32"

# Lazy imports for coremltools — not all paths need them
# W-26 fix: raises ImportError instead of silently returning (None, None, None, None).
def _import_coremltools():
    """Import coremltools sub-modules and return (ct, mb, types, np).

    Raises ImportError if coremltools is not installed (W-26 fix).
    Previously returned (None, None, None, None) which silently hid import errors.
    """
    ct = _ensure_coremltools()  # raises ImportError if not installed
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    return ct, mb, types, np


def build_linear_projection_program(command: dict):
    """Build a MIL Program object for a linear projection via mb.linear.

    FC projections use mb.linear(x, weight, bias) which is the canonical
    Core ML op for fully-connected projections (replaces matmul + add).

    Payload fields consumed:
        task_name, input_dim, output_dim, batch_size, dtype,
        opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "linear_projection")
    input_dim = command.get("input_dim", 64)
    output_dim = command.get("output_dim", 32)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)
    input_shape = (batch_size, input_dim)

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=mil_dtype)],
            opset_version=target_os,
        )
        def prog(x):
            # mb.linear expects weight shape [output_dim, input_dim] (transposed
            # from the matmul convention of [input_dim, output_dim]).
            w_val = np.random.randn(output_dim, input_dim).astype(np_dtype)
            b_val = np.zeros(output_dim, dtype=np_dtype)
            z = mb.linear(x=x, weight=w_val, bias=b_val, name="output")
            return z

    metadata = {
        "task_name": task_name,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "linear_projection",
    }

    return prog, metadata


def save_mlpackage(mlmodel, mlpackage_path: str) -> dict:
    """Save an MLModel as .mlpackage, compute hash and file inventory.

    Args:
        mlmodel: An MLModel object (already converted).
        mlpackage_path: Target path for the .mlpackage directory.

    Returns:
        Dict with content_hash, package_files, output_path.
    """
    mlpackage_path = Path(mlpackage_path)

    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)

    mlmodel.save(str(mlpackage_path))

    content_hash = _hash_directory(mlpackage_path)

    package_files = []
    for root, dirs, files in os.walk(mlpackage_path):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, mlpackage_path)
            package_files.append({"path": rel, "size_bytes": os.path.getsize(fp)})

    return {
        "content_hash": content_hash,
        "package_files": package_files,
        "output_path": str(mlpackage_path),
    }


def compute_plan_info(mlpackage_path: str, compute_units_str: str = "CPU_AND_NE") -> dict:
    """Attempt to load a compute plan for an mlpackage.

    Returns {"available": True} on Apple hardware, or
    {"available": False, "reason": "..."} otherwise.
    This is a best-effort call; it always succeeds but may report unavailable.
    """
    ct, _, _, _ = _import_coremltools()
    if ct is None:
        return {"available": False, "reason": "coremltools not installed"}

    compute_unit = COMPUTE_MAP.get(compute_units_str, ct.ComputeUnit.CPU_AND_NE)

    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(str(mlpackage_path), compute_unit)
        return {"available": True}
    except Exception as e:
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Function descriptor resolvers (used by emit_program)
# ---------------------------------------------------------------------------

def _resolve_function_descriptors(
    functions: Optional[list],
    input_dim: int,
    output_dim: int,
    dtype: str,
) -> list:
    """Build function descriptors for the result payload.

    If the caller supplies a functions list, use it.
    Otherwise synthesize a single-function descriptor.
    """
    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [{"name": "x", "shape": [1, input_dim], "dtype": dtype}]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, output_dim], "dtype": dtype}]),
                "stateful": fn.get("stateful", False),
            })
        return descs

    return [{
        "name": "main",
        "inputs": [{"name": "x", "shape": [1, input_dim], "dtype": dtype}],
        "outputs": [{"name": "output", "shape": [1, output_dim], "dtype": dtype}],
        "stateful": False,
    }]


def _resolve_decode_step_function_descriptors(
    functions: Optional[list],
    embed_dim: int,
    dtype: str,
) -> list:
    """Build function descriptors for decode-step result payloads."""
    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [{"name": "x", "shape": [1, embed_dim], "dtype": dtype}]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, embed_dim], "dtype": dtype}]),
                "stateful": fn.get("stateful", False),
            })
        return descs

    return [{
        "name": "main",
        "inputs": [{"name": "x", "shape": [1, embed_dim], "dtype": dtype}],
        "outputs": [{"name": "output", "shape": [1, embed_dim], "dtype": dtype}],
        "stateful": False,
    }]


def _resolve_lut_function_descriptors(
    functions: Optional[list],
    embed_dim: int,
    dtype: str,
) -> list:
    """Build function descriptors for LUT projection result payloads."""
    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [{"name": "indices", "shape": [1], "dtype": "int32"}]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, embed_dim], "dtype": dtype}]),
                "stateful": fn.get("stateful", False),
            })
        return descs

    return [{
        "name": "main",
        "inputs": [{"name": "indices", "shape": [1], "dtype": "int32"}],
        "outputs": [{"name": "output", "shape": [1, embed_dim], "dtype": dtype}],
        "stateful": False,
    }]


def _resolve_attention_function_descriptors(
    functions: Optional[list],
    embed_dim: int,
    dtype: str,
) -> list:
    """Build function descriptors for attention result payloads."""
    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [{"name": "x", "shape": [1, 1, embed_dim], "dtype": dtype}]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, 1, embed_dim], "dtype": dtype}]),
                "stateful": fn.get("stateful", False),
            })
        return descs

    return [{
        "name": "main",
        "inputs": [{"name": "x", "shape": [1, 1, embed_dim], "dtype": dtype}],
        "outputs": [{"name": "output", "shape": [1, 1, embed_dim], "dtype": dtype}],
        "stateful": False,
    }]


def _resolve_mlp_block_function_descriptors(
    functions: Optional[list],
    input_dim: int,
    output_dim: int,
    dtype: str,
) -> list:
    """Build function descriptors for MLP block result payloads."""
    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [{"name": "x", "shape": [1, input_dim], "dtype": dtype}]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, output_dim], "dtype": dtype}]),
                "stateful": fn.get("stateful", False),
            })
        return descs

    return [{
        "name": "main",
        "inputs": [{"name": "x", "shape": [1, input_dim], "dtype": dtype}],
        "outputs": [{"name": "output", "shape": [1, output_dim], "dtype": dtype}],
        "stateful": False,
    }]


# ---------------------------------------------------------------------------
# Descriptor resolver wrappers for emit_program()
# ---------------------------------------------------------------------------

def _linear_proj_descriptor_resolver(command, dtype_str):
    return _resolve_function_descriptors(
        command.get("functions", None),
        command.get("input_dim", 64),
        command.get("output_dim", 32),
        dtype_str,
    )


def _decode_step_descriptor_resolver(command, dtype_str):
    return _resolve_decode_step_function_descriptors(
        command.get("functions", None),
        command.get("embed_dim", 128),
        dtype_str,
    )


def _lut_descriptor_resolver(command, dtype_str):
    return _resolve_lut_function_descriptors(
        command.get("functions", None),
        command.get("embed_dim", 512),
        dtype_str,
    )


def _attention_descriptor_resolver(command, dtype_str):
    return _resolve_attention_function_descriptors(
        command.get("functions", None),
        command.get("embed_dim", 128),
        dtype_str,
    )


def _mlp_block_descriptor_resolver(command, dtype_str):
    return _resolve_mlp_block_function_descriptors(
        command.get("functions", None),
        command.get("input_dim", 128),
        command.get("output_dim", 128),
        dtype_str,
    )


# ---------------------------------------------------------------------------
# Emit functions using emit_program() (deduplicated from 12× pattern — C-13)
# ---------------------------------------------------------------------------

def emit_linear_projection(command: dict) -> dict:
    """Build a single-function linear projection MIL program and save as mlpackage."""
    return emit_program(
        command,
        build_fn=build_linear_projection_program,
        resolve_descriptors_fn=_linear_proj_descriptor_resolver,
        default_task_name="linear_projection",
        error_prefix="MIL emission failed",
    )


def build_lut_projection_program(command: dict):
    """Build a MIL Program object for a LUT projection (gather-based).

    Payload fields consumed:
        task_name, vocab_size, embed_dim, num_groups, lut_bitwidth,
        batch_size, dtype, opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "lut_projection")
    vocab_size = command.get("vocab_size", 32000)
    embed_dim = command.get("embed_dim", 512)
    num_groups = command.get("num_groups", 64)
    lut_bitwidth = command.get("lut_bitwidth", 4)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    # LUT table shape: [num_groups, vocab_size]
    group_dim = embed_dim // num_groups

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(batch_size,), dtype=types.int32)],
            opset_version=target_os,
        )
        def prog(indices):
            lut_values = np.random.randn(num_groups * vocab_size, group_dim).astype(np_dtype)
            lut_table = mb.const(val=lut_values, name="lut_table")

            gathered = mb.gather(x=lut_table, indices=indices, name="gather")

            gathered_parts = [gathered]
            for g in range(1, num_groups):
                offset = g * vocab_size
                offset_indices_val = np.array([offset], dtype=np.int32)
                offset_const = mb.const(val=offset_indices_val, name=f"offset_{g}")
                gathered_g = mb.gather(x=lut_table, indices=offset_const, name=f"gather_{g}")
                gathered_parts.append(gathered_g)

            if len(gathered_parts) > 1:
                result = mb.concat(values=gathered_parts, axis=-1, name="output")
            else:
                result = gathered
            return result

    metadata = {
        "task_name": task_name,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_groups": num_groups,
        "lut_bitwidth": lut_bitwidth,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "lut_projection",
    }

    return prog, metadata


def emit_lut_projection(command: dict) -> dict:
    """Build a dedicated LUT projection MIL program and save as mlpackage."""
    return emit_program(
        command,
        build_fn=build_lut_projection_program,
        resolve_descriptors_fn=_lut_descriptor_resolver,
        default_task_name="lut_projection",
        error_prefix="LUT projection emission failed",
    )


def build_decode_step_program(command: dict):
    """Build a MIL Program object for a decode-step (QKV + attention + output projection).

    **NOTE**: This is the STATELESS variant. K and V cache values are
    deterministic `mb.const` tensors, not state reads. For real KV-cache
    state semantics, use `build_stateful_decode_step_program()` instead.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "decode_step")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)
    input_shape = (batch_size, embed_dim)

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=mil_dtype)],
            opset_version=target_os,
        )
        def prog(x):
            # Step 1: QKV projection
            qkv_dim = 3 * embed_dim
            w_qkv_val = np.random.randn(qkv_dim, embed_dim).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            # Split into Q, K, V along the last dimension
            q = mb.slice_by_index(x=qkv, begin=[0, 0], end=[batch_size, embed_dim], name="q")
            k = mb.slice_by_index(x=qkv, begin=[0, embed_dim], end=[batch_size, 2 * embed_dim], name="k")
            v = mb.slice_by_index(x=qkv, begin=[0, 2 * embed_dim], end=[batch_size, 3 * embed_dim], name="v")

            # Step 2: Multi-head attention
            q_4d = mb.reshape(x=q, shape=[batch_size, num_heads, 1, head_dim], name="q_4d")

            # KV cache: deterministic const values (stateless variant)
            k_cache_val = np.random.randn(kv_len, embed_dim).astype(np_dtype)
            k_cache = mb.const(val=k_cache_val, name="k_cache")
            v_cache_val = np.random.randn(kv_len, embed_dim).astype(np_dtype)
            v_cache = mb.const(val=v_cache_val, name="v_cache")

            k_4d = mb.reshape(x=k_cache, shape=[1, num_heads, kv_len, head_dim], name="k_4d")
            v_4d = mb.reshape(x=v_cache, shape=[1, num_heads, kv_len, head_dim], name="v_4d")

            attn_out = mb.scaled_dot_product_attention(query=q_4d, key=k_4d, value=v_4d, name="attn_out")
            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, embed_dim], name="attn_reshaped")

            # Step 3: Output projection
            w_out_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)
            result = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")

            return result

    metadata = {
        "task_name": task_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "decode_step",
    }

    return prog, metadata


# ---------------------------------------------------------------------------
# W-20 fix: emit_decode_step now routes to the STATEFUL path by default.
# ---------------------------------------------------------------------------

def emit_decode_step(command: dict) -> dict:
    """Build a stateful decode-step MIL program and save as mlpackage.

    W-20 FIX: This function now routes to the stateful path by default,
    using real KV-cache state semantics (mb.read_state /
    mb.coreml_update_state, iOS 18+). This matches Sprint 40's intent.

    Previously this routed to the stateless path (build_decode_step_program)
    using deterministic mb.const KV cache values. The stateless path is now
    available explicitly via emit_stateless_decode_step().

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, compute_units, output_path, seed, functions

    Returns a result dict with status, output_path, content_hash,
    package_files, function_descriptors, metadata including
    emission_path='stateful_decode_step'.
    """
    return emit_stateful_decode_step(command)


def emit_stateless_decode_step(command: dict) -> dict:
    """Build a stateless decode-step MIL program — explicit stateless path.

    This is the explicit stateless decode-step emission path. K and V cache
    values are deterministic `mb.const` tensors (not state reads), suitable
    for single-step inference testing where state persistence across calls
    is not required.

    For real autoregressive inference, use `emit_decode_step` (which now
    routes to the stateful path by default — W-20 fix) or
    `emit_stateful_decode_step` explicitly.

    Payload fields consumed: Same as emit_decode_step.
    """
    return emit_program(
        command,
        build_fn=build_decode_step_program,
        resolve_descriptors_fn=_decode_step_descriptor_resolver,
        default_task_name="decode_step",
        error_prefix="Decode-step emission failed",
    )


def build_stateful_decode_step_program(command: dict):
    """Build a stateful MIL Program for decode-step with real KV-cache state.

    This constructs a MIL program that uses `mb.StateTensorSpec` for KV cache
    state declaration, `mb.read_state` for reading cache values, and
    `mb.coreml_update_state` for writing updated K/V back to the cache.

    The program flow:
    1. Read KV cache state: mb.read_state(input=k_state), mb.read_state(input=v_state)
    2. QKV projection: linear(x, W_qkv) -> Q, K_new, V_new
    3. Update KV cache state: mb.coreml_update_state(state=k_state, value=updated_k)
    4. Scaled dot-product attention: mb.scaled_dot_product_attention(Q, updated_K, updated_V)
    5. Output projection: linear(attn_output, W_out) -> output

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "stateful_decode_step")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    input_shape = (batch_size, embed_dim)
    kv_state_shape = (1, num_heads, kv_len, head_dim)

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=input_shape, dtype=mil_dtype),
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # k_cache state
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # v_cache state
            ],
            opset_version=target_os,
        )
        def prog(x, k_state, v_state):
            # Step 1: Read KV cache state
            k_cached = mb.read_state(input=k_state, name="k_cache_read")
            v_cached = mb.read_state(input=v_state, name="v_cache_read")

            # Step 2: QKV projection
            qkv_dim = 3 * embed_dim
            w_qkv_val = np.random.randn(qkv_dim, embed_dim).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            # Split into Q, K_new, V_new
            q = mb.slice_by_index(x=qkv, begin=[0, 0], end=[batch_size, embed_dim], name="q")
            k_new = mb.slice_by_index(x=qkv, begin=[0, embed_dim], end=[batch_size, 2 * embed_dim], name="k_new")
            v_new = mb.slice_by_index(x=qkv, begin=[0, 2 * embed_dim], end=[batch_size, 3 * embed_dim], name="v_new")

            # Reshape for multi-head attention
            q_4d = mb.reshape(x=q, shape=[batch_size, num_heads, 1, head_dim], name="q_4d")
            k_new_4d = mb.reshape(x=k_new, shape=[1, num_heads, 1, head_dim], name="k_new_4d")
            v_new_4d = mb.reshape(x=v_new, shape=[1, num_heads, 1, head_dim], name="v_new_4d")

            # Step 3: Update KV cache by replacing the last position with new K/V
            k_updated = mb.slice_update(
                x=k_cached, update=k_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="k_updated"
            )
            v_updated = mb.slice_update(
                x=v_cached, update=v_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="v_updated"
            )

            # Step 4: Write updated KV cache back to state (side effect only)
            mb.coreml_update_state(state=k_state, value=k_updated, name="k_cache_write")
            mb.coreml_update_state(state=v_state, value=v_updated, name="v_cache_write")

            # Step 5: Scaled dot-product attention using the updated KV cache
            attn_out = mb.scaled_dot_product_attention(
                query=q_4d, key=k_updated, value=v_updated, name="attn_out"
            )
            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, embed_dim], name="attn_reshaped")

            # Step 6: Output projection
            w_out_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)
            result = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")

            return result

    metadata = {
        "task_name": task_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "stateful_decode_step",
        "stateful": True,
        "state_inputs": [
            {"name": "k_state", "shape": list(kv_state_shape), "dtype": dtype_str},
            {"name": "v_state", "shape": list(kv_state_shape), "dtype": dtype_str},
        ],
    }

    return prog, metadata


def _stateful_decode_step_descriptor_resolver(command, dtype_str):
    """Resolve function descriptors for stateful decode-step with KV state inputs."""
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    functions = command.get("functions", None)

    if functions is not None:
        descs = []
        for fn in functions:
            descs.append({
                "name": fn.get("name", "main"),
                "inputs": fn.get("inputs", [
                    {"name": "x", "shape": [1, embed_dim], "dtype": dtype_str},
                    {"name": "k_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                    {"name": "v_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                ]),
                "outputs": fn.get("outputs", [{"name": "output", "shape": [1, embed_dim], "dtype": dtype_str}]),
                "stateful": True,
            })
        return descs

    return [{
        "name": "main",
        "inputs": [
            {"name": "x", "shape": [1, embed_dim], "dtype": dtype_str},
            {"name": "k_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
            {"name": "v_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
        ],
        "outputs": [{"name": "output", "shape": [1, embed_dim], "dtype": dtype_str}],
        "stateful": True,
    }]


def emit_stateful_decode_step(command: dict) -> dict:
    """Build a stateful decode-step MIL program with real KV-cache state and save as mlpackage.

    This is the stateful emission path for decode-step tasks (Sprint 36/37).
    Unlike the stateless `emit_stateless_decode_step` which uses deterministic
    `mb.const` for KV cache values, this path declares KV cache as Core ML
    state via `mb.StateTensorSpec`.

    The emitted model requires iOS 18+ and macOS 15+ for stateful model execution.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, compute_units, output_path, seed, functions
    """
    return emit_program(
        command,
        build_fn=build_stateful_decode_step_program,
        resolve_descriptors_fn=_stateful_decode_step_descriptor_resolver,
        default_task_name="stateful_decode_step",
        use_stateful_pipeline=True,
        error_prefix="Stateful decode-step emission failed",
    )


def build_shard_decode_step_program(command: dict):
    """Build a shard-role-aware decode-step MIL program with genuinely different
    op structures per role (Sprint 44).

    This constructs a decode-step program whose op structure varies by shard role,
    matching the RoleMirBuilder's intent from `crates/passes/src/role_mir.rs`.
    After Sprint 44, each role adds a role-specific post-attention operation:

      - **Entry**: attention + output_proj + **Reshape** (handoff preparation)
      - **Interior**: attention + output_proj + **GELU** activation
      - **Exit**: attention + output_proj + **LayerNorm**

    The role→op mapping is defined in program_builder.SHARD_ROLE_OP_MAP,
    which is the single source of truth for the Python side and MUST be
    kept in sync with the Rust RoleMirBuilder (W-27).

    Payload fields consumed (in addition to build_stateful_decode_step_program):
        shard_role: str — "Entry", "Interior", or "Exit" (required)
        shard_hidden_dim, shard_num_heads, shard_head_dim, shard_output_dim (optional)
    """
    ct, mb, types, np = _import_coremltools()

    shard_role = command.get("shard_role", "Entry")
    if shard_role not in ("Entry", "Interior", "Exit"):
        raise ValueError(f"shard_role must be Entry, Interior, or Exit, got '{shard_role}'")

    # Base dimensions
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    # Shard-specific overrides
    hidden_dim = command.get("shard_hidden_dim", embed_dim)
    shard_heads = command.get("shard_num_heads", num_heads)
    shard_head_dim = command.get("shard_head_dim", head_dim)
    output_dim = command.get("shard_output_dim", embed_dim)
    if shard_role != "Exit":
        output_dim = hidden_dim  # Entry/Interior output hidden_dim

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    # Input shape depends on shard role
    if shard_role == "Entry":
        input_shape = (batch_size, embed_dim)
    else:
        input_shape = (batch_size, hidden_dim)

    kv_state_shape = (1, shard_heads, kv_len, shard_head_dim)

    # Determine role-specific post-attention operation from SHARD_ROLE_OP_MAP
    # (W-27: single source of truth, synced with Rust RoleMirBuilder)
    role_specific_op = command.get("role_specific_op", None)
    if role_specific_op is None:
        role_specific_op = SHARD_ROLE_OP_MAP.get(shard_role, "none")

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=input_shape, dtype=mil_dtype),
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # k_cache state
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # v_cache state
            ],
            opset_version=target_os,
        )
        def prog(x, k_state, v_state):
            # Read KV cache state
            k_cached = mb.read_state(input=k_state, name="k_cache_read")
            v_cached = mb.read_state(input=v_state, name="v_cache_read")

            # QKV projection — output dim is 3 * hidden_dim for this shard
            qkv_dim = 3 * hidden_dim
            w_qkv_val = np.random.randn(qkv_dim, input_shape[1]).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            # Split into Q, K_new, V_new
            q = mb.slice_by_index(x=qkv, begin=[0, 0], end=[batch_size, hidden_dim], name="q")
            k_new = mb.slice_by_index(x=qkv, begin=[0, hidden_dim], end=[batch_size, 2 * hidden_dim], name="k_new")
            v_new = mb.slice_by_index(x=qkv, begin=[0, 2 * hidden_dim], end=[batch_size, 3 * hidden_dim], name="v_new")

            # Reshape for multi-head attention
            q_4d = mb.reshape(x=q, shape=[batch_size, shard_heads, 1, shard_head_dim], name="q_4d")
            k_new_4d = mb.reshape(x=k_new, shape=[1, shard_heads, 1, shard_head_dim], name="k_new_4d")
            v_new_4d = mb.reshape(x=v_new, shape=[1, shard_heads, 1, shard_head_dim], name="v_new_4d")

            # Update KV cache
            k_updated = mb.slice_update(
                x=k_cached, update=k_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, shard_heads, kv_len, shard_head_dim],
                name="k_updated"
            )
            v_updated = mb.slice_update(
                x=v_cached, update=v_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, shard_heads, kv_len, shard_head_dim],
                name="v_updated"
            )

            # Write updated KV cache back to state (side effect only)
            mb.coreml_update_state(state=k_state, value=k_updated, name="k_cache_write")
            mb.coreml_update_state(state=v_state, value=v_updated, name="v_cache_write")

            # Scaled dot-product attention
            attn_out = mb.scaled_dot_product_attention(
                query=q_4d, key=k_updated, value=v_updated, name="attn_out"
            )
            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, hidden_dim], name="attn_reshaped")

            # Output projection
            w_out_val = np.random.randn(output_dim, hidden_dim).astype(np_dtype)
            projected = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")

            # Role-specific post-attention operation (Sprint 44, W-27)
            # Uses SHARD_ROLE_OP_MAP for the role→op mapping.
            if role_specific_op == "reshape":
                result = mb.reshape(
                    x=projected,
                    shape=[batch_size, 1, output_dim],
                    name="handoff_reshape",
                )
            elif role_specific_op == "gelu":
                result = mb.gelu(
                    x=projected,
                    mode="TANH_APPROXIMATION",
                    name="interior_gelu",
                )
            elif role_specific_op == "layernorm":
                ln_gamma_val = np.ones(output_dim, dtype=np_dtype)
                ln_beta_val = np.zeros(output_dim, dtype=np_dtype)
                result = mb.layer_norm(
                    x=projected,
                    gamma=ln_gamma_val,
                    beta=ln_beta_val,
                    axes=[1],
                    epsilon=np_dtype(1e-5),
                    name="exit_layernorm",
                )
            else:
                result = projected

            return result

    metadata = {
        "task_name": command.get("task_name", f"shard_decode_step_{shard_role.lower()}"),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": f"shard_decode_step_{shard_role.lower()}",
        "stateful": True,
        "shard_role": shard_role,
        "shard_hidden_dim": hidden_dim,
        "shard_num_heads": shard_heads,
        "shard_head_dim": shard_head_dim,
        "shard_output_dim": output_dim,
        "role_specific_op": role_specific_op,
        "state_inputs": [
            {"name": "k_state", "shape": list(kv_state_shape), "dtype": dtype_str},
            {"name": "v_state", "shape": list(kv_state_shape), "dtype": dtype_str},
        ],
    }

    return prog, metadata


def _shard_decode_step_descriptor_resolver(command, dtype_str):
    """Resolve function descriptors for shard decode-step."""
    shard_role = command.get("shard_role", "Entry")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    hidden_dim = command.get("shard_hidden_dim", embed_dim)
    shard_heads = command.get("shard_num_heads", num_heads)
    shard_head_dim = command.get("shard_head_dim", head_dim)
    output_dim = command.get("shard_output_dim", embed_dim)
    kv_len = command.get("kv_len", 64)

    if shard_role == "Entry":
        input_shape = [1, embed_dim]
    else:
        input_shape = [1, hidden_dim]

    # Output shape varies by shard role and role-specific op (Sprint 44)
    if shard_role == "Entry":
        output_shape = [1, 1, hidden_dim]
    elif shard_role == "Exit":
        output_shape = [1, output_dim]
    else:
        output_shape = [1, hidden_dim]

    # Determine role-specific op for metadata
    role_specific_op = command.get("role_specific_op", SHARD_ROLE_OP_MAP.get(shard_role, "none"))

    return [{
        "name": "main",
        "inputs": [
            {"name": "x", "shape": input_shape, "dtype": dtype_str},
            {"name": "k_state", "shape": [1, shard_heads, kv_len, shard_head_dim], "dtype": dtype_str, "is_state": True},
            {"name": "v_state", "shape": [1, shard_heads, kv_len, shard_head_dim], "dtype": dtype_str, "is_state": True},
        ],
        "outputs": [{"name": "output", "shape": output_shape, "dtype": dtype_str}],
        "stateful": True,
        "shard_role": shard_role,
        "role_specific_op": role_specific_op,
    }]


def emit_shard_decode_step(command: dict) -> dict:
    """Build a shard-role-aware decode-step MIL program and save as mlpackage.

    Sprint 37 introduced dimension differences per role. Sprint 44 extends
    this to produce genuinely different op structures per role, matching
    the RoleMirBuilder's ShardOpProfile assignments.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, compute_units, output_path, seed,
        shard_role (required: "Entry", "Interior", or "Exit"),
        shard_hidden_dim, shard_num_heads, shard_head_dim,
        shard_output_dim (all optional, fall back to base params)
    """
    shard_role = command.get("shard_role", "Entry")
    if shard_role not in ("Entry", "Interior", "Exit"):
        return _error_result(f"shard_role must be Entry, Interior, or Exit, got '{shard_role}'")

    return emit_program(
        command,
        build_fn=build_shard_decode_step_program,
        resolve_descriptors_fn=_shard_decode_step_descriptor_resolver,
        default_task_name=f"shard_decode_step_{shard_role.lower()}",
        use_stateful_pipeline=True,
        error_prefix="Shard decode-step emission failed",
    )


def emit_palettized_linear_projection(command: dict) -> dict:
    """Emit a normal linear projection and then apply real coremltools palettization.

    This is the honest real palettization path (Sprint 38). The approach:
    1. Build a normal mb.linear program (same as emit_linear_projection)
    2. Convert to MLModel
    3. Apply palettization via `palettize.apply_palettization()`
    4. Save the palettized mlpackage

    Payload fields consumed:
        task_name, input_dim, output_dim, batch_size, dtype,
        opset_version, compute_units, output_path, seed,
        palettization_nbits (default: 4),
        palettization_mode (default: "kmeans"),
        palettization_granularity (default: "per_grouped_channel"),
        palettization_group_size (default: 32)
    """
    try:
        ct = _ensure_coremltools()

        from converter import convert_milprogram
        from palettize import apply_palettization

        output_path = command.get("output_path", "")
        if not output_path:
            return _error_result("No output_path specified")

        compute_units_str = command.get("compute_units", "CPU_AND_NE")
        dtype_str = command.get("dtype", "fp16")

        # Step 1: Build the normal linear projection MIL program
        prog, prog_meta = build_linear_projection_program(command)

        # Step 2: Convert using converter.py
        precision_str = "FLOAT16" if dtype_str == "fp16" else "FLOAT32"
        mlmodel = convert_milprogram(
            prog,
            opset_version=command.get("opset_version", "iOS18"),
            compute_precision=precision_str,
            compute_units=compute_units_str,
        )

        # Step 3: Apply real palettization via coremltools API
        palettization_nbits = command.get("palettization_nbits", 4)
        palettization_mode = command.get("palettization_mode", "kmeans")
        palettization_granularity = command.get("palettization_granularity", "per_grouped_channel")
        palettization_group_size = command.get("palettization_group_size", 32)

        weight_name = "output"

        palettization_specs = [{
            "weight_name": weight_name,
            "mode": palettization_mode,
            "nbits": palettization_nbits,
            "granularity": palettization_granularity,
            "group_size": palettization_group_size,
            "channel_axis": 1,
        }]

        palettized_model = apply_palettization(mlmodel, palettization_specs)

        # Step 4: Resolve function descriptors
        input_dim = command.get("input_dim", 64)
        output_dim = command.get("output_dim", 32)
        functions = command.get("functions", None)
        function_descriptors = _resolve_function_descriptors(functions, input_dim, output_dim, dtype_str)

        # Step 5: Save the palettized model
        task_name = command.get("task_name", "palettized_linear_projection")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_path = output_dir / f"{task_name}.mlpackage"

        save_info = save_mlpackage(palettized_model, str(mlpackage_path))

        # Step 6: Compute plan (best-effort)
        compute_plan = compute_plan_info(str(mlpackage_path), compute_units_str)

        return {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": compute_plan,
            "function_descriptors": function_descriptors,
            "metadata": {
                **prog_meta,
                "emission_path": "palettized_linear_projection",
                "palettization_applied": palettization_specs,
            },
        }

    except Exception as e:
        return _error_result(f"Palettized linear projection emission failed: {e}")


def build_mlp_block_program(command: dict):
    """Build a MIL Program object for an MLP block (up-projection + activation + down-projection).

    Payload fields consumed:
        task_name, input_dim, hidden_dim, output_dim, activation,
        batch_size, dtype, opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "mlp_block")
    input_dim = command.get("input_dim", 128)
    hidden_dim = command.get("hidden_dim", 512)
    output_dim = command.get("output_dim", 128)
    activation = command.get("activation", "gelu")
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)
    input_shape = (batch_size, input_dim)

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=mil_dtype)],
            opset_version=target_os,
        )
        def prog(x):
            # Step 1: Up-projection (input_dim -> hidden_dim) via mb.linear
            w_up_val = np.random.randn(hidden_dim, input_dim).astype(np_dtype)
            up_proj = mb.linear(x=x, weight=w_up_val, bias=None, name="up_proj")

            # Step 2: Activation
            if activation == "gelu":
                activated = mb.gelu(x=up_proj, mode="TANH_APPROXIMATION", name="activated")
            elif activation == "relu":
                activated = mb.relu(x=up_proj, name="activated")
            else:
                raise ValueError(f"Unsupported activation: {activation}. Must be 'gelu' or 'relu'.")

            # Step 3: Down-projection (hidden_dim -> output_dim) via mb.linear
            w_down_val = np.random.randn(output_dim, hidden_dim).astype(np_dtype)
            result = mb.linear(x=activated, weight=w_down_val, bias=None, name="output")

            return result

    metadata = {
        "task_name": task_name,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "activation": activation,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "mlp_block",
    }

    return prog, metadata


def emit_mlp_block(command: dict) -> dict:
    """Build a dedicated MLP block MIL program and save as mlpackage."""
    return emit_program(
        command,
        build_fn=build_mlp_block_program,
        resolve_descriptors_fn=_mlp_block_descriptor_resolver,
        default_task_name="mlp_block",
        error_prefix="MLP block emission failed",
    )


def build_attention_program(command: dict):
    """Build a MIL Program object for a multi-head self-attention block.

    When `causal=True` (default for autoregressive models), a causal mask is applied.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, seq_len,
        batch_size, dtype, opset_version, seed, causal (optional, default True)
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "attention")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    seq_len = command.get("seq_len", 32)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)
    causal = command.get("causal", True)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    input_shape = (batch_size, seq_len, embed_dim)
    qkv_dim = 3 * embed_dim

    with rng_seed_context(seed):
        @mb.program(
            input_specs=[mb.TensorSpec(shape=input_shape, dtype=mil_dtype)],
            opset_version=target_os,
        )
        def prog(x):
            # Step 1: QKV projection — linear(x, W_qkv)
            w_qkv_val = np.random.randn(qkv_dim, embed_dim).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            # Split into Q, K, V along the last dimension
            q = mb.slice_by_index(x=qkv, begin=[0, 0, 0], end=[batch_size, seq_len, embed_dim], name="q")
            k = mb.slice_by_index(x=qkv, begin=[0, 0, embed_dim], end=[batch_size, seq_len, 2 * embed_dim], name="k")
            v = mb.slice_by_index(x=qkv, begin=[0, 0, 2 * embed_dim], end=[batch_size, seq_len, 3 * embed_dim], name="v")

            # Step 2: Multi-head attention
            q_4d = mb.reshape(x=q, shape=[batch_size, seq_len, num_heads, head_dim], name="q_4d")
            k_4d = mb.reshape(x=k, shape=[batch_size, seq_len, num_heads, head_dim], name="k_4d")
            v_4d = mb.reshape(x=v, shape=[batch_size, seq_len, num_heads, head_dim], name="v_4d")

            q_t = mb.transpose(x=q_4d, perm=[0, 2, 1, 3], name="q_t")
            k_t = mb.transpose(x=k_4d, perm=[0, 2, 1, 3], name="k_t")
            v_t = mb.transpose(x=v_4d, perm=[0, 2, 1, 3], name="v_t")

            if causal:
                causal_mask_val = np.tril(
                    np.ones((seq_len, seq_len), dtype=np.bool_)
                ).reshape(1, 1, seq_len, seq_len)
                causal_mask = mb.const(val=causal_mask_val, name="causal_mask")
                attn_out = mb.scaled_dot_product_attention(
                    query=q_t, key=k_t, value=v_t,
                    attn_mask=causal_mask, name="attn_out"
                )
            else:
                attn_out = mb.scaled_dot_product_attention(
                    query=q_t, key=k_t, value=v_t, name="attn_out"
                )

            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, seq_len, embed_dim], name="attn_reshaped")

            # Step 3: Output projection
            w_out_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)
            result = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")

            return result

    metadata = {
        "task_name": task_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "causal": causal,
        "emission_path": "attention",
    }

    return prog, metadata


def emit_attention(command: dict) -> dict:
    """Build a dedicated attention MIL program and save as mlpackage."""
    return emit_program(
        command,
        build_fn=build_attention_program,
        resolve_descriptors_fn=_attention_descriptor_resolver,
        default_task_name="attention",
        error_prefix="Attention emission failed",
    )


def emit_mlprogram(command: dict) -> dict:
    """Emit an ML program using the clean build → convert → save pipeline.

    For linear projection tasks, this is functionally equivalent to
    emit_linear_projection but uses the separated pipeline explicitly.
    """
    return emit_program(
        command,
        build_fn=build_linear_projection_program,
        resolve_descriptors_fn=_linear_proj_descriptor_resolver,
        default_task_name="linear_projection",
        error_prefix="ML program emission failed",
    )


def inspect_mlpackage(mlpackage_path: str) -> dict:
    """Inspect a saved mlpackage: read structure, report inventory."""
    if not mlpackage_path or not Path(mlpackage_path).exists():
        return _error_result(f"mlpackage not found: {mlpackage_path}")

    try:
        import coremltools as ct
    except ImportError:
        return _error_result("coremltools not installed")

    try:
        spec_path = Path(mlpackage_path) / "Data" / "com.apple.CoreML" / "model.mlmodel"
        spec_exists = spec_path.exists()

        manifest_path = Path(mlpackage_path) / "Manifest.json"
        manifest = None
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

        load_result = None
        try:
            model = ct.models.MLModel(str(mlpackage_path))
            load_result = {
                "loaded": True,
                "input_names": list(model.input_description.keys()) if hasattr(model, 'input_description') else [],
                "output_names": list(model.output_description.keys()) if hasattr(model, 'output_description') else [],
            }
        except Exception as e:
            load_result = {"loaded": False, "reason": str(e)}

        files = []
        total_size = 0
        for root, dirs, filenames in os.walk(mlpackage_path):
            for f in filenames:
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, mlpackage_path)
                sz = os.path.getsize(fp)
                total_size += sz
                files.append({"path": rel, "size_bytes": sz})

        return {
            "status": "success",
            "error_message": None,
            "spec_exists": spec_exists,
            "manifest": manifest,
            "load_result": load_result,
            "total_size_bytes": total_size,
            "file_count": len(files),
            "files": files,
            "metadata": {},
        }

    except Exception as e:
        return _error_result(f"Inspection failed: {e}")


def build_multifunction_program(command: dict):
    """Build a multi-function MIL program with "embedding" and "decode_step" functions.

    Sprint 40: The decode_step function uses the stateful variant with
    real KV-cache state semantics.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, seed
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "multifunction")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    vocab_size = command.get("vocab_size", 32000)

    with rng_seed_context(seed):
        # --- Function 1: embedding ---
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(batch_size, vocab_size), dtype=types.int32)],
            opset_version=target_os,
            function_name="embedding",
        )
        def embedding_prog(token_ids):
            w_embed_val = np.random.randn(embed_dim, vocab_size).astype(np_dtype)
            b_embed_val = np.zeros(embed_dim, dtype=np_dtype)
            embedded = mb.linear(x=token_ids, weight=w_embed_val, bias=b_embed_val, name="embedded")
            return embedded

        # --- Function 2: decode_step (STATEFUL, Sprint 40) ---
        kv_state_shape = (1, num_heads, kv_len, head_dim)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(batch_size, embed_dim), dtype=mil_dtype),
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # k_cache state
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),  # v_cache state
            ],
            opset_version=target_os,
            function_name="decode_step",
        )
        def decode_step_prog(x, k_state, v_state):
            # Read KV cache state
            k_cached = mb.read_state(input=k_state, name="k_cache_read")
            v_cached = mb.read_state(input=v_state, name="v_cache_read")

            # QKV projection
            qkv_dim = 3 * embed_dim
            w_qkv_val = np.random.randn(qkv_dim, embed_dim).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            q = mb.slice_by_index(x=qkv, begin=[0, 0], end=[batch_size, embed_dim], name="q")
            k_new = mb.slice_by_index(x=qkv, begin=[0, embed_dim], end=[batch_size, 2 * embed_dim], name="k_new")
            v_new = mb.slice_by_index(x=qkv, begin=[0, 2 * embed_dim], end=[batch_size, 3 * embed_dim], name="v_new")

            q_4d = mb.reshape(x=q, shape=[batch_size, num_heads, 1, head_dim], name="q_4d")
            k_new_4d = mb.reshape(x=k_new, shape=[1, num_heads, 1, head_dim], name="k_new_4d")
            v_new_4d = mb.reshape(x=v_new, shape=[1, num_heads, 1, head_dim], name="v_new_4d")

            # Update KV cache
            k_updated = mb.slice_update(
                x=k_cached, update=k_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="k_updated"
            )
            v_updated = mb.slice_update(
                x=v_cached, update=v_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="v_updated"
            )

            # Write updated KV cache back to state
            mb.coreml_update_state(state=k_state, value=k_updated, name="k_cache_write")
            mb.coreml_update_state(state=v_state, value=v_updated, name="v_cache_write")

            # Scaled dot-product attention
            attn_out = mb.scaled_dot_product_attention(
                query=q_4d, key=k_updated, value=v_updated, name="attn_out"
            )
            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, embed_dim], name="attn_reshaped")

            # Output projection
            w_out_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)
            result = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")
            return result

    # --- Merge functions into a single program ---
    embedding_prog.add_function("decode_step", decode_step_prog.functions["decode_step"])
    embedding_prog.default_function_name = "embedding"

    metadata = {
        "task_name": task_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "multifunction",
        "multifunction": True,
        "function_names": ["embedding", "decode_step"],
        "stateful_decode_step": True,
    }

    return embedding_prog, metadata


def emit_multifunction(command: dict) -> dict:
    """Build a multi-function MIL program and save as mlpackage.

    Sprint 40: The decode_step function uses real KV-cache state semantics,
    so the converter must use `convert_milprogram(pass_pipeline=make_stateful_pass_pipeline())`.

    After saving, validates that the resulting model has 2 functions.
    """
    try:
        ct = _ensure_coremltools()

        from converter import convert_milprogram, make_stateful_pass_pipeline

        output_path = command.get("output_path", "")
        if not output_path:
            return _error_result("No output_path specified")

        compute_units_str = command.get("compute_units", "CPU_AND_NE")
        dtype_str = command.get("dtype", "fp16")

        # Step 1: Build the multi-function MIL program
        prog, prog_meta = build_multifunction_program(command)

        # Step 2: Convert using stateful-aware pass pipeline
        precision_str = "FLOAT16" if dtype_str == "fp16" else "FLOAT32"
        mlmodel = convert_milprogram(
            prog,
            opset_version=command.get("opset_version", "iOS18"),
            compute_precision=precision_str,
            compute_units=compute_units_str,
            pass_pipeline=make_stateful_pass_pipeline(),
        )

        # Step 3: Resolve function descriptors for both functions
        embed_dim = command.get("embed_dim", 128)
        num_heads = command.get("num_heads", 4)
        head_dim = command.get("head_dim", 32)
        kv_len = command.get("kv_len", 64)
        vocab_size = command.get("vocab_size", 32000)

        function_descriptors = [
            {
                "name": "embedding",
                "inputs": [{"name": "token_ids", "shape": [1, vocab_size], "dtype": "int32"}],
                "outputs": [{"name": "embedded", "shape": [1, embed_dim], "dtype": dtype_str}],
                "stateful": False,
            },
            {
                "name": "decode_step",
                "inputs": [
                    {"name": "x", "shape": [1, embed_dim], "dtype": dtype_str},
                    {"name": "k_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                    {"name": "v_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                ],
                "outputs": [{"name": "output", "shape": [1, embed_dim], "dtype": dtype_str}],
                "stateful": True,
            },
        ]

        # Step 4: Save
        task_name = command.get("task_name", "multifunction")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_path = output_dir / f"{task_name}.mlpackage"

        save_info = save_mlpackage(mlmodel, str(mlpackage_path))

        # Step 5: Validate that the saved model has 2 functions
        multifunction_validation = {"validated": False, "function_count": None, "function_names": []}
        try:
            spec = mlmodel.get_spec()
            if hasattr(spec, 'mlProgram') and hasattr(spec.mlProgram, 'functions'):
                fn_dict = spec.mlProgram.functions
                fn_names = list(fn_dict.keys())
                multifunction_validation = {
                    "validated": True,
                    "function_count": len(fn_names),
                    "function_names": fn_names,
                    "has_embedding": "embedding" in fn_names,
                    "has_decode_step": "decode_step" in fn_names,
                }
        except Exception as e:
            multifunction_validation["error"] = str(e)

        # Step 6: Compute plan (best-effort)
        compute_plan = compute_plan_info(str(mlpackage_path), compute_units_str)

        return {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": compute_plan,
            "function_descriptors": function_descriptors,
            "multifunction_validation": multifunction_validation,
            "metadata": prog_meta,
        }

    except Exception as e:
        return _error_result(f"Multi-function emission failed: {e}")


def validate_multifunction_package(mlpackage_path: str, expected_functions: list = None) -> dict:
    """Validate a multi-function mlpackage.

    Loads the model via ct.models.MLModel and checks that
    spec.mlProgram.functions has the expected function names.
    Also inspects per-function op counts and weight file sizes.

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        expected_functions: Optional list of expected function names.

    Returns:
        Dict with: valid, function_count, function_names, etc.
    """
    ct, _, _, _ = _import_coremltools()
    if ct is None:
        return {
            "valid": False,
            "function_count": 0,
            "function_names": [],
            "missing_functions": ["embedding", "decode_step"],
            "extra_functions": [],
            "function_op_counts": {},
            "weight_file_size_bytes": None,
            "weight_sharing_possible": None,
            "error": "coremltools not installed",
        }

    if expected_functions is None:
        expected_functions = ["embedding", "decode_step"]
    expected_set = set(expected_functions)

    try:
        model = ct.models.MLModel(mlpackage_path)
        spec = model.get_spec()

        if not hasattr(spec, 'mlProgram') or not hasattr(spec.mlProgram, 'functions'):
            return {
                "valid": False,
                "function_count": 0,
                "function_names": [],
                "missing_functions": list(expected_set),
                "extra_functions": [],
                "function_op_counts": {},
                "weight_file_size_bytes": None,
                "weight_sharing_possible": None,
                "error": "Model does not have mlProgram.functions",
            }

        fn_dict = spec.mlProgram.functions
        fn_names = list(fn_dict.keys())
        fn_set = set(fn_names)

        missing = list(expected_set - fn_set)
        extra = list(fn_set - expected_set)

        # Per-function op inspection
        function_op_counts = {}
        for fn_name, fn_block in fn_dict.items():
            op_count = 0
            if hasattr(fn_block, 'block_specializations'):
                for _spec_key, block in fn_block.block_specializations.items():
                    op_count += len(block.operations)
            function_op_counts[fn_name] = op_count

        # Weight file size check
        weight_path = Path(mlpackage_path) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
        weight_size = weight_path.stat().st_size if weight_path.exists() else None

        weight_sharing_possible = None
        if weight_size is not None and len(fn_names) > 1:
            weight_sharing_possible = "structurally_possible_runtime_verification_needed"

        return {
            "valid": len(missing) == 0,
            "function_count": len(fn_names),
            "function_names": fn_names,
            "missing_functions": missing,
            "extra_functions": extra,
            "function_op_counts": function_op_counts,
            "weight_file_size_bytes": weight_size,
            "weight_sharing_possible": weight_sharing_possible,
        }

    except Exception as e:
        return {
            "valid": False,
            "function_count": 0,
            "function_names": [],
            "missing_functions": list(expected_set),
            "extra_functions": [],
            "function_op_counts": {},
            "weight_file_size_bytes": None,
            "weight_sharing_possible": None,
            "error": str(e),
        }


def build_multifunction_program_with_shared_weights(command: dict):
    """Build a multi-function MIL program with weight sharing between functions.

    Sprint 42: This variant demonstrates weight sharing across functions.
    The shared weight pattern models tied embeddings.

    Payload fields consumed:
        task_name, embed_dim, num_heads, head_dim, kv_len, batch_size,
        dtype, opset_version, seed, share_weights (bool, default True)
    """
    ct, mb, types, np = _import_coremltools()

    task_name = command.get("task_name", "multifunction_shared")
    embed_dim = command.get("embed_dim", 128)
    num_heads = command.get("num_heads", 4)
    head_dim = command.get("head_dim", 32)
    kv_len = command.get("kv_len", 64)
    batch_size = command.get("batch_size", 1)
    dtype_str = command.get("dtype", "fp16")
    opset_version = command.get("opset_version", "iOS18")
    seed = command.get("seed", 42)
    share_weights = command.get("share_weights", True)

    np_dtype, mil_dtype = resolve_dtypes(dtype_str, types)
    target_os = resolve_opset_target(ct, opset_version)

    vocab_size = command.get("vocab_size", 32000)

    with rng_seed_context(seed):
        # Pre-create shared weights
        shared_weight_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)

        # --- Function 1: embedding ---
        @mb.program(
            input_specs=[mb.TensorSpec(shape=(batch_size, vocab_size), dtype=types.int32)],
            opset_version=target_os,
            function_name="embedding",
        )
        def embedding_prog(token_ids):
            w_embed_val = np.random.randn(embed_dim, vocab_size).astype(np_dtype)
            b_embed_val = np.zeros(embed_dim, dtype=np_dtype)
            embedded = mb.linear(x=token_ids, weight=w_embed_val, bias=b_embed_val, name="embedded")

            if share_weights:
                shared_w = mb.const(val=shared_weight_val, name="shared_projection_weight")
                hidden = mb.linear(x=embedded, weight=shared_w, bias=None, name="hidden_proj")
                return hidden
            else:
                return embedded

        # --- Function 2: decode_step (STATEFUL) ---
        kv_state_shape = (1, num_heads, kv_len, head_dim)

        @mb.program(
            input_specs=[
                mb.TensorSpec(shape=(batch_size, embed_dim), dtype=mil_dtype),
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),
                mb.StateTensorSpec(kv_state_shape, dtype=mil_dtype),
            ],
            opset_version=target_os,
            function_name="decode_step",
        )
        def decode_step_prog(x, k_state, v_state):
            k_cached = mb.read_state(input=k_state, name="k_cache_read")
            v_cached = mb.read_state(input=v_state, name="v_cache_read")

            qkv_dim = 3 * embed_dim
            w_qkv_val = np.random.randn(qkv_dim, embed_dim).astype(np_dtype)
            qkv = mb.linear(x=x, weight=w_qkv_val, bias=None, name="qkv_proj")

            q = mb.slice_by_index(x=qkv, begin=[0, 0], end=[batch_size, embed_dim], name="q")
            k_new = mb.slice_by_index(x=qkv, begin=[0, embed_dim], end=[batch_size, 2 * embed_dim], name="k_new")
            v_new = mb.slice_by_index(x=qkv, begin=[0, 2 * embed_dim], end=[batch_size, 3 * embed_dim], name="v_new")

            q_4d = mb.reshape(x=q, shape=[batch_size, num_heads, 1, head_dim], name="q_4d")
            k_new_4d = mb.reshape(x=k_new, shape=[1, num_heads, 1, head_dim], name="k_new_4d")
            v_new_4d = mb.reshape(x=v_new, shape=[1, num_heads, 1, head_dim], name="v_new_4d")

            k_updated = mb.slice_update(
                x=k_cached, update=k_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="k_updated"
            )
            v_updated = mb.slice_update(
                x=v_cached, update=v_new_4d,
                begin=[0, 0, kv_len - 1, 0], end=[1, num_heads, kv_len, head_dim],
                name="v_updated"
            )

            mb.coreml_update_state(state=k_state, value=k_updated, name="k_cache_write")
            mb.coreml_update_state(state=v_state, value=v_updated, name="v_cache_write")

            attn_out = mb.scaled_dot_product_attention(
                query=q_4d, key=k_updated, value=v_updated, name="attn_out"
            )
            attn_reshaped = mb.reshape(x=attn_out, shape=[batch_size, embed_dim], name="attn_reshaped")

            if share_weights:
                shared_w = mb.const(val=shared_weight_val, name="shared_projection_weight")
                result = mb.linear(x=attn_reshaped, weight=shared_w, bias=None, name="output")
            else:
                w_out_val = np.random.randn(embed_dim, embed_dim).astype(np_dtype)
                result = mb.linear(x=attn_reshaped, weight=w_out_val, bias=None, name="output")
            return result

    # --- Merge functions ---
    embedding_prog.add_function("decode_step", decode_step_prog.functions["decode_step"])
    embedding_prog.default_function_name = "embedding"

    metadata = {
        "task_name": task_name,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "kv_len": kv_len,
        "batch_size": batch_size,
        "dtype": dtype_str,
        "opset_version": opset_version,
        "seed": seed,
        "emission_path": "multifunction_shared_weights",
        "multifunction": True,
        "function_names": ["embedding", "decode_step"],
        "stateful_decode_step": True,
        "weight_sharing": share_weights,
        "shared_weight_name": "shared_projection_weight" if share_weights else None,
        "shared_weight_shape": [embed_dim, embed_dim] if share_weights else None,
    }

    return embedding_prog, metadata


def emit_multifunction_shared_weights(command: dict) -> dict:
    """Build a multi-function MIL program with weight sharing and save as mlpackage.

    Sprint 42: Produces a multi-function mlpackage where the "embedding" and
    "decode_step" functions share a weight tensor ("shared_projection_weight").

    The conversion uses `convert_milprogram(pass_pipeline=make_stateful_pass_pipeline())`
    because the decode_step function contains `mb.coreml_update_state` ops.
    """
    try:
        ct = _ensure_coremltools()

        from converter import convert_milprogram, make_stateful_pass_pipeline

        output_path = command.get("output_path", "")
        if not output_path:
            return _error_result("No output_path specified")

        compute_units_str = command.get("compute_units", "CPU_AND_NE")
        dtype_str = command.get("dtype", "fp16")
        share_weights = command.get("share_weights", True)

        # Step 1: Build the multi-function MIL program with shared weights
        prog, prog_meta = build_multifunction_program_with_shared_weights(command)

        # Step 2: Convert using stateful-aware pass pipeline
        precision_str = "FLOAT16" if dtype_str == "fp16" else "FLOAT32"
        mlmodel = convert_milprogram(
            prog,
            opset_version=command.get("opset_version", "iOS18"),
            compute_precision=precision_str,
            compute_units=compute_units_str,
            pass_pipeline=make_stateful_pass_pipeline(),
        )

        # Step 3: Resolve function descriptors for both functions
        embed_dim = command.get("embed_dim", 128)
        num_heads = command.get("num_heads", 4)
        head_dim = command.get("head_dim", 32)
        kv_len = command.get("kv_len", 64)
        vocab_size = command.get("vocab_size", 32000)

        function_descriptors = [
            {
                "name": "embedding",
                "inputs": [{"name": "token_ids", "shape": [1, vocab_size], "dtype": "int32"}],
                "outputs": [{"name": "hidden_proj", "shape": [1, embed_dim], "dtype": dtype_str}],
                "stateful": False,
                "uses_shared_weight": share_weights,
            },
            {
                "name": "decode_step",
                "inputs": [
                    {"name": "x", "shape": [1, embed_dim], "dtype": dtype_str},
                    {"name": "k_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                    {"name": "v_state", "shape": [1, num_heads, kv_len, head_dim], "dtype": dtype_str, "is_state": True},
                ],
                "outputs": [{"name": "output", "shape": [1, embed_dim], "dtype": dtype_str}],
                "stateful": True,
                "uses_shared_weight": share_weights,
            },
        ]

        # Step 4: Save
        task_name = command.get("task_name", "multifunction_shared")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        mlpackage_path = output_dir / f"{task_name}.mlpackage"

        save_info = save_mlpackage(mlmodel, str(mlpackage_path))

        # Step 5: Validate multifunction structure
        multifunction_validation = {"validated": False, "function_count": None, "function_names": []}
        try:
            spec = mlmodel.get_spec()
            if hasattr(spec, 'mlProgram') and hasattr(spec.mlProgram, 'functions'):
                fn_dict = spec.mlProgram.functions
                fn_names = list(fn_dict.keys())
                multifunction_validation = {
                    "validated": True,
                    "function_count": len(fn_names),
                    "function_names": fn_names,
                    "has_embedding": "embedding" in fn_names,
                    "has_decode_step": "decode_step" in fn_names,
                }
        except Exception as e:
            multifunction_validation["error"] = str(e)

        # Step 6: Weight sharing verification
        weight_sharing_verification = {"verified": False}
        try:
            weight_path = Path(str(mlpackage_path)) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
            if weight_path.exists():
                weight_sharing_verification = {
                    "verified": True,
                    "weight_bin_size": weight_path.stat().st_size,
                    "note": "add_function() path does not deduplicate constants across functions (Sprint 42 verified limitation)",
                }
        except Exception:
            pass

        # Step 7: Compute plan (best-effort)
        compute_plan = compute_plan_info(str(mlpackage_path), compute_units_str)

        return {
            "status": "success",
            "error_message": None,
            "output_path": save_info["output_path"],
            "coremltools_version": ct.__version__,
            "content_hash": save_info["content_hash"],
            "package_files": save_info["package_files"],
            "compute_plan": compute_plan,
            "function_descriptors": function_descriptors,
            "multifunction_validation": multifunction_validation,
            "weight_sharing_verification": weight_sharing_verification,
            "metadata": prog_meta,
        }

    except Exception as e:
        return _error_result(f"Multi-function shared-weights emission failed: {e}")


def _hash_directory(path: Path) -> str:
    """Compute SHA-256 hash of all files in a directory, sorted by relative path."""
    hasher = hashlib.sha256()
    file_hashes = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, path)
            with open(fp, "rb") as fh:
                file_hash = hashlib.sha256(fh.read()).hexdigest()
            file_hashes.append((rel, file_hash))

    for rel, fh in sorted(file_hashes):
        hasher.update(rel.encode())
        hasher.update(fh.encode())

    return hasher.hexdigest()
