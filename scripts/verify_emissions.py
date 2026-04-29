#!/usr/bin/env python3
"""Verification harness: test all emission paths in the MILLer project.

For each emission path this script:
  1. Calls the build_* function to construct the MIL Program
  2. Calls the appropriate converter function to produce an MLModel
  3. Inspects the resulting spec for function count and op count
  4. Reports success/failure with details

No mlpackages are written to disk — this is a pure build+convert+inspect check.

Usage:
    cd /path/to/MILLer
    python3 scripts/verify_emissions.py

Exit code 0 if all paths pass, 1 otherwise.
"""

import os
import sys
import traceback

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_ops_in_function(func):
    """Count the number of operations in a single ML Program function.

    The protobuf structure uses func.block_specializations, a dict-like
    mapping from opset name (e.g. "CoreML8") to a Block that contains
    .operations.
    """
    op_count = 0
    op_types = []
    for block_name in func.block_specializations:
        block = func.block_specializations[block_name]
        ops = block.operations
        op_count += len(ops)
        for op in ops:
            op_types.append(op.type)
    return op_count, op_types


def inspect_model(mlmodel, label):
    """Inspect a converted MLModel and return a dict of structural info."""
    info = {"function_count": 0, "functions": {}}

    try:
        spec = mlmodel.get_spec()
    except Exception as e:
        info["spec_error"] = str(e)
        return info

    # mlprogram functions live in spec.mlProgram.functions (dict-like)
    try:
        ml_prog = spec.mlProgram
        fn_dict = {}
        # Iterate over the functions
        for fn_name in ml_prog.functions:
            fn_dict[fn_name] = ml_prog.functions[fn_name]

        info["function_count"] = len(fn_dict)
        for fn_name, fn_proto in fn_dict.items():
            op_count, op_types = count_ops_in_function(fn_proto)
            info["functions"][fn_name] = {"op_count": op_count, "op_types": op_types}
    except Exception as e:
        info["functions_error"] = str(e)

    return info


def run_test(name, build_fn, build_kwargs, convert_fn, convert_kwargs):
    """Run a single emission-path test: build → convert → inspect.

    Returns (success: bool, detail: str).
    """
    try:
        # Step 1: Build the MIL program
        prog, metadata = build_fn(build_kwargs)

        # Step 2: Convert
        mlmodel = convert_fn(prog, **convert_kwargs)

        # Step 3: Inspect the spec
        info = inspect_model(mlmodel, name)

        # Format a nice detail string
        fn_count = info.get("function_count", "?")
        funcs = info.get("functions", {})
        if funcs:
            fn_details = ", ".join(
                f"{fn}({d['op_count']} ops: {', '.join(d['op_types'])})"
                for fn, d in funcs.items()
            )
        else:
            fn_details = "N/A"

        spec_err = info.get("spec_error") or info.get("functions_error")
        if spec_err:
            detail = f"fn_count={fn_count} ({fn_details}) [spec_warning: {spec_err}]"
        else:
            detail = f"fn_count={fn_count} ({fn_details})"

        return True, detail

    except Exception as exc:
        tb_lines = traceback.format_exc().splitlines()
        # Keep last 3 lines for brevity
        short_tb = " | ".join(tb_lines[-3:])
        return False, f"{type(exc).__name__}: {exc}  [{short_tb}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("MILLer — Emission Path Verification Harness")
    print("=" * 72)
    print()

    # ── Prerequisite: coremltools availability ──────────────────────────
    try:
        import coremltools as ct
        print(f"coremltools version: {ct.__version__}")
    except ImportError:
        print("FATAL: coremltools is not installed. Cannot run verification.")
        return 1

    # Also verify numpy
    try:
        import numpy as np
        print(f"numpy version: {np.__version__}")
    except ImportError:
        print("FATAL: numpy is not installed. Cannot run verification.")
        return 1

    # Import build functions and converters
    from mil_emitter import (
        build_linear_projection_program,
        build_lut_projection_program,
        build_decode_step_program,
        build_stateful_decode_step_program,
        build_mlp_block_program,
        build_attention_program,
        build_multifunction_program,
        build_shard_decode_step_program,
    )
    from converter import (
        convert_milprogram,
        convert_stateful_milprogram,
    )

    print()
    print("-" * 72)

    # ── Test definitions ────────────────────────────────────────────────
    # Each entry: (display_name, build_fn, build_kwargs, convert_fn, convert_kwargs)
    tests = []

    # 1. Linear Projection
    tests.append((
        "linear_projection → convert_milprogram",
        build_linear_projection_program,
        {"task_name": "verify_linear", "input_dim": 64, "output_dim": 32,
         "batch_size": 1, "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 2. LUT Projection
    tests.append((
        "lut_projection → convert_milprogram",
        build_lut_projection_program,
        {"task_name": "verify_lut", "vocab_size": 32000, "embed_dim": 512,
         "num_groups": 64, "lut_bitwidth": 4, "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 3. Decode Step (stateless)
    tests.append((
        "decode_step (stateless) → convert_milprogram",
        build_decode_step_program,
        {"task_name": "verify_decode", "embed_dim": 128, "num_heads": 4,
         "head_dim": 32, "kv_len": 64, "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 4. Stateful Decode Step
    tests.append((
        "stateful_decode_step → convert_stateful_milprogram",
        build_stateful_decode_step_program,
        {"task_name": "verify_stateful", "embed_dim": 128, "num_heads": 4,
         "head_dim": 32, "kv_len": 64, "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_stateful_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 5. MLP Block
    tests.append((
        "mlp_block → convert_milprogram",
        build_mlp_block_program,
        {"task_name": "verify_mlp", "input_dim": 128, "hidden_dim": 512,
         "output_dim": 128, "activation": "gelu", "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 6. Attention
    tests.append((
        "attention → convert_milprogram",
        build_attention_program,
        {"task_name": "verify_attn", "embed_dim": 128, "num_heads": 4,
         "head_dim": 32, "seq_len": 32, "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 7. Multi-function
    tests.append((
        "multifunction → convert_stateful_milprogram",
        build_multifunction_program,
        {"task_name": "verify_mfn", "embed_dim": 128, "num_heads": 4,
         "head_dim": 32, "kv_len": 64, "batch_size": 1,
         "dtype": "fp16", "opset_version": "iOS18", "seed": 42},
        convert_stateful_milprogram,
        {"opset_version": "iOS18", "compute_precision": "FLOAT16",
         "compute_units": "CPU_AND_NE"},
    ))

    # 8a-c. Shard Decode Step (Entry, Interior, Exit)
    for role in ("Entry", "Interior", "Exit"):
        tests.append((
            f"shard_decode_step ({role}) → convert_stateful_milprogram",
            build_shard_decode_step_program,
            {"task_name": f"verify_shard_{role.lower()}",
             "embed_dim": 128, "num_heads": 4, "head_dim": 32,
             "kv_len": 64, "batch_size": 1, "dtype": "fp16",
             "opset_version": "iOS18", "seed": 42,
             "shard_role": role},
            convert_stateful_milprogram,
            {"opset_version": "iOS18", "compute_precision": "FLOAT16",
             "compute_units": "CPU_AND_NE"},
        ))

    # ── Run tests ───────────────────────────────────────────────────────
    results = []
    for name, build_fn, build_kw, convert_fn, convert_kw in tests:
        success, detail = run_test(name, build_fn, build_kw, convert_fn, convert_kw)
        status_str = "PASS" if success else "FAIL"
        print(f"  {status_str}  {name}")
        print(f"        {detail}")
        results.append((name, success, detail))

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    pass_count = sum(1 for _, s, _ in results if s)
    fail_count = len(results) - pass_count
    total = len(results)
    print(f"Summary: {pass_count}/{total} passed, {fail_count}/{total} failed")

    if fail_count > 0:
        print()
        print("Failed paths:")
        for name, success, detail in results:
            if not success:
                print(f"  ✗ {name}: {detail}")

    print()
    if fail_count == 0:
        print("ALL EMISSION PATHS VERIFIED SUCCESSFULLY")
        return 0
    else:
        print("SOME EMISSION PATHS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
