#!/usr/bin/env python3
"""Smoke test: verify all Python MIL emitters produce valid mlpackages.

This script exercises every emitter path in MILLer and reports
success/failure with content hashes. It runs on any platform with coremltools
installed — no Apple hardware required for construction and conversion.

Usage:
    cd /path/to/MILLer
    PYTHONPATH=python python3 scripts/smoke_test_emitters.py

Exit code 0 if all emitters pass, 1 otherwise.
"""

import os
import sys
import shutil
import tempfile

# Ensure the python/ directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from mil_emitter import (
    emit_linear_projection,
    emit_decode_step,
    emit_stateful_decode_step,
    emit_stateless_decode_step,
    emit_attention,
    emit_mlp_block,
    emit_lut_projection,
    emit_multifunction,
    validate_multifunction_package,
)


def test_emitter(name: str, fn, cmd: dict) -> bool:
    """Run an emitter function and report result. Returns True on success."""
    result = fn(cmd)
    status = result.get("status")
    content_hash = result.get("content_hash", "N/A")
    short_hash = content_hash[:16] if content_hash and content_hash != "N/A" else "N/A"
    error = result.get("error_message", "")

    if status == "success":
        print(f"  PASS  {name}: hash={short_hash}")
        return True
    else:
        print(f"  FAIL  {name}: {error}")
        return False


def main():
    tmpdir = tempfile.mkdtemp(prefix="ane_smoke_")
    all_pass = True

    print(f"MILLer — Emitter Smoke Test")
    print(f"Output directory: {tmpdir}")
    print()

    # 1. Linear projection
    all_pass &= test_emitter("Linear Projection", emit_linear_projection, {
        "task_name": "smoke_linear", "input_dim": 64, "output_dim": 32,
        "batch_size": 1, "dtype": "fp16", "opset_version": "iOS18",
        "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "linear"), "seed": 42,
    })

    # 2. Attention
    all_pass &= test_emitter("Attention", emit_attention, {
        "task_name": "smoke_attn", "embed_dim": 128, "num_heads": 4,
        "head_dim": 32, "seq_len": 32, "batch_size": 1, "dtype": "fp16",
        "opset_version": "iOS18", "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "attn"), "seed": 42,
    })

    # 3. Stateful decode-step (default path, Sprint 40)
    all_pass &= test_emitter("Stateful Decode Step", emit_stateful_decode_step, {
        "task_name": "smoke_stateful", "embed_dim": 128, "num_heads": 4,
        "head_dim": 32, "kv_len": 64, "batch_size": 1, "dtype": "fp16",
        "opset_version": "iOS18", "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "stateful"), "seed": 42,
    })

    # 4. Stateless decode-step (testing path)
    all_pass &= test_emitter("Stateless Decode Step", emit_stateless_decode_step, {
        "task_name": "smoke_stateless", "embed_dim": 128, "num_heads": 4,
        "head_dim": 32, "kv_len": 64, "batch_size": 1, "dtype": "fp16",
        "opset_version": "iOS18", "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "stateless"), "seed": 42,
    })

    # 5. MLP block
    all_pass &= test_emitter("MLP Block", emit_mlp_block, {
        "task_name": "smoke_mlp", "input_dim": 128, "hidden_dim": 512,
        "output_dim": 128, "activation": "gelu", "batch_size": 1,
        "dtype": "fp16", "opset_version": "iOS18",
        "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "mlp"), "seed": 42,
    })

    # 6. LUT projection
    all_pass &= test_emitter("LUT Projection", emit_lut_projection, {
        "task_name": "smoke_lut", "vocab_size": 32000, "embed_dim": 512,
        "num_groups": 64, "lut_bitwidth": 4, "batch_size": 1,
        "dtype": "fp16", "opset_version": "iOS18",
        "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "lut"), "seed": 42,
    })

    # 7. Multi-function package
    all_pass &= test_emitter("Multi-Function", emit_multifunction, {
        "task_name": "smoke_mfn", "embed_dim": 128, "num_heads": 4,
        "head_dim": 32, "kv_len": 64, "batch_size": 1, "dtype": "fp16",
        "opset_version": "iOS18", "compute_units": "CPU_AND_NE",
        "output_path": os.path.join(tmpdir, "mfn"), "seed": 42,
    })

    # 8. Validate the multi-function package
    mfn_path = os.path.join(tmpdir, "mfn", "smoke_mfn.mlpackage")
    if os.path.exists(mfn_path):
        mfn_result = validate_multifunction_package(mfn_path)
        valid = mfn_result.get("valid", False)
        fn_names = mfn_result.get("function_names", [])
        fn_count = mfn_result.get("function_count", 0)
        if valid and "embedding" in fn_names and "decode_step" in fn_names:
            print(f"  PASS  Multi-Function Validation: {fn_count} functions {fn_names}")
        else:
            print(f"  FAIL  Multi-Function Validation: valid={valid} functions={fn_names}")
            all_pass = False
    else:
        print(f"  SKIP  Multi-Function Validation: mlpackage not found")
        all_pass = False

    print()
    if all_pass:
        print("ALL EMITTERS PASS")
    else:
        print("SOME EMITTERS FAILED")

    # Clean up
    shutil.rmtree(tmpdir, ignore_errors=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
