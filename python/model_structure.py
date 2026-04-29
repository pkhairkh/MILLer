"""Model Structure — Structural introspection of emitted mlpackages via MLModelStructure.

This module replaces weak host-side package checks (file existence, Manifest.json
readability) with real structural introspection using MLModelStructure.load_from_path().

On Apple hardware with coremltools >= 8.0, this provides:
  - Function names and signatures
  - Per-operation names, inputs, outputs
  - State declaration inventory
  - Weight tensor names and shapes
  - Nested block structure

On non-Apple platforms (Linux), MLModelStructure is unavailable because it
requires the Core ML runtime. The module handles this gracefully and reports
the exact reason, making it possible to distinguish "structurally verified"
from "structurally unverifiable on this platform".

Rust/Python boundary:
  - Python OWNS: MLModelStructure API calls, op graph walking, structural analysis
  - Rust OWNS: deciding what MIR ops were expected, consuming the comparison result

Architecture:
  inspect_model_structure()  — main entry: walks mlpackage structure
  compare_mir_vs_structure() — compares expected MIR ops against actual structure
"""

from typing import Any, Dict, List, Optional


def inspect_model_structure(mlpackage_path: str) -> Dict[str, Any]:
    """Inspect an mlpackage using MLModelStructure for structural verification.

    This is the correct way to verify that an emitted mlpackage contains the
    expected op graph, replacing the previous approach of checking file
    existence and Manifest.json readability.

    On Apple hardware, this uses MLModelStructure.load_from_path() to walk
    the program structure without loading the model into the ML runtime.
    On non-Apple platforms, the API is unavailable but the code path is real.

    Args:
        mlpackage_path: Path to the .mlpackage directory.

    Returns:
        Dict with:
          - available: bool — whether structural inspection succeeded
          - reason: str (if unavailable) — why inspection failed
          - functions: list of function descriptors (if available)
          - operations: list of operation descriptors (if available)
          - state_declarations: list of state descriptors (if available)
          - total_operation_count: int — number of operations found
          - inspection_method: str — "mlmodel_structure" or "fallback_file_check"
    """
    try:
        from coremltools.models.model_structure import MLModelStructure
    except ImportError:
        return _unavailable_result(
            "MLModelStructure not available in this coremltools version "
            "(requires coremltools >= 8.0)"
        )

    try:
        structure = MLModelStructure.load_from_path(str(mlpackage_path))
    except Exception as e:
        error_str = str(e)
        if "libcoremlpython" in error_str or "CoreML" in error_str:
            reason = (
                "Apple Core ML runtime not available on this platform — "
                "MLModelStructure requires macOS with Core ML framework"
            )
        else:
            reason = f"MLModelStructure.load_from_path() failed: {error_str}"
        return _unavailable_result(reason)

    # Walk the program structure
    functions = []
    all_operations = []
    state_declarations = []

    try:
        program = structure.program
        if program is not None and hasattr(program, 'functions'):
            for func_name, func_obj in program.functions.items():
                func_desc = _walk_function(func_name, func_obj)
                functions.append(func_desc)
                all_operations.extend(func_desc.get("operations", []))

                # Extract state declarations from function inputs
                if hasattr(func_obj, 'inputs'):
                    for inp in func_obj.inputs:
                        inp_desc = _describe_value(inp)
                        if inp_desc.get("is_state", False):
                            state_declarations.append(inp_desc)
    except Exception as e:
        # If we can't walk the structure, report what we got
        return {
            "available": True,
            "functions": functions,
            "operations": all_operations,
            "state_declarations": state_declarations,
            "total_operation_count": len(all_operations),
            "inspection_method": "mlmodel_structure_partial",
            "walk_error": str(e),
        }

    return {
        "available": True,
        "functions": functions,
        "operations": all_operations,
        "state_declarations": state_declarations,
        "total_operation_count": len(all_operations),
        "inspection_method": "mlmodel_structure",
    }


def compare_mir_vs_structure(
    mir_ops: List[Dict[str, Any]],
    structure_ops: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare expected MIR op list against emitted model structure.

    This function takes the MIR ops that the compiler intended to emit and
    the actual ops found in the model structure, then computes:
    - which MIR ops have matching structure ops
    - which MIR ops are missing from the structure
    - which structure ops were not expected by the MIR
    - an op fidelity score (0-1)

    The comparison is by op type name. MIR ops use names like "MILLinear",
    "MILGelu", etc., while structure ops use Core ML names like "linear",
    "gelu", etc. This function maps between them.

    Args:
        mir_ops: List of MIR op dicts, each with at least "op_type" key.
        structure_ops: List of structure op dicts, each with at least
            "op_type" key.

    Returns:
        Dict with:
          - op_fidelity_score: float — fraction of MIR ops matched (0-1)
          - matched_ops: list of matched op pairs
          - missing_from_structure: list of MIR ops not found in structure
          - extra_in_structure: list of structure ops not expected by MIR
          - mir_op_count: int
          - structure_op_count: int
    """
    # Mapping from MIR op type names to Core ML MIL op names
    MIR_TO_MIL = {
        "MILConst": "const",
        "MILMatMul": "matmul",
        "MILLinear": "linear",
        "MILConv": "conv",
        "MILAdd": "add",
        "MILMul": "mul",
        "MILSub": "sub",
        "MILAbs": "abs",
        "MILReshape": "reshape",
        "MILTranspose": "transpose",
        "MILSplit": "split",
        "MILConcat": "concat",
        "MILSoftmax": "softmax",
        "MILScaledDotProductAttention": "scaled_dot_product_attention",
        "MILSliceByIndex": "slice_by_index",
        "MILGelu": "gelu",
        "MILReadState": "read_state",
        "MILCoremlUpdateState": "coreml_update_state",
        "MILSliceUpdate": "slice_update",
        "MILWriteState": "write_state",
        "MILStateWrite": "state_write",
        "MILReduceSum": "reduce_sum",
        "MILReduceMean": "reduce_mean",
        "MILRsqrt": "rsqrt",
        "MILRealDiv": "real_div",
        "MILLayerNorm": "layer_norm",
        "MILTopk": "topk",
        "MILGather": "gather",
        "MILCos": "cos",
        "MILSin": "sin",
        "MILCast": "cast",
    }

    # Also accept short names without MIL prefix (e.g., "Linear" → "linear")
    SHORT_TO_MIL = {
        "Const": "const",
        "MatMul": "matmul",
        "Linear": "linear",
        "Conv": "conv",
        "Add": "add",
        "Mul": "mul",
        "Sub": "sub",
        "Abs": "abs",
        "Reshape": "reshape",
        "Transpose": "transpose",
        "Split": "split",
        "Concat": "concat",
        "Softmax": "softmax",
        "ScaledDotProductAttention": "scaled_dot_product_attention",
        "SliceByIndex": "slice_by_index",
        "Gelu": "gelu",
        "ReadState": "read_state",
        "CoremlUpdateState": "coreml_update_state",
        "WriteState": "write_state",
        "SliceUpdate": "slice_update",
        "ReduceSum": "reduce_sum",
        "ReduceMean": "reduce_mean",
        "Rsqrt": "rsqrt",
        "RealDiv": "real_div",
        "LayerNorm": "layer_norm",
        "Topk": "topk",
        "Gather": "gather",
        "Cos": "cos",
        "Sin": "sin",
        "Cast": "cast",
    }

    # Build expected MIL op names from MIR ops
    expected_mil_names = []
    for mir_op in mir_ops:
        op_type = mir_op.get("op_type", "")
        mil_name = MIR_TO_MIL.get(op_type) or SHORT_TO_MIL.get(op_type)
        if mil_name:
            expected_mil_names.append(mil_name)
        else:
            # Unknown MIR op type — try lowercasing as last resort
            lower = op_type.lower()
            if lower in [v for v in MIR_TO_MIL.values()]:
                expected_mil_names.append(lower)
            else:
                # Unknown MIR op type — include as-is for honest reporting
                expected_mil_names.append(op_type)

    # Build actual MIL op names from structure ops
    actual_mil_names = []
    for struct_op in structure_ops:
        op_type = struct_op.get("op_type", "unknown")
        actual_mil_names.append(op_type)

    # Compare: for each expected name, find a matching actual name
    # Use a multiset comparison since ops can repeat
    from collections import Counter
    expected_counts = Counter(expected_mil_names)
    actual_counts = Counter(actual_mil_names)

    matched_ops = []
    missing_from_structure = []
    extra_in_structure = []

    # Find matches
    all_op_names = set(list(expected_counts.keys()) + list(actual_counts.keys()))
    for op_name in sorted(all_op_names):
        expected_n = expected_counts.get(op_name, 0)
        actual_n = actual_counts.get(op_name, 0)
        matched_n = min(expected_n, actual_n)
        if matched_n > 0:
            matched_ops.append({
                "op_type": op_name,
                "expected_count": expected_n,
                "actual_count": actual_n,
                "matched_count": matched_n,
            })
        if expected_n > actual_n:
            missing_from_structure.append({
                "op_type": op_name,
                "expected_count": expected_n,
                "actual_count": actual_n,
                "deficit": expected_n - actual_n,
            })
        if actual_n > expected_n:
            extra_in_structure.append({
                "op_type": op_name,
                "expected_count": expected_n,
                "actual_count": actual_n,
                "surplus": actual_n - expected_n,
            })

    # Op fidelity score: fraction of expected ops that are matched
    total_expected = sum(expected_counts.values())
    total_matched = sum(min(expected_counts[k], actual_counts.get(k, 0)) for k in expected_counts)
    op_fidelity_score = total_matched / total_expected if total_expected > 0 else 0.0

    return {
        "op_fidelity_score": round(op_fidelity_score, 4),
        "matched_ops": matched_ops,
        "missing_from_structure": missing_from_structure,
        "extra_in_structure": extra_in_structure,
        "mir_op_count": total_expected,
        "structure_op_count": sum(actual_counts.values()),
    }


def inspect_model_structure_with_mir_comparison(
    mlpackage_path: str,
    mir_ops: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Combined inspection: structural introspection + MIR comparison.

    This is the primary entry point for Sprint 34 structural verification.
    It performs both the MLModelStructure inspection and the MIR-vs-structure
    comparison in a single call, returning a unified result.

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        mir_ops: Optional list of MIR op dicts for comparison. Each dict
            should have at least an "op_type" key with the MIR op variant
            name (e.g., "MILLinear", "MILGelu").

    Returns:
        Dict with all fields from inspect_model_structure(), plus:
          - mir_comparison: result of compare_mir_vs_structure() if mir_ops
            was provided and structure inspection succeeded, or None
          - mir_comparison_possible: bool — whether comparison was attempted
    """
    structure_result = inspect_model_structure(mlpackage_path)

    mir_comparison = None
    mir_comparison_possible = False

    if mir_ops is not None and structure_result.get("available", False):
        structure_ops = structure_result.get("operations", [])
        mir_comparison = compare_mir_vs_structure(mir_ops, structure_ops)
        mir_comparison_possible = True
    elif mir_ops is not None and not structure_result.get("available", False):
        # MIR comparison not possible because structure inspection unavailable
        mir_comparison_possible = False
    # If mir_ops is None, no comparison requested

    return {
        **structure_result,
        "mir_comparison": mir_comparison,
        "mir_comparison_possible": mir_comparison_possible,
    }


def fallback_file_structure(mlpackage_path: str) -> Dict[str, Any]:
    """Perform a fallback structural check when MLModelStructure is unavailable.

    This walks the mlpackage directory and extracts whatever structural
    information can be obtained from files alone (without the Core ML runtime).
    It explicitly labels itself as a fallback method.

    This is NOT a replacement for MLModelStructure — it provides weaker
    verification. The purpose is to give the host inspector something
    structural to report on platforms where MLModelStructure is unavailable.

    Args:
        mlpackage_path: Path to the .mlpackage directory.

    Returns:
        Dict with:
          - available: bool — always True (file checks always work)
          - inspection_method: str — always "fallback_file_check"
          - file_inventory: list of {path, size_bytes}
          - has_manifest: bool
          - has_model_spec: bool
          - has_weights: bool
          - weight_file_count: int
          - estimated_weight_bytes: int
    """
    from pathlib import Path
    import os

    pkg_path = Path(mlpackage_path)
    if not pkg_path.exists() or not pkg_path.is_dir():
        return {
            "available": False,
            "inspection_method": "fallback_file_check",
            "reason": "mlpackage directory does not exist",
            "file_inventory": [],
            "has_manifest": False,
            "has_model_spec": False,
            "has_weights": False,
            "weight_file_count": 0,
            "estimated_weight_bytes": 0,
        }

    # File inventory
    file_inventory = []
    for root, dirs, files in os.walk(pkg_path):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, pkg_path)
            sz = os.path.getsize(fp)
            file_inventory.append({"path": rel, "size_bytes": sz})

    # Check for key structural files
    manifest_path = pkg_path / "Manifest.json"
    model_spec_path = pkg_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
    weights_dir = pkg_path / "Data" / "com.apple.CoreML" / "weights"

    has_manifest = manifest_path.exists()
    has_model_spec = model_spec_path.exists()
    has_weights = weights_dir.exists() and any(weights_dir.iterdir()) if weights_dir.exists() else False

    weight_file_count = 0
    estimated_weight_bytes = 0
    if weights_dir.exists():
        for f in weights_dir.iterdir():
            if f.is_file():
                weight_file_count += 1
                estimated_weight_bytes += f.stat().st_size

    # Attempt to parse model.mlmodel for op names (protobuf text format)
    # This works without the Core ML runtime since it's just text parsing
    op_names_from_spec = []
    if has_model_spec:
        try:
            with open(model_spec_path, 'rb') as f:
                content = f.read()
            # The model.mlmodel file is compiled protobuf, but we can try
            # to extract op type strings from it by scanning for known patterns.
            # This is a best-effort heuristic, not a reliable method.
            known_ops = [
                b"linear", b"matmul", b"conv", b"gelu", b"relu",
                b"softmax", b"reshape", b"transpose", b"concat",
                b"split", b"gather", b"const", b"add", b"mul",
                b"sub", b"abs", b"cast", b"reduce_sum", b"reduce_mean",
                b"rsqrt", b"real_div", b"layer_norm", b"topk",
                b"cos", b"sin", b"slice_by_index",
                b"scaled_dot_product_attention",
                b"read_state", b"coreml_update_state",
            ]
            for op_name in known_ops:
                if op_name in content:
                    op_names_from_spec.append(op_name.decode('ascii'))
        except Exception:
            pass

    return {
        "available": True,
        "inspection_method": "fallback_file_check",
        "file_inventory": file_inventory,
        "has_manifest": has_manifest,
        "has_model_spec": has_model_spec,
        "has_weights": has_weights,
        "weight_file_count": weight_file_count,
        "estimated_weight_bytes": estimated_weight_bytes,
        "op_names_heuristic": op_names_from_spec,
        "heuristic_note": (
            "Op names extracted by byte-scanning compiled protobuf — "
            "this is NOT reliable structural verification. Use MLModelStructure "
            "on Apple hardware for authoritative op inventory."
        ),
    }


# --- Internal helpers ---

def _walk_function(func_name: str, func_obj: Any) -> Dict[str, Any]:
    """Walk a single function from MLModelStructure and extract its structure.

    Args:
        func_name: Name of the function (e.g., "main").
        func_obj: The Function object from MLModelStructure.

    Returns:
        Dict with function name, input/output specs, and operations list.
    """
    operations = []
    input_specs = []
    output_specs = []

    # Extract input specifications
    if hasattr(func_obj, 'inputs'):
        for inp in func_obj.inputs:
            input_specs.append(_describe_value(inp))

    # Extract output specifications
    if hasattr(func_obj, 'outputs'):
        for outp in func_obj.outputs:
            output_specs.append(_describe_value(outp))

    # Walk operations in the function's blocks
    if hasattr(func_obj, 'block') and func_obj.block is not None:
        _walk_block(func_obj.block, operations)
    elif hasattr(func_obj, 'operations'):
        # Some versions expose operations directly
        for op in func_obj.operations:
            operations.append(_describe_operation(op))

    return {
        "name": func_name,
        "inputs": input_specs,
        "outputs": output_specs,
        "operations": operations,
        "operation_count": len(operations),
    }


def _walk_block(block: Any, operations: list) -> None:
    """Recursively walk a block's operations, including nested blocks.

    Args:
        block: A Block object from MLModelStructure.
        operations: List to append operation descriptors to.
    """
    if hasattr(block, 'operations'):
        for op in block.operations:
            op_desc = _describe_operation(op)
            operations.append(op_desc)

            # Recurse into nested blocks (e.g., if/else branches)
            if hasattr(op, 'blocks') and op.blocks is not None:
                for nested_block in op.blocks:
                    _walk_block(nested_block, operations)


def _describe_operation(op: Any) -> Dict[str, Any]:
    """Describe a single operation from MLModelStructure.

    Args:
        op: An operation object from MLModelStructure.

    Returns:
        Dict with op name, type, inputs, outputs.
    """
    desc = {
        "name": getattr(op, 'name', 'unknown'),
        "op_type": getattr(op, 'op_type', getattr(op, 'type', 'unknown')),
    }

    # Extract input names
    if hasattr(op, 'inputs'):
        desc["inputs"] = [
            inp.name if hasattr(inp, 'name') else str(inp)
            for inp in op.inputs
        ]

    # Extract output names
    if hasattr(op, 'outputs'):
        desc["outputs"] = [
            out.name if hasattr(out, 'name') else str(out)
            for out in op.outputs
        ]

    # Extract attributes (e.g., axis, keep_dims, mode)
    if hasattr(op, 'attributes'):
        desc["attributes"] = {
            k: str(v) for k, v in op.attributes.items()
        }

    return desc


def _describe_value(value: Any) -> Dict[str, Any]:
    """Describe a value (input/output/state) from MLModelStructure.

    Args:
        value: A value descriptor from MLModelStructure.

    Returns:
        Dict with name, shape, dtype, and state flag.
    """
    desc = {
        "name": getattr(value, 'name', 'unknown'),
        "is_state": False,
    }

    # Extract shape information
    if hasattr(value, 'shape'):
        shape = value.shape
        if hasattr(shape, 'shape'):
            desc["shape"] = list(shape.shape)
        elif isinstance(shape, (list, tuple)):
            desc["shape"] = list(shape)
        else:
            desc["shape"] = str(shape)
    elif hasattr(value, 'type') and hasattr(value.type, 'shape'):
        shape = value.type.shape
        if hasattr(shape, 'shape'):
            desc["shape"] = list(shape.shape)
        elif isinstance(shape, (list, tuple)):
            desc["shape"] = list(shape)

    # Extract dtype
    if hasattr(value, 'dtype'):
        desc["dtype"] = str(value.dtype)
    elif hasattr(value, 'type'):
        desc["dtype"] = str(value.type)

    # Detect state types
    type_str = str(getattr(value, 'type', '')).lower()
    if 'state' in type_str or 'statetype' in type_str:
        desc["is_state"] = True

    return desc


def _unavailable_result(reason: str) -> Dict[str, Any]:
    """Create a standardized unavailable result dict."""
    return {
        "available": False,
        "reason": reason,
        "functions": [],
        "operations": [],
        "state_declarations": [],
        "total_operation_count": 0,
        "inspection_method": "unavailable",
    }
