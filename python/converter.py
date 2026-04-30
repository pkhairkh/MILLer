"""Core ML Converter - Handles ct.convert() invocation.

This module encapsulates the ct.convert() step for MIL program → MLModel
conversion. It is the single place where ct.convert() is called.

Rust/Python boundary:
  - Python OWNS: ct.convert() invocation, coremltools API calls, mlprogram
    conversion settings (opset, compute_precision, compute_units).
  - Rust OWNS: deciding what to compile, providing the MIR that describes
    the MIL graph, orchestrating the compile pipeline, consuming results.

Multifunction support (Sprint 39):
  - convert_multifunction_milprogram() converts multi-function MIL programs
    that were built using mb.program(function_name=...) and prog.add_function().
  - Multi-function programs carry their function structure through ct.convert()
    naturally; the converter ensures default_function_name is set correctly.
"""

from typing import Dict, Any, Optional, List

# Lazy import to avoid ImportError on non-macOS systems
ct = None

def _ensure_coremltools():
    global ct
    if ct is None:
        import coremltools
        ct = coremltools
    return ct


def convert_milprogram(
    program,
    opset_version: str = "iOS18",
    compute_precision: str = "FLOAT16",
    compute_units: str = "CPU_AND_NE",
    optimization_hints: Optional[Dict] = None,
    inputs: Optional[list] = None,
) -> Any:
    """Convert a MIL program to an MLModel mlprogram.
    
    Args:
        program: The MIL Program object.
        opset_version: Target opset (e.g., "iOS18").
        compute_precision: "FLOAT16" or "FLOAT32".
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        optimization_hints: Dict of optimization hints.
        inputs: List of ct.TensorType/ct.StateType input specs.
    
    Returns:
        An MLModel object.
    """
    ct = _ensure_coremltools()
    target_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    precision_map = {
        "FLOAT16": ct.precision.FLOAT16,
        "FLOAT32": ct.precision.FLOAT32,
    }
    compute_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    
    kwargs = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": target_map.get(opset_version, ct.target.iOS18),
        "compute_precision": precision_map.get(compute_precision, ct.precision.FLOAT16),
        "debug": True,
    }
    
    if inputs is not None:
        kwargs["inputs"] = inputs
    
    if optimization_hints:
        kwargs["optimization_hints"] = optimization_hints
    
    mlmodel = ct.convert(program, **kwargs)
    return mlmodel


def convert_multifunction_milprogram(
    program,
    opset_version: str = "iOS18",
    compute_precision: str = "FLOAT16",
    compute_units: str = "CPU_AND_NE",
    default_function_name: Optional[str] = None,
    optimization_hints: Optional[Dict] = None,
    inputs: Optional[list] = None,
) -> Any:
    """Convert a multi-function MIL program to an MLModel mlprogram.

    This is the same as convert_milprogram but additionally supports
    setting the default_function_name on the converted model. Multi-function
    programs created via mb.program(function_name=...) and prog.add_function()
    carry their function structure through ct.convert(); this function
    ensures the default_function_name is set correctly after conversion.

    Args:
        program: The multi-function MIL Program object.
        opset_version: Target opset (e.g., "iOS18").
        compute_precision: "FLOAT16" or "FLOAT32".
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        default_function_name: Name of the default function (e.g., "embedding").
            If None, the program's own default_function_name is used.
        optimization_hints: Dict of optimization hints.
        inputs: List of ct.TensorType/ct.StateType input specs.

    Returns:
        An MLModel object with multiple functions.
    """
    # Use the program's default_function_name if not explicitly overridden
    if default_function_name is None and hasattr(program, 'default_function_name'):
        default_function_name = program.default_function_name

    # Convert using the same path as single-function programs
    mlmodel = convert_milprogram(
        program,
        opset_version=opset_version,
        compute_precision=compute_precision,
        compute_units=compute_units,
        optimization_hints=optimization_hints,
        inputs=inputs,
    )

    if default_function_name:
        mlmodel.spec.defaultFunctionName = default_function_name

    return mlmodel


def convert_stateful_milprogram(
    program,
    opset_version: str = "iOS18",
    compute_precision: str = "FLOAT16",
    compute_units: str = "CPU_AND_NE",
    optimization_hints: Optional[Dict] = None,
    inputs: Optional[list] = None,
) -> Any:
    """Convert a stateful MIL program to an MLModel mlprogram.

    This is the same as convert_milprogram but removes the
    `common::canonicalize_inplace_pattern` pass from the default
    pipeline, which does not handle `coreml_update_state` ops
    correctly in coremltools 9.0. The pass attempts to rewrite
    inplace patterns but fails on state update ops, producing
    a spurious error. Removing it is safe because:
    (1) the pass is a canonicalization optimization, not a
        correctness requirement;
    (2) stateful programs use coreml_update_state for its side
        effects, not as an inplace mutation pattern;
    (3) all other passes in the default pipeline remain active.

    Args:
        program: The MIL Program object (containing state ops).
        opset_version: Target opset (e.g., "iOS18").
        compute_precision: "FLOAT16" or "FLOAT32".
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        optimization_hints: Dict of optimization hints.
        inputs: List of ct.TensorType/ct.StateType input specs.

    Returns:
        An MLModel object with state declarations.
    """
    ct = _ensure_coremltools()
    target_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    precision_map = {
        "FLOAT16": ct.precision.FLOAT16,
        "FLOAT32": ct.precision.FLOAT32,
    }

    # Build a custom pass pipeline that removes the problematic
    # canonicalize_inplace_pattern pass. This pass doesn't handle
    # coreml_update_state correctly, causing conversion to fail.
    pipeline = ct.PassPipeline.DEFAULT
    # Find and remove the problematic pass by index
    inplace_idx = None
    for i, name in enumerate(pipeline.passes):
        if name == "common::canonicalize_inplace_pattern":
            inplace_idx = i
            break
    if inplace_idx is not None:
        pipeline.remove_pass(inplace_idx)

    kwargs = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": target_map.get(opset_version, ct.target.iOS18),
        "compute_precision": precision_map.get(compute_precision, ct.precision.FLOAT16),
        "debug": True,
        "pass_pipeline": pipeline,
    }

    if inputs is not None:
        kwargs["inputs"] = inputs

    if optimization_hints:
        kwargs["optimization_hints"] = optimization_hints

    mlmodel = ct.convert(program, **kwargs)
    return mlmodel
