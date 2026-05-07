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

from typing import Any, Dict, Optional

from common import _ensure_coremltools


def convert_milprogram(
    program,
    opset_version: str = "iOS18",
    compute_precision: str = "FLOAT16",
    compute_units: str = "CPU_AND_NE",
    optimization_hints: Optional[Dict] = None,
    inputs: Optional[list] = None,
    pass_pipeline: Optional[Any] = None,
) -> Any:
    """Convert a MIL program to an MLModel mlprogram.

    Args:
        program: The MIL Program object.
        opset_version: Target opset (e.g., "iOS18").
        compute_precision: "FLOAT16" or "FLOAT32".
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        optimization_hints: Dict of optimization hints.
        inputs: List of ct.TensorType/ct.StateType input specs.
        pass_pipeline: Optional ct.PassPipeline to use instead of the default.
            When provided, this pipeline is passed to ct.convert() via the
            pass_pipeline keyword argument. This replaces the former
            convert_stateful_milprogram() function (W-25 fix): callers that
            need the stateful-aware pipeline (which removes
            canonicalize_inplace_pattern) should pass the result of
            make_stateful_pass_pipeline() here.

    Returns:
        An MLModel object.

    macOS SIGABRT workaround:
        ct.convert() internally creates an MLModel from the converted spec
        for validation. On macOS, the MLModel constructor triggers C++ model
        compilation which can SIGABRT with "coremldata.bin is not a valid
        .mlmodelc file". We monkey-patch MLModel.__init__ to inject
        skip_model_load=True during ct.convert() to prevent this. The 3 MIL
        pipeline stages still validate the model structurally; we only skip
        the C++ runtime compilation step. The returned MLModel has a valid
        spec (accessible via .get_spec()) but predict() won't work — which
        is fine since the bridge only needs to save the spec.
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

    kwargs = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": target_map.get(opset_version, ct.target.iOS18),
        "compute_precision": precision_map.get(compute_precision, ct.precision.FLOAT16),
        "debug": True,
    }

    if pass_pipeline is not None:
        kwargs["pass_pipeline"] = pass_pipeline

    if inputs is not None:
        kwargs["inputs"] = inputs

    if optimization_hints:
        kwargs["optimization_hints"] = optimization_hints

    # Monkey-patch MLModel.__init__ to use skip_model_load=True during
    # ct.convert(). This prevents the macOS C++ model compilation step
    # that causes SIGABRT ("coremldata.bin is not a valid .mlmodelc file").
    # ct.convert() internally creates an MLModel from the spec for
    # validation; on macOS this triggers compilation. With skip_model_load,
    # the MLModel is created without C++ compilation — the spec is still
    # valid and can be saved via save_mlpackage(). The try/finally ensures
    # the original __init__ is always restored, even if ct.convert() raises.
    _original_init = ct.models.MLModel.__init__

    def _patched_init(self, *args, **kwargs_inner):
        if "skip_model_load" not in kwargs_inner:
            kwargs_inner["skip_model_load"] = True
        return _original_init(self, *args, **kwargs_inner)

    ct.models.MLModel.__init__ = _patched_init
    try:
        mlmodel = ct.convert(program, **kwargs)
    finally:
        ct.models.MLModel.__init__ = _original_init

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


def make_stateful_pass_pipeline():
    """Build a ct.PassPipeline suitable for stateful MIL programs.

    Removes the `common::canonicalize_inplace_pattern` pass from the default
    pipeline, which does not handle `coreml_update_state` ops correctly in
    coremltools 9.0. The pass attempts to rewrite inplace patterns but fails
    on state update ops, producing a spurious error. Removing it is safe
    because:
    (1) the pass is a canonicalization optimization, not a correctness
        requirement;
    (2) stateful programs use coreml_update_state for its side effects,
        not as an inplace mutation pattern;
    (3) all other passes in the default pipeline remain active.

    This replaces the former convert_stateful_milprogram() function (W-25).

    Returns:
        A ct.PassPipeline with canonicalize_inplace_pattern removed.
    """
    ct = _ensure_coremltools()
    pipeline = ct.PassPipeline.DEFAULT
    # Find and remove the problematic pass by index
    inplace_idx = None
    for i, name in enumerate(pipeline.passes):
        if name == "common::canonicalize_inplace_pattern":
            inplace_idx = i
            break
    if inplace_idx is not None:
        pipeline.remove_pass(inplace_idx)
    return pipeline
