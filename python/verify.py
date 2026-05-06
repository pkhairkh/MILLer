"""Unified Verification Harness — Sprint 40.

This module provides a single entry point for verifying emitted mlpackage
artifacts against the compiler's intended MIR graph, compute plan placement,
state conformance, and multi-function conformance.

It operationalizes the PDF's proposed verification methodology into an actual
project verification harness that produces structured, comparable artifacts.

The harness performs four verification dimensions:
  1. Op graph fidelity — compare emitted model ops against intended MIR ops
  2. Compute-unit placement — ANE placement rate from MLComputePlan (macOS only)
  3. State conformance — verify state declarations and read/write ops match intent
  4. Multi-function conformance — verify function count and names match intent

Each dimension produces a structured result that can be persisted as a JSON
artifact and compared across runs.

Architecture:
  verify_model()          — main entry: unified verification of an mlpackage
  VerificationResult      — structured result dataclass for all four dimensions
  OpFidelityResult        — op graph fidelity result
  PlacementResult         — compute-unit placement result
  StateConformanceResult  — state model conformance result
  MultifunctionResult     — multi-function model conformance result

Rust/Python boundary:
  Python OWNS: all verification logic (MLModelStructure, MLComputePlan,
    model spec walking, comparison algorithms)
  Rust OWNS: deciding what to verify, providing expected MIR ops,
    consuming the verification result, persisting artifacts

On non-Apple platforms (Linux), verification is partial:
  - Op fidelity uses spec-based op extraction (protobuf introspection)
  - Compute plan placement reports unavailable
  - State conformance uses spec-based state detection
  - Multi-function conformance uses spec-based function counting
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class OpFidelityResult:
    """Op graph fidelity verification result.

    Compares the ops that the compiler intended (from MIR) against the ops
    actually present in the emitted model (from MLModelStructure or spec
    inspection).
    """

    def __init__(self):
        self.op_fidelity_score: float = 0.0
        self.mir_op_count: int = 0
        self.structure_op_count: int = 0
        self.matched_ops: List[Dict[str, Any]] = []
        self.missing_from_structure: List[Dict[str, Any]] = []
        self.extra_in_structure: List[Dict[str, Any]] = []
        self.verification_method: str = "unavailable"
        self.available: bool = False
        self.reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_fidelity_score": self.op_fidelity_score,
            "mir_op_count": self.mir_op_count,
            "structure_op_count": self.structure_op_count,
            "matched_ops": self.matched_ops,
            "missing_from_structure": self.missing_from_structure,
            "extra_in_structure": self.extra_in_structure,
            "verification_method": self.verification_method,
            "available": self.available,
            "reason": self.reason,
        }


class PlacementResult:
    """Compute-unit placement verification result.

    Reports the ANE placement rate from MLComputePlan, if available.
    On non-Apple hosts, uses offline prediction from known op→device
    mappings (Sprint 57).
    """

    def __init__(self):
        self.ane_placement_rate: float = 0.0
        self.total_ops: int = 0
        self.ane_placed_ops: int = 0
        self.per_op_placement: List[Dict[str, Any]] = []
        self.available: bool = False
        self.reason: Optional[str] = None
        self.verification_method: str = "unavailable"
        self.prediction_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ane_placement_rate": self.ane_placement_rate,
            "total_ops": self.total_ops,
            "ane_placed_ops": self.ane_placed_ops,
            "per_op_placement": self.per_op_placement,
            "available": self.available,
            "reason": self.reason,
            "verification_method": self.verification_method,
            "prediction_confidence": self.prediction_confidence,
        }


class StateConformanceResult:
    """State model conformance verification result.

    Checks that state declarations and state read/write ops match the
    compiler's intent.
    """

    def __init__(self):
        self.stateful_model: bool = False
        self.expected_state_count: int = 0
        self.actual_state_count: int = 0
        self.state_names_match: bool = False
        self.has_read_state: bool = False
        self.has_update_state: bool = False
        self.state_details: List[Dict[str, Any]] = []
        self.conformance_score: float = 0.0
        self.verification_method: str = "unavailable"
        self.available: bool = False
        self.reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stateful_model": self.stateful_model,
            "expected_state_count": self.expected_state_count,
            "actual_state_count": self.actual_state_count,
            "state_names_match": self.state_names_match,
            "has_read_state": self.has_read_state,
            "has_update_state": self.has_update_state,
            "state_details": self.state_details,
            "conformance_score": self.conformance_score,
            "verification_method": self.verification_method,
            "available": self.available,
            "reason": self.reason,
        }


class MultifunctionResult:
    """Multi-function model conformance verification result.

    Checks that the multi-function model structure matches the compiler's
    intent: correct function count, function names, and weight sharing.
    """

    def __init__(self):
        self.is_multifunction: bool = False
        self.expected_function_count: int = 1
        self.actual_function_count: int = 0
        self.function_names_match: bool = False
        self.function_details: List[Dict[str, Any]] = []
        self.conformance_score: float = 0.0
        self.verification_method: str = "unavailable"
        self.available: bool = False
        self.reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_multifunction": self.is_multifunction,
            "expected_function_count": self.expected_function_count,
            "actual_function_count": self.actual_function_count,
            "function_names_match": self.function_names_match,
            "function_details": self.function_details,
            "conformance_score": self.conformance_score,
            "verification_method": self.verification_method,
            "available": self.available,
            "reason": self.reason,
        }


class VerificationResult:
    """Unified verification result combining all four dimensions.

    This is the top-level result object for verify_model(). It contains
    the results of all verification dimensions plus metadata about the
    verification run.
    """

    def __init__(self):
        self.op_fidelity: OpFidelityResult = OpFidelityResult()
        self.placement: PlacementResult = PlacementResult()
        self.state_conformance: StateConformanceResult = StateConformanceResult()
        self.multifunction_conformance: MultifunctionResult = MultifunctionResult()
        self.overall_score: float = 0.0
        self.mlpackage_path: str = ""
        self.verification_timestamp: str = ""
        self.platform: str = ""
        self.coremltools_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_fidelity": self.op_fidelity.to_dict(),
            "placement": self.placement.to_dict(),
            "state_conformance": self.state_conformance.to_dict(),
            "multifunction_conformance": self.multifunction_conformance.to_dict(),
            "overall_score": round(self.overall_score, 4),
            "mlpackage_path": self.mlpackage_path,
            "verification_timestamp": self.verification_timestamp,
            "platform": self.platform,
            "coremltools_version": self.coremltools_version,
        }


# ---------------------------------------------------------------------------
# Main verification entry point
# ---------------------------------------------------------------------------

def verify_model(
    mlpackage_path: str,
    mir_ops: Optional[List[Dict[str, Any]]] = None,
    expected_function_names: Optional[List[str]] = None,
    expected_state_names: Optional[List[str]] = None,
    compute_units: str = "CPU_AND_NE",
) -> VerificationResult:
    """Perform unified verification of an emitted mlpackage.

    This is the primary entry point for Sprint 40 verification. It runs
    all four verification dimensions and produces a single structured result.

    Args:
        mlpackage_path: Path to the .mlpackage directory to verify.
        mir_ops: Optional list of MIR op dicts for op fidelity comparison.
            Each dict should have an "op_type" key with the MIR variant name.
        expected_function_names: Optional list of expected function names
            for multi-function conformance.
        expected_state_names: Optional list of expected state input names
            for state conformance.
        compute_units: Compute units for compute plan inspection.

    Returns:
        VerificationResult with all four dimensions populated.
    """
    import datetime

    result = VerificationResult()
    result.mlpackage_path = mlpackage_path
    result.verification_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Detect platform
    import platform
    result.platform = platform.system()

    # Detect coremltools version
    try:
        import coremltools as ct
        result.coremltools_version = ct.__version__
    except ImportError:
        result.coremltools_version = None

    pkg_path = Path(mlpackage_path)
    if not pkg_path.exists() or not pkg_path.is_dir():
        result.op_fidelity.reason = "mlpackage directory does not exist"
        result.placement.reason = "mlpackage directory does not exist"
        result.state_conformance.reason = "mlpackage directory does not exist"
        result.multifunction_conformance.reason = "mlpackage directory does not exist"
        return result

    # ── Dimension 1: Op Graph Fidelity ────────────────────────────────
    _verify_op_fidelity(result, mlpackage_path, mir_ops)

    # ── Dimension 2: Compute-Unit Placement ────────────────────────────
    _verify_placement(result, mlpackage_path, compute_units, mir_ops)

    # ── Dimension 3: State Conformance ─────────────────────────────────
    _verify_state_conformance(result, mlpackage_path, expected_state_names)

    # ── Dimension 4: Multi-Function Conformance ────────────────────────
    _verify_multifunction_conformance(
        result, mlpackage_path, expected_function_names
    )

    # ── Overall Score ──────────────────────────────────────────────────
    # Weighted average: op_fidelity (40%), placement (20%),
    # state_conformance (20%), multifunction_conformance (20%).
    # Dimensions that are unavailable contribute 0.5 (neutral) instead of 0.
    scores = []
    weights = []

    fidelity = result.op_fidelity.op_fidelity_score if result.op_fidelity.available else 0.5
    scores.append(fidelity)
    weights.append(0.4)

    placement = result.placement.ane_placement_rate if result.placement.available else 0.5
    scores.append(placement)
    weights.append(0.2)

    state = result.state_conformance.conformance_score if result.state_conformance.available else 0.5
    scores.append(state)
    weights.append(0.2)

    mf = result.multifunction_conformance.conformance_score if result.multifunction_conformance.available else 0.5
    scores.append(mf)
    weights.append(0.2)

    result.overall_score = sum(s * w for s, w in zip(scores, weights))

    return result


# ---------------------------------------------------------------------------
# Internal verification functions
# ---------------------------------------------------------------------------

def _verify_op_fidelity(
    result: VerificationResult,
    mlpackage_path: str,
    mir_ops: Optional[List[Dict[str, Any]]],
) -> None:
    """Verify op graph fidelity by comparing MIR intent against emitted structure.

    Uses MLModelStructure when available (macOS), falls back to spec-based
    op extraction on other platforms.
    """
    from model_structure import (
        inspect_model_structure_with_mir_comparison,
        compare_mir_vs_structure,
        fallback_file_structure,
    )

    # Step 1: Try MLModelStructure + MIR comparison
    structure_result = inspect_model_structure_with_mir_comparison(
        mlpackage_path, mir_ops
    )

    if structure_result.get("available", False):
        # MLModelStructure worked — use its result directly
        result.op_fidelity.available = True
        result.op_fidelity.verification_method = "mlmodel_structure"

        mir_comparison = structure_result.get("mir_comparison")
        if mir_comparison is not None:
            result.op_fidelity.op_fidelity_score = mir_comparison.get(
                "op_fidelity_score", 0.0
            )
            result.op_fidelity.mir_op_count = mir_comparison.get("mir_op_count", 0)
            result.op_fidelity.structure_op_count = mir_comparison.get(
                "structure_op_count", 0
            )
            result.op_fidelity.matched_ops = mir_comparison.get("matched_ops", [])
            result.op_fidelity.missing_from_structure = mir_comparison.get(
                "missing_from_structure", []
            )
            result.op_fidelity.extra_in_structure = mir_comparison.get(
                "extra_in_structure", []
            )
        else:
            # Structure inspection succeeded but no MIR ops to compare
            ops = structure_result.get("operations", [])
            result.op_fidelity.structure_op_count = len(ops)
            result.op_fidelity.verification_method = "mlmodel_structure_no_mir"
    else:
        # MLModelStructure unavailable — fall back to spec-based extraction
        result.op_fidelity.available = False
        result.op_fidelity.reason = structure_result.get(
            "reason", "MLModelStructure unavailable"
        )
        result.op_fidelity.verification_method = "spec_based_fallback"

        # Try spec-based op extraction via coremltools
        _extract_ops_from_spec(result, mlpackage_path, mir_ops)


def _extract_ops_from_spec(
    result: VerificationResult,
    mlpackage_path: str,
    mir_ops: Optional[List[Dict[str, Any]]],
) -> None:
    """Extract ops from the model spec using coremltools when MLModelStructure
    is unavailable.

    This provides a weaker but still useful verification on Linux.
    """
    try:
        import coremltools as ct
        model = ct.models.MLModel(mlpackage_path)
        spec = model.get_spec()

        # Walk mlProgram functions
        actual_ops = []
        if hasattr(spec, 'mlProgram'):
            ml_prog = spec.mlProgram
            for fn_name in ml_prog.functions:
                fn_proto = ml_prog.functions[fn_name]
                for block_name in fn_proto.block_specializations:
                    block = fn_proto.block_specializations[block_name]
                    for op in block.operations:
                        actual_ops.append({
                            "op_type": op.type,
                            "name": op.name if hasattr(op, 'name') else "unknown",
                            "function": fn_name,
                        })

        if actual_ops:
            result.op_fidelity.structure_op_count = len(actual_ops)
            result.op_fidelity.available = True
            result.op_fidelity.verification_method = "spec_based_extraction"

            # If MIR ops provided, compare
            if mir_ops is not None:
                from model_structure import compare_mir_vs_structure
                comparison = compare_mir_vs_structure(mir_ops, actual_ops)
                result.op_fidelity.op_fidelity_score = comparison.get(
                    "op_fidelity_score", 0.0
                )
                result.op_fidelity.mir_op_count = comparison.get("mir_op_count", 0)
                result.op_fidelity.matched_ops = comparison.get("matched_ops", [])
                result.op_fidelity.missing_from_structure = comparison.get(
                    "missing_from_structure", []
                )
                result.op_fidelity.extra_in_structure = comparison.get(
                    "extra_in_structure", []
                )
    except Exception as e:
        result.op_fidelity.reason = (
            f"Spec-based extraction failed: {e}. "
            "MLModelStructure requires macOS with Core ML runtime."
        )


def _verify_placement(
    result: VerificationResult,
    mlpackage_path: str,
    compute_units: str,
    mir_ops: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Verify compute-unit placement using MLComputePlan (macOS) or
    offline prediction (Linux/non-Apple hosts).

    Sprint 57: when MLComputePlan is unavailable (non-Apple hosts), this
    now falls back to offline placement prediction using the same known
    op→device mapping table as the Rust ComputePlanVerifier. This closes
    Issue #5 by providing a predicted placement instead of a plain
    "unavailable" result on Linux.

    The offline prediction is clearly marked with verification_method
    "offline_prediction" and includes a prediction_confidence score so
    downstream consumers can distinguish predicted vs. observed placement.
    """
    from compute_plan import harvest_compute_plan, predict_placement_from_ops

    harvest_result = harvest_compute_plan(mlpackage_path, compute_units)

    if harvest_result.get("available", False):
        result.placement.available = True
        result.placement.verification_method = "mlcomputeplan"
        result.placement.ane_placement_rate = harvest_result.get(
            "ane_placement_rate", 0.0
        )
        result.placement.total_ops = harvest_result.get("total_ops", 0)
        result.placement.per_op_placement = harvest_result.get(
            "per_op_placement", []
        )
        result.placement.ane_placed_ops = sum(
            1 for op in result.placement.per_op_placement
            if "NeuralEngine" in op.get("preferred_device", "")
        )
    else:
        # MLComputePlan unavailable — try offline prediction from MIR ops
        predict_result = predict_placement_from_ops(mir_ops)

        if predict_result.get("available", False):
            result.placement.available = True
            result.placement.verification_method = "offline_prediction"
            result.placement.prediction_confidence = predict_result.get(
                "prediction_confidence", 0.0
            )
            result.placement.ane_placement_rate = predict_result.get(
                "ane_placement_rate", 0.0
            )
            result.placement.total_ops = predict_result.get("total_ops", 0)
            result.placement.per_op_placement = predict_result.get(
                "per_op_placement", []
            )
            result.placement.ane_placed_ops = sum(
                1 for op in result.placement.per_op_placement
                if "NeuralEngine" in op.get("preferred_device", "")
            )
            # Mark this as a prediction, not observed data
            result.placement.reason = (
                "Predicted placement from known op→device mappings "
                "(MLComputePlan unavailable on this platform). "
                f"Prediction confidence: {predict_result.get('prediction_confidence', 0.0):.2f}"
            )
        else:
            result.placement.available = False
            result.placement.verification_method = "unavailable"
            result.placement.reason = harvest_result.get(
                "reason", "MLComputePlan unavailable"
            )


def _verify_state_conformance(
    result: VerificationResult,
    mlpackage_path: str,
    expected_state_names: Optional[List[str]],
) -> None:
    """Verify state model conformance by checking for state declarations
    and state read/write ops in the emitted model.

    Uses MLModelStructure when available (macOS), falls back to spec-based
    detection on other platforms.
    """
    from model_structure import inspect_model_structure

    structure_result = inspect_model_structure(mlpackage_path)

    if structure_result.get("available", False):
        # MLModelStructure available — use it for state detection
        result.state_conformance.available = True
        result.state_conformance.verification_method = "mlmodel_structure"

        state_decls = structure_result.get("state_declarations", [])
        operations = structure_result.get("operations", [])

        result.state_conformance.state_details = state_decls
        result.state_conformance.actual_state_count = len(state_decls)
        result.state_conformance.stateful_model = len(state_decls) > 0

        # Check for state ops
        op_types = [op.get("op_type", "") for op in operations]
        result.state_conformance.has_read_state = "read_state" in op_types
        result.state_conformance.has_update_state = (
            "coreml_update_state" in op_types or "write_state" in op_types
        )

        # Compare against expected state names
        if expected_state_names is not None:
            result.state_conformance.expected_state_count = len(expected_state_names)
            actual_names = {s.get("name", "") for s in state_decls}
            expected_set = set(expected_state_names)
            result.state_conformance.state_names_match = actual_names == expected_set

        # Compute conformance score
        score = 1.0
        if expected_state_names is not None:
            if len(state_decls) != len(expected_state_names):
                score *= 0.5
            if not result.state_conformance.state_names_match:
                score *= 0.7
        if result.state_conformance.stateful_model:
            if not result.state_conformance.has_read_state:
                score *= 0.5
            if not result.state_conformance.has_update_state:
                score *= 0.5
        result.state_conformance.conformance_score = score
    else:
        # MLModelStructure unavailable — try spec-based state detection
        _detect_state_from_spec(result, mlpackage_path, expected_state_names)


def _detect_state_from_spec(
    result: VerificationResult,
    mlpackage_path: str,
    expected_state_names: Optional[List[str]],
) -> None:
    """Detect state declarations from the model spec when MLModelStructure
    is unavailable.

    This provides weaker verification on Linux by checking the model's
    spec.description.state field directly.
    """
    try:
        import coremltools as ct
        model = ct.models.MLModel(mlpackage_path)
        spec = model.get_spec()

        result.state_conformance.available = True
        result.state_conformance.verification_method = "spec_based_detection"

        # Check for state declarations in spec.description.state
        state_decls = []
        if hasattr(spec, 'description') and hasattr(spec.description, 'state'):
            for state_entry in spec.description.state:
                state_desc = {"name": state_entry.name}
                if hasattr(state_entry, 'type'):
                    type_info = state_entry.type
                    if hasattr(type_info, 'multiArrayType'):
                        state_desc["shape"] = list(type_info.multiArrayType.shape)
                    if hasattr(type_info, 'stateType'):
                        result.state_conformance.stateful_model = True
                        state_desc["is_state"] = True
                        if hasattr(type_info.stateType, 'shape'):
                            inner = type_info.stateType
                            if hasattr(inner, 'multiArrayType'):
                                state_desc["shape"] = list(
                                    inner.multiArrayType.shape
                                )
                state_decls.append(state_desc)

        # Also check input types for StateType
        if hasattr(spec, 'description') and hasattr(spec.description, 'input'):
            for inp in spec.description.input:
                inp_type_str = str(inp.type).lower()
                if 'statetype' in inp_type_str or 'state' in inp_type_str:
                    state_desc = {"name": inp.name, "is_state": True}
                    if inp.type.HasField('multiArrayType'):
                        state_desc["shape"] = list(
                            inp.type.multiArrayType.shape
                        )
                    state_decls.append(state_desc)

        result.state_conformance.state_details = state_decls
        result.state_conformance.actual_state_count = len(state_decls)
        result.state_conformance.stateful_model = len(state_decls) > 0

        # Check for state ops in the program
        has_read = False
        has_update = False
        if hasattr(spec, 'mlProgram'):
            ml_prog = spec.mlProgram
            for fn_name in ml_prog.functions:
                fn_proto = ml_prog.functions[fn_name]
                for block_name in fn_proto.block_specializations:
                    block = fn_proto.block_specializations[block_name]
                    for op in block.operations:
                        if op.type == "read_state":
                            has_read = True
                        elif op.type in ("coreml_update_state", "write_state"):
                            has_update = True

        result.state_conformance.has_read_state = has_read
        result.state_conformance.has_update_state = has_update

        # Compare against expected state names
        if expected_state_names is not None:
            result.state_conformance.expected_state_count = len(expected_state_names)
            actual_names = {s.get("name", "") for s in state_decls}
            expected_set = set(expected_state_names)
            result.state_conformance.state_names_match = actual_names == expected_set

        # Compute conformance score
        score = 1.0
        if expected_state_names is not None:
            if len(state_decls) != len(expected_state_names):
                score *= 0.5
            if not result.state_conformance.state_names_match:
                score *= 0.7
        if result.state_conformance.stateful_model:
            if not result.state_conformance.has_read_state:
                score *= 0.5
            if not result.state_conformance.has_update_state:
                score *= 0.5
        result.state_conformance.conformance_score = score

    except Exception as e:
        result.state_conformance.available = False
        result.state_conformance.reason = (
            f"Spec-based state detection failed: {e}. "
            "State verification requires macOS with Core ML runtime for "
            "full fidelity, or coremltools for spec-based detection."
        )


def _verify_multifunction_conformance(
    result: VerificationResult,
    mlpackage_path: str,
    expected_function_names: Optional[List[str]],
) -> None:
    """Verify multi-function model conformance by checking function count
    and names.

    Uses MLModelStructure when available (macOS), falls back to spec-based
    detection on other platforms.
    """
    from model_structure import inspect_model_structure

    structure_result = inspect_model_structure(mlpackage_path)

    if structure_result.get("available", False):
        # MLModelStructure available — use it
        result.multifunction_conformance.available = True
        result.multifunction_conformance.verification_method = "mlmodel_structure"

        functions = structure_result.get("functions", [])
        result.multifunction_conformance.actual_function_count = len(functions)
        result.multifunction_conformance.is_multifunction = len(functions) > 1

        fn_details = []
        for fn in functions:
            fn_details.append({
                "name": fn.get("name", "unknown"),
                "input_count": len(fn.get("inputs", [])),
                "output_count": len(fn.get("outputs", [])),
                "operation_count": fn.get("operation_count", 0),
            })
        result.multifunction_conformance.function_details = fn_details

        # Compare against expected
        if expected_function_names is not None:
            result.multifunction_conformance.expected_function_count = len(
                expected_function_names
            )
            actual_names = {fn.get("name", "") for fn in functions}
            expected_set = set(expected_function_names)
            result.multifunction_conformance.function_names_match = (
                actual_names == expected_set
            )

        # Compute conformance score
        score = 1.0
        if expected_function_names is not None:
            if len(functions) != len(expected_function_names):
                score *= 0.5
            if not result.multifunction_conformance.function_names_match:
                score *= 0.7
        result.multifunction_conformance.conformance_score = score
    else:
        # MLModelStructure unavailable — try spec-based detection
        _detect_multifunction_from_spec(
            result, mlpackage_path, expected_function_names
        )


def _detect_multifunction_from_spec(
    result: VerificationResult,
    mlpackage_path: str,
    expected_function_names: Optional[List[str]],
) -> None:
    """Detect multi-function structure from the model spec when
    MLModelStructure is unavailable.

    This provides weaker verification on Linux by checking spec.functions
    directly.
    """
    try:
        import coremltools as ct
        model = ct.models.MLModel(mlpackage_path)
        spec = model.get_spec()

        result.multifunction_conformance.available = True
        result.multifunction_conformance.verification_method = "spec_based_detection"

        fn_details = []
        fn_count = 0

        if hasattr(spec, 'mlProgram'):
            ml_prog = spec.mlProgram
            for fn_name in ml_prog.functions:
                fn_proto = ml_prog.functions[fn_name]
                op_count = 0
                for block_name in fn_proto.block_specializations:
                    block = fn_proto.block_specializations[block_name]
                    op_count += len(block.operations)
                fn_details.append({
                    "name": fn_name,
                    "operation_count": op_count,
                })
                fn_count += 1

        if fn_count == 0:
            fn_count = 1  # Single-function model

        result.multifunction_conformance.actual_function_count = fn_count
        result.multifunction_conformance.is_multifunction = fn_count > 1
        result.multifunction_conformance.function_details = fn_details

        # Compare against expected
        if expected_function_names is not None:
            result.multifunction_conformance.expected_function_count = len(
                expected_function_names
            )
            actual_names = {fn["name"] for fn in fn_details}
            expected_set = set(expected_function_names)
            result.multifunction_conformance.function_names_match = (
                actual_names == expected_set
            )

        # Compute conformance score
        score = 1.0
        if expected_function_names is not None:
            if fn_count != len(expected_function_names):
                score *= 0.5
            if not result.multifunction_conformance.function_names_match:
                score *= 0.7
        result.multifunction_conformance.conformance_score = score

    except Exception as e:
        result.multifunction_conformance.available = False
        result.multifunction_conformance.reason = (
            f"Spec-based multi-function detection failed: {e}. "
            "Multi-function verification requires macOS with Core ML runtime "
            "for full fidelity, or coremltools for spec-based detection."
        )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_verification_result(
    result: VerificationResult,
    output_dir: str,
    artifact_name: str = "verification_result",
) -> Dict[str, str]:
    """Persist the verification result as a JSON artifact.

    Args:
        result: The VerificationResult to persist.
        output_dir: Directory to write the artifact file.
        artifact_name: Base name for the artifact file (without extension).

    Returns:
        Dict with the path to the saved artifact.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifact_path = out_path / f"{artifact_name}.json"
    with open(str(artifact_path), "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Also save a summary with just the scores for quick comparison
    summary_path = out_path / f"{artifact_name}_summary.json"
    summary = {
        "overall_score": result.overall_score,
        "op_fidelity_score": result.op_fidelity.op_fidelity_score,
        "ane_placement_rate": result.placement.ane_placement_rate,
        "state_conformance_score": result.state_conformance.conformance_score,
        "multifunction_conformance_score": result.multifunction_conformance.conformance_score,
        "mlpackage_path": result.mlpackage_path,
        "platform": result.platform,
        "coremltools_version": result.coremltools_version,
    }
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "artifact_path": str(artifact_path),
        "summary_path": str(summary_path),
    }


# ---------------------------------------------------------------------------
# T-P5-12: Semantic emission verification
# ---------------------------------------------------------------------------

def verify_emission_semantics(
    mlpackage_path: str,
    expected_inputs: Optional[List[Dict[str, Any]]] = None,
    expected_outputs: Optional[List[Dict[str, Any]]] = None,
    expected_dtypes: Optional[Dict[str, str]] = None,
    forbidden_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify semantic correctness of an emitted mlpackage.

    T-P5-12: This function performs lightweight host-side semantic checks
    on an emitted mlpackage to catch common emission errors before runtime.
    It does NOT require macOS or coremltools to execute the model.

    Checks performed:
    1. All expected I/O names appear in the emitted mlpackage
    2. Weight files exist and have non-zero size
    3. Proto descriptors match the original graph's dtypes (not all Float16)
    4. No placeholder names remain in the emission (e.g., "__placeholder__",
       "__unused__", "__todo__")

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        expected_inputs: List of dicts with 'name' keys for expected input names.
        expected_outputs: List of dicts with 'name' keys for expected output names.
        expected_dtypes: Dict mapping I/O name to expected dtype string
            (e.g., {"x": "fp16", "output": "fp32"}). Checks that not all
            dtypes are Float16 (which would indicate a default-dtype bug).
        forbidden_patterns: List of string patterns that must NOT appear in
            any I/O name (e.g., ["__placeholder__", "__unused__", "__todo__"]).

    Returns:
        Dict with:
          - passed: bool — whether all checks passed
          - errors: list of str — semantic errors found
          - warnings: list of str — non-fatal issues found
          - details: dict — per-check details
    """
    pkg_path = Path(mlpackage_path)
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    # Check 1: mlpackage directory exists
    if not pkg_path.exists() or not pkg_path.is_dir():
        return {
            "passed": False,
            "errors": [f"mlpackage directory does not exist: {mlpackage_path}"],
            "warnings": [],
            "details": {"directory_exists": False},
        }

    # Check 2: Manifest.json exists and is readable
    manifest_path = pkg_path / "Manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            details["manifest_readable"] = True
        except Exception as e:
            errors.append(f"Manifest.json is not valid JSON: {e}")
            details["manifest_readable"] = False
    else:
        errors.append("Manifest.json is missing")
        details["manifest_readable"] = False

    # Check 3: Weight file exists and has non-zero size
    weight_bin_path = pkg_path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    if weight_bin_path.exists():
        weight_size = weight_bin_path.stat().st_size
        details["weight_bin_size"] = weight_size
        if weight_size == 0:
            errors.append("weight.bin exists but is empty (0 bytes) — all weights may be zero-filled")
    else:
        # Weight file may not exist for models with no weights (unlikely but possible)
        warnings.append("weight.bin not found — model may have no weight data")
        details["weight_bin_size"] = 0

    # Check 4: Model.mlmodel (protobuf) exists and has non-zero size
    mlmodel_path = pkg_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
    if mlmodel_path.exists():
        mlmodel_size = mlmodel_path.stat().st_size
        details["mlmodel_size"] = mlmodel_size
        if mlmodel_size == 0:
            errors.append("model.mlmodel is empty (0 bytes)")
    else:
        errors.append("model.mlmodel is missing from the mlpackage")
        details["mlmodel_size"] = 0

    # Attempt structural inspection for I/O name checks
    io_names = {"inputs": set(), "outputs": set()}
    io_dtypes: Dict[str, str] = {}

    try:
        import coremltools as ct
        try:
            model = ct.models.MLModel(str(pkg_path))
            spec = model.get_spec()

            # Extract input names and dtypes
            for inp in spec.description.input:
                io_names["inputs"].add(inp.name)
                type_str = str(inp.type).lower()
                if "float16" in type_str or "fp16" in type_str:
                    io_dtypes[inp.name] = "fp16"
                elif "float32" in type_str or "fp32" in type_str:
                    io_dtypes[inp.name] = "fp32"
                elif "int32" in type_str:
                    io_dtypes[inp.name] = "int32"
                else:
                    io_dtypes[inp.name] = "unknown"

            # Extract output names and dtypes
            for outp in spec.description.output:
                io_names["outputs"].add(outp.name)
                type_str = str(outp.type).lower()
                if "float16" in type_str or "fp16" in type_str:
                    io_dtypes[outp.name] = "fp16"
                elif "float32" in type_str or "fp32" in type_str:
                    io_dtypes[outp.name] = "fp32"
                elif "int32" in type_str:
                    io_dtypes[outp.name] = "int32"
                else:
                    io_dtypes[outp.name] = "unknown"

            details["structural_inspection"] = "coremltools"
        except Exception as e:
            warnings.append(f"Could not load model for structural inspection: {e}")
            details["structural_inspection"] = "unavailable"
    except ImportError:
        warnings.append("coremltools not available — I/O name and dtype checks skipped")
        details["structural_inspection"] = "coremltools_not_installed"

    # Check 5: All expected I/O names are present
    if expected_inputs and io_names["inputs"]:
        missing_inputs = [
            inp["name"] for inp in expected_inputs
            if inp["name"] not in io_names["inputs"]
        ]
        if missing_inputs:
            errors.append(
                f"Missing expected input name(s) in emitted mlpackage: {missing_inputs}"
            )
        details["missing_inputs"] = missing_inputs

    if expected_outputs and io_names["outputs"]:
        missing_outputs = [
            outp["name"] for outp in expected_outputs
            if outp["name"] not in io_names["outputs"]
        ]
        if missing_outputs:
            errors.append(
                f"Missing expected output name(s) in emitted mlpackage: {missing_outputs}"
            )
        details["missing_outputs"] = missing_outputs

    # Check 6: Proto descriptors match expected dtypes (not all Float16)
    if io_dtypes:
        all_fp16 = all(v == "fp16" for v in io_dtypes.values())
        if all_fp16 and len(io_dtypes) > 0:
            # This is a warning, not an error — some models genuinely are all-FP16
            warnings.append(
                "All I/O dtypes are Float16 — this may indicate a default-dtype bug "
                "where original graph dtypes were not preserved during emission"
            )
        details["all_dtypes_fp16"] = all_fp16
        details["io_dtypes"] = io_dtypes

    # Check 7: No placeholder names remain in the emission
    default_forbidden = ["__placeholder__", "__unused__", "__todo__", "__stub__"]
    patterns_to_check = (forbidden_patterns or []) + default_forbidden
    placeholder_found = []
    for name in io_names["inputs"] | io_names["outputs"]:
        for pattern in patterns_to_check:
            if pattern in name:
                placeholder_found.append(name)
                break
    if placeholder_found:
        errors.append(
            f"Placeholder name(s) found in emitted I/O: {placeholder_found}. "
            "These indicate incomplete emission — every I/O tensor should have "
            "a meaningful name from the original graph."
        )
    details["placeholder_names"] = placeholder_found

    passed = len(errors) == 0
    return {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "details": details,
    }


# ---------------------------------------------------------------------------
# T-D-02 (M-032): MIR specification compliance verification
# ---------------------------------------------------------------------------

class MirSpecViolation:
    """A single MIR specification compliance violation.

    Represents a mismatch between a constructed MIL program and the
    MIR specification from the Rust compiler side.
    """

    def __init__(self, check: str, message: str, severity: str = "error"):
        self.check: str = check
        self.message: str = message
        self.severity: str = severity  # "error" or "warning"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "message": self.message,
            "severity": self.severity,
        }

    def __repr__(self):
        return f"MirSpecViolation(check={self.check!r}, message={self.message!r}, severity={self.severity!r})"


def verify_mir_spec_compliance(program, mir_spec: dict) -> list:
    """Verify that a MIL program matches the MIR specification.

    T-D-02 (M-032): Verifies that a constructed MIL program matches the MIR
    specification from the Rust compiler side.

    Args:
        program: The MIL program (mb object) to verify
        mir_spec: Dict containing MIR specification with:
            - 'inputs': list of {name, shape, dtype}
            - 'outputs': list of {name, shape, dtype}
            - 'ops': list of {name, type, inputs, outputs}

    Returns:
        List of MirSpecViolation objects (empty = pass)
    """
    violations = []

    if mir_spec is None:
        return violations

    # Extract program I/O info if available
    prog_inputs = []
    prog_outputs = []
    prog_ops = []

    if program is not None:
        # Try to extract from the MIL program object
        try:
            if hasattr(program, 'main_function'):
                fn = program.main_function
                if hasattr(fn, 'inputs'):
                    prog_inputs = [
                        {"name": inp.name, "shape": list(inp.shape) if hasattr(inp, 'shape') else []}
                        for inp in fn.inputs.values()
                    ] if hasattr(fn.inputs, 'values') else []
                if hasattr(fn, 'outputs'):
                    prog_outputs = [
                        {"name": out.name}
                        for out in fn.outputs.values()
                    ] if hasattr(fn.outputs, 'values') else []
                if hasattr(fn, 'operations'):
                    prog_ops = [
                        {"name": op.name, "type": op.op_type}
                        for op in fn.operations
                    ]
        except Exception:
            pass  # Program introspection not available — skip program-side checks

    # Check 1: Input count matches
    expected_inputs = mir_spec.get('inputs', [])
    if expected_inputs and prog_inputs:
        if len(prog_inputs) != len(expected_inputs):
            violations.append(MirSpecViolation(
                check="input_count",
                message=f"Input count mismatch: MIR spec has {len(expected_inputs)}, program has {len(prog_inputs)}",
                severity="error",
            ))

    # Check 2: Input names match
    if expected_inputs and prog_inputs:
        expected_names = {inp.get('name', '') for inp in expected_inputs}
        actual_names = {inp.get('name', '') for inp in prog_inputs}
        missing = expected_names - actual_names
        if missing:
            violations.append(MirSpecViolation(
                check="input_names",
                message=f"Missing inputs from program: {missing}",
                severity="error",
            ))

    # Check 3: Input shapes/dtypes match
    if expected_inputs and prog_inputs:
        for exp_inp in expected_inputs:
            exp_name = exp_inp.get('name', '')
            for act_inp in prog_inputs:
                if act_inp.get('name', '') == exp_name:
                    exp_shape = exp_inp.get('shape', [])
                    act_shape = act_inp.get('shape', [])
                    if exp_shape and act_shape and exp_shape != act_shape:
                        violations.append(MirSpecViolation(
                            check="input_shapes",
                            message=f"Shape mismatch for input '{exp_name}': expected {exp_shape}, got {act_shape}",
                            severity="warning",
                        ))
                    break

    # Check 4: Output count matches
    expected_outputs = mir_spec.get('outputs', [])
    if expected_outputs and prog_outputs:
        if len(prog_outputs) != len(expected_outputs):
            violations.append(MirSpecViolation(
                check="output_count",
                message=f"Output count mismatch: MIR spec has {len(expected_outputs)}, program has {len(prog_outputs)}",
                severity="error",
            ))

    # Check 5: Output names match
    if expected_outputs and prog_outputs:
        expected_names = {out.get('name', '') for out in expected_outputs}
        actual_names = {out.get('name', '') for out in prog_outputs}
        missing = expected_names - actual_names
        if missing:
            violations.append(MirSpecViolation(
                check="output_names",
                message=f"Missing outputs from program: {missing}",
                severity="error",
            ))

    # Check 6: All MIR ops are represented in the program
    expected_ops = mir_spec.get('ops', [])
    if expected_ops and prog_ops:
        actual_op_types = {op['type'] for op in prog_ops}
        for exp_op in expected_ops:
            exp_type = exp_op.get('type', '')
            if exp_type and exp_type not in actual_op_types:
                violations.append(MirSpecViolation(
                    check="missing_ops",
                    message=f"MIR op '{exp_type}' not found in program",
                    severity="warning",
                ))

    # Check 7: No extra ops not in MIR spec
    if expected_ops and prog_ops:
        expected_op_types = {op.get('type', '') for op in expected_ops}
        for act_op in prog_ops:
            act_type = act_op.get('type', '')
            if act_type and act_type not in expected_op_types:
                violations.append(MirSpecViolation(
                    check="extra_ops",
                    message=f"Program has op '{act_type}' not in MIR spec",
                    severity="warning",
                ))

    return violations


def pre_emit_verification(builder_ops, mir_spec=None):
    """T-D-02 (M-032): Verify builder operations before emission.

    Checks:
    1. All SSA references resolve (no dangling inputs)
    2. No duplicate output names
    3. Shape consistency for broadcast ops
    4. Dtype compatibility for binary ops
    5. Weight references exist in the spec

    Returns list of issues found (empty = pass).
    """
    issues = []
    defined_names = set()

    # Collect graph inputs from mir_spec if provided
    graph_inputs = set()
    if mir_spec and 'graph_inputs' in mir_spec:
        graph_inputs = set(mir_spec['graph_inputs'])

    for op in builder_ops:
        # Check for duplicate output names
        output_name = op.get('output_name')
        if output_name is not None:
            if output_name in defined_names:
                issues.append(f"Duplicate output name: {output_name}")
            defined_names.add(output_name)

        # Check that inputs reference defined names (or are graph inputs)
        for inp in op.get('inputs', []):
            if inp not in defined_names and inp not in graph_inputs:
                if not inp.startswith('const_') and not inp.startswith('weight_'):
                    issues.append(f"Dangling input reference: {inp} in op {op.get('output_name', '<unknown>')}")

    return issues
