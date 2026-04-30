"""Compute Plan — Inspects compute device assignment for ML programs.

Provides detailed compute plan inspection using MLComputePlan.
On Apple hardware, this reports per-operation compute device assignment.
On non-Apple platforms, the API is unavailable but the code path is real.

Sprint 35 additions:
  - harvest_compute_plan(): extracts per-op placement + estimated cost
  - harvest_to_observations(): converts placement data into knowledge store observations
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


def inspect_compute_plan(
    mlpackage_path: str,
    compute_units: str = "CPU_AND_NE",
) -> Dict[str, Any]:
    """Inspect the compute plan for an ML program.

    On Apple hardware with coremltools >= 8.0, this uses MLComputePlan
    to report per-operation compute device assignment.

    On non-Apple platforms (Linux), MLComputePlan.load_from_path will
    fail because it requires the Core ML runtime. The function handles
    this gracefully and reports the reason.

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        compute_units: Compute units to use for planning.

    Returns:
        Dict with:
          - available: bool — whether compute plan inspection succeeded
          - reason: str (if unavailable) — why inspection failed
          - operations: list (if available) — per-op compute device assignments
          - total_operations: int — number of operations in the model
    """
    try:
        from coremltools.models.compute_plan import MLComputePlan
    except ImportError:
        return {
            "available": False,
            "reason": "MLComputePlan not available in this coremltools version",
            "operations": [],
            "total_operations": 0,
        }

    ct = _ensure_coremltools()
    compute_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    compute_unit = compute_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)

    try:
        plan = MLComputePlan.load_from_path(str(mlpackage_path), compute_unit)

        # If we get here, we're on Apple hardware with runtime support
        operations = []
        if hasattr(plan, 'operations_by_name'):
            for op_name, op_info in plan.operations_by_name.items():
                op_entry = {
                    "name": op_name,
                    "compute_device": str(op_info.compute_device) if hasattr(op_info, 'compute_device') else "unknown",
                }
                if hasattr(op_info, 'compute_device_memory_bytes'):
                    op_entry["memory_bytes"] = op_info.compute_device_memory_bytes
                operations.append(op_entry)

        return {
            "available": True,
            "operations": operations,
            "total_operations": len(operations),
        }

    except Exception as e:
        error_str = str(e)
        # Classify the failure reason
        if "libcoremlpython" in error_str or "CoreML" in error_str:
            reason = "Apple Core ML runtime not available on this platform"
        else:
            reason = f"Compute plan inspection failed: {error_str}"

        return {
            "available": False,
            "reason": reason,
            "operations": [],
            "total_operations": 0,
        }


def harvest_compute_plan(
    mlpackage_path: str,
    compute_units: str = "CPU_AND_NE",
) -> Dict[str, Any]:
    """Harvest per-op placement and cost data from MLComputePlan.

    On Apple hardware, this loads MLComputePlan and extracts per-operation
    device placement (preferred device, supported devices) and estimated
    execution cost using the MLComputePlan cost/usage query APIs.

    On non-Apple platforms (Linux), MLComputePlan is unavailable and this
    returns available=False with a clear reason string.

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        compute_units: Compute units to use for planning (default "CPU_AND_NE").

    Returns:
        Dict with:
          - available: bool — whether compute plan harvesting succeeded
          - reason: str (if unavailable) — why harvesting failed
          - per_op_placement: list of dicts, each with:
              - op_name: str — operation name in the ML program
              - preferred_device: str — preferred compute device (e.g. "NeuralEngine")
              - supported_devices: list of str — all supported compute devices
              - estimated_cost: float or None — estimated execution cost (0.0 if unavailable)
          - ane_placement_rate: float — fraction of ops preferring NeuralEngine (0.0–1.0)
          - total_ops: int — total number of operations
          - source: str — "mlcomputeplan" if available, "unavailable" otherwise
    """
    try:
        from coremltools.models.compute_plan import MLComputePlan
    except ImportError:
        return {
            "available": False,
            "reason": "MLComputePlan not available in this coremltools version",
            "per_op_placement": [],
            "ane_placement_rate": 0.0,
            "total_ops": 0,
            "source": "unavailable",
        }

    ct = _ensure_coremltools()
    compute_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    compute_unit = compute_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)

    try:
        plan = MLComputePlan.load_from_path(str(mlpackage_path), compute_unit)
    except Exception as e:
        error_str = str(e)
        if "libcoremlpython" in error_str or "CoreML" in error_str:
            reason = "Apple Core ML runtime not available on this platform"
        else:
            reason = f"Compute plan harvesting failed: {error_str}"
        return {
            "available": False,
            "reason": reason,
            "per_op_placement": [],
            "ane_placement_rate": 0.0,
            "total_ops": 0,
            "source": "unavailable",
        }

    # We are on Apple hardware with runtime support
    per_op_placement: List[Dict[str, Any]] = []
    ane_count = 0

    if hasattr(plan, 'operations_by_name'):
        for op_name, op_info in plan.operations_by_name.items():
            # Extract preferred compute device
            preferred_device = "unknown"
            if hasattr(op_info, 'compute_device') and op_info.compute_device is not None:
                preferred_device = str(op_info.compute_device)

            # Extract supported devices via get_compute_device_usage_for_mlprogram_operation
            supported_devices: List[str] = []
            try:
                if hasattr(plan, 'get_compute_device_usage_for_mlprogram_operation'):
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op_name)
                    if usage is not None:
                        if hasattr(usage, 'supported_devices'):
                            supported_devices = [str(d) for d in usage.supported_devices]
                        elif hasattr(usage, 'compute_device'):
                            # Some coremltools versions return a single device
                            supported_devices = [str(usage.compute_device)]
            except Exception:
                pass

            # Extract estimated cost via get_estimated_cost_for_mlprogram_operation
            estimated_cost: Optional[float] = None
            try:
                if hasattr(plan, 'get_estimated_cost_for_mlprogram_operation'):
                    cost_val = plan.get_estimated_cost_for_mlprogram_operation(op_name)
                    if cost_val is not None:
                        estimated_cost = float(cost_val)
            except Exception:
                pass

            entry = {
                "op_name": op_name,
                "preferred_device": preferred_device,
                "supported_devices": supported_devices,
                "estimated_cost": estimated_cost,
            }
            per_op_placement.append(entry)

            if "NeuralEngine" in preferred_device:
                ane_count += 1

    total_ops = len(per_op_placement)
    ane_placement_rate = ane_count / total_ops if total_ops > 0 else 0.0

    return {
        "available": True,
        "per_op_placement": per_op_placement,
        "ane_placement_rate": ane_placement_rate,
        "total_ops": total_ops,
        "source": "mlcomputeplan",
    }


def predict_placement_from_ops(
    mir_ops: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Predict per-op placement from MIR ops using known op→device mappings.

    This is the offline/host-side counterpart to MLComputePlan harvesting.
    It mirrors the Rust ComputePlanVerifier::predict_proof() logic, using
    the same known op→device mapping table. On non-Apple hosts where
    MLComputePlan is unavailable, this provides a predicted placement
    instead of a plain "unavailable" result.

    The prediction is conservative: only ops well-documented as ANE-friendly
    are predicted for NeuralEngine placement. Unknown ops default to CPU.

    Sprint 57: this closes Issue #5 by giving Linux/non-Apple hosts a
    placement prediction that is stronger than a plain unavailable result.

    Args:
        mir_ops: Optional list of MIR op dicts, each with an "op_type" key.
            If None, returns unavailable.

    Returns:
        Dict with:
          - available: bool — True if prediction succeeded
          - source: str — "offline_prediction" (always)
          - per_op_placement: list of dicts with predicted placement
          - ane_placement_rate: float — predicted ANE placement fraction
          - total_ops: int — number of ops processed
          - prediction_confidence: float — overall confidence (0.0–1.0)
    """
    if mir_ops is None or len(mir_ops) == 0:
        return {
            "available": False,
            "source": "offline_prediction",
            "reason": "No MIR ops provided for placement prediction",
            "per_op_placement": [],
            "ane_placement_rate": 0.0,
            "total_ops": 0,
            "prediction_confidence": 0.0,
        }

    # Known op→device mapping — mirrors Rust ComputePlanVerifier::default_known_placements()
    # Each entry: (op_pattern, predicted_device, confidence)
    KNOWN_PLACEMENTS: List[tuple] = [
        # ANE-friendly ops
        ("linear", "NeuralEngine", 0.85),
        ("matmul", "NeuralEngine", 0.80),
        ("gelu", "NeuralEngine", 0.75),
        ("scaled_dot_product_attention", "NeuralEngine", 0.70),
        ("sdpa", "NeuralEngine", 0.70),
        ("softmax", "NeuralEngine", 0.75),
        ("layer_norm", "NeuralEngine", 0.80),
        ("layernorm", "NeuralEngine", 0.80),
        ("reshape", "NeuralEngine", 0.90),
        ("transpose", "NeuralEngine", 0.90),
        ("relu", "NeuralEngine", 0.75),
        ("rsqrt", "NeuralEngine", 0.70),
        ("reduce_mean", "NeuralEngine", 0.65),
        ("reduce_sum", "NeuralEngine", 0.65),
        ("concat", "NeuralEngine", 0.80),
        ("split", "NeuralEngine", 0.70),
        ("real_div", "NeuralEngine", 0.60),
        ("add", "NeuralEngine", 0.70),
        ("mul", "NeuralEngine", 0.70),
        # CPU-bound ops
        ("embedding", "CPU", 0.90),
        ("gather", "CPU", 0.70),
        ("topk", "CPU", 0.85),
        ("slice_by_index", "CPU", 0.60),
        ("slice_update", "CPU", 0.55),
        ("cast", "CPU", 0.60),
        # State ops — typically CPU
        ("read_state", "CPU", 0.80),
        ("coreml_update_state", "CPU", 0.80),
        ("state_write", "CPU", 0.80),
    ]

    # Build a lookup dict: op_type_lower → (device, confidence)
    placement_lookup: Dict[str, tuple] = {}
    for pattern, device, conf in KNOWN_PLACEMENTS:
        placement_lookup[pattern.lower()] = (device, conf)

    per_op_placement: List[Dict[str, Any]] = []
    total_confidence = 0.0
    ane_count = 0

    for op in mir_ops:
        op_type_raw = op.get("op_type", "unknown")
        op_name = op.get("name", op_type_raw)

        # Normalize op type for lookup: strip "MIL" prefix and lowercase
        op_type_key = op_type_raw.lower()
        if op_type_key.startswith("mil"):
            op_type_key = op_type_key[3:]

        # Look up predicted device
        if op_type_key in placement_lookup:
            device, conf = placement_lookup[op_type_key]
        else:
            # Unknown ops default to CPU with low confidence
            device = "CPU"
            conf = 0.30

        entry = {
            "op_name": op_name,
            "op_type": op_type_raw,
            "preferred_device": device,
            "supported_devices": [device],
            "estimated_cost": None,
            "prediction_confidence": conf,
        }
        per_op_placement.append(entry)
        total_confidence += conf

        if "NeuralEngine" in device:
            ane_count += 1

    total_ops = len(per_op_placement)
    ane_placement_rate = ane_count / total_ops if total_ops > 0 else 0.0
    avg_confidence = total_confidence / total_ops if total_ops > 0 else 0.0

    return {
        "available": True,
        "source": "offline_prediction",
        "per_op_placement": per_op_placement,
        "ane_placement_rate": ane_placement_rate,
        "total_ops": total_ops,
        "prediction_confidence": round(avg_confidence, 4),
    }


def harvest_to_observations(
    harvest_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert harvested compute plan data into knowledge store observation format.

    For each op in the per_op_placement list, creates a SurvivalMatrixEntry
    observation indicating whether the op was placed on the NeuralEngine or not.
    Compute plan observations have confidence 0.9 because the placement data
    is deterministic for a given hardware+OS combination.

    Args:
        harvest_result: The dict returned by harvest_compute_plan().

    Returns:
        List of observation dicts, each with:
          - observation_type: "SurvivalMatrixEntry"
          - op_pattern: str — the op name from the compute plan
          - device_class: str — the preferred device class
          - ane_placed: bool — True if preferred_device is NeuralEngine
          - confidence: float — 0.9 (deterministic for given hardware+OS)
          - evidence_source: "compute_plan"
          - evidence_count: 1
    """
    observations: List[Dict[str, Any]] = []

    if not harvest_result.get("available", False):
        return observations

    per_op_placement = harvest_result.get("per_op_placement", [])
    for op_entry in per_op_placement:
        op_name = op_entry.get("op_name", "unknown")
        preferred_device = op_entry.get("preferred_device", "unknown")
        ane_placed = "NeuralEngine" in preferred_device

        observations.append({
            "observation_type": "SurvivalMatrixEntry",
            "op_pattern": op_name,
            "device_class": preferred_device,
            "ane_placed": ane_placed,
            "confidence": 0.9,
            "evidence_source": "compute_plan",
            "evidence_count": 1,
        })

    return observations
