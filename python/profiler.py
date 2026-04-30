"""Profiler - Runs Core ML models and captures timing/metrics."""

import numpy as np
import time
from typing import Dict, Any, List, Optional
import json

from common import _ensure_coremltools, COMPUTE_MAP


def generate_inputs(
    mlpackage_path: str,
    compute_units: str = "CPU_AND_NE",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate random inputs for a Core ML model based on its spec.

    This is the single place where profiling input generation happens
    (W-22 fix). Previously, bridge.py's handle_profile duplicated this
    logic inline.

    Args:
        mlpackage_path: Path to the .mlpackage directory.
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        seed: Random seed for reproducible input generation.

    Returns:
        Dict of input name -> numpy array (fp16).

    Raises:
        ImportError: If coremltools is not installed.
    """
    ct = _ensure_coremltools()
    compute_unit = COMPUTE_MAP.get(compute_units, ct.ComputeUnit.CPU_AND_NE)
    model = ct.models.MLModel(mlpackage_path, compute_units=compute_unit)

    np.random.seed(seed)
    spec = model.get_spec()
    desc = spec.description
    inputs = {}
    for inp in desc.input:
        shape = list(inp.type.multiArrayType.shape) if hasattr(inp.type, 'multiArrayType') else [1, 64]
        inputs[inp.name] = np.random.randn(*shape).astype(np.float16)
    return inputs


def profile_model(
    mlpackage_path: str,
    inputs: Dict[str, np.ndarray],
    compute_units: str = "CPU_AND_NE",
    warmup_iterations: int = 5,
    measured_iterations: int = 20,
    stateful: bool = False,
) -> Dict[str, Any]:
    """Profile a Core ML model and capture timing metrics.
    
    Args:
        mlpackage_path: Path to the .mlpackage directory.
        inputs: Dict of input name -> numpy array.
        compute_units: "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY", "ALL".
        warmup_iterations: Number of warmup predict() calls.
        measured_iterations: Number of measured predict() calls.
        stateful: Whether the model uses stateful prediction.
    
    Returns:
        Dict with latency statistics and output snapshots.
    """
    ct = _ensure_coremltools()
    compute_unit = COMPUTE_MAP.get(compute_units, ct.ComputeUnit.CPU_AND_NE)
    
    model = ct.models.MLModel(
        mlpackage_path,
        compute_units=compute_unit,
    )
    
    state = None
    if stateful:
        state = model.make_state()
    
    # Warmup
    for _ in range(warmup_iterations):
        if stateful:
            model.predict(inputs, state=state)
        else:
            model.predict(inputs)
    
    # Measured runs
    latencies = []
    outputs_list = []
    
    for _ in range(measured_iterations):
        start = time.perf_counter_ns()
        if stateful:
            result = model.predict(inputs, state=state)
        else:
            result = model.predict(inputs)
        end = time.perf_counter_ns()
        
        latencies.append(end - start)
        outputs_list.append(result)
    
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    
    return {
        "latency": {
            "median_ns": latencies_sorted[n // 2],
            "p90_ns": latencies_sorted[int(n * 0.9)],
            "p99_ns": latencies_sorted[int(n * 0.99)],
            "min_ns": latencies_sorted[0],
            "max_ns": latencies_sorted[-1],
            "iterations": n,
        },
        "last_predict_duration_ns": model.last_predict_duration_in_nano_seconds,
        "output_keys": list(outputs_list[-1].keys()) if outputs_list else [],
    }
