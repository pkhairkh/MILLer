"""Profiler - Runs Core ML models and captures timing/metrics."""

import coremltools as ct
import numpy as np
import time
from typing import Dict, Any, List, Optional
import json


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
    compute_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    
    model = ct.models.MLModel(
        mlpackage_path,
        compute_units=compute_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE),
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
