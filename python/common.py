"""Common constants and helpers shared across MILLer Python modules.

Centralizes:
  - COMPUTE_MAP: string → ct.ComputeUnit mapping (populated after _ensure_coremltools())
  - _error_result(): standardized error result dict
  - _ensure_coremltools(): lazy coremltools import with explicit ImportError

Fixes:
  - W-17: compute_map dict was duplicated 5× across bridge.py, profiler.py,
    compute_plan.py (2×), and converter.py
  - W-18: _error_result() was defined independently in bridge.py and
    mil_emitter.py with identical shapes
  - W-19: per-module mutable global state (ct = None; def _ensure_coremltools())
    duplicated across converter.py, profiler.py, compute_plan.py, palettize.py
  - W-26: modules silently swallowed ImportError — _import_coremltools()
    returned (None, None, None, None) on failure; now raises explicitly
"""

# --- Lazy coremltools import (W-19, W-26) ---

_ct = None
COMPUTE_MAP = None


def _ensure_coremltools():
    """Import and cache coremltools module.

    Unlike the previous per-module patterns that silently returned None on
    ImportError, this raises ImportError explicitly so callers don't silently
    proceed with None values (W-26).

    Also populates COMPUTE_MAP on first successful import (W-17).

    Returns:
        The coremltools module.

    Raises:
        ImportError: If coremltools is not installed.
    """
    global _ct, COMPUTE_MAP
    if _ct is None:
        try:
            import coremltools
        except ImportError as e:
            raise ImportError(
                f"coremltools is required but not installed: {e}"
            ) from e
        _ct = coremltools
        COMPUTE_MAP = {
            "CPU_AND_NE": _ct.ComputeUnit.CPU_AND_NE,
            "CPU_AND_GPU": _ct.ComputeUnit.CPU_AND_GPU,
            "CPU_ONLY": _ct.ComputeUnit.CPU_ONLY,
            "ALL": _ct.ComputeUnit.ALL,
        }
    return _ct


def _error_result(message: str) -> dict:
    """Standardized error result dict for bridge/subprocess communication.

    This is the canonical error result format (W-18). Previously defined
    independently in bridge.py and mil_emitter.py with identical shapes.
    The bridge.py version is used as canonical.

    Args:
        message: Error description.

    Returns:
        Dict with status="error" and all standard result fields set to None/empty.
    """
    return {
        "status": "error",
        "error_message": message,
        "output_path": None,
        "coremltools_version": None,
        "content_hash": None,
        "package_files": [],
        "compute_plan": None,
        "function_descriptors": [],
        "metadata": {},
    }
