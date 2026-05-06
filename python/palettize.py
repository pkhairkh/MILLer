"""Palettization - Applies post-training palettization to MLModel weights.

Uses coremltools.optimize.coreml palettization APIs.
Compatible with coremltools 9.0 (op_name_configs, CompressionGranularity).
"""

from typing import Any, Dict, List

from common import _ensure_coremltools

# Palettize-specific optimization imports (cached separately from ct)
_OptimizationConfig = None
_OpPalettizerConfig = None
_palettize_weights = None


def _ensure_palettize_deps():
    """Ensure coremltools and palettization dependencies are loaded.

    Uses common._ensure_coremltools() for the core coremltools import,
    then lazily imports the optimization sub-modules on first use.

    Raises:
        ImportError: If coremltools (or its optimize sub-module) is not installed.
    """
    global _OptimizationConfig, _OpPalettizerConfig, _palettize_weights
    if _OptimizationConfig is None:
        _ensure_coremltools()  # raises ImportError if coremltools missing
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )
        _OptimizationConfig = OptimizationConfig
        _OpPalettizerConfig = OpPalettizerConfig
        _palettize_weights = palettize_weights


def apply_palettization(
    mlmodel: Any,
    palettization_specs: List[Dict[str, Any]],
) -> Any:
    """Apply palettization to an MLModel per the given specifications.

    Args:
        mlmodel: The MLModel to palettize.
        palettization_specs: List of palettization specifications from the Rust side.
            Each spec contains: weight_name, mode, nbits, granularity, group_size, channel_axis.

    Returns:
        The palettized MLModel.
    """
    _ensure_palettize_deps()
    op_name_configs = {}

    for spec in palettization_specs:
        weight_name = spec["weight_name"]
        op_config = _OpPalettizerConfig(
            mode=spec.get("mode", "kmeans"),
            nbits=spec.get("nbits", 4),
            granularity=spec.get("granularity", "per_grouped_channel"),
            group_size=spec.get("group_size", 32),
            channel_axis=spec.get("channel_axis", 1),
        )
        op_name_configs[weight_name] = op_config

    # coremltools 9.0 uses op_name_configs (not op_configs)
    config = _OptimizationConfig(
        global_config=None,
        op_type_configs=None,
        op_name_configs=op_name_configs,
    )

    return _palettize_weights(mlmodel, config)
