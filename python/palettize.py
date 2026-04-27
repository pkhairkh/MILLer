"""Palettization - Applies post-training palettization to MLModel weights.

Uses coremltools.optimize.coreml palettization APIs.
Compatible with coremltools 9.0 (op_name_configs, CompressionGranularity).
"""

import coremltools as ct
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpPalettizerConfig,
    palettize_weights,
)
from typing import Dict, Any, List, Optional


def apply_palettization(
    mlmodel: ct.models.MLModel,
    palettization_specs: List[Dict[str, Any]],
) -> ct.models.MLModel:
    """Apply palettization to an MLModel per the given specifications.

    Args:
        mlmodel: The MLModel to palettize.
        palettization_specs: List of palettization specifications from the Rust side.
            Each spec contains: weight_name, mode, nbits, granularity, group_size, channel_axis.

    Returns:
        The palettized MLModel.
    """
    op_name_configs = {}

    for spec in palettization_specs:
        weight_name = spec["weight_name"]
        op_config = OpPalettizerConfig(
            mode=spec.get("mode", "kmeans"),
            nbits=spec.get("nbits", 4),
            granularity=spec.get("granularity", "per_grouped_channel"),
            group_size=spec.get("group_size", 32),
            channel_axis=spec.get("channel_axis", 1),
        )
        op_name_configs[weight_name] = op_config

    # coremltools 9.0 uses op_name_configs (not op_configs)
    config = OptimizationConfig(
        global_config=None,
        op_type_configs=None,
        op_name_configs=op_name_configs,
    )

    return palettize_weights(mlmodel, config)
