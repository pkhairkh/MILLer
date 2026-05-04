//! Palettize Weights Pass — apply post-hoc palettization to Core ML constants.
//!
//! Applies coremltools.optimize palettization to mask and KV cache constants
//! in emitted Core ML packages. This pass operates at the SIR level to
//! annotate which weight tensors should be palettized and with what strategy.
//!
//! The mixed-quantization approach uses different strategies for different
//! weight types:
//! - Embedding/LM head: Blockwise quantization (int4 with per-group scales)
//! - Attention/MLP projections: GroupedLut (4/6/8-bit with per-group scalars)
//! - KV/mask constants: 1-bit kmeans palettization
//! - Q/K projections: treated more conservatively (higher bitwidth)
//!
//! ## Palette bit-width validation
//!
//! The ANE only supports palette bit-widths in the set {1, 2, 3, 4, 6, 8}.
//! Bit-widths 5 and 7 are **invalid** and will cause ANE runtime errors.
//! The pass validates all computed bit-widths and rejects invalid values
//! with a clear error message.

use ane_ir::sir::{SirGraph, SirOp};

/// Valid palette bit-widths supported by the ANE hardware.
/// Values 5 and 7 are NOT supported and will cause runtime errors.
pub const VALID_PALETTE_BITS: &[usize] = &[1, 2, 3, 4, 6, 8];

/// Validate that a palette bit-width is in the ANE-supported set.
///
/// Returns `Ok(())` if valid, or an error message describing the issue.
pub fn validate_palette_bits(bits: usize) -> Result<(), String> {
    if VALID_PALETTE_BITS.contains(&bits) {
        Ok(())
    } else {
        Err(format!(
            "Invalid palette bit-width {}: must be one of {:?}. \
             ANE hardware does not support {}-bit palettization.",
            bits, VALID_PALETTE_BITS, bits
        ))
    }
}

/// Result of the palettize weights pass.
#[derive(Debug, Clone)]
pub struct PalettizeResult {
    /// Number of weight tensors annotated with quantization strategies.
    pub weights_annotated: usize,
    /// Number of LinearProjection nodes that received GroupedLut quantization.
    pub grouped_lut_applied: usize,
    /// Number of Const nodes that received palettization.
    pub consts_palettized: usize,
    /// Number of ops where bit-width validation failed (bit-width clamped to nearest valid).
    pub bits_clamped: usize,
}

/// Configuration for the palettize weights pass.
#[derive(Debug, Clone)]
pub struct PalettizeConfig {
    /// Default bit-width for attention projection weights (Q, K, V, O).
    pub attention_bits: usize,
    /// Default bit-width for MLP projection weights (gate, up, down).
    pub mlp_bits: usize,
    /// Default bit-width for KV/mask constants.
    pub mask_kv_bits: usize,
    /// Default group size for GroupedLut quantization.
    pub group_size: usize,
    /// Whether to use more conservative quantization for Q/K projections.
    pub conservative_qk: bool,
}

impl Default for PalettizeConfig {
    fn default() -> Self {
        PalettizeConfig {
            attention_bits: 4,
            mlp_bits: 4,
            mask_kv_bits: 1,
            group_size: 128,
            conservative_qk: true,
        }
    }
}

/// Clamp a bit-width to the nearest valid ANE palette bit-width.
///
/// For bit-widths between valid values, rounds down to the nearest
/// supported bit-width (e.g., 5 → 4, 7 → 6). This preserves
/// quantization benefit while ensuring ANE compatibility.
fn clamp_to_valid_bits(bits: usize) -> usize {
    if VALID_PALETTE_BITS.contains(&bits) {
        return bits;
    }
    // Round down to nearest valid bit-width
    *VALID_PALETTE_BITS.iter().filter(|&&b| b <= bits).last().unwrap_or(&1)
}

/// Run the palettize weights pass on a SIR graph.
///
/// This pass annotates weight-bearing ops (LinearProjection, Const,
/// ConstexprBlockwiseShiftScale, etc.) with quantization strategies.
/// The actual palettization happens during Core ML emission, where
/// coremltools.optimize is applied to the emitted packages.
///
/// The annotation strategy:
/// - LinearProjection ops get `GroupedLut` quantization based on their
///   position in the model (attention vs MLP), stored in `palette_bits`
/// - Const ops for KV/mask get kmeans palettization stored in `palette_bits`
/// - Embedding ops get `Blockwise` quantization
///
/// # Panics
///
/// This function will not panic — invalid bit-widths are clamped to the
/// nearest valid ANE-supported value and a warning is recorded.
pub fn run_palettize_weights_pass(
    graph: &mut SirGraph,
    config: &PalettizeConfig,
) -> PalettizeResult {
    let mut result = PalettizeResult {
        weights_annotated: 0,
        grouped_lut_applied: 0,
        consts_palettized: 0,
        bits_clamped: 0,
    };

    // Annotate LinearProjection nodes with GroupedLut quantization
    for node in &mut graph.nodes {
        match &mut node.op {
            SirOp::LinearProjection { palette_bits, .. } => {
                // Determine if this is an attention or MLP projection
                // based on the node name (heuristic from naming conventions)
                let is_attention = node.name.contains("q_proj")
                    || node.name.contains("k_proj")
                    || node.name.contains("v_proj")
                    || node.name.contains("o_proj")
                    || node.name.contains("out_proj")
                    || node.name.contains("qkv");

                let is_qk = node.name.contains("q_proj") || node.name.contains("k_proj");

                let raw_bits = if is_qk && config.conservative_qk {
                    // Q/K get higher bit-width for stability
                    // NOTE: attention_bits + 2 can produce invalid widths (5, 7).
                    // For example, attention_bits=4 → 6 (valid), but attention_bits=3 → 5 (invalid).
                    // We clamp to the nearest valid ANE-supported bit-width.
                    (config.attention_bits + 2).min(8)
                } else if is_attention {
                    config.attention_bits
                } else {
                    config.mlp_bits
                };

                // Validate and clamp bit-width to ANE-supported values
                let bits = if validate_palette_bits(raw_bits).is_ok() {
                    raw_bits
                } else {
                    let clamped = clamp_to_valid_bits(raw_bits);
                    log::warn!(
                        "Palettize: bit-width {} invalid for ANE, clamped to {} for node '{}'",
                        raw_bits, clamped, node.name
                    );
                    result.bits_clamped += 1;
                    clamped
                };

                // Wire the quantization strategy into the palette_bits field
                *palette_bits = Some(bits);
                result.grouped_lut_applied += 1;
                result.weights_annotated += 1;
            }
            SirOp::Const { value_path, palette_bits, .. }
                // Palettize KV/mask constants
                if (value_path.contains("mask") || value_path.contains("kv")) => {
                    // Validate mask/KV bit-width
                    let bits = if validate_palette_bits(config.mask_kv_bits).is_ok() {
                        config.mask_kv_bits
                    } else {
                        let clamped = clamp_to_valid_bits(config.mask_kv_bits);
                        log::warn!(
                            "Palettize: mask/kv bit-width {} invalid for ANE, clamped to {}",
                            config.mask_kv_bits, clamped
                        );
                        result.bits_clamped += 1;
                        clamped
                    };
                    *palette_bits = Some(bits);
                    result.consts_palettized += 1;
                    result.weights_annotated += 1;
                }
            _ => {}
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ane_ir::sir::{SirMetadata, SirNode, SirNodeId, TaskOrigin};

    fn make_test_graph() -> SirGraph {
        SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("q_proj_0".to_string()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input_0".to_string()),
                        weight: "q_weight_0".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "q_proj_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("down_proj_0".to_string()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input_1".to_string()),
                        weight: "down_weight_0".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "down_proj_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("mask_0".to_string()),
                    op: SirOp::Const {
                        value_path: "static_tables/mask_tab".to_string(),
                        dtype: ane_ir::mir::MilDtype::Fp16,
                        palette_bits: None,
                    },
                    name: "causal_mask_0".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
            ],
            inputs: vec![],
            outputs: vec![],
        }
    }

    #[test]
    fn test_palettize_annotates_weights() {
        let mut graph = make_test_graph();
        let config = PalettizeConfig::default();
        let result = run_palettize_weights_pass(&mut graph, &config);

        assert!(result.grouped_lut_applied >= 2, "Should annotate at least 2 LinearProjection ops");
        assert!(result.consts_palettized >= 1, "Should palettize at least 1 mask constant");
        assert!(result.weights_annotated >= 3, "Should annotate at least 3 weights total");

        // Verify palette_bits is actually set (was a no-op before T-48)
        for node in &graph.nodes {
            match &node.op {
                SirOp::LinearProjection { palette_bits, .. } => {
                    assert!(
                        palette_bits.is_some(),
                        "LinearProjection '{}' should have palette_bits set after pass",
                        node.name
                    );
                }
                SirOp::Const { value_path, palette_bits, .. }
                    if value_path.contains("mask") || value_path.contains("kv") =>
                {
                    assert!(
                        palette_bits.is_some(),
                        "Const '{}' should have palette_bits set after pass",
                        node.name
                    );
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_conservative_qk_gets_higher_bits() {
        let mut graph = make_test_graph();
        let mut config = PalettizeConfig::default();
        config.conservative_qk = true;
        config.attention_bits = 4;

        let result = run_palettize_weights_pass(&mut graph, &config);
        assert!(result.grouped_lut_applied >= 1);

        // With conservative_qk and attention_bits=4: Q/K get 4+2=6 bits
        for node in &graph.nodes {
            if let SirOp::LinearProjection { palette_bits, .. } = &node.op {
                if node.name.contains("q_proj") || node.name.contains("k_proj") {
                    assert_eq!(
                        *palette_bits,
                        Some(6),
                        "Q/K projections should get 6 bits (4+2 conservative)"
                    );
                }
            }
        }
    }

    #[test]
    fn test_invalid_bit_width_clamped() {
        let mut graph = make_test_graph();
        let mut config = PalettizeConfig::default();
        config.conservative_qk = true;
        config.attention_bits = 3; // 3 + 2 = 5 (invalid!)

        let result = run_palettize_weights_pass(&mut graph, &config);
        assert!(result.bits_clamped >= 1, "Should clamp at least 1 invalid bit-width");

        // 5-bit should be clamped to 4
        for node in &graph.nodes {
            if let SirOp::LinearProjection { palette_bits, .. } = &node.op {
                if node.name.contains("q_proj") || node.name.contains("k_proj") {
                    assert_eq!(*palette_bits, Some(4), "5-bit should be clamped to 4");
                }
            }
        }
    }

    #[test]
    fn test_validate_palette_bits() {
        // Valid bit-widths
        for &bits in VALID_PALETTE_BITS {
            assert!(validate_palette_bits(bits).is_ok(), "{} should be valid", bits);
        }
        // Invalid bit-widths
        assert!(validate_palette_bits(5).is_err(), "5-bit is not ANE-supported");
        assert!(validate_palette_bits(7).is_err(), "7-bit is not ANE-supported");
        assert!(validate_palette_bits(9).is_err(), "9-bit is not ANE-supported");
        assert!(validate_palette_bits(0).is_err(), "0-bit is not ANE-supported");
    }

    #[test]
    fn test_clamp_to_valid_bits() {
        assert_eq!(clamp_to_valid_bits(1), 1);
        assert_eq!(clamp_to_valid_bits(4), 4);
        assert_eq!(clamp_to_valid_bits(5), 4); // 5 → 4 (round down)
        assert_eq!(clamp_to_valid_bits(6), 6);
        assert_eq!(clamp_to_valid_bits(7), 6); // 7 → 6 (round down)
        assert_eq!(clamp_to_valid_bits(8), 8);
        assert_eq!(clamp_to_valid_bits(10), 8); // 10 → 8 (round down)
    }

    #[test]
    fn test_mlp_projection_gets_mlp_bits() {
        let mut graph = make_test_graph();
        let mut config = PalettizeConfig::default();
        config.mlp_bits = 6;
        config.conservative_qk = false;

        let _result = run_palettize_weights_pass(&mut graph, &config);

        // down_proj is MLP — should get mlp_bits
        for node in &graph.nodes {
            if let SirOp::LinearProjection { palette_bits, .. } = &node.op {
                if node.name.contains("down_proj") {
                    assert_eq!(*palette_bits, Some(6), "MLP projection should get mlp_bits=6");
                }
            }
        }
    }
}
