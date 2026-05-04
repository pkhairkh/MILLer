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

use ane_ir::ane_layout::clamp_to_valid_palette_bits;
use ane_ir::common::ModelArchitecture;
use ane_ir::sir::{SirGraph, SirOp};

/// Re-export centralized palette bit-width validation from `ane_ir::ane_layout`.
///
/// T-64 (I-38): Previously defined locally in this module. Now centralized
/// in `ane_ir::ane_layout` so that `ane-lab` and `ane-ir` can also use it
/// without depending on `ane-passes`.
///
/// T-118: Also re-exports `validate_palette_bits_for_family` for
/// version-conditional palette bit-width validation.
pub use ane_ir::ane_layout::{validate_palette_bits, validate_palette_bits_for_family, VALID_PALETTE_BITS};

/// T-121: Re-export vector palettization constraint validation from
/// `op_constraints`.
///
/// Vector palettization has three ANEC-enforced constraints:
/// 1. Palettization dimension must be Cout (not Cin, etc.)
/// 2. Zero point is not supported for vector palettized kernels
/// 3. Palette size 256 (8-bit full LUT) is not supported
///
/// Re-exporting from this module makes the validation accessible alongside
/// other palette-related utilities, so consumers don't need to import from
/// the `op_constraints` module directly.
pub use crate::op_constraints::validate_vector_palettization_constraints;

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
/// Delegates to [`ane_ir::ane_layout::clamp_to_valid_palette_bits`].
/// T-64 (I-38): Previously defined locally. Now centralized in `ane-ir`.
fn clamp_to_valid_bits(bits: usize) -> usize {
    clamp_to_valid_palette_bits(bits)
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
/// # Architecture-aware weight classification (T-72 / I-47)
///
/// When `architecture` is provided, the pass uses `ModelArchitecture`
/// pattern methods (`q_proj_pattern()`, `k_proj_pattern()`, etc.) to
/// classify weights as attention vs MLP. When `None`, a Qwen3 default
/// is assumed with a logged warning (backward-compatible behavior).
///
/// # Panics
///
/// This function will not panic — invalid bit-widths are clamped to the
/// nearest valid ANE-supported value and a warning is recorded.
pub fn run_palettize_weights_pass(
    graph: &mut SirGraph,
    config: &PalettizeConfig,
) -> PalettizeResult {
    run_palettize_weights_pass_with_arch(graph, config, None)
}

/// Architecture-aware version of [`run_palettize_weights_pass`].
///
/// T-72 (I-47): Previously, the pass used hardcoded Qwen3/LLaMA name
/// patterns (`q_proj`, `k_proj`, etc.) to classify weights. Other
/// architectures (GPT-2, T5, BART) wouldn't match, getting wrong
/// bit-width assignments. Now uses `ModelArchitecture` for
/// architecture-aware pattern resolution.
pub fn run_palettize_weights_pass_with_arch(
    graph: &mut SirGraph,
    config: &PalettizeConfig,
    architecture: Option<&ModelArchitecture>,
) -> PalettizeResult {
    let mut result = PalettizeResult {
        weights_annotated: 0,
        grouped_lut_applied: 0,
        consts_palettized: 0,
        bits_clamped: 0,
    };

    // T-72: Use architecture-specific patterns for weight classification.
    // When no architecture is specified, default to Qwen3 with a warning.
    let arch = match architecture.cloned() {
        Some(a) => a,
        None => {
            log::warn!(
                "palettize_weights: no architecture specified, defaulting to Qwen3 \
                 weight-name patterns. Pass an explicit architecture to avoid \
                 incorrect weight classification for non-Qwen3 models."
            );
            ModelArchitecture::Qwen3
        }
    };

    let q_pat = arch.q_proj_pattern();
    let k_pat = arch.k_proj_pattern();
    let v_pat = arch.v_proj_pattern();
    let o_pat = arch.o_proj_pattern();
    let gate_pat = arch.gate_proj_pattern();
    let up_pat = arch.up_proj_pattern();
    let down_pat = arch.down_proj_pattern();

    // Annotate LinearProjection nodes with GroupedLut quantization
    for node in &mut graph.nodes {
        match &mut node.op {
            SirOp::LinearProjection { palette_bits, .. } => {
                // T-72 (I-47): Use architecture-specific patterns instead of
                // hardcoded Qwen3 name heuristics. Each architecture defines
                // its own weight naming convention via ModelArchitecture methods.
                let is_q = node.name.contains(q_pat);
                let is_k = node.name.contains(k_pat);
                let is_v = node.name.contains(v_pat);
                let is_o = node.name.contains(o_pat);
                let is_gate = node.name.contains(gate_pat);
                let is_up = node.name.contains(up_pat);
                let is_down = node.name.contains(down_pat);

                let is_attention = is_q || is_k || is_v || is_o
                    || node.name.contains("out_proj")
                    || node.name.contains("qkv");
                let is_qk = is_q || is_k;

                let raw_bits = if is_qk && config.conservative_qk {
                    // Q/K get higher bit-width for stability
                    // NOTE: attention_bits + 2 can produce invalid widths (5, 7).
                    // For example, attention_bits=4 → 6 (valid), but attention_bits=3 → 5 (invalid).
                    // We clamp to the nearest valid ANE-supported bit-width.
                    (config.attention_bits + 2).min(8)
                } else if is_attention {
                    config.attention_bits
                } else if is_gate || is_up || is_down {
                    // Explicitly-identified MLP projections
                    config.mlp_bits
                } else {
                    // Unknown projection type — default to MLP bits.
                    // Log a warning so users know a projection wasn't classified.
                    log::warn!(
                        "Palettize: node '{}' doesn't match any known attention or MLP \
                         pattern for the configured architecture. Defaulting to mlp_bits={}.",
                        node.name, config.mlp_bits
                    );
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

/// T-98 (V-110): Populate quantized conv weight attributes on MILConv nodes.
///
/// After SIR→AIR→MIR lowering, this pass scans MIR graphs for `MILConv` nodes
/// that have palettized weight entries and populates their `kernel_scale`,
/// `kernel_zero_point`, and `kernel_palettized_lut` fields. These fields map
/// to the ANEC `kernel_scale`, `kernel_zero_point`, and `kernel_palettized_LUT`
/// convolution attributes required for quantized/palettized convolution emission.
///
/// # When to call
///
/// Call after lowering to MIR and after weight resolution, when the palettized
/// weight entries are available. If no palettized conv weights exist, this
/// function is a no-op.
///
/// # Palettization metadata
///
/// The `palettized_conv_weights` parameter maps conv node names to their
/// quantization metadata. Each entry provides:
/// - `kernel_scale`: the per-op scale factor for dequantization
/// - `kernel_zero_point`: the zero-point offset (0 for symmetric)
/// - `kernel_palettized_lut`: the name of the LUT weight entry
///
/// # Returns
///
/// The number of MILConv nodes that had their quantization fields populated.
pub fn populate_conv_quantization_fields(
    mir_nodes: &mut [ane_ir::mir::MirNode],
    palettized_conv_weights: &std::collections::HashMap<String, ConvQuantizationInfo>,
) -> usize {
    let mut populated = 0;
    for node in mir_nodes.iter_mut() {
        if let ane_ir::mir::MirOp::MILConv { .. } = node.op {
            // Extract name first to avoid borrow conflicts
            let name = if let ane_ir::mir::MirOp::MILConv { ref name, .. } = node.op {
                name.clone()
            } else {
                continue;
            };

            if let Some(info) = palettized_conv_weights.get(&name) {
                if let ane_ir::mir::MirOp::MILConv {
                    ref mut kernel_scale,
                    ref mut kernel_zero_point,
                    ref mut kernel_palettized_lut,
                    ..
                } = node.op
                {
                    *kernel_scale = Some(info.kernel_scale);
                    *kernel_zero_point = Some(info.kernel_zero_point);
                    *kernel_palettized_lut = Some(info.kernel_palettized_lut.clone());
                    populated += 1;
                }
            }
        }
    }
    populated
}

/// T-98 (V-110): Quantization metadata for a single conv weight tensor.
///
/// Maps to the ANEC `kernel_scale`, `kernel_zero_point`, and
/// `kernel_palettized_LUT` attributes on the convolution operation.
#[derive(Debug, Clone)]
pub struct ConvQuantizationInfo {
    /// Per-op scale factor for quantized/palettized conv weights.
    /// Used to dequantize int4/int8 weights back to floating point.
    pub kernel_scale: f32,
    /// Zero-point offset for quantized conv weights.
    /// Zero indicates symmetric quantization.
    pub kernel_zero_point: i32,
    /// Name of the palettized LUT weight entry for this conv.
    /// References a weight entry in the weight blob that contains
    /// the lookup table for palettized (kmeans) weights.
    pub kernel_palettized_lut: String,
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
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
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
                    name: "model.layers.0.mlp.down_proj.weight".to_string(),
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

    // T-72 tests: architecture-aware weight classification
    #[test]
    fn test_palettize_with_explicit_qwen3_architecture() {
        let mut graph = make_test_graph();
        let mut config = PalettizeConfig::default();
        config.conservative_qk = true;
        config.attention_bits = 4;

        let result = run_palettize_weights_pass_with_arch(
            &mut graph,
            &config,
            Some(&ModelArchitecture::Qwen3),
        );

        assert!(result.grouped_lut_applied >= 2);
        // Q projection should get 6 bits (4+2 conservative)
        for node in &graph.nodes {
            if let SirOp::LinearProjection { palette_bits, .. } = &node.op {
                if node.name.contains(".self_attn.q_proj.weight") {
                    assert_eq!(
                        *palette_bits,
                        Some(6),
                        "Q proj should get 6 bits with conservative_qk"
                    );
                }
            }
        }
    }

    #[test]
    fn test_palettize_with_generic_architecture() {
        // Test that a Generic architecture with GPT-2-like patterns works
        let mut graph = SirGraph {
            nodes: vec![
                SirNode {
                    id: SirNodeId("attn_q_0".to_string()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input_0".to_string()),
                        weight: "q_weight_0".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "transformer.h.0.attn.c_attn.q_proj.weight".to_string(),
                    metadata: SirMetadata {
                        task_origin: TaskOrigin::Synthetic,
                        model_id: None,
                        quality_contract: None,
                        precision_override: None,
                    },
                },
                SirNode {
                    id: SirNodeId("mlp_fc_0".to_string()),
                    op: SirOp::LinearProjection {
                        input: SirNodeId("input_1".to_string()),
                        weight: "mlp_fc_weight_0".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                    name: "transformer.h.0.mlp.c_fc.weight".to_string(),
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
        };

        let gpt2_arch = ModelArchitecture::Generic {
            q_proj_pattern: ".attn.q_proj.".to_string(),
            k_proj_pattern: ".attn.k_proj.".to_string(),
            v_proj_pattern: ".attn.v_proj.".to_string(),
            o_proj_pattern: ".attn.out_proj.".to_string(),
            gate_proj_pattern: ".mlp.gate.".to_string(),
            up_proj_pattern: ".mlp.up.".to_string(),
            down_proj_pattern: ".mlp.down.".to_string(),
        };

        let mut config = PalettizeConfig::default();
        config.conservative_qk = false;
        config.attention_bits = 6;
        config.mlp_bits = 4;

        let result = run_palettize_weights_pass_with_arch(&mut graph, &config, Some(&gpt2_arch));

        assert!(result.grouped_lut_applied >= 2);

        // Q projection should be classified as attention (6 bits)
        for node in &graph.nodes {
            if let SirOp::LinearProjection { palette_bits, .. } = &node.op {
                if node.name.contains(".attn.q_proj.") {
                    assert_eq!(*palette_bits, Some(6), "GPT-2 Q proj should get attention_bits=6");
                }
                // Unrecognized MLP name falls through to default mlp_bits
                if node.name.contains("mlp.c_fc") {
                    assert_eq!(*palette_bits, Some(4), "Unrecognized name defaults to mlp_bits=4");
                }
            }
        }
    }

    // ─── T-98: Conv quantization field population tests ─────────────────

    #[test]
    fn test_t98_populate_conv_quantization_fields() {
        use ane_ir::mir::{MilDtype, MirNode, MirNodeId, MirOp};
        use ane_ir::common::ComputeUnitHint;
        use std::collections::HashMap;

        let mut nodes = vec![
            MirNode {
                id: MirNodeId("conv1".into()),
                op: MirOp::MILConv {
                    name: "conv1".into(),
                    x: MirNodeId("x".into()),
                    weight: MirNodeId("w1".into()),
                    pad_type: "valid".into(),
                    groups: 1,
                    strides: vec![1, 1],
                    pad_amounts: vec![0, 0, 0, 0],
                    dilations: vec![1, 1],
                    kernel_scale: None,
                    kernel_zero_point: None,
                    kernel_palettized_lut: None,
                },
                dtype: MilDtype::Fp16,
                shape: vec![1, 3, 32, 32],
                compute_unit_hint: None,
                air_source: None,
            },
            MirNode {
                id: MirNodeId("conv2".into()),
                op: MirOp::MILConv {
                    name: "conv2".into(),
                    x: MirNodeId("x2".into()),
                    weight: MirNodeId("w2".into()),
                    pad_type: "same".into(),
                    groups: 2,
                    strides: vec![2, 2],
                    pad_amounts: vec![1, 1, 1, 1],
                    dilations: vec![1, 1],
                    kernel_scale: None,
                    kernel_zero_point: None,
                    kernel_palettized_lut: None,
                },
                dtype: MilDtype::Fp16,
                shape: vec![1, 64, 16, 16],
                compute_unit_hint: None,
                air_source: None,
            },
        ];

        let mut palettized = HashMap::new();
        palettized.insert(
            "conv1".to_string(),
            ConvQuantizationInfo {
                kernel_scale: 0.0078,
                kernel_zero_point: 0,
                kernel_palettized_lut: "conv1_weight_lut_4bit".to_string(),
            },
        );

        let populated = populate_conv_quantization_fields(&mut nodes, &palettized);
        assert_eq!(populated, 1, "Only conv1 should be populated");

        // Verify conv1 has quantization fields set
        if let MirOp::MILConv { name, kernel_scale, kernel_zero_point, kernel_palettized_lut, .. } = &nodes[0].op {
            assert_eq!(name, "conv1");
            assert_eq!(*kernel_scale, Some(0.0078));
            assert_eq!(*kernel_zero_point, Some(0));
            assert_eq!(kernel_palettized_lut.as_deref(), Some("conv1_weight_lut_4bit"));
        } else {
            panic!("Expected MILConv");
        }

        // Verify conv2 is untouched
        if let MirOp::MILConv { kernel_scale, kernel_zero_point, kernel_palettized_lut, .. } = &nodes[1].op {
            assert_eq!(*kernel_scale, None, "conv2 should not be populated");
            assert_eq!(*kernel_zero_point, None);
            assert_eq!(*kernel_palettized_lut, None);
        }
    }

    #[test]
    fn test_t98_populate_conv_quantization_empty_map() {
        use ane_ir::mir::{MilDtype, MirNode, MirNodeId, MirOp};
        use std::collections::HashMap;

        let mut nodes = vec![
            MirNode {
                id: MirNodeId("conv1".into()),
                op: MirOp::MILConv {
                    name: "conv1".into(),
                    x: MirNodeId("x".into()),
                    weight: MirNodeId("w1".into()),
                    pad_type: "valid".into(),
                    groups: 1,
                    strides: vec![1],
                    pad_amounts: vec![0],
                    dilations: vec![1],
                    kernel_scale: None,
                    kernel_zero_point: None,
                    kernel_palettized_lut: None,
                },
                dtype: MilDtype::Fp16,
                shape: vec![1, 3, 32, 32],
                compute_unit_hint: None,
                air_source: None,
            },
        ];

        let palettized: HashMap<String, ConvQuantizationInfo> = HashMap::new();
        let populated = populate_conv_quantization_fields(&mut nodes, &palettized);
        assert_eq!(populated, 0, "No convs should be populated with empty map");
    }

    #[test]
    fn test_t98_conv_quantization_info_fields() {
        let info = ConvQuantizationInfo {
            kernel_scale: 0.0156,
            kernel_zero_point: -128,
            kernel_palettized_lut: "conv_weight_lut_8bit".to_string(),
        };
        assert!((info.kernel_scale - 0.0156).abs() < f32::EPSILON);
        assert_eq!(info.kernel_zero_point, -128);
        assert_eq!(info.kernel_palettized_lut, "conv_weight_lut_8bit");
    }
}
