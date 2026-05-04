//! Staticize pass — **REMOVED FROM PIPELINE** (T-107).
//!
//! This pass was originally intended to resolve dynamic constructs in SIR
//! into static equivalents based on profiling knowledge and shape inference:
//! - Replace symbolic dimensions with concrete values from the task spec
//! - Replace runtime-computed indices with static lookup tables
//! - Resolve variable-length sequences to fixed lengths
//! - Record staticization decisions in SIR metadata
//!
//! However, none of these features were ever implemented. The pass was a
//! pure pass-through (`Ok(input)`) that consumed a pipeline step while doing
//! nothing, wasting developer trust and obscuring the actual pipeline.
//!
//! **Removal rationale** (T-107): A phantom pass that claims capabilities it
//! doesn't have is worse than no pass at all — it misleads developers into
//! thinking dynamic SIR constructs are being resolved when they are not.
//! The pass has been removed from the compile pipeline in `main.rs`. If
//! staticization is needed in the future, it should be implemented as a new
//! pass with clear scope and tests before being wired into the pipeline.
//!
//! The module and its tests are preserved as a historical reference and to
//! serve as a scaffold for a future implementation if one becomes necessary.

use ane_ir::sir::SirGraph;
use anyhow::Result;

/// Staticize pass — **DEPRECATED, removed from pipeline** (T-107).
///
/// This struct is preserved for backward compatibility and as a scaffold
/// for future implementation. It is NOT wired into the compile pipeline.
/// See module-level documentation for removal rationale.
#[deprecated(
    since = "0.1.0",
    note = "StaticizePass was a phantom no-op pass and has been removed from the pipeline. \
           See module docs for rationale. If staticization is needed, implement a new pass."
)]
pub struct StaticizePass {
    // No configuration needed — pass is a no-op
}

impl Default for StaticizePass {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticizePass {
    pub fn new() -> Self {
        Self {}
    }

    /// Run the staticize pass — **DEPRECATED, no-op** (T-107).
    ///
    /// This method returns the input graph unchanged. It is preserved for
    /// backward compatibility but is no longer called from the pipeline.
    /// See module-level documentation for the removal rationale.
    #[allow(deprecated)]
    pub fn run(&self, input: SirGraph) -> Result<SirGraph> {
        // Pass-through: this pass was never implemented. Removed from pipeline
        // in T-107 because a phantom no-op pass misleads developers.
        Ok(input)
    }
}

#[cfg(test)]
#[allow(deprecated)] // T-107: StaticizePass is deprecated but tests preserve behavior
mod tests {
    use super::*;
    use ane_ir::common::MilDtype;
    use ane_ir::sir::{QualityContract, SirMetadata, SirNode, SirNodeId, SirOp, TaskOrigin};

    // ─── Helpers ──────────────────────────────────────────────────────

    fn make_metadata() -> SirMetadata {
        SirMetadata {
            task_origin: TaskOrigin::Synthetic,
            model_id: None,
            quality_contract: None,
            precision_override: None,
        }
    }

    fn make_node(id: &str, op: SirOp) -> SirNode {
        SirNode {
            id: SirNodeId(id.to_string()),
            op,
            name: id.to_string(),
            metadata: make_metadata(),
        }
    }

    fn make_metadata_with(origin: TaskOrigin, model_id: Option<String>) -> SirMetadata {
        SirMetadata {
            task_origin: origin,
            model_id,
            quality_contract: None,
            precision_override: None,
        }
    }

    /// Assert that two SirGraphs are structurally identical.
    /// Since SirOp does not derive PartialEq, we compare node IDs,
    /// node count, graph inputs/outputs, and spot-check op variants.
    fn assert_graphs_identical(left: &SirGraph, right: &SirGraph) {
        assert_eq!(
            left.nodes.len(),
            right.nodes.len(),
            "Node count mismatch: left={}, right={}",
            left.nodes.len(),
            right.nodes.len()
        );
        assert_eq!(left.inputs, right.inputs, "Graph inputs differ");
        assert_eq!(left.outputs, right.outputs, "Graph outputs differ");

        for (l, r) in left.nodes.iter().zip(right.nodes.iter()) {
            assert_eq!(l.id, r.id, "Node ID mismatch: {:?} vs {:?}", l.id, r.id);
            assert_eq!(l.name, r.name, "Node name mismatch for {:?}", l.id);
            assert_eq!(
                format!("{:?}", l.op),
                format!("{:?}", r.op),
                "SirOp mismatch for node {:?}: {:?} vs {:?}",
                l.id,
                l.op,
                r.op
            );
            assert_eq!(
                format!("{:?}", l.metadata),
                format!("{:?}", r.metadata),
                "SirMetadata mismatch for node {:?}",
                l.id
            );
        }
    }

    // ─── 1. Empty / Minimal Graphs ────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let graph = SirGraph { nodes: vec![], inputs: vec![], outputs: vec![] };

        let pass = StaticizePass::new();
        let result = pass.run(graph).unwrap();

        assert_eq!(result.nodes.len(), 0);
        assert_eq!(result.inputs.len(), 0);
        assert_eq!(result.outputs.len(), 0);
    }

    #[test]
    fn test_empty_graph_default_new_equivalent() {
        // StaticizePass::new() and StaticizePass::default() should behave identically.
        let graph = SirGraph { nodes: vec![], inputs: vec![], outputs: vec![] };

        let r1 = StaticizePass::new().run(graph.clone()).unwrap();
        let r2 = StaticizePass::default().run(graph).unwrap();
        assert_graphs_identical(&r1, &r2);
    }

    // ─── 2. Single-Node Graphs ────────────────────────────────────────

    #[test]
    fn test_single_const_node() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "const_0",
                SirOp::Const {
                    value_path: "weights/embedding.bin".to_string(),
                    dtype: MilDtype::Fp16,
                    palette_bits: None,
                },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("const_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_single_fill_node() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "fill_0",
                SirOp::Fill { shape: vec![1, 512, 896], value: 0.0, dtype: MilDtype::Fp16 },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("fill_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_single_identity_placeholder() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "input_0",
                SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
            )],
            inputs: vec![SirNodeId("input_0".to_string())],
            outputs: vec![SirNodeId("input_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 3. Linear Projection (Core Vertical Slice) ──────────────────

    #[test]
    fn test_linear_projection_chain() {
        // The primary use case: input → LinearProjection → output
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "linear_0",
                    SirOp::LinearProjection {
                        input: SirNodeId("input".to_string()),
                        weight: "model.layers.0.self_attn.q_proj.weight".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("linear_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_linear_projection_with_bias() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "linear_0",
                    SirOp::LinearProjection {
                        input: SirNodeId("input".to_string()),
                        weight: "lm_head.weight".to_string(),
                        bias: Some("lm_head.bias".to_string()),
                        palette_bits: None,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("linear_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 4. RMSNorm + RoPE (Common Attention Building Blocks) ─────────

    #[test]
    fn test_rms_norm_node() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "rms_0",
                    SirOp::RMSNorm {
                        input: SirNodeId("input".to_string()),
                        weight: "model.layers.0.input_layernorm.weight".to_string(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("rms_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_rope_transform_node() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "rope_0",
                    SirOp::RoPETransform {
                        input: SirNodeId("input".to_string()),
                        tables: "rope_tables_shared".to_string(),
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("rope_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 5. DecodeStep (Full Attention Step) ──────────────────────────

    #[test]
    fn test_decode_step_minimal() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "token",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "decode_0",
                    SirOp::DecodeStep {
                        token: SirNodeId("token".to_string()),
                        state_map: vec!["k_cache_0".to_string(), "v_cache_0".to_string()],
                        q_weight: Some("q_proj.weight".to_string()),
                        k_weight: Some("k_proj.weight".to_string()),
                        v_weight: Some("v_proj.weight".to_string()),
                        out_weight: Some("o_proj.weight".to_string()),
                        rope_tables: Some("rope_tables".to_string()),
                        position: Some(SirNodeId("pos_0".to_string())),
                        q_norm_weight: None,
                        k_norm_weight: None,
                        norm_epsilon: 1e-6,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: None,
                    },
                ),
            ],
            inputs: vec![SirNodeId("token".to_string())],
            outputs: vec![SirNodeId("decode_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_decode_step_with_qk_norm() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "token",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "decode_0",
                    SirOp::DecodeStep {
                        token: SirNodeId("token".to_string()),
                        state_map: vec!["k_cache_0".to_string(), "v_cache_0".to_string()],
                        q_weight: Some("q_proj.weight".to_string()),
                        k_weight: Some("k_proj.weight".to_string()),
                        v_weight: Some("v_proj.weight".to_string()),
                        out_weight: Some("o_proj.weight".to_string()),
                        rope_tables: None,
                        position: None,
                        q_norm_weight: Some("q_norm.weight".to_string()),
                        k_norm_weight: Some("k_norm.weight".to_string()),
                        norm_epsilon: 1e-5,
                        qk_norm_type: "rms".to_string(),
                        mask_ref: Some("causal_mask".to_string()),
                    },
                ),
            ],
            inputs: vec![SirNodeId("token".to_string())],
            outputs: vec![SirNodeId("decode_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 6. SDPA and AttentionBlock ───────────────────────────────────

    #[test]
    fn test_sdpa_without_mask() {
        let graph = SirGraph {
            nodes: vec![
                make_node("q", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("k", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("v", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "sdpa",
                    SirOp::ScaledDotProductAttention {
                        query: SirNodeId("q".to_string()),
                        key: SirNodeId("k".to_string()),
                        value: SirNodeId("v".to_string()),
                        attention_mask: None,
                        scale: Some(0.0884),
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("q".to_string()),
                SirNodeId("k".to_string()),
                SirNodeId("v".to_string()),
            ],
            outputs: vec![SirNodeId("sdpa".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_sdpa_with_mask_and_scale() {
        let graph = SirGraph {
            nodes: vec![
                make_node("q", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("k", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("v", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "mask",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "sdpa",
                    SirOp::ScaledDotProductAttention {
                        query: SirNodeId("q".to_string()),
                        key: SirNodeId("k".to_string()),
                        value: SirNodeId("v".to_string()),
                        attention_mask: Some(SirNodeId("mask".to_string())),
                        scale: Some(0.125),
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("q".to_string()),
                SirNodeId("k".to_string()),
                SirNodeId("v".to_string()),
                SirNodeId("mask".to_string()),
            ],
            outputs: vec![SirNodeId("sdpa".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_attention_block() {
        let graph = SirGraph {
            nodes: vec![
                make_node("q", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("k", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("v", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "attn",
                    SirOp::AttentionBlock {
                        q: SirNodeId("q".to_string()),
                        k: SirNodeId("k".to_string()),
                        v: SirNodeId("v".to_string()),
                        mask: None,
                        rope: Some(SirNodeId("rope_ref".to_string())),
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("q".to_string()),
                SirNodeId("k".to_string()),
                SirNodeId("v".to_string()),
            ],
            outputs: vec![SirNodeId("attn".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 7. State Operations (KV Cache) ──────────────────────────────

    #[test]
    fn test_state_read() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "k_cache",
                SirOp::StateRead {
                    state_id: "kv_cache_k_0".to_string(),
                    offset: 0,
                    shape: vec![1, 14, 512, 64],
                },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("k_cache".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_state_write() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "write_k",
                    SirOp::StateWrite {
                        state_id: "kv_cache_k_0".to_string(),
                        offset: 0,
                        value: SirNodeId("input".to_string()),
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("write_k".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 8. Elementwise Ops ──────────────────────────────────────────

    #[test]
    fn test_unary_elementwise_ops() {
        // Test several representative unary ops: Silu, Relu, Sigmoid, Tanh, Gelu, Exp, Neg, Abs
        let ops: Vec<(&str, SirOp)> = vec![
            ("silu_0", SirOp::Silu { input: SirNodeId("input".to_string()) }),
            ("relu_0", SirOp::Relu { input: SirNodeId("input".to_string()) }),
            ("sigmoid_0", SirOp::Sigmoid { input: SirNodeId("input".to_string()) }),
            ("tanh_0", SirOp::Tanh { input: SirNodeId("input".to_string()) }),
            (
                "gelu_0",
                SirOp::Gelu { input: SirNodeId("input".to_string()), mode: "TANH_APPROXIMATION".to_string() },
            ),
            ("exp_0", SirOp::Exp { input: SirNodeId("input".to_string()) }),
            ("neg_0", SirOp::Neg { input: SirNodeId("input".to_string()) }),
            ("abs_0", SirOp::Abs { input: SirNodeId("input".to_string()) }),
            ("sqrt_0", SirOp::Sqrt { input: SirNodeId("input".to_string()) }),
            ("rsqrt_0", SirOp::Rsqrt { input: SirNodeId("input".to_string()) }),
            ("cos_0", SirOp::Cos { input: SirNodeId("input".to_string()) }),
            ("sin_0", SirOp::Sin { input: SirNodeId("input".to_string()) }),
            ("ceil_0", SirOp::Ceil { input: SirNodeId("input".to_string()) }),
            ("floor_0", SirOp::Floor { input: SirNodeId("input".to_string()) }),
            ("round_0", SirOp::Round { input: SirNodeId("input".to_string()) }),
            ("sign_0", SirOp::Sign { input: SirNodeId("input".to_string()) }),
            ("log_0", SirOp::Log { input: SirNodeId("input".to_string()), epsilon: 1e-12 }),
            ("logical_not_0", SirOp::LogicalNot { input: SirNodeId("input".to_string()) }),
            ("softplus_0", SirOp::Softplus { input: SirNodeId("input".to_string()) }),
            ("softsign_0", SirOp::Softsign { input: SirNodeId("input".to_string()) }),
        ];

        let mut nodes = vec![make_node(
            "input",
            SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
        )];

        for (id, op) in ops {
            nodes.push(make_node(id, op));
        }

        let graph = SirGraph {
            nodes,
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("softsign_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_binary_elementwise_ops() {
        let ops: Vec<(&str, SirOp)> = vec![
            (
                "add_0",
                SirOp::Add { x: SirNodeId("input".to_string()), y: SirNodeId("input".to_string()) },
            ),
            (
                "mul_0",
                SirOp::Mul { x: SirNodeId("input".to_string()), y: SirNodeId("input".to_string()) },
            ),
            (
                "sub_0",
                SirOp::Sub { x: SirNodeId("input".to_string()), y: SirNodeId("input".to_string()) },
            ),
            (
                "max_0",
                SirOp::Maximum {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "min_0",
                SirOp::Minimum {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "div_0",
                SirOp::RealDiv {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "pow_0",
                SirOp::Pow { x: SirNodeId("input".to_string()), y: SirNodeId("input".to_string()) },
            ),
            (
                "eq_0",
                SirOp::Equal {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "ne_0",
                SirOp::NotEqual {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "gt_0",
                SirOp::Greater {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "ge_0",
                SirOp::GreaterEqual {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "lt_0",
                SirOp::Less {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
            (
                "le_0",
                SirOp::LessEqual {
                    x: SirNodeId("input".to_string()),
                    y: SirNodeId("input".to_string()),
                },
            ),
        ];

        let mut nodes = vec![make_node(
            "input",
            SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
        )];

        for (id, op) in ops {
            nodes.push(make_node(id, op));
        }

        let graph = SirGraph {
            nodes,
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("le_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 9. Reduction Ops ────────────────────────────────────────────

    #[test]
    fn test_reduction_ops() {
        let ops: Vec<(&str, SirOp)> = vec![
            (
                "sum_0",
                SirOp::ReduceSum {
                    input: SirNodeId("input".to_string()),
                    axes: vec![2],
                    keep_dims: true,
                },
            ),
            (
                "mean_0",
                SirOp::ReduceMean {
                    input: SirNodeId("input".to_string()),
                    axes: vec![2],
                    keep_dims: false,
                },
            ),
            (
                "max_0",
                SirOp::ReduceMax {
                    input: SirNodeId("input".to_string()),
                    axes: vec![1, 2],
                    keep_dims: true,
                },
            ),
            (
                "min_0",
                SirOp::ReduceMin {
                    input: SirNodeId("input".to_string()),
                    axes: vec![2],
                    keep_dims: false,
                },
            ),
            (
                "prod_0",
                SirOp::ReduceProd {
                    input: SirNodeId("input".to_string()),
                    axes: vec![2],
                    keep_dims: false,
                },
            ),
            (
                "argmax_0",
                SirOp::ReduceArgmax {
                    input: SirNodeId("input".to_string()),
                    axis: 2,
                    keep_dims: false,
                },
            ),
            (
                "argmin_0",
                SirOp::ReduceArgmin {
                    input: SirNodeId("input".to_string()),
                    axis: 1,
                    keep_dims: true,
                },
            ),
        ];

        let mut nodes = vec![make_node(
            "input",
            SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
        )];

        for (id, op) in ops {
            nodes.push(make_node(id, op));
        }

        let graph = SirGraph {
            nodes,
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("argmin_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 10. Tensor Transform Ops ────────────────────────────────────

    #[test]
    fn test_reshape_and_transpose() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "reshape_0",
                    SirOp::Reshape {
                        input: SirNodeId("input".to_string()),
                        target_shape: vec![1, 14, 512, 64],
                    },
                ),
                make_node(
                    "transpose_0",
                    SirOp::Transpose {
                        input: SirNodeId("reshape_0".to_string()),
                        perm: vec![0, 2, 1, 3],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("transpose_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_concat_split() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "split_0",
                    SirOp::Split { input: SirNodeId("input".to_string()), axis: 2, num_splits: 2 },
                ),
                make_node(
                    "concat_0",
                    SirOp::Concat { inputs: vec![SirNodeId("split_0".to_string())], axis: 2 },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("concat_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_expand_dims_squeeze() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "expand_0",
                    SirOp::ExpandDims { input: SirNodeId("input".to_string()), axis: vec![1] },
                ),
                make_node(
                    "squeeze_0",
                    SirOp::Squeeze { input: SirNodeId("expand_0".to_string()), axis: vec![1] },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("squeeze_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_tile_and_pad() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "tile_0",
                    SirOp::Tile { input: SirNodeId("input".to_string()), reps: vec![1, 1, 4, 1] },
                ),
                make_node(
                    "pad_0",
                    SirOp::Pad {
                        input: SirNodeId("tile_0".to_string()),
                        pad_amounts: vec![0, 0, 1, 1, 0, 0, 0, 0],
                        mode: "constant".to_string(),
                        constant_value: 0.0,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("pad_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 11. Gather / Scatter ────────────────────────────────────────

    #[test]
    fn test_gather() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "indices",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "gather_0",
                    SirOp::Gather {
                        input: SirNodeId("input".to_string()),
                        indices: SirNodeId("indices".to_string()),
                        axis: 0,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("indices".to_string())],
            outputs: vec![SirNodeId("gather_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 12. MatMul ──────────────────────────────────────────────────

    #[test]
    fn test_matmul() {
        let graph = SirGraph {
            nodes: vec![
                make_node("a", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("b", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "mm_0",
                    SirOp::MatMul { a: SirNodeId("a".to_string()), b: SirNodeId("b".to_string()) },
                ),
            ],
            inputs: vec![SirNodeId("a".to_string()), SirNodeId("b".to_string())],
            outputs: vec![SirNodeId("mm_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 13. Cast and Select ─────────────────────────────────────────

    #[test]
    fn test_cast_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "cast_0",
                    SirOp::Cast { input: SirNodeId("input".to_string()), dtype: MilDtype::Fp16 },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("cast_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_select_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "cond",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node("x", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("y", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "select_0",
                    SirOp::Select {
                        condition: SirNodeId("cond".to_string()),
                        x: SirNodeId("x".to_string()),
                        y: SirNodeId("y".to_string()),
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("cond".to_string()),
                SirNodeId("x".to_string()),
                SirNodeId("y".to_string()),
            ],
            outputs: vec![SirNodeId("select_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_where_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "cond",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node("x", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("y", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "where_0",
                    SirOp::Where {
                        condition: SirNodeId("cond".to_string()),
                        x: SirNodeId("x".to_string()),
                        y: SirNodeId("y".to_string()),
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("cond".to_string()),
                SirNodeId("x".to_string()),
                SirNodeId("y".to_string()),
            ],
            outputs: vec![SirNodeId("where_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 14. Softmax ─────────────────────────────────────────────────

    #[test]
    fn test_softmax() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "softmax_0",
                    SirOp::Softmax { input: SirNodeId("input".to_string()), axis: -1 },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("softmax_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 15. Conv ────────────────────────────────────────────────────

    #[test]
    fn test_conv_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "weight",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "conv_0",
                    SirOp::Conv {
                        input: SirNodeId("input".to_string()),
                        weight: SirNodeId("weight".to_string()),
                        pad_type: "custom".to_string(),
                        groups: 1,
                        strides: vec![1, 1],
                        pad_amounts: vec![0, 0, 0, 0],
                        dilations: vec![1, 1],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("weight".to_string())],
            outputs: vec![SirNodeId("conv_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 16. SliceByIndex ────────────────────────────────────────────

    #[test]
    fn test_slice_by_index() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "slice_0",
                    SirOp::SliceByIndex {
                        input: SirNodeId("input".to_string()),
                        begin: vec![0, 0, 0, 0],
                        end: vec![-1, -1, -1, 64],
                        stride: vec![1, 1, 1, 1],
                        begin_mask: vec![true, true, true, false],
                        end_mask: vec![true, true, true, false],
                        squeeze_mask: vec![false, false, false, false],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("slice_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 17. Quantization / Constexpr Ops ────────────────────────────

    #[test]
    fn test_quantize_dequantize() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "quant_0",
                    SirOp::Quantize {
                        input: SirNodeId("input".to_string()),
                        scale: 0.1,
                        zero_point: 0,
                        axis: -1,
                        output_dtype: MilDtype::Int8,
                    },
                ),
                make_node(
                    "dequant_0",
                    SirOp::Dequantize {
                        input: SirNodeId("quant_0".to_string()),
                        scale: 0.1,
                        zero_point: 0,
                        axis: -1,
                        output_dtype: MilDtype::Fp16,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("dequant_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_constexpr_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "affine_0",
                    SirOp::ConstexprAffineDequantize {
                        quantized_data: "weight_q.bin".to_string(),
                        scale: 0.05,
                        zero_point: 128,
                        axis: 0,
                    },
                ),
                make_node(
                    "lut_0",
                    SirOp::ConstexprLutToDense {
                        indices: "palette_idx.bin".to_string(),
                        lut: "palette_lut.bin".to_string(),
                        num_bits: 4,
                    },
                ),
                make_node(
                    "sparse_0",
                    SirOp::ConstexprSparseToDense {
                        nonzero_data: "sparse_data.bin".to_string(),
                        shape: vec![256, 512],
                        default_value: 0.0,
                    },
                ),
                make_node(
                    "cast_0",
                    SirOp::ConstexprCast {
                        data: "weight_fp32.bin".to_string(),
                        dtype: MilDtype::Fp16,
                    },
                ),
            ],
            inputs: vec![],
            outputs: vec![SirNodeId("cast_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 18. Sampler ────────────────────────────────────────────────

    #[test]
    fn test_sampler_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "logits",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "sampler_0",
                    SirOp::Sampler {
                        logits: SirNodeId("logits".to_string()),
                        temperature: 0.8,
                        top_p: 0.95,
                        rep_penalty: 1.1,
                        min_p: 0.05,
                        top_k: 64,
                        gumbel_noise: true,
                    },
                ),
            ],
            inputs: vec![SirNodeId("logits".to_string())],
            outputs: vec![SirNodeId("sampler_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 19. Normalization Ops ──────────────────────────────────────

    #[test]
    fn test_layer_norm_and_batch_norm() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "ln_0",
                    SirOp::LayerNorm {
                        input: SirNodeId("input".to_string()),
                        weight: "ln_weight.bin".to_string(),
                        bias: Some("ln_bias.bin".to_string()),
                        epsilon: 1e-5,
                        axes: vec![2],
                    },
                ),
                make_node(
                    "bn_0",
                    SirOp::BatchNorm {
                        input: SirNodeId("ln_0".to_string()),
                        mean: "bn_mean.bin".to_string(),
                        variance: "bn_var.bin".to_string(),
                        gamma: Some("bn_gamma.bin".to_string()),
                        beta: Some("bn_beta.bin".to_string()),
                        epsilon: 1e-5,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("bn_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 20. Multi-Node Realistic Pipeline ───────────────────────────

    #[test]
    fn test_realistic_decode_pipeline() {
        // Simulates a realistic decode-step pipeline:
        // input → StateRead (K cache) + StateRead (V cache) + LinearProjection (Q) →
        // SDPA → LinearProjection (O) → RMSNorm → output
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "token",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "k_cache",
                    SirOp::StateRead {
                        state_id: "kv_k_0".to_string(),
                        offset: 0,
                        shape: vec![1, 14, 512, 64],
                    },
                ),
                make_node(
                    "v_cache",
                    SirOp::StateRead {
                        state_id: "kv_v_0".to_string(),
                        offset: 0,
                        shape: vec![1, 14, 512, 64],
                    },
                ),
                make_node(
                    "q_proj",
                    SirOp::LinearProjection {
                        input: SirNodeId("token".to_string()),
                        weight: "q_proj.weight".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                ),
                make_node(
                    "sdpa",
                    SirOp::ScaledDotProductAttention {
                        query: SirNodeId("q_proj".to_string()),
                        key: SirNodeId("k_cache".to_string()),
                        value: SirNodeId("v_cache".to_string()),
                        attention_mask: None,
                        scale: Some(0.125),
                    },
                ),
                make_node(
                    "o_proj",
                    SirOp::LinearProjection {
                        input: SirNodeId("sdpa".to_string()),
                        weight: "o_proj.weight".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                ),
                make_node(
                    "rms_norm",
                    SirOp::RMSNorm {
                        input: SirNodeId("o_proj".to_string()),
                        weight: "norm.weight".to_string(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                ),
            ],
            inputs: vec![SirNodeId("token".to_string())],
            outputs: vec![SirNodeId("rms_norm".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 21. Metadata Preservation ───────────────────────────────────

    #[test]
    fn test_metadata_preserved() {
        // Verify that all metadata fields survive the pass-through unchanged.
        let graph = SirGraph {
            nodes: vec![SirNode {
                id: SirNodeId("node_0".to_string()),
                op: SirOp::Relu { input: SirNodeId("input".to_string()) },
                name: "my_relu".to_string(),
                metadata: SirMetadata {
                    task_origin: TaskOrigin::RealModel { name: "qwen3-0.6b".to_string() },
                    model_id: Some("qwen3-0.6b-instruct".to_string()),
                    quality_contract: Some(QualityContract {
                        max_perplexity_delta: Some(0.1),
                        max_latency_ms: Some(50.0),
                    }),
                    precision_override: Some("fp16".to_string()),
                },
            }],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("node_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_task_origin_variants_preserved() {
        // Ensure all TaskOrigin variants survive pass-through.
        let origins: Vec<TaskOrigin> = vec![
            TaskOrigin::Synthetic,
            TaskOrigin::RealModel { name: "llama-2-7b".to_string() },
            TaskOrigin::MilImport { source: "model.mlpackage".to_string() },
            TaskOrigin::Manual,
            TaskOrigin::TransformersTrace { name: "qwen3-1.8b".to_string() },
        ];

        for (i, origin) in origins.into_iter().enumerate() {
            let id = format!("node_{}", i);
            let graph = SirGraph {
                nodes: vec![SirNode {
                    id: SirNodeId(id.clone()),
                    op: SirOp::Relu { input: SirNodeId("input".to_string()) },
                    name: id.clone(),
                    metadata: make_metadata_with(origin, None),
                }],
                inputs: vec![SirNodeId("input".to_string())],
                outputs: vec![SirNodeId(id.clone())],
            };

            let result = StaticizePass::new().run(graph.clone()).unwrap();
            assert_graphs_identical(&graph, &result);
        }
    }

    // ─── 22. Graph I/O Preservation ──────────────────────────────────

    #[test]
    fn test_graph_io_preserved_multiple_inputs() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "in_a",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "in_b",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "add",
                    SirOp::Add {
                        x: SirNodeId("in_a".to_string()),
                        y: SirNodeId("in_b".to_string()),
                    },
                ),
            ],
            inputs: vec![SirNodeId("in_a".to_string()), SirNodeId("in_b".to_string())],
            outputs: vec![SirNodeId("add".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_graph_io_preserved_multiple_outputs() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node("relu_0", SirOp::Relu { input: SirNodeId("input".to_string()) }),
                make_node("sigmoid_0", SirOp::Sigmoid { input: SirNodeId("input".to_string()) }),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("relu_0".to_string()), SirNodeId("sigmoid_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_graph_io_preserved_empty_inputs() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "const_0",
                SirOp::Fill { shape: vec![1, 128], value: 1.0, dtype: MilDtype::Fp16 },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("const_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 23. Return Type Consistency ─────────────────────────────────

    #[test]
    fn test_run_returns_ok() {
        // StaticizePass::run() should always return Ok for the current
        // pass-through implementation — never Err.
        let graph = SirGraph {
            nodes: vec![make_node(
                "input",
                SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
            )],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("input".to_string())],
        };

        let result = StaticizePass::new().run(graph);
        assert!(result.is_ok(), "StaticizePass::run() should always return Ok for pass-through");
    }

    #[test]
    fn test_run_result_unwraps_cleanly() {
        // Verify the Result<SirGraph> unwraps to a valid SirGraph.
        let graph = SirGraph {
            nodes: vec![make_node(
                "node_0",
                SirOp::Const {
                    value_path: "w.bin".to_string(),
                    dtype: MilDtype::Fp32,
                    palette_bits: None,
                },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("node_0".to_string())],
        };

        let result = StaticizePass::new().run(graph).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.outputs.len(), 1);
    }

    // ─── 24. Idempotency ─────────────────────────────────────────────

    #[test]
    fn test_idempotent_single_pass() {
        // Running the pass once should produce the same graph.
        // Running it again on the output should produce the same result.
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "linear_0",
                    SirOp::LinearProjection {
                        input: SirNodeId("input".to_string()),
                        weight: "w_q.bin".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                ),
                make_node("relu_0", SirOp::Relu { input: SirNodeId("linear_0".to_string()) }),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("relu_0".to_string())],
        };

        let pass = StaticizePass::new();
        let result1 = pass.run(graph.clone()).unwrap();
        let result2 = pass.run(result1.clone()).unwrap();
        assert_graphs_identical(&result1, &result2);
    }

    #[test]
    fn test_idempotent_multi_node() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "rms_0",
                    SirOp::RMSNorm {
                        input: SirNodeId("input".to_string()),
                        weight: "norm.weight".to_string(),
                        epsilon: 1e-6,
                        axes: vec![2],
                    },
                ),
                make_node(
                    "linear_0",
                    SirOp::LinearProjection {
                        input: SirNodeId("rms_0".to_string()),
                        weight: "q_proj.weight".to_string(),
                        bias: None,
                        palette_bits: None,
                    },
                ),
                make_node(
                    "sdpa_0",
                    SirOp::ScaledDotProductAttention {
                        query: SirNodeId("linear_0".to_string()),
                        key: SirNodeId("k_cache".to_string()),
                        value: SirNodeId("v_cache".to_string()),
                        attention_mask: None,
                        scale: Some(0.0884),
                    },
                ),
                make_node(
                    "k_cache",
                    SirOp::StateRead {
                        state_id: "kv_k".to_string(),
                        offset: 0,
                        shape: vec![1, 14, 512, 64],
                    },
                ),
                make_node(
                    "v_cache",
                    SirOp::StateRead {
                        state_id: "kv_v".to_string(),
                        offset: 0,
                        shape: vec![1, 14, 512, 64],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("sdpa_0".to_string())],
        };

        let pass = StaticizePass::new();
        let r1 = pass.run(graph).unwrap();
        let r2 = pass.run(r1.clone()).unwrap();
        let r3 = pass.run(r2.clone()).unwrap();
        assert_graphs_identical(&r1, &r3);
    }

    // ─── 25. Large Graph Stress Test ─────────────────────────────────

    #[test]
    fn test_large_graph_many_nodes() {
        // Build a graph with many nodes to verify no accidental mutations
        // accumulate. This simulates a realistic multi-layer model.
        let mut nodes = vec![make_node(
            "input",
            SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
        )];

        for i in 0..50 {
            let prev = if i == 0 { "input".to_string() } else { format!("relu_{}", i - 1) };
            nodes.push(make_node(&format!("relu_{}", i), SirOp::Relu { input: SirNodeId(prev) }));
        }

        let graph = SirGraph {
            nodes,
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("relu_49".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
        assert_eq!(result.nodes.len(), 51);
    }

    // ─── 26. Topk ────────────────────────────────────────────────────

    #[test]
    fn test_topk_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "topk_0",
                    SirOp::Topk { input: SirNodeId("input".to_string()), k: 5, axis: -1 },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("topk_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 27. Pooling Ops ─────────────────────────────────────────────

    #[test]
    fn test_pooling_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "maxpool_0",
                    SirOp::MaxPool {
                        input: SirNodeId("input".to_string()),
                        kernel_sizes: vec![3, 3],
                        strides: vec![2, 2],
                        pad_types: vec!["custom".to_string(), "custom".to_string()],
                        pad_amounts: vec![1, 1, 1, 1],
                    },
                ),
                make_node(
                    "avgpool_0",
                    SirOp::AvgPool {
                        input: SirNodeId("maxpool_0".to_string()),
                        kernel_sizes: vec![2, 2],
                        strides: vec![2, 2],
                        pad_types: vec!["valid".to_string(), "valid".to_string()],
                        pad_amounts: vec![0, 0, 0, 0],
                        count_include_padding: false,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("avgpool_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 28. Recurrent Ops (RNN/GRU/LSTM) ───────────────────────────

    #[test]
    fn test_recurrent_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "h0",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "rnn_0",
                    SirOp::Rnn {
                        input: SirNodeId("input".to_string()),
                        initial_h: SirNodeId("h0".to_string()),
                        weight_ih: "rnn_w_ih.bin".to_string(),
                        weight_hh: "rnn_w_hh.bin".to_string(),
                        bias: Some("rnn_bias.bin".to_string()),
                        mode: "relu".to_string(),
                        output_sequence: true,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("h0".to_string())],
            outputs: vec![SirNodeId("rnn_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_gru_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "h0",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "gru_0",
                    SirOp::Gru {
                        input: SirNodeId("input".to_string()),
                        initial_h: SirNodeId("h0".to_string()),
                        weight_ih: "gru_w_ih.bin".to_string(),
                        weight_hh: "gru_w_hh.bin".to_string(),
                        bias: None,
                        reset_after: true,
                        output_sequence: false,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("h0".to_string())],
            outputs: vec![SirNodeId("gru_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_lstm_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "h0",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "c0",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "lstm_0",
                    SirOp::Lstm {
                        input: SirNodeId("input".to_string()),
                        initial_h: SirNodeId("h0".to_string()),
                        initial_c: SirNodeId("c0".to_string()),
                        weight_ih: "lstm_w_ih.bin".to_string(),
                        weight_hh: "lstm_w_hh.bin".to_string(),
                        bias: Some("lstm_bias.bin".to_string()),
                        output_sequence: true,
                    },
                ),
            ],
            inputs: vec![
                SirNodeId("input".to_string()),
                SirNodeId("h0".to_string()),
                SirNodeId("c0".to_string()),
            ],
            outputs: vec![SirNodeId("lstm_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 29. Control Flow (Cond/WhileLoop) ───────────────────────────

    #[test]
    fn test_cond_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "pred",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "cond_0",
                    SirOp::Cond {
                        pred: SirNodeId("pred".to_string()),
                        true_graph: "true_branch".to_string(),
                        false_graph: "false_branch".to_string(),
                    },
                ),
            ],
            inputs: vec![SirNodeId("pred".to_string())],
            outputs: vec![SirNodeId("cond_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_while_loop_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "var_0",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "while_0",
                    SirOp::WhileLoop {
                        condition: "cond_fn".to_string(),
                        body: "body_fn".to_string(),
                        loop_vars: vec![SirNodeId("var_0".to_string())],
                    },
                ),
            ],
            inputs: vec![SirNodeId("var_0".to_string())],
            outputs: vec![SirNodeId("while_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 30. Random Ops ──────────────────────────────────────────────

    #[test]
    fn test_random_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "bernoulli_0",
                    SirOp::RandomBernoulli {
                        shape: vec![1, 128],
                        prob: 0.5,
                        seed: Some(42),
                        dtype: MilDtype::Fp16,
                    },
                ),
                make_node(
                    "normal_0",
                    SirOp::RandomNormal {
                        shape: vec![1, 128],
                        mean: 0.0,
                        stddev: 1.0,
                        seed: None,
                        dtype: MilDtype::Fp32,
                    },
                ),
                make_node(
                    "uniform_0",
                    SirOp::RandomUniform {
                        shape: vec![1, 64],
                        low: -1.0,
                        high: 1.0,
                        seed: Some(123),
                        dtype: MilDtype::Fp16,
                    },
                ),
            ],
            inputs: vec![],
            outputs: vec![SirNodeId("uniform_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 31. ConvTranspose ───────────────────────────────────────────

    #[test]
    fn test_conv_transpose() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "weight",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "conv_t_0",
                    SirOp::ConvTranspose {
                        input: SirNodeId("input".to_string()),
                        weight: SirNodeId("weight".to_string()),
                        pad_type: "custom".to_string(),
                        groups: 1,
                        strides: vec![2, 2],
                        pad_amounts: vec![1, 1, 1, 1],
                        dilations: vec![1, 1],
                        output_shape: vec![1, 3, 224, 224],
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("weight".to_string())],
            outputs: vec![SirNodeId("conv_t_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 32. ReshapeLike, Flatten2d, Reverse ─────────────────────────

    #[test]
    fn test_reshape_like_and_flatten() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "ref",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "reshape_like_0",
                    SirOp::ReshapeLike {
                        input: SirNodeId("input".to_string()),
                        ref_tensor: SirNodeId("ref".to_string()),
                    },
                ),
                make_node(
                    "flatten_0",
                    SirOp::Flatten2d { input: SirNodeId("reshape_like_0".to_string()), axis: 1 },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string()), SirNodeId("ref".to_string())],
            outputs: vec![SirNodeId("flatten_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    #[test]
    fn test_reverse_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "reverse_0",
                    SirOp::Reverse { input: SirNodeId("input".to_string()), axes: vec![1] },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("reverse_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 33. Space/Depth rearrangement ops ───────────────────────────

    #[test]
    fn test_space_depth_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "d2s_0",
                    SirOp::DepthToSpace { input: SirNodeId("input".to_string()), block_size: 2 },
                ),
                make_node(
                    "s2d_0",
                    SirOp::SpaceToDepth { input: SirNodeId("d2s_0".to_string()), block_size: 2 },
                ),
                make_node(
                    "shuffle_0",
                    SirOp::PixelShuffle {
                        input: SirNodeId("s2d_0".to_string()),
                        upscale_factor: 2,
                    },
                ),
                make_node(
                    "unshuffle_0",
                    SirOp::PixelUnshuffle {
                        input: SirNodeId("shuffle_0".to_string()),
                        downscale_factor: 2,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("unshuffle_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 34. Cumsum, FillLike, OneHot, Range1d ──────────────────────

    #[test]
    fn test_misc_tensor_ops() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "cumsum_0",
                    SirOp::Cumsum {
                        input: SirNodeId("input".to_string()),
                        axis: 1,
                        exclusive: false,
                        reverse: false,
                    },
                ),
                make_node(
                    "fill_like_0",
                    SirOp::FillLike {
                        ref_tensor: SirNodeId("cumsum_0".to_string()),
                        value: 0.0,
                        dtype: MilDtype::Fp16,
                    },
                ),
                make_node("range_0", SirOp::Range1d { start: 0.0, end: 512.0, step: 1.0 }),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("range_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 35. LeakyRelu, ScaledTanh, ThresholdedRelu ─────────────────

    #[test]
    fn test_parametric_activations() {
        let graph = SirGraph {
            nodes: vec![
                make_node(
                    "input",
                    SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) },
                ),
                make_node(
                    "leaky_0",
                    SirOp::LeakyRelu { input: SirNodeId("input".to_string()), alpha: 0.01 },
                ),
                make_node(
                    "scaled_tanh_0",
                    SirOp::ScaledTanh {
                        input: SirNodeId("leaky_0".to_string()),
                        alpha: 1.0,
                        beta: 0.5,
                    },
                ),
                make_node(
                    "thresh_0",
                    SirOp::ThresholdedRelu {
                        input: SirNodeId("scaled_tanh_0".to_string()),
                        alpha: 1.0,
                    },
                ),
            ],
            inputs: vec![SirNodeId("input".to_string())],
            outputs: vec![SirNodeId("thresh_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 36. Einsum ──────────────────────────────────────────────────

    #[test]
    fn test_einsum_op() {
        let graph = SirGraph {
            nodes: vec![
                make_node("a", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node("b", SirOp::Identity { input: SirNodeId("__placeholder__".to_string()) }),
                make_node(
                    "einsum_0",
                    SirOp::Einsum {
                        inputs: vec![SirNodeId("a".to_string()), SirNodeId("b".to_string())],
                        equation: "bij,bjk->bik".to_string(),
                    },
                ),
            ],
            inputs: vec![SirNodeId("a".to_string()), SirNodeId("b".to_string())],
            outputs: vec![SirNodeId("einsum_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }

    // ─── 37. ConstexprBlockwiseShiftScale ────────────────────────────

    #[test]
    fn test_constexpr_blockwise_shift_scale() {
        let graph = SirGraph {
            nodes: vec![make_node(
                "bwss_0",
                SirOp::ConstexprBlockwiseShiftScale {
                    data: "quant_data.bin".to_string(),
                    scale: "quant_scale.bin".to_string(),
                    offset: "quant_offset.bin".to_string(),
                    block_size: vec![128],
                },
            )],
            inputs: vec![],
            outputs: vec![SirNodeId("bwss_0".to_string())],
        };

        let result = StaticizePass::new().run(graph.clone()).unwrap();
        assert_graphs_identical(&graph, &result);
    }
}
