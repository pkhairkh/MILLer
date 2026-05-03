//! `ToProto` Trait — Unified Proto-Emission Interface for MirOp
//!
//! T-38 (I-17 / CQ-20, CQ-21): This trait replaces the per-variant match-arm
//! boilerplate spread across `From<MirOp>` for `MirOpCompat`, `compat_input_names()`,
//! `remap_compat_inputs()`, `rename_compat_output()`, and `op_output_names()`.
//!
//! ## Design
//!
//! The trait provides four methods that every MirOp variant must implement:
//!
//! - **`proto_op_type()`** → Core ML MIL op type string (e.g., "linear", "add")
//! - **`proto_output_name()`** → SSA output value name
//! - **`proto_input_refs()`** → SSA input references (MirNodeId→String, weight names)
//! - **`is_proto_supported()`** → Whether this variant has a specialized emission path
//!
//! Adding a new MirOp variant now requires updating these trait methods in ONE place
//! (mir.rs), instead of the previous 5 separate match expressions.
//!
//! ## Boilerplate Reduction
//!
//! Before T-38:
//! - `From<MirOp>` for `MirOpCompat`  (~80 lines, 167 arms)
//! - `mir_op_to_compat()`              (~300 lines)
//! - `compat_input_names()`            (~120 lines, 79 arms)
//! - `remap_compat_inputs()`           (~200 lines, 79 arms)
//! - `rename_compat_output()`          (~80 lines, 79 arms)
//! - `op_output_names()`               (~50 lines, 79 arms)
//!
//! After T-38: Each variant's proto-relevant data is defined ONCE in the trait impl,
//! and `MirOpCompat` methods are generated from a single macro specification.

/// Trait for MIR operations that can be emitted as Core ML proto operations.
///
/// This trait provides a unified interface for extracting proto-relevant
/// information from `MirOp` variants, replacing the previous pattern of
/// 5+ separate per-variant match expressions that had to be updated in
/// lockstep.
///
/// # Implementation
///
/// Every `MirOp` variant implements this trait. The implementation is
/// consolidated in `mir.rs` as part of the `impl MirOp` block. Adding a
/// new variant requires updating:
/// 1. The enum definition
/// 2. The `ToProto` trait methods (this trait)
/// 3. The `default_engine()` method
/// 4. The `mil_op_name()` method
///
/// This is 4 places instead of the previous 7+.
pub trait ToProto {
    /// Returns the Core ML MIL operation type string.
    ///
    /// This is the `type` field in the MIL `Operation` protobuf message.
    /// Examples: "const", "linear", "add", "reshape", "unsupported_tile".
    ///
    /// For ops without a specialized emission path, this returns the same
    /// value as `mil_op_name()` — the proto emission layer will reject
    /// these with a clear error message at validation time.
    fn proto_op_type(&self) -> &'static str;

    /// Returns the SSA output value name for this operation.
    ///
    /// Every `MirOp` variant has a `name` field that serves as the unique
    /// SSA value name in the MIL block. Other operations reference this
    /// name as an input.
    fn proto_output_name(&self) -> &str;

    /// Returns all SSA input references as owned Strings.
    ///
    /// This extracts every field that references another SSA value:
    /// - `MirNodeId` fields are unwrapped to their inner `String`
    /// - `Option<MirNodeId>` fields are flattened (Some→push, None→skip)
    /// - `Vec<MirNodeId>` fields are extended (each element unwrapped)
    /// - `String` weight-name fields (e.g., `weight`, `bias`) are included
    /// - `Option<String>` weight-name fields are flattened
    ///
    /// Non-reference fields (scalars, shapes, axes, flags) are NOT included.
    fn proto_input_refs(&self) -> Vec<String>;

    /// Returns whether this MirOp variant has a specialized proto emission path.
    ///
    /// Supported variants have a 1:1 mapping to `MirOpCompat` variants and
    /// can be emitted directly. Unsupported variants fall through to
    /// `MirOpCompat::Unsupported` and will be rejected by the proto
    /// validation gate with a clear error message.
    fn is_proto_supported(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::ToProto;
    use crate::mir::{MilDtype, MirNodeId, MirOp};

    /// Helper: create a MirNodeId from &str.
    fn nid(s: &str) -> MirNodeId {
        MirNodeId(s.to_string())
    }

    // ─── proto_op_type() tests ─────────────────────────────────────

    #[test]
    fn test_proto_op_type_const() {
        let op = MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 };
        assert_eq!(op.proto_op_type(), "const");
    }

    #[test]
    fn test_proto_op_type_linear() {
        let op = MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None };
        assert_eq!(op.proto_op_type(), "linear");
    }

    #[test]
    fn test_proto_op_type_add() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") };
        assert_eq!(op.proto_op_type(), "add");
    }

    #[test]
    fn test_proto_op_type_unsupported_einsum() {
        let op = MirOp::MILEinsum { name: "e".into(), inputs: vec![], equation: "".into() };
        assert_eq!(op.proto_op_type(), "einsum");
    }

    #[test]
    fn test_proto_op_type_matches_mil_op_name() {
        // proto_op_type should always equal mil_op_name
        let ops: Vec<MirOp> = vec![
            MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 },
            MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None },
            MirOp::MILMatMul { name: "m".into(), x: nid("x"), y: nid("y"), transpose_y: false },
            MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") },
            MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![1, 2] },
            MirOp::MILSoftmax { name: "s".into(), x: nid("x"), axis: -1 },
            MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![1], keep_dims: false },
            MirOp::MILEinsum { name: "e".into(), inputs: vec![], equation: "".into() },
            MirOp::MILConvTranspose { name: "ct".into(), x: nid("x"), weight: nid("w"), pad_type: "valid".into(), groups: 1, strides: vec![1], pad_amounts: vec![0], dilations: vec![1], output_shape: vec![] },
        ];
        for op in &ops {
            assert_eq!(op.proto_op_type(), op.mil_op_name(),
                "proto_op_type mismatch for {:?}", op.mil_op_name());
        }
    }

    // ─── proto_output_name() tests ─────────────────────────────────

    #[test]
    fn test_proto_output_name_const() {
        let op = MirOp::MILConst { name: "my_const".into(), value_path: "v".into(), dtype: MilDtype::Fp16 };
        assert_eq!(op.proto_output_name(), "my_const");
    }

    #[test]
    fn test_proto_output_name_linear() {
        let op = MirOp::MILLinear { name: "output".into(), x: nid("x"), weight: "w".into(), bias: None };
        assert_eq!(op.proto_output_name(), "output");
    }

    #[test]
    fn test_proto_output_name_all_variants_have_name() {
        // Spot-check a variety of variants
        let cases: Vec<(MirOp, &str)> = vec![
            (MirOp::MILAdd { name: "add_out".into(), x: nid("x"), y: nid("y") }, "add_out"),
            (MirOp::MILReshape { name: "reshape_out".into(), x: nid("x"), shape: vec![] }, "reshape_out"),
            (MirOp::MILScaledDotProductAttention { name: "sdpa_out".into(), query: nid("q"), key: nid("k"), value: nid("v"), attention_mask: None, scale: None }, "sdpa_out"),
            (MirOp::MILReadState { name: "rs_out".into(), state_id: "s".into(), shape: vec![], dtype: MilDtype::Fp16 }, "rs_out"),
            (MirOp::MILFill { name: "fill_out".into(), shape: vec![], value: 0.0, dtype: MilDtype::Fp16 }, "fill_out"),
        ];
        for (op, expected) in &cases {
            assert_eq!(op.proto_output_name(), *expected);
        }
    }

    // ─── proto_input_refs() tests ──────────────────────────────────

    #[test]
    fn test_proto_input_refs_const_no_inputs() {
        let op = MirOp::MILConst { name: "c".into(), value_path: "vp".into(), dtype: MilDtype::Fp16 };
        assert!(op.proto_input_refs().is_empty());
    }

    #[test]
    fn test_proto_input_refs_linear() {
        let op = MirOp::MILLinear { name: "l".into(), x: nid("input_x"), weight: "weight_w".into(), bias: Some("bias_b".into()) };
        let refs = op.proto_input_refs();
        assert_eq!(refs, vec!["input_x", "weight_w", "bias_b"]);
    }

    #[test]
    fn test_proto_input_refs_linear_no_bias() {
        let op = MirOp::MILLinear { name: "l".into(), x: nid("input_x"), weight: "weight_w".into(), bias: None };
        let refs = op.proto_input_refs();
        assert_eq!(refs, vec!["input_x", "weight_w"]);
    }

    #[test]
    fn test_proto_input_refs_binary_op() {
        let op = MirOp::MILAdd { name: "a".into(), x: nid("left"), y: nid("right") };
        assert_eq!(op.proto_input_refs(), vec!["left", "right"]);
    }

    #[test]
    fn test_proto_input_refs_unary_op() {
        let op = MirOp::MILRelu { name: "r".into(), x: nid("input") };
        assert_eq!(op.proto_input_refs(), vec!["input"]);
    }

    #[test]
    fn test_proto_input_refs_concat() {
        let op = MirOp::MILConcat { name: "c".into(), values: vec![nid("a"), nid("b"), nid("c_val")], axis: 0 };
        assert_eq!(op.proto_input_refs(), vec!["a", "b", "c_val"]);
    }

    #[test]
    fn test_proto_input_refs_sdpa_with_mask() {
        let op = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"), key: nid("k"), value: nid("v"),
            attention_mask: Some(nid("mask")),
            scale: Some(0.125),
        };
        assert_eq!(op.proto_input_refs(), vec!["q", "k", "v", "mask"]);
    }

    #[test]
    fn test_proto_input_refs_sdpa_no_mask() {
        let op = MirOp::MILScaledDotProductAttention {
            name: "sdpa".into(),
            query: nid("q"), key: nid("k"), value: nid("v"),
            attention_mask: None,
            scale: None,
        };
        assert_eq!(op.proto_input_refs(), vec!["q", "k", "v"]);
    }

    #[test]
    fn test_proto_input_refs_fill_no_inputs() {
        let op = MirOp::MILFill { name: "f".into(), shape: vec![1, 512], value: 1.0, dtype: MilDtype::Fp16 };
        assert!(op.proto_input_refs().is_empty());
    }

    #[test]
    fn test_proto_input_refs_reshape() {
        let op = MirOp::MILReshape { name: "r".into(), x: nid("input"), shape: vec![1, 2, 3] };
        assert_eq!(op.proto_input_refs(), vec!["input"]);
    }

    #[test]
    fn test_proto_input_refs_layer_norm() {
        let op = MirOp::MILLayerNorm { name: "ln".into(), x: nid("input"), weight: "w".into(), bias: Some("b".into()), epsilon: 1e-5, axes: vec![2] };
        assert_eq!(op.proto_input_refs(), vec!["input", "w", "b"]);
    }

    #[test]
    fn test_proto_input_refs_layer_norm_no_bias() {
        let op = MirOp::MILLayerNorm { name: "ln".into(), x: nid("input"), weight: "w".into(), bias: None, epsilon: 1e-5, axes: vec![2] };
        assert_eq!(op.proto_input_refs(), vec!["input", "w"]);
    }

    #[test]
    fn test_proto_input_refs_read_state_no_ssa_inputs() {
        let op = MirOp::MILReadState { name: "rs".into(), state_id: "kv_cache".into(), shape: vec![1, 8, 512, 128], dtype: MilDtype::Fp16 };
        // state_id is a state reference, not an SSA input — handled specially by the compat layer
        assert!(op.proto_input_refs().is_empty());
    }

    #[test]
    fn test_proto_input_refs_coreml_update_state() {
        let op = MirOp::MILCoremlUpdateState { name: "us".into(), state_id: "kv_cache".into(), value: nid("new_val") };
        let refs = op.proto_input_refs();
        assert_eq!(refs, vec!["new_val"]);
    }

    #[test]
    fn test_proto_input_refs_state_write() {
        let op = MirOp::MILStateWrite { name: "sw".into(), state_ref: "kv_cache".into(), value: nid("new_val") };
        let refs = op.proto_input_refs();
        assert_eq!(refs, vec!["new_val"]);
    }

    #[test]
    fn test_proto_input_refs_cond() {
        let op = MirOp::MILCond { name: "c".into(), pred: nid("flag"), true_graph: "true_fn".into(), false_graph: "false_fn".into() };
        // true_graph/false_graph are function names, not SSA inputs
        assert_eq!(op.proto_input_refs(), vec!["flag"]);
    }

    #[test]
    fn test_proto_input_refs_while_loop() {
        let op = MirOp::MILWhileLoop { name: "wl".into(), condition: "cond_fn".into(), body: "body_fn".into(), loop_vars: vec![nid("v1"), nid("v2")] };
        assert_eq!(op.proto_input_refs(), vec!["v1", "v2"]);
    }

    #[test]
    fn test_proto_input_refs_conv() {
        let op = MirOp::MILConv { name: "c".into(), x: nid("input"), weight: nid("w"), pad_type: "valid".into(), groups: 1, strides: vec![1], pad_amounts: vec![0], dilations: vec![1] };
        assert_eq!(op.proto_input_refs(), vec!["input", "w"]);
    }

    #[test]
    fn test_proto_input_refs_constexpr_lut_to_dense() {
        let op = MirOp::MILConstexprLutToDense { name: "cltd".into(), indices: "idx_weight".into(), lut: "lut_weight".into(), num_bits: 4 };
        assert_eq!(op.proto_input_refs(), vec!["idx_weight", "lut_weight"]);
    }

    #[test]
    fn test_proto_input_refs_gather() {
        let op = MirOp::MILGather { name: "g".into(), x: nid("data"), indices: nid("idx"), axis: 0 };
        assert_eq!(op.proto_input_refs(), vec!["data", "idx"]);
    }

    #[test]
    fn test_proto_input_refs_slice_update() {
        let op = MirOp::MILSliceUpdate { name: "su".into(), x: nid("data"), update: nid("upd"), begin: vec![0], end: vec![5] };
        assert_eq!(op.proto_input_refs(), vec!["data", "upd"]);
    }

    #[test]
    fn test_proto_input_refs_select_where() {
        let select = MirOp::MILSelect { name: "s".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        assert_eq!(select.proto_input_refs(), vec!["c", "a", "b"]);
        let where_op = MirOp::MILWhere { name: "w".into(), condition: nid("c"), x: nid("a"), y: nid("b") };
        assert_eq!(where_op.proto_input_refs(), vec!["c", "a", "b"]);
    }

    // ─── is_proto_supported() tests ────────────────────────────────

    #[test]
    fn test_is_proto_supported_const() {
        let op = MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 };
        assert!(op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_linear() {
        let op = MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None };
        assert!(op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_einsum_unsupported() {
        let op = MirOp::MILEinsum { name: "e".into(), inputs: vec![], equation: "".into() };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_conv_transpose_unsupported() {
        let op = MirOp::MILConvTranspose { name: "ct".into(), x: nid("x"), weight: nid("w"), pad_type: "valid".into(), groups: 1, strides: vec![1], pad_amounts: vec![0], dilations: vec![1], output_shape: vec![] };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_relu6_unsupported() {
        let op = MirOp::MILRelu6 { name: "r6".into(), x: nid("x") };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_batch_norm_unsupported() {
        let op = MirOp::MILBatchNorm { name: "bn".into(), x: nid("x"), mean: "m".into(), variance: "v".into(), gamma: None, beta: None, epsilon: 1e-5 };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_max_pool_unsupported() {
        let op = MirOp::MILMaxPool { name: "mp".into(), x: nid("x"), kernel_sizes: vec![3], strides: vec![1], pad_types: vec!["valid".into()], pad_amounts: vec![0] };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_sdpa() {
        let op = MirOp::MILScaledDotProductAttention { name: "sdpa".into(), query: nid("q"), key: nid("k"), value: nid("v"), attention_mask: None, scale: None };
        assert!(op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_layer_norm() {
        let op = MirOp::MILLayerNorm { name: "ln".into(), x: nid("x"), weight: "w".into(), bias: None, epsilon: 1e-5, axes: vec![] };
        assert!(op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_constexpr_variants() {
        let op = MirOp::MILConstexprAffineDequantize { name: "cad".into(), quantized_data: "qd".into(), scale: 1.0, zero_point: 0, axis: 0 };
        assert!(op.is_proto_supported());
        let op2 = MirOp::MILConstexprLutToDense { name: "cltd".into(), indices: "i".into(), lut: "l".into(), num_bits: 4 };
        assert!(op2.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_quantize_unsupported() {
        let op = MirOp::MILQuantize { name: "q".into(), x: nid("x"), scale: 1.0, zero_point: 0, axis: 0, output_dtype: MilDtype::UInt8 };
        // Quantize has no MirOpCompat variant — falls to Unsupported
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_control_flow_unsupported() {
        let op = MirOp::MILCond { name: "c".into(), pred: nid("p"), true_graph: "t".into(), false_graph: "f".into() };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_random_unsupported() {
        let op = MirOp::MILRandomBernoulli { name: "rb".into(), shape: vec![1], prob: 0.5, seed: None, dtype: MilDtype::Fp16 };
        assert!(!op.is_proto_supported());
    }

    #[test]
    fn test_is_proto_supported_supported_ops_count() {
        // Verify that a reasonable number of ops are supported
        let supported_ops: Vec<MirOp> = vec![
            MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 },
            MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None },
            MirOp::MILMatMul { name: "mm".into(), x: nid("x"), y: nid("y"), transpose_y: false },
            MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") },
            MirOp::MILMul { name: "m".into(), x: nid("x"), y: nid("y") },
            MirOp::MILSub { name: "s".into(), x: nid("x"), y: nid("y") },
            MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![] },
            MirOp::MILTranspose { name: "t".into(), x: nid("x"), perm: vec![] },
            MirOp::MILSoftmax { name: "sm".into(), x: nid("x"), axis: -1 },
            MirOp::MILRelu { name: "rl".into(), x: nid("x") },
        ];
        assert!(supported_ops.iter().all(|op| op.is_proto_supported()),
            "All basic supported ops should return true");
    }

    // ─── Cross-consistency tests ───────────────────────────────────

    #[test]
    fn test_proto_op_type_consistency_with_mil_op_name() {
        // For ALL MirOp variants, proto_op_type() must equal mil_op_name()
        // (they delegate to the same implementation)
        let sample_ops: Vec<MirOp> = vec![
            MirOp::MILConst { name: "c".into(), value_path: "v".into(), dtype: MilDtype::Fp16 },
            MirOp::MILLinear { name: "l".into(), x: nid("x"), weight: "w".into(), bias: None },
            MirOp::MILMatMul { name: "m".into(), x: nid("x"), y: nid("y"), transpose_y: false },
            MirOp::MILEinsum { name: "e".into(), inputs: vec![], equation: "".into() },
            MirOp::MILConv { name: "c".into(), x: nid("x"), weight: nid("w"), pad_type: "valid".into(), groups: 1, strides: vec![1], pad_amounts: vec![0], dilations: vec![1] },
            MirOp::MILAdd { name: "a".into(), x: nid("x"), y: nid("y") },
            MirOp::MILAbs { name: "ab".into(), x: nid("x") },
            MirOp::MILReshape { name: "r".into(), x: nid("x"), shape: vec![1] },
            MirOp::MILReduceMean { name: "rm".into(), x: nid("x"), axes: vec![1], keep_dims: false },
            MirOp::MILLayerNorm { name: "ln".into(), x: nid("x"), weight: "w".into(), bias: None, epsilon: 1e-5, axes: vec![] },
            MirOp::MILScaledDotProductAttention { name: "sdpa".into(), query: nid("q"), key: nid("k"), value: nid("v"), attention_mask: None, scale: None },
            MirOp::MILReadState { name: "rs".into(), state_id: "s".into(), shape: vec![], dtype: MilDtype::Fp16 },
            MirOp::MILConvTranspose { name: "ct".into(), x: nid("x"), weight: nid("w"), pad_type: "valid".into(), groups: 1, strides: vec![1], pad_amounts: vec![0], dilations: vec![1], output_shape: vec![] },
            MirOp::MILRelu6 { name: "r6".into(), x: nid("x") },
            MirOp::MILBatchNorm { name: "bn".into(), x: nid("x"), mean: "m".into(), variance: "v".into(), gamma: None, beta: None, epsilon: 1e-5 },
            MirOp::MILMaxPool { name: "mp".into(), x: nid("x"), kernel_sizes: vec![3], strides: vec![1], pad_types: vec!["valid".into()], pad_amounts: vec![0] },
            MirOp::MILFill { name: "f".into(), shape: vec![], value: 0.0, dtype: MilDtype::Fp16 },
            MirOp::MILCond { name: "c".into(), pred: nid("p"), true_graph: "t".into(), false_graph: "f".into() },
            MirOp::MILRandomBernoulli { name: "rb".into(), shape: vec![], prob: 0.5, seed: None, dtype: MilDtype::Fp16 },
        ];
        for op in &sample_ops {
            assert_eq!(op.proto_op_type(), op.mil_op_name(),
                "proto_op_type != mil_op_name for {:?}", op.mil_op_name());
        }
    }

    #[test]
    fn test_proto_input_refs_no_name_in_output() {
        // Ensure the output name is NEVER included in input refs
        let op = MirOp::MILAdd { name: "my_add_output".into(), x: nid("left"), y: nid("right") };
        let refs = op.proto_input_refs();
        assert!(!refs.contains(&"my_add_output".to_string()),
            "Output name should not appear in input refs");
    }
}
