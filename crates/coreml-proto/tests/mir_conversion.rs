//! Test for the From<ane_ir::mir::MirOp> for MirOpCompat conversion (T-17).
//!
//! This test verifies that the conversion covers all MirOp variants.
//! If a new MirOp variant is added without updating the From impl,
//! this test will fail to compile (exhaustive match check).
//!
//! Run with: cargo test -p ane-coreml-proto --features mir-conversion

use ane_coreml_proto::mir_compat::{MilDtypeCompat, MirOpCompat};
use ane_ir::mir::{MirNodeId, MirOp, MilDtype};

fn nid(s: &str) -> MirNodeId {
    MirNodeId(s.to_string())
}

/// Test that a representative sample of MirOp variants convert to the
/// correct MirOpCompat variant. This also ensures the From impl's
/// match is exhaustive — if a new MirOp variant is added, the compiler
/// will reject the match until it's handled.
#[test]
fn test_mir_op_to_compat_exhaustive_coverage() {
    // ─── Constants ───────────────────────────────────────────────
    let op = MirOp::MILConst {
        name: "w".into(),
        value_path: "/tmp/w.bin".into(),
        dtype: MilDtype::Fp16,
    };
    let compat: MirOpCompat = op.into();
    assert!(matches!(compat, MirOpCompat::Const { .. }));

    // ─── Linear / FC ─────────────────────────────────────────────
    let op = MirOp::MILLinear {
        name: "out".into(),
        x: nid("x"),
        weight: "w".into(),
        bias: Some("b".into()),
    };
    let compat: MirOpCompat = op.into();
    assert!(matches!(compat, MirOpCompat::Linear { .. }));

    let op = MirOp::MILMatMul {
        name: "mm".into(),
        x: nid("a"),
        y: nid("b"),
        transpose_y: false,
    };
    let compat: MirOpCompat = op.into();
    assert!(matches!(compat, MirOpCompat::MatMul { .. }));

    // ─── Elementwise Binary ──────────────────────────────────────
    let op = MirOp::MILAdd {
        name: "add1".into(),
        x: nid("a"),
        y: nid("b"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Add { .. }));

    let op = MirOp::MILMul {
        name: "mul1".into(),
        x: nid("a"),
        y: nid("b"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Mul { .. }));

    let op = MirOp::MILRealDiv {
        name: "div1".into(),
        x: nid("a"),
        y: nid("b"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::RealDiv { .. }));

    let op = MirOp::MILEqual {
        name: "eq1".into(),
        x: nid("a"),
        y: nid("b"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Equal { .. }));

    // ─── Elementwise Unary ───────────────────────────────────────
    let op = MirOp::MILRelu {
        name: "relu1".into(),
        x: nid("a"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Relu { .. }));

    let op = MirOp::MILSilu {
        name: "silu1".into(),
        x: nid("a"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Silu { .. }));

    let op = MirOp::MILGelu {
        name: "gelu1".into(),
        x: nid("a"),
        mode: "exact".into(),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Gelu { .. }));

    let op = MirOp::MILExp {
        name: "exp1".into(),
        x: nid("a"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Exp { .. }));

    // ─── Reduction ───────────────────────────────────────────────
    let op = MirOp::MILReduceSum {
        name: "rs".into(),
        x: nid("a"),
        axes: vec![1],
        keep_dims: false,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReduceSum { .. }));

    let op = MirOp::MILReduceMean {
        name: "rm".into(),
        x: nid("a"),
        axes: vec![1],
        keep_dims: true,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReduceMean { .. }));

    // ─── Tensor Transform ────────────────────────────────────────
    let op = MirOp::MILReshape {
        name: "r1".into(),
        x: nid("a"),
        shape: vec![1, 2, 3],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Reshape { .. }));

    let op = MirOp::MILTranspose {
        name: "t1".into(),
        x: nid("a"),
        perm: vec![1, 0],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Transpose { .. }));

    let op = MirOp::MILConcat {
        name: "c1".into(),
        values: vec![nid("a"), nid("b")],
        axis: 0,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Concat { .. }));

    // ─── State ───────────────────────────────────────────────────
    let op = MirOp::MILReadState {
        name: "rs".into(),
        state_id: "kv".into(),
        shape: vec![1, 128],
        dtype: MilDtype::Fp16,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReadState { .. }));

    let op = MirOp::MILCoremlUpdateState {
        name: "us".into(),
        state_id: "kv".into(),
        value: nid("v"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::CoremlUpdateState { .. }));

    let op = MirOp::MILStateWrite {
        name: "sw".into(),
        state_ref: "kv".into(),
        value: nid("v"),
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::StateWrite { .. }));

    // ─── Attention ───────────────────────────────────────────────
    let op = MirOp::MILScaledDotProductAttention {
        name: "attn".into(),
        query: nid("q"),
        key: nid("k"),
        value: nid("v"),
        attention_mask: None,
        scale: Some(0.125),
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::ScaledDotProductAttention { scale, attention_mask, .. } = compat {
        assert_eq!(scale, Some(0.125), "scale should be preserved through MirOp → MirOpCompat conversion");
        assert!(attention_mask.is_none(), "attention_mask should be None when source is None");
    } else {
        panic!("Expected MirOpCompat::ScaledDotProductAttention");
    }

    // Test with attention_mask present
    let op_with_mask = MirOp::MILScaledDotProductAttention {
        name: "attn_masked".into(),
        query: nid("q"),
        key: nid("k"),
        value: nid("v"),
        attention_mask: Some(nid("mask")),
        scale: Some(0.0625),
    };
    let compat_masked = MirOpCompat::from(op_with_mask);
    if let MirOpCompat::ScaledDotProductAttention { attention_mask, scale, .. } = compat_masked {
        assert_eq!(attention_mask, Some("mask".to_string()), "attention_mask should be preserved through conversion");
        assert_eq!(scale, Some(0.0625), "scale should be preserved through conversion");
    } else {
        panic!("Expected MirOpCompat::ScaledDotProductAttention");
    }

    // ─── Normalization ───────────────────────────────────────────
    let op = MirOp::MILLayerNorm {
        name: "ln".into(),
        x: nid("a"),
        weight: "w".into(),
        bias: Some("b".into()),
        epsilon: 1e-5,
        axes: vec![2],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::LayerNorm { .. }));

    // ─── Conv ────────────────────────────────────────────────────
    let op = MirOp::MILConv {
        name: "conv1".into(),
        x: nid("a"),
        weight: nid("w"),
        pad_type: "valid".into(),
        groups: 1,
        strides: vec![1],
        pad_amounts: vec![0],
        dilations: vec![1],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Conv { .. }));

    // ─── Ops that map to Unsupported ─────────────────────────────
    let op = MirOp::MILConvTranspose {
        name: "ct".into(),
        x: nid("a"),
        weight: nid("w"),
        pad_type: "valid".into(),
        groups: 1,
        strides: vec![1],
        pad_amounts: vec![0],
        dilations: vec![1],
        output_shape: vec![1],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Unsupported { .. }));

    let op = MirOp::MILRandomNormal {
        name: "rn".into(),
        shape: vec![1],
        mean: 0.0,
        stddev: 1.0,
        seed: None,
        dtype: MilDtype::Fp32,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Unsupported { .. }));
}

/// Test that dtype conversion maps correctly.
#[test]
fn test_mir_op_dtype_conversion() {
    let op = MirOp::MILCast {
        name: "cast_fp16".into(),
        x: nid("a"),
        dtype: MilDtype::Fp16,
    };
    if let MirOpCompat::Cast { dtype, .. } = MirOpCompat::from(op) {
        assert_eq!(dtype, MilDtypeCompat::Fp16);
    }

    let op = MirOp::MILCast {
        name: "cast_fp32".into(),
        x: nid("a"),
        dtype: MilDtype::Fp32,
    };
    if let MirOpCompat::Cast { dtype, .. } = MirOpCompat::from(op) {
        assert_eq!(dtype, MilDtypeCompat::Fp32);
    }

    let op = MirOp::MILCast {
        name: "cast_int32".into(),
        x: nid("a"),
        dtype: MilDtype::Int32,
    };
    if let MirOpCompat::Cast { dtype, .. } = MirOpCompat::from(op) {
        assert_eq!(dtype, MilDtypeCompat::Int32);
    }
}

/// Test that specific field values survive the conversion.
#[test]
fn test_mir_op_field_values_preserved() {
    let op = MirOp::MILLinear {
        name: "projection".into(),
        x: nid("input_tensor"),
        weight: "weight_matrix".into(),
        bias: Some("bias_vector".into()),
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::Linear { name, x, weight_name, bias_name } = compat {
        assert_eq!(name, "projection");
        assert_eq!(x, "input_tensor");
        assert_eq!(weight_name, "weight_matrix");
        assert_eq!(bias_name, Some("bias_vector".into()));
    } else {
        panic!("Expected MirOpCompat::Linear");
    }

    let op = MirOp::MILSoftmax {
        name: "sm".into(),
        x: nid("logits"),
        axis: -1,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::Softmax { axis, .. } = compat {
        assert_eq!(axis, -1);
    } else {
        panic!("Expected MirOpCompat::Softmax");
    }
}
