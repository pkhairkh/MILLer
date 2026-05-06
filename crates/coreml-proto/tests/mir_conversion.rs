//! Test for the From<ane_ir::mir::MirOp> for MirOpCompat conversion (T-17).
//!
//! This test verifies that the conversion covers all MirOp variants.
//! If a new MirOp variant is added without updating the From impl,
//! this test will fail to compile (exhaustive match check).
//!
//! Run with: cargo test -p ane-coreml-proto --features mir-conversion

use ane_coreml_proto::mir_compat::{MilDtypeCompat, MirOpCompat};
use ane_ir::mir::{MilDtype, MirNodeId, MirOp};
use ane_ir::toproto::ToProto;

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

    let op = MirOp::MILMatMul { name: "mm".into(), x: nid("a"), y: nid("b"), transpose_y: false };
    let compat: MirOpCompat = op.into();
    assert!(matches!(compat, MirOpCompat::MatMul { .. }));

    // ─── Elementwise Binary ──────────────────────────────────────
    let op = MirOp::MILAdd { name: "add1".into(), x: nid("a"), y: nid("b") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Add { .. }));

    let op = MirOp::MILMul { name: "mul1".into(), x: nid("a"), y: nid("b") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Mul { .. }));

    let op = MirOp::MILRealDiv { name: "div1".into(), x: nid("a"), y: nid("b") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::RealDiv { .. }));

    let op = MirOp::MILEqual { name: "eq1".into(), x: nid("a"), y: nid("b") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Equal { .. }));

    // ─── Elementwise Unary ───────────────────────────────────────
    let op = MirOp::MILRelu { name: "relu1".into(), x: nid("a") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Relu { .. }));

    let op = MirOp::MILSilu { name: "silu1".into(), x: nid("a") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Silu { .. }));

    let op = MirOp::MILGelu { name: "gelu1".into(), x: nid("a"), mode: "exact".into() };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Gelu { .. }));

    let op = MirOp::MILExp { name: "exp1".into(), x: nid("a") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Exp { .. }));

    // ─── Reduction ───────────────────────────────────────────────
    let op =
        MirOp::MILReduceSum { name: "rs".into(), x: nid("a"), axes: vec![1], keep_dims: false };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReduceSum { .. }));

    let op =
        MirOp::MILReduceMean { name: "rm".into(), x: nid("a"), axes: vec![1], keep_dims: true };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReduceMean { .. }));

    // ─── Tensor Transform ────────────────────────────────────────
    let op = MirOp::MILReshape { name: "r1".into(), x: nid("a"), shape: vec![1, 2, 3] };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Reshape { .. }));

    let op = MirOp::MILTranspose { name: "t1".into(), x: nid("a"), perm: vec![1, 0] };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Transpose { .. }));

    let op = MirOp::MILConcat { name: "c1".into(), values: vec![nid("a"), nid("b")], axis: 0 };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::Concat { .. }));

    // ─── State ───────────────────────────────────────────────────
    let op = MirOp::MILReadState {
        name: "rs".into(),
        state_id: "kv".into(),
        shape: vec![1, 128],
        dtype: MilDtype::Fp16,
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::ReadState { .. }));

    let op =
        MirOp::MILCoremlUpdateState { name: "us".into(), state_id: "kv".into(), value: nid("v") };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::CoremlUpdateState { .. }));

    let op = MirOp::MILStateWrite { name: "sw".into(), state_ref: "kv".into(), value: nid("v") };
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
        assert_eq!(
            scale,
            Some(0.125),
            "scale should be preserved through MirOp → MirOpCompat conversion"
        );
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
        assert_eq!(
            attention_mask,
            Some("mask".to_string()),
            "attention_mask should be preserved through conversion"
        );
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

    // ─── Pooling (T-66 / I-40) ───────────────────────────────────
    let op = MirOp::MILMaxPool {
        name: "mp".into(),
        x: nid("a"),
        kernel_sizes: vec![3, 3],
        strides: vec![1, 1],
        pad_types: vec!["valid".into()],
        pad_amounts: vec![0, 0, 0, 0],
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::MaxPool { name, x, kernel_sizes, strides, pad_type, pad_amounts } = compat {
        assert_eq!(name, "mp");
        assert_eq!(x, "a");
        assert_eq!(kernel_sizes, vec![3, 3]);
        assert_eq!(strides, vec![1, 1]);
        assert_eq!(pad_type, "valid");
        assert_eq!(pad_amounts, vec![0, 0, 0, 0]);
    } else {
        panic!("Expected MirOpCompat::MaxPool");
    }

    let op = MirOp::MILAvgPool {
        name: "ap".into(),
        x: nid("a"),
        kernel_sizes: vec![2, 2],
        strides: vec![2, 2],
        pad_types: vec!["same".into()],
        pad_amounts: vec![1, 1, 1, 1],
        count_include_padding: true,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::AvgPool {
        name,
        x,
        kernel_sizes,
        strides,
        pad_type,
        pad_amounts,
        count_include_padding,
    } = compat
    {
        assert_eq!(name, "ap");
        assert_eq!(x, "a");
        assert_eq!(kernel_sizes, vec![2, 2]);
        assert_eq!(strides, vec![2, 2]);
        assert_eq!(pad_type, "same");
        assert_eq!(pad_amounts, vec![1, 1, 1, 1]);
        assert!(count_include_padding);
    } else {
        panic!("Expected MirOpCompat::AvgPool");
    }

    let op = MirOp::MILL2Pool {
        name: "lp".into(),
        x: nid("a"),
        kernel_sizes: vec![3, 3],
        strides: vec![2, 2],
        pad_types: vec!["valid".into()],
        pad_amounts: vec![0, 0, 0, 0],
    };
    assert!(matches!(MirOpCompat::from(op), MirOpCompat::L2Pool { .. }));

    // ─── Spatial Rearrangement (T-66 / I-40) ─────────────────────
    let op = MirOp::MILDepthToSpace { name: "d2s".into(), x: nid("a"), block_size: 4 };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::DepthToSpace { name, x, block_size } = compat {
        assert_eq!(name, "d2s");
        assert_eq!(x, "a");
        assert_eq!(block_size, 4);
    } else {
        panic!("Expected MirOpCompat::DepthToSpace");
    }

    let op = MirOp::MILSpaceToDepth { name: "s2d".into(), x: nid("a"), block_size: 2 };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::SpaceToDepth { name, x, block_size } = compat {
        assert_eq!(name, "s2d");
        assert_eq!(x, "a");
        assert_eq!(block_size, 2);
    } else {
        panic!("Expected MirOpCompat::SpaceToDepth");
    }

    let op = MirOp::MILPixelShuffle { name: "ps".into(), x: nid("a"), upscale_factor: 3 };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::PixelShuffle { name, x, upscale_factor } = compat {
        assert_eq!(name, "ps");
        assert_eq!(x, "a");
        assert_eq!(upscale_factor, 3);
    } else {
        panic!("Expected MirOpCompat::PixelShuffle");
    }

    let op = MirOp::MILPixelUnshuffle { name: "pu".into(), x: nid("a"), downscale_factor: 2 };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::PixelUnshuffle { name, x, downscale_factor } = compat {
        assert_eq!(name, "pu");
        assert_eq!(x, "a");
        assert_eq!(downscale_factor, 2);
    } else {
        panic!("Expected MirOpCompat::PixelUnshuffle");
    }

    // ─── Normalization (T-66 / I-40) ─────────────────────────────
    let op = MirOp::MILBatchNorm {
        name: "bn".into(),
        x: nid("a"),
        mean: "mean_w".into(),
        variance: "var_w".into(),
        gamma: Some("gamma_w".into()),
        beta: Some("beta_w".into()),
        epsilon: 1e-5,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::BatchNorm { name, x, mean, variance, gamma, beta, epsilon } = compat {
        assert_eq!(name, "bn");
        assert_eq!(x, "a");
        assert_eq!(mean, "mean_w");
        assert_eq!(variance, "var_w");
        assert_eq!(gamma, Some("gamma_w".to_string()));
        assert_eq!(beta, Some("beta_w".to_string()));
        assert!((epsilon - 1e-5).abs() < 1e-10);
    } else {
        panic!("Expected MirOpCompat::BatchNorm");
    }

    let op = MirOp::MILInstanceNorm {
        name: "in".into(),
        x: nid("a"),
        gamma: None,
        beta: None,
        epsilon: 1e-4,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::InstanceNorm { name, x, gamma, beta, epsilon } = compat {
        assert_eq!(name, "in");
        assert_eq!(x, "a");
        assert!(gamma.is_none());
        assert!(beta.is_none());
        assert!((epsilon - 1e-4).abs() < 1e-10);
    } else {
        panic!("Expected MirOpCompat::InstanceNorm");
    }

    let op = MirOp::MILL2Norm { name: "l2".into(), x: nid("a"), epsilon: 1e-6, axes: vec![1, 2] };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::L2Norm { name, x, epsilon, axes } = compat {
        assert_eq!(name, "l2");
        assert_eq!(x, "a");
        assert!((epsilon - 1e-6).abs() < 1e-10);
        assert_eq!(axes, vec![1, 2]);
    } else {
        panic!("Expected MirOpCompat::L2Norm");
    }

    // ─── Quantize / Dequantize (T-66 / I-40) ─────────────────────
    let op = MirOp::MILQuantize {
        name: "q".into(),
        x: nid("a"),
        scale: 0.1,
        zero_point: 128,
        axis: 0,
        output_dtype: MilDtype::UInt8,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::Quantize { name, x, scale, zero_point, axis, output_dtype } = compat {
        assert_eq!(name, "q");
        assert_eq!(x, "a");
        assert!((scale - 0.1).abs() < 1e-10);
        assert_eq!(zero_point, 128);
        assert_eq!(axis, 0);
        assert_eq!(output_dtype, MilDtypeCompat::UInt8);
    } else {
        panic!("Expected MirOpCompat::Quantize");
    }

    let op = MirOp::MILDequantize {
        name: "dq".into(),
        x: nid("a"),
        scale: 0.05,
        zero_point: 0,
        axis: 1,
        output_dtype: MilDtype::Fp16,
    };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::Dequantize { name, x, scale, zero_point, axis, output_dtype } = compat {
        assert_eq!(name, "dq");
        assert_eq!(x, "a");
        assert!((scale - 0.05).abs() < 1e-10);
        assert_eq!(zero_point, 0);
        assert_eq!(axis, 1);
        assert_eq!(output_dtype, MilDtypeCompat::Fp16);
    } else {
        panic!("Expected MirOpCompat::Dequantize");
    }
}

/// Test that dtype conversion maps correctly.
#[test]
fn test_mir_op_dtype_conversion() {
    let op = MirOp::MILCast { name: "cast_fp16".into(), x: nid("a"), dtype: MilDtype::Fp16 };
    if let MirOpCompat::Cast { dtype, .. } = MirOpCompat::from(op) {
        assert_eq!(dtype, MilDtypeCompat::Fp16);
    }

    let op = MirOp::MILCast { name: "cast_fp32".into(), x: nid("a"), dtype: MilDtype::Fp32 };
    if let MirOpCompat::Cast { dtype, .. } = MirOpCompat::from(op) {
        assert_eq!(dtype, MilDtypeCompat::Fp32);
    }

    let op = MirOp::MILCast { name: "cast_int32".into(), x: nid("a"), dtype: MilDtype::Int32 };
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

    let op = MirOp::MILSoftmax { name: "sm".into(), x: nid("logits"), axis: -1 };
    let compat = MirOpCompat::from(op);
    if let MirOpCompat::Softmax { axis, .. } = compat {
        assert_eq!(axis, -1);
    } else {
        panic!("Expected MirOpCompat::Softmax");
    }
}

/// Test that input_names() returns correct values for T-66 (I-40) variants.
#[test]
fn test_t66_input_names() {
    // Pooling ops: single input x
    let mp = MirOpCompat::MaxPool {
        name: "mp".into(),
        x: "a".into(),
        kernel_sizes: vec![3, 3],
        strides: vec![1, 1],
        pad_type: "valid".into(),
        pad_amounts: vec![0, 0, 0, 0],
    };
    assert_eq!(mp.input_names(), vec!["a"]);

    let ap = MirOpCompat::AvgPool {
        name: "ap".into(),
        x: "b".into(),
        kernel_sizes: vec![2, 2],
        strides: vec![2, 2],
        pad_type: "same".into(),
        pad_amounts: vec![1, 1, 1, 1],
        count_include_padding: true,
    };
    assert_eq!(ap.input_names(), vec!["b"]);

    let lp = MirOpCompat::L2Pool {
        name: "lp".into(),
        x: "c".into(),
        kernel_sizes: vec![3, 3],
        strides: vec![2, 2],
        pad_type: "valid".into(),
        pad_amounts: vec![0, 0, 0, 0],
    };
    assert_eq!(lp.input_names(), vec!["c"]);

    // Spatial rearrangement ops: single input x
    let d2s = MirOpCompat::DepthToSpace { name: "d2s".into(), x: "d".into(), block_size: 4 };
    assert_eq!(d2s.input_names(), vec!["d"]);

    let s2d = MirOpCompat::SpaceToDepth { name: "s2d".into(), x: "e".into(), block_size: 2 };
    assert_eq!(s2d.input_names(), vec!["e"]);

    let ps = MirOpCompat::PixelShuffle { name: "ps".into(), x: "f".into(), upscale_factor: 3 };
    assert_eq!(ps.input_names(), vec!["f"]);

    let pu = MirOpCompat::PixelUnshuffle { name: "pu".into(), x: "g".into(), downscale_factor: 2 };
    assert_eq!(pu.input_names(), vec!["g"]);

    // Normalization ops
    let l2n =
        MirOpCompat::L2Norm { name: "l2".into(), x: "h".into(), epsilon: 1e-6, axes: vec![1, 2] };
    assert_eq!(l2n.input_names(), vec!["h"]);

    let bn = MirOpCompat::BatchNorm {
        name: "bn".into(),
        x: "i".into(),
        mean: "m".into(),
        variance: "v".into(),
        gamma: Some("g".into()),
        beta: Some("b".into()),
        epsilon: 1e-5,
    };
    assert_eq!(bn.input_names(), vec!["i", "m", "v", "g", "b"]);

    let bn_no_params = MirOpCompat::BatchNorm {
        name: "bn2".into(),
        x: "i".into(),
        mean: "m".into(),
        variance: "v".into(),
        gamma: None,
        beta: None,
        epsilon: 1e-5,
    };
    assert_eq!(bn_no_params.input_names(), vec!["i", "m", "v"]);

    let inorm = MirOpCompat::InstanceNorm {
        name: "in".into(),
        x: "j".into(),
        gamma: Some("g".into()),
        beta: None,
        epsilon: 1e-4,
    };
    assert_eq!(inorm.input_names(), vec!["j", "g"]);

    let inorm_no_params = MirOpCompat::InstanceNorm {
        name: "in2".into(),
        x: "j".into(),
        gamma: None,
        beta: None,
        epsilon: 1e-4,
    };
    assert_eq!(inorm_no_params.input_names(), vec!["j"]);

    // Quantize / Dequantize ops: single input x
    let q = MirOpCompat::Quantize {
        name: "q".into(),
        x: "k".into(),
        scale: 0.1,
        zero_point: 128,
        axis: 0,
        output_dtype: MilDtypeCompat::UInt8,
    };
    assert_eq!(q.input_names(), vec!["k"]);

    let dq = MirOpCompat::Dequantize {
        name: "dq".into(),
        x: "l".into(),
        scale: 0.05,
        zero_point: 0,
        axis: 1,
        output_dtype: MilDtypeCompat::Fp16,
    };
    assert_eq!(dq.input_names(), vec!["l"]);
}

/// Test that output_name() returns the name field for T-66 (I-40) variants.
#[test]
fn test_t66_output_name() {
    let mp = MirOpCompat::MaxPool {
        name: "mp_out".into(),
        x: "a".into(),
        kernel_sizes: vec![3, 3],
        strides: vec![1, 1],
        pad_type: "valid".into(),
        pad_amounts: vec![0, 0, 0, 0],
    };
    assert_eq!(mp.output_name(), "mp_out");

    let d2s = MirOpCompat::DepthToSpace { name: "d2s_out".into(), x: "d".into(), block_size: 4 };
    assert_eq!(d2s.output_name(), "d2s_out");

    let bn = MirOpCompat::BatchNorm {
        name: "bn_out".into(),
        x: "i".into(),
        mean: "m".into(),
        variance: "v".into(),
        gamma: None,
        beta: None,
        epsilon: 1e-5,
    };
    assert_eq!(bn.output_name(), "bn_out");

    let q = MirOpCompat::Quantize {
        name: "q_out".into(),
        x: "k".into(),
        scale: 0.1,
        zero_point: 128,
        axis: 0,
        output_dtype: MilDtypeCompat::UInt8,
    };
    assert_eq!(q.output_name(), "q_out");
}

/// Test that remap_inputs() correctly remaps input references for T-66 (I-40) variants.
#[test]
fn test_t66_remap_inputs() {
    let remap = |name: String| -> String { format!("remapped_{}", name) };

    // Pooling: remaps x
    let mp = MirOpCompat::MaxPool {
        name: "mp".into(),
        x: "a".into(),
        kernel_sizes: vec![3, 3],
        strides: vec![1, 1],
        pad_type: "valid".into(),
        pad_amounts: vec![0, 0, 0, 0],
    };
    let remapped = mp.remap_inputs(remap);
    if let MirOpCompat::MaxPool { x, .. } = remapped {
        assert_eq!(x, "remapped_a");
    } else {
        panic!("Expected MaxPool after remap");
    }

    // DepthToSpace: remaps x
    let d2s = MirOpCompat::DepthToSpace { name: "d2s".into(), x: "d".into(), block_size: 4 };
    let remapped = d2s.remap_inputs(remap);
    if let MirOpCompat::DepthToSpace { x, .. } = remapped {
        assert_eq!(x, "remapped_d");
    } else {
        panic!("Expected DepthToSpace after remap");
    }

    // BatchNorm: remaps x, mean, variance, gamma, beta
    let bn = MirOpCompat::BatchNorm {
        name: "bn".into(),
        x: "i".into(),
        mean: "m".into(),
        variance: "v".into(),
        gamma: Some("g".into()),
        beta: Some("b".into()),
        epsilon: 1e-5,
    };
    let remapped = bn.remap_inputs(remap);
    if let MirOpCompat::BatchNorm { x, mean, variance, gamma, beta, .. } = remapped {
        assert_eq!(x, "remapped_i");
        assert_eq!(mean, "remapped_m");
        assert_eq!(variance, "remapped_v");
        assert_eq!(gamma, Some("remapped_g".to_string()));
        assert_eq!(beta, Some("remapped_b".to_string()));
    } else {
        panic!("Expected BatchNorm after remap");
    }

    // InstanceNorm with None gamma/beta
    let inorm = MirOpCompat::InstanceNorm {
        name: "in".into(),
        x: "j".into(),
        gamma: None,
        beta: None,
        epsilon: 1e-4,
    };
    let remapped = inorm.remap_inputs(remap);
    if let MirOpCompat::InstanceNorm { x, gamma, beta, .. } = remapped {
        assert_eq!(x, "remapped_j");
        assert!(gamma.is_none());
        assert!(beta.is_none());
    } else {
        panic!("Expected InstanceNorm after remap");
    }

    // L2Norm: remaps x
    let l2n =
        MirOpCompat::L2Norm { name: "l2".into(), x: "h".into(), epsilon: 1e-6, axes: vec![1, 2] };
    let remapped = l2n.remap_inputs(remap);
    if let MirOpCompat::L2Norm { x, .. } = remapped {
        assert_eq!(x, "remapped_h");
    } else {
        panic!("Expected L2Norm after remap");
    }

    // Quantize: remaps x
    let q = MirOpCompat::Quantize {
        name: "q".into(),
        x: "k".into(),
        scale: 0.1,
        zero_point: 128,
        axis: 0,
        output_dtype: MilDtypeCompat::UInt8,
    };
    let remapped = q.remap_inputs(remap);
    if let MirOpCompat::Quantize { x, .. } = remapped {
        assert_eq!(x, "remapped_k");
    } else {
        panic!("Expected Quantize after remap");
    }

    // Dequantize: remaps x
    let dq = MirOpCompat::Dequantize {
        name: "dq".into(),
        x: "l".into(),
        scale: 0.05,
        zero_point: 0,
        axis: 1,
        output_dtype: MilDtypeCompat::Fp16,
    };
    let remapped = dq.remap_inputs(remap);
    if let MirOpCompat::Dequantize { x, .. } = remapped {
        assert_eq!(x, "remapped_l");
    } else {
        panic!("Expected Dequantize after remap");
    }
}

// ─── T-P5-10: Unsupported input_names() weight materialization gap ─────

/// Test that Unsupported ops carry input names from the source MirOp,
/// so weight materialization does not silently drop referenced weights.
#[test]
fn test_unsupported_input_names_from_mir_op() {
    // MILConvTranspose has inputs x (MirNodeId) and weight (MirNodeId).
    // When it maps to Unsupported, those inputs should be preserved.
    let op = MirOp::MILConvTranspose {
        name: "ct".into(),
        x: nid("input_x"),
        weight: nid("weight_w"),
        pad_type: "valid".into(),
        groups: 1,
        strides: vec![1],
        pad_amounts: vec![0],
        dilations: vec![1],
        output_shape: vec![1],
    };
    let compat: MirOpCompat = op.into();
    if let MirOpCompat::Unsupported { inputs, op_kind, .. } = &compat {
        assert_eq!(op_kind, "conv_transpose");
        assert_eq!(
            *inputs,
            vec!["input_x", "weight_w"],
            "Unsupported from MILConvTranspose should carry input names"
        );
    } else {
        panic!("Expected MirOpCompat::Unsupported from MILConvTranspose");
    }

    // Verify input_names() returns the same thing as inputs
    assert_eq!(compat.input_names(), vec!["input_x", "weight_w"]);
}

/// Test that an Unsupported op from a unary MirOp carries a single input name.
#[test]
fn test_unsupported_input_names_unary_op() {
    // MILRelu6 has a single input x
    let op = MirOp::MILRelu6 { name: "r6".into(), x: nid("input_a") };
    let compat: MirOpCompat = op.into();
    if let MirOpCompat::Unsupported { inputs, op_kind, .. } = &compat {
        assert_eq!(op_kind, "relu6");
        assert_eq!(*inputs, vec!["input_a"]);
    } else {
        panic!("Expected MirOpCompat::Unsupported from MILRelu6");
    }
    assert_eq!(compat.input_names(), vec!["input_a"]);
}

/// Test that an Unsupported op from a MirOp with no inputs (e.g., random)
/// carries an empty inputs vec.
#[test]
fn test_unsupported_input_names_no_input_op() {
    let op = MirOp::MILRandomNormal {
        name: "rn".into(),
        shape: vec![1],
        mean: 0.0,
        stddev: 1.0,
        seed: None,
        dtype: MilDtype::Fp32,
    };
    let compat: MirOpCompat = op.into();
    if let MirOpCompat::Unsupported { inputs, .. } = &compat {
        assert!(
            inputs.is_empty(),
            "RandomNormal has no SSA inputs — Unsupported inputs should be empty"
        );
    } else {
        panic!("Expected MirOpCompat::Unsupported from MILRandomNormal");
    }
    assert!(compat.input_names().is_empty());
}

/// Test that Unsupported ops' input_names() matches MirOp::proto_input_refs().
#[test]
fn test_unsupported_input_names_matches_proto_input_refs() {
    // MILPrelu has x (MirNodeId) and alpha (String weight name)
    let op =
        MirOp::MILPrelu { name: "prelu".into(), x: nid("input_x"), alpha: "alpha_weight".into() };
    let expected_inputs = op.proto_input_refs();
    let compat: MirOpCompat = op.into();
    if let MirOpCompat::Unsupported { inputs, .. } = &compat {
        assert_eq!(*inputs, expected_inputs, "Unsupported inputs should match proto_input_refs()");
    } else {
        panic!("Expected MirOpCompat::Unsupported from MILPrelu");
    }
    assert_eq!(compat.input_names(), expected_inputs);
}

/// Test that remap_inputs correctly remaps Unsupported ops' inputs.
#[test]
fn test_unsupported_remap_inputs() {
    let op = MirOp::MILConvTranspose {
        name: "ct".into(),
        x: nid("input_x"),
        weight: nid("weight_w"),
        pad_type: "valid".into(),
        groups: 1,
        strides: vec![1],
        pad_amounts: vec![0],
        dilations: vec![1],
        output_shape: vec![1],
    };
    let compat: MirOpCompat = op.into();

    let remap = |name: String| format!("remapped_{}", name);
    let remapped = compat.remap_inputs(remap);

    if let MirOpCompat::Unsupported { inputs, .. } = &remapped {
        assert_eq!(*inputs, vec!["remapped_input_x", "remapped_weight_w"]);
    } else {
        panic!("Expected MirOpCompat::Unsupported after remap");
    }
    assert_eq!(remapped.input_names(), vec!["remapped_input_x", "remapped_weight_w"]);
}

/// Test that rename_output preserves inputs in Unsupported ops.
#[test]
fn test_unsupported_rename_output_preserves_inputs() {
    let op = MirOp::MILConvTranspose {
        name: "ct".into(),
        x: nid("input_x"),
        weight: nid("weight_w"),
        pad_type: "valid".into(),
        groups: 1,
        strides: vec![1],
        pad_amounts: vec![0],
        dilations: vec![1],
        output_shape: vec![1],
    };
    let compat: MirOpCompat = op.into();
    let renamed = compat.rename_output("new_output_name".into());

    if let MirOpCompat::Unsupported { name, inputs, op_kind, .. } = &renamed {
        assert_eq!(*name, "new_output_name");
        assert_eq!(*op_kind, "conv_transpose");
        assert_eq!(*inputs, vec!["input_x", "weight_w"], "rename_output should preserve inputs");
    } else {
        panic!("Expected MirOpCompat::Unsupported after rename_output");
    }
}

/// Test that an Unsupported op constructed manually with explicit inputs
/// returns those inputs from input_names().
#[test]
fn test_unsupported_manual_construction_input_names() {
    let unsupported = MirOpCompat::Unsupported {
        op_kind: "custom_op".into(),
        name: "custom_out".into(),
        params_json: "{}".into(),
        inputs: vec!["input_a".into(), "input_b".into(), "weight_c".into()],
    };
    assert_eq!(unsupported.input_names(), vec!["input_a", "input_b", "weight_c"]);
    assert_eq!(unsupported.output_name(), "custom_out");
}
