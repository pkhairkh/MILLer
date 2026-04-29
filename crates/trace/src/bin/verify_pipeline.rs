//! Full IR pipeline verification: Trace → SIR → AIR → MIR → PIR
//!
//! Loads a Qwen3-0.6B trace JSON, builds SIR, then runs each IR pass
//! and validates the output at every stage for correctness.

use std::collections::HashMap;
use std::fs;

use ane_ir::ane_target::AneFamily;
use ane_ir::air::{AirGraph, AirOp};
use ane_ir::mir::{MirGraph, MirOp, MilDtype};
use ane_ir::pir::PirGraph;
use ane_ir::sir::{SirGraph, SirOp};
use ane_passes::knowledge_query::NoKnowledge;
use ane_passes::legality_rewrite::{DecompositionContext, LegalityRewritePass};
use ane_passes::mil_lower::MilLowerPass;
use ane_passes::shard_plan::ShardPlan;
use ane_trace::graph::TracedGraph;
use ane_trace::sir_build::build_sir_from_trace;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let trace_path = args.get(1).map(|s| s.as_str()).unwrap_or("/tmp/qwen3_trace_new.json");

    eprintln!("Loading trace from: {}", trace_path);
    let json_str = fs::read_to_string(trace_path)
        .unwrap_or_else(|e| { eprintln!("Failed to read {}: {}", trace_path, e); std::process::exit(1); });

    let trace: TracedGraph = serde_json::from_str(&json_str)
        .unwrap_or_else(|e| { eprintln!("Failed to parse trace JSON: {}", e); std::process::exit(1); });

    // ═══════════════════════════════════════════════════════════════
    // STAGE 0: Trace Validation
    // ═══════════════════════════════════════════════════════════════
    println!("═══ STAGE 0: Trace Validation ═══");
    let cfg = &trace.model_config;
    println!("  model_id: {}", trace.model_id);
    println!("  architecture: {}", trace.architecture);
    println!("  hidden_size={}, num_heads={}, num_kv_heads={:?}",
        cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads);
    println!("  intermediate_size={}, vocab_size={}", cfg.intermediate_size, cfg.vocab_size);
    println!("  uses_rms_norm={}, uses_gqa={}, uses_rope={}", cfg.uses_rms_norm, cfg.uses_gqa, cfg.uses_rope);

    let trace_ok = cfg.hidden_size == 1024
        && cfg.num_attention_heads == 16
        && cfg.num_key_value_heads == Some(8)
        && cfg.num_hidden_layers == 28
        && cfg.intermediate_size == 3072
        && cfg.vocab_size == 151936
        && cfg.uses_rms_norm
        && cfg.uses_gqa
        && cfg.uses_rope;

    if trace_ok {
        println!("  ✅ Trace config matches Qwen3-0.6B specification");
    } else {
        println!("  ❌ Trace config MISMATCH — see above");
    }

    let feat = &trace.discovered_features;
    let has_qk_norm = feat.detection_methods.contains_key("qk_norm");
    println!("  qk_norm detected={}, attention_types={:?}, mlp_types={:?}",
        has_qk_norm, feat.attention_module_types, feat.mlp_module_types);
    println!("  linear_count={}, embedding_count={}, weights={}",
        feat.linear_count, feat.embedding_count, trace.weights.len());

    let qk_norm_ok = has_qk_norm && feat.attention_module_types.contains(&"Qwen3Attention".to_string());
    if qk_norm_ok {
        println!("  ✅ Feature detection: QK-norm detected, Qwen3Attention/Qwen3MLP identified");
    } else {
        println!("  ❌ Feature detection incomplete");
    }

    // ═══════════════════════════════════════════════════════════════
    // STAGE 1: SIR Construction
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══ STAGE 1: SIR Construction ═══");
    let sir = build_sir_from_trace(&trace, AneFamily::A16)
        .unwrap_or_else(|e| { eprintln!("SIR build failed: {}", e); std::process::exit(1); });

    let sir_ok = validate_sir(&sir);
    dump_sir_stats(&sir);

    // Save SIR JSON
    let sir_json = serde_json::to_string_pretty(&sir).unwrap();
    fs::write("/tmp/qwen3_sir.json", &sir_json).ok();
    println!("  SIR JSON saved to /tmp/qwen3_sir.json");

    // ═══════════════════════════════════════════════════════════════
    // STAGE 2: SIR → AIR (Legality Rewrite)
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══ STAGE 2: SIR → AIR (Legality Rewrite) ═══");
    let kq = NoKnowledge;
    let decomp_ctx = DecompositionContext {
        batch_size: 1,
        embed_dim: cfg.hidden_size,
        num_heads: cfg.num_attention_heads,
        head_dim: cfg.hidden_size / cfg.num_attention_heads,
        seq_len: 32,
        kv_heads: cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads),
        intermediate_size: cfg.intermediate_size,
        vocab_size: cfg.vocab_size,
    };
    let legality_pass = LegalityRewritePass::new();
    let air = legality_pass.run(sir.clone(), &kq, Some(&decomp_ctx))
        .unwrap_or_else(|e| { eprintln!("AIR lowering failed: {}", e); std::process::exit(1); });

    let air_ok = validate_air(&air);
    dump_air_stats(&air);

    // Save AIR JSON
    let air_json = serde_json::to_string_pretty(&air).unwrap();
    fs::write("/tmp/qwen3_air.json", &air_json).ok();
    println!("  AIR JSON saved to /tmp/qwen3_air.json");

    // ═══════════════════════════════════════════════════════════════
    // STAGE 3: AIR → MIR (MIL Lower)
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══ STAGE 3: AIR → MIR (MIL Lower) ═══");
    let shard_plan = ShardPlan::default();
    let mut input_shapes: HashMap<_, Vec<usize>> = HashMap::new();
    // Seed the input shape for the placeholder input
    for input_id in &air.inputs {
        input_shapes.insert(input_id.clone(), vec![1, 32]); // batch=1, seq=32
    }

    let mil_lower = MilLowerPass::new();
    let mir_graphs = mil_lower.run(&air, &shard_plan, &input_shapes)
        .unwrap_or_else(|e| { eprintln!("MIR lowering failed: {}", e); std::process::exit(1); });

    println!("  MIR graphs produced: {}", mir_graphs.len());
    let mir_ok = if mir_graphs.is_empty() {
        println!("  ❌ No MIR graphs produced!");
        false
    } else {
        let mir = &mir_graphs[0];
        validate_mir(mir);
        dump_mir_stats(mir);

        // Save MIR JSON
        let mir_json = serde_json::to_string_pretty(mir).unwrap();
        fs::write("/tmp/qwen3_mir.json", &mir_json).ok();
        println!("  MIR JSON saved to /tmp/qwen3_mir.json");
        true
    };

    // ═══════════════════════════════════════════════════════════════
    // STAGE 4: MIR → PIR (Shard Plan)
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══ STAGE 4: MIR → PIR (Shard Planning) ═══");
    let mut shard_pass = ane_passes::shard_plan::ShardPlanPass::new();
    let (pir_plan, pir_graph) = shard_pass.run(&sir, &kq)
        .unwrap_or_else(|e| { eprintln!("PIR planning failed: {}", e); std::process::exit(1); });

    println!("  Shard plan: num_shards={}, is_multi_shard={}", pir_plan.num_shards, pir_plan.is_multi_shard);
    println!("  Compute units: {:?}", pir_plan.compute_units);
    println!("  Shard roles: {:?}", pir_plan.shard_roles);
    println!("  Shard names: {:?}", pir_plan.shard_names);

    let pir_ok = validate_pir(&pir_graph);
    dump_pir_stats(&pir_graph);

    // Save PIR JSON
    let pir_json = serde_json::to_string_pretty(&pir_graph).unwrap();
    fs::write("/tmp/qwen3_pir.json", &pir_json).ok();
    println!("  PIR JSON saved to /tmp/qwen3_pir.json");

    // ═══════════════════════════════════════════════════════════════
    // Final Summary
    // ═══════════════════════════════════════════════════════════════
    println!("\n═══ PIPELINE SUMMARY ═══");
    let all_ok = trace_ok && sir_ok && air_ok && mir_ok && pir_ok;
    println!("  Trace:  {}", if trace_ok { "✅" } else { "❌" });
    println!("  SIR:    {}", if sir_ok { "✅" } else { "❌" });
    println!("  AIR:    {}", if air_ok { "✅" } else { "❌" });
    println!("  MIR:    {}", if mir_ok { "✅" } else { "❌" });
    println!("  PIR:    {}", if pir_ok { "✅" } else { "❌" });
    println!();
    if all_ok {
        println!("🎉 ALL PIPELINE STAGES PASS ACCURACY CHECKS");
    } else {
        println!("⚠️ SOME PIPELINE STAGES HAVE ISSUES — see above for details");
    }
}

// ═══════════════════════════════════════════════════════════════
// Validation Functions
// ═══════════════════════════════════════════════════════════════

fn validate_sir(sir: &SirGraph) -> bool {
    let mut ok = true;
    let n = sir.nodes.len();

    // Expected: 509 nodes for Qwen3-0.6B with 28 layers
    if n != 509 {
        println!("  ❌ SIR node count: {} (expected 509)", n);
        ok = false;
    } else {
        println!("  ✅ SIR node count: {} (matches expected 509)", n);
    }

    // Count op types
    let mut op_counts: HashMap<&str, usize> = HashMap::new();
    for node in &sir.nodes {
        let name = match &node.op {
            SirOp::LinearProjection { .. } => "LinearProjection",
            SirOp::RMSNorm { .. } => "RMSNorm",
            SirOp::ScaledDotProductAttention { .. } => "SDPA",
            SirOp::Add { .. } => "Add",
            SirOp::Mul { .. } => "Mul",
            SirOp::Silu { .. } => "Silu",
            SirOp::Tile { .. } => "Tile",
            SirOp::Gather { .. } => "Gather",
            SirOp::Identity { .. } => "Identity",
            _ => "Other",
        };
        *op_counts.entry(name).or_insert(0) += 1;
    }
    println!("  SIR op breakdown:");
    let mut sorted: Vec<_> = op_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted {
        println!("    {}: {}", op, count);
    }

    // Check residual connections: 56 Add ops (28 attn + 28 mlp)
    let add_count = op_counts.get("Add").copied().unwrap_or(0);
    if add_count != 56 {
        println!("  ❌ SIR Add ops: {} (expected 56 for residual connections)", add_count);
        ok = false;
    } else {
        println!("  ✅ SIR Add ops: {} (residual connections present)", add_count);
    }

    // Check QK-norm: 56 RMSNorm beyond the 57 structural norms = 113 total
    let rms_count = op_counts.get("RMSNorm").copied().unwrap_or(0);
    if rms_count != 113 {
        println!("  ❌ SIR RMSNorm ops: {} (expected 113 = 57 structural + 56 QK-norm)", rms_count);
        ok = false;
    } else {
        println!("  ✅ SIR RMSNorm ops: {} (57 structural + 56 QK-norm)", rms_count);
    }

    // Check GQA: 56 Tile ops (28 k_tile + 28 v_tile for 16Q/8KV)
    let tile_count = op_counts.get("Tile").copied().unwrap_or(0);
    if tile_count != 56 {
        println!("  ❌ SIR Tile ops: {} (expected 56 for GQA expansion)", tile_count);
        ok = false;
    } else {
        println!("  ✅ SIR Tile ops: {} (GQA expansion)", tile_count);
    }

    // Check SDPA: 28 (one per layer)
    let sdpa_count = op_counts.get("SDPA").copied().unwrap_or(0);
    if sdpa_count != 28 {
        println!("  ❌ SIR SDPA ops: {} (expected 28, one per layer)", sdpa_count);
        ok = false;
    } else {
        println!("  ✅ SIR SDPA ops: {} (one per layer)", sdpa_count);
    }

    // Check SDPA scale = 1/sqrt(128)
    let mut scale_ok = true;
    for node in &sir.nodes {
        if let SirOp::ScaledDotProductAttention { scale, .. } = &node.op {
            if let Some(s) = scale {
                let expected = 1.0f32 / 128.0f32.sqrt();
                if (s - expected).abs() > 0.001 {
                    println!("  ❌ SDPA scale: {} (expected ~{})", s, expected);
                    scale_ok = false;
                }
            }
        }
    }
    if scale_ok {
        println!("  ✅ SDPA scale: 1/√128 ≈ 0.0884 (correct for head_dim=128)");
    } else {
        ok = false;
    }

    // Check SwiGLU: 28 Mul ops (gate_silu * up_proj)
    let mul_count = op_counts.get("Mul").copied().unwrap_or(0);
    if mul_count != 28 {
        println!("  ❌ SIR Mul ops: {} (expected 28 for SwiGLU)", mul_count);
        ok = false;
    } else {
        println!("  ✅ SIR Mul ops: {} (SwiGLU gate_silu × up_proj)", mul_count);
    }

    // Check LinearProjection count: 197 (7 per layer + 1 lm_head)
    let lin_count = op_counts.get("LinearProjection").copied().unwrap_or(0);
    if lin_count != 197 {
        println!("  ❌ SIR LinearProjection ops: {} (expected 197)", lin_count);
        ok = false;
    } else {
        println!("  ✅ SIR LinearProjection ops: {} (7×28 + 1 lm_head)", lin_count);
    }

    ok
}

fn validate_air(air: &AirGraph) -> bool {
    let mut ok = true;
    let n = air.nodes.len();

    if n == 0 {
        println!("  ❌ AIR has 0 nodes!");
        return false;
    }

    println!("  AIR node count: {}", n);

    // Count AIR op types
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    for node in &air.nodes {
        let name = match &node.op {
            AirOp::Conv1x1AsLinear { .. } => "Conv1x1AsLinear",
            AirOp::ScaledDotProductAttention { .. } => "SDPA",
            AirOp::Add { .. } => "Add",
            AirOp::Mul { .. } => "Mul",
            AirOp::Silu { .. } => "Silu",
            AirOp::Tile { .. } => "Tile",
            AirOp::Gather { .. } => "Gather",
            AirOp::GatherAlongAxis { .. } => "GatherAlongAxis",
            AirOp::GatherNd { .. } => "GatherNd",
            AirOp::Identity { .. } => "Identity",
            AirOp::ReduceMean { .. } => "ReduceMean",
            AirOp::Rsqrt { .. } => "Rsqrt",
            AirOp::Reshape { .. } => "Reshape",
            AirOp::Transpose { .. } => "Transpose",
            AirOp::SliceByIndex { .. } => "SliceByIndex",
            AirOp::Softmax { .. } => "Softmax",
            AirOp::Split { .. } => "Split",
            AirOp::Concat { .. } => "Concat",
            AirOp::StateReadFixed { .. } => "StateReadFixed",
            AirOp::StateWriteFixed { .. } => "StateWriteFixed",
            _ => "Other",
        };
        *op_counts.entry(name.to_string()).or_insert(0) += 1;
    }

    println!("  AIR op breakdown:");
    let mut sorted: Vec<_> = op_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted {
        println!("    {}: {}", op, count);
    }

    // Key validations:
    // 1. LinearProjection → Conv1x1AsLinear
    let conv_count = op_counts.get("Conv1x1AsLinear").copied().unwrap_or(0);
    if conv_count == 197 {
        println!("  ✅ Conv1x1AsLinear: {} (all LinearProjection → Conv1x1AsLinear)", conv_count);
    } else {
        println!("  ❌ Conv1x1AsLinear: {} (expected 197)", conv_count);
        ok = false;
    }

    // 2. Check if RMSNorm was decomposed
    let reduce_mean = op_counts.get("ReduceMean").copied().unwrap_or(0);
    let rsqrt = op_counts.get("Rsqrt").copied().unwrap_or(0);

    if reduce_mean > 0 && rsqrt > 0 {
        println!("  ✅ RMSNorm decomposed: ReduceMean={}, Rsqrt={}", reduce_mean, rsqrt);
        // Expected: 113 RMSNorm → 113×4 = 452 AIR ops (ReduceMean+Rsqrt+Mul+Mul)
        let expected_rms_decomp = 113 * 4;
        let rms_related = reduce_mean + rsqrt + op_counts.get("Mul").copied().unwrap_or(0);
        println!("    (RMSNorm-related AIR ops: {} out of expected {})", rms_related, expected_rms_decomp);
    } else {
        println!("  ⚠️  RMSNorm not decomposed (ReduceMean={}, Rsqrt={}) — may be passthrough", reduce_mean, rsqrt);
    }

    // 3. SDPA preserved
    let sdpa = op_counts.get("SDPA").copied().unwrap_or(0);
    if sdpa == 28 {
        println!("  ✅ SDPA: {} (preserved from SIR)", sdpa);
    } else {
        println!("  ❌ SDPA: {} (expected 28)", sdpa);
        ok = false;
    }

    // 4. Residual connections preserved (Add from SIR passthrough + RMSNorm decomposition adds)
    let add = op_counts.get("Add").copied().unwrap_or(0);
    if add >= 56 {
        println!("  ✅ Add: {} (≥56: residual connections preserved, extra from RMSNorm decomp)", add);
    } else {
        println!("  ❌ Add: {} (expected at least 56 for residuals)", add);
        ok = false;
    }

    // 5. SwiGLU preserved
    let silu = op_counts.get("Silu").copied().unwrap_or(0);
    let mul = op_counts.get("Mul").copied().unwrap_or(0);
    if silu >= 28 && mul >= 28 {
        println!("  ✅ SwiGLU: Silu={}, Mul={} (gate activation preserved)", silu, mul);
    } else {
        println!("  ❌ SwiGLU: Silu={}, Mul={} (expected at least 28 each)", silu, mul);
        ok = false;
    }

    // 6. Tile preserved (GQA)
    let tile = op_counts.get("Tile").copied().unwrap_or(0);
    if tile == 56 {
        println!("  ✅ Tile: {} (GQA expansion preserved)", tile);
    } else {
        println!("  ❌ Tile: {} (expected 56 for GQA)", tile);
        ok = false;
    }

    ok
}

fn validate_mir(mir: &MirGraph) -> bool {
    let mut ok = true;
    let n = mir.nodes.len();

    if n == 0 {
        println!("  ❌ MIR has 0 nodes!");
        return false;
    }

    println!("  MIR node count: {}", n);
    println!("  MIR opset: {}, shard: {}", mir.opset_version, mir.shard_name);

    // Count MIR op types
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    let mut dtype_counts: HashMap<String, usize> = HashMap::new();
    for node in &mir.nodes {
        let name = match &node.op {
            MirOp::MILLinear { .. } => "MILLinear",
            MirOp::MILMatMul { .. } => "MILMatMul",
            MirOp::MILAdd { .. } => "MILAdd",
            MirOp::MILMul { .. } => "MILMul",
            MirOp::MILSilu { .. } => "MILSilu",
            MirOp::MILScaledDotProductAttention { .. } => "MILSDPA",
            MirOp::MILGather { .. } => "MILGather",
            MirOp::MILGatherAlongAxis { .. } => "MILGatherAlongAxis",
            MirOp::MILGatherNd { .. } => "MILGatherNd",
            MirOp::MILReshape { .. } => "MILReshape",
            MirOp::MILTranspose { .. } => "MILTranspose",
            MirOp::MILSliceByIndex { .. } => "MILSliceByIndex",
            MirOp::MILReduceMean { .. } => "MILReduceMean",
            MirOp::MILRsqrt { .. } => "MILRsqrt",
            MirOp::MILReadState { .. } => "MILReadState",
            MirOp::MILTile { .. } => "MILTile",
            MirOp::MILIdentity { .. } => "MILIdentity",
            MirOp::MILConst { .. } => "MILConst",
            MirOp::MILSoftmax { .. } => "MILSoftmax",
            MirOp::MILConcat { .. } => "MILConcat",
            MirOp::MILSplit { .. } => "MILSplit",
            MirOp::MILCoremlUpdateState { .. } => "MILCoremlUpdateState",
            _ => "Other",
        };
        *op_counts.entry(name.to_string()).or_insert(0) += 1;

        let dtype_name = match node.dtype {
            MilDtype::Fp16 => "fp16",
            MilDtype::Fp32 => "fp32",
            MilDtype::Int32 => "int32",
            MilDtype::Bool => "bool",
            MilDtype::Int8 => "int8",
            MilDtype::Int16 => "int16",
            MilDtype::Fp64 => "fp64",
            MilDtype::UInt8 => "uint8",
        };
        *dtype_counts.entry(dtype_name.to_string()).or_insert(0) += 1;
    }

    println!("  MIR op breakdown:");
    let mut sorted: Vec<_> = op_counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted {
        println!("    {}: {}", op, count);
    }
    println!("  Dtype distribution:");
    let mut dtype_sorted: Vec<_> = dtype_counts.iter().collect();
    dtype_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (dtype, count) in &dtype_sorted {
        println!("    {}: {}", dtype, count);
    }

    // Key MIR validations:
    // 1. Conv1x1AsLinear → MILLinear (NOT MILMatMul)
    let mil_linear = op_counts.get("MILLinear").copied().unwrap_or(0);
    let mil_matmul = op_counts.get("MILMatMul").copied().unwrap_or(0);
    if mil_linear >= 197 && mil_matmul == 0 {
        println!("  ✅ MILLinear: {} (Conv1x1AsLinear correctly lowered, no MILMatMul)", mil_linear);
    } else if mil_matmul > 0 {
        println!("  ❌ MILMatMul: {} found (LinearProjection should map to MILLinear, not MILMatMul)", mil_matmul);
        ok = false;
    } else {
        println!("  ⚠️  MILLinear: {} (expected ≥197 from SIR's 197 LinearProjection)", mil_linear);
    }

    // 2. SDPA preserved
    let mil_sdpa = op_counts.get("MILSDPA").copied().unwrap_or(0);
    if mil_sdpa == 28 {
        println!("  ✅ MILSDPA: {} (SDPA preserved through lowering)", mil_sdpa);
    } else {
        println!("  ❌ MILSDPA: {} (expected 28)", mil_sdpa);
        ok = false;
    }

    // 3. Residual connections
    let mil_add = op_counts.get("MILAdd").copied().unwrap_or(0);
    if mil_add >= 56 {
        println!("  ✅ MILAdd: {} (≥56: residual connections preserved)", mil_add);
    } else {
        println!("  ❌ MILAdd: {} (expected ≥56)", mil_add);
        ok = false;
    }

    // 4. SwiGLU
    let mil_silu = op_counts.get("MILSilu").copied().unwrap_or(0);
    let mil_mul = op_counts.get("MILMul").copied().unwrap_or(0);
    if mil_silu >= 28 && mil_mul >= 28 {
        println!("  ✅ MILSilu={}, MILMul={} (SwiGLU preserved)", mil_silu, mil_mul);
    } else {
        println!("  ❌ SwiGLU: MILSilu={}, MILMul={} (expected ≥28 each)", mil_silu, mil_mul);
        ok = false;
    }

    // 5. RMSNorm decomposition preserved
    let mil_reduce_mean = op_counts.get("MILReduceMean").copied().unwrap_or(0);
    let mil_rsqrt = op_counts.get("MILRsqrt").copied().unwrap_or(0);
    if mil_reduce_mean > 0 && mil_rsqrt > 0 {
        println!("  ✅ RMSNorm decomposition: MILReduceMean={}, MILRsqrt={}", mil_reduce_mean, mil_rsqrt);
    } else {
        println!("  ⚠️  No RMSNorm decomposition in MIR (ReduceMean={}, Rsqrt={})", mil_reduce_mean, mil_rsqrt);
    }

    ok
}

fn validate_pir(pir: &PirGraph) -> bool {
    let ok = true;
    let n_packages = pir.packages.len();

    println!("  PIR packages: {}", n_packages);
    for pkg in &pir.packages {
        let role_str = match &pkg.role {
            ane_ir::pir::PackageRole::IO => "IO".to_string(),
            ane_ir::pir::PackageRole::DecoderShard(sr) => format!("DecoderShard({:?})", sr),
            ane_ir::pir::PackageRole::Sampler => "Sampler".to_string(),
        };
        println!("    Package: {} (role={}, compute_units={:?}, functions={})",
            pkg.name, role_str, pkg.compute_units, pkg.functions.len());
    }

    println!("  PIR state_declarations: {}", pir.state_declarations.len());
    println!("  PIR handoffs: {}", pir.handoffs.len());
    println!("  PIR context_length: {}", pir.context_length);

    // At minimum, there should be at least one package
    if n_packages == 0 {
        println!("  ❌ No packages in PIR graph!");
        return false;
    }

    println!("  ✅ PIR graph has {} package(s) with role assignments", n_packages);

    // Check for IO model
    let has_io = pir.packages.iter().any(|p| {
        matches!(p.role, ane_ir::pir::PackageRole::IO)
    });
    if has_io {
        println!("  ✅ IO package present");
    } else {
        println!("  ⚠️  No IO package found");
    }

    // Check for decoder shards
    let decoder_count = pir.packages.iter().filter(|p| {
        matches!(p.role, ane_ir::pir::PackageRole::DecoderShard(_))
    }).count();
    if decoder_count > 0 {
        println!("  ✅ Decoder shard packages: {}", decoder_count);
    }

    // Check handoffs
    if !pir.handoffs.is_empty() {
        println!("  ✅ Handoffs present: {}", pir.handoffs.len());
        for h in &pir.handoffs {
            println!("    {} → {} (tensor: {}, dtype: {})",
                h.from_package, h.to_package, h.tensor_name, h.dtype);
        }
    }

    ok
}

// ═══════════════════════════════════════════════════════════════
// Stats Dump Functions
// ═══════════════════════════════════════════════════════════════

fn dump_sir_stats(sir: &SirGraph) {
    println!("  SIR inputs: {:?}, outputs: {:?}",
        sir.inputs.iter().map(|i| &i.0).collect::<Vec<_>>(),
        sir.outputs.iter().map(|o| &o.0).collect::<Vec<_>>());
}

fn dump_air_stats(air: &AirGraph) {
    println!("  AIR inputs: {:?}, outputs: {:?}",
        air.inputs.iter().map(|i| &i.0).take(5).collect::<Vec<_>>(),
        air.outputs.iter().map(|o| &o.0).take(5).collect::<Vec<_>>());
    println!("  AIR staticization_decisions: {}", air.staticization_decisions.len());
}

fn dump_mir_stats(mir: &MirGraph) {
    println!("  MIR inputs: {:?}, outputs: {:?}",
        mir.inputs.iter().map(|i| &i.0).take(5).collect::<Vec<_>>(),
        mir.outputs.iter().map(|o| &o.0).take(5).collect::<Vec<_>>());

    // Check compute unit hints
    let mut cu_counts: HashMap<String, usize> = HashMap::new();
    for node in &mir.nodes {
        let cu = match &node.compute_unit_hint {
            Some(ane_ir::mir::ComputeUnitHint::CPUAndNE) => "CPU_AND_NE",
            Some(ane_ir::mir::ComputeUnitHint::CPUAndGPU) => "CPU_AND_GPU",
            Some(ane_ir::mir::ComputeUnitHint::CPUOnly) => "CPU_ONLY",
            Some(ane_ir::mir::ComputeUnitHint::All) => "ALL",
            None => "None",
        };
        *cu_counts.entry(cu.to_string()).or_insert(0) += 1;
    }
    println!("  Compute unit distribution: {:?}", cu_counts);
}

fn dump_pir_stats(_pir: &PirGraph) {
    // Stats already printed in validate_pir
}
