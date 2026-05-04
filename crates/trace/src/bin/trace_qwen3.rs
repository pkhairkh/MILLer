use ane_ir::ane_target::AneFamily;
use ane_ir::sir::SirOp;
use ane_trace::graph::TracedGraph;
use ane_trace::sir_build::build_sir_from_trace;
use std::fs;

fn main() {
    let json_str = fs::read_to_string("/tmp/qwen3_trace.json").expect("Failed to read trace JSON");

    let trace: TracedGraph = serde_json::from_str(&json_str).expect("Failed to parse trace JSON");

    println!("=== TracedGraph Config ===");
    println!(
        "hidden_size={}, num_heads={}, num_kv_heads={:?}, intermediate={}",
        trace.model_config.hidden_size,
        trace.model_config.num_attention_heads,
        trace.model_config.num_key_value_heads,
        trace.model_config.intermediate_size,
    );
    println!(
        "uses_rms_norm={}, uses_gqa={}, uses_rope={}",
        trace.model_config.uses_rms_norm, trace.model_config.uses_gqa, trace.model_config.uses_rope,
    );
    println!("layer_norm_epsilon: {}", trace.model_config.layer_norm_epsilon);
    println!("nodes: {}, weights: {}", trace.nodes.len(), trace.weights.len());

    // Build SIR
    let sir = build_sir_from_trace(&trace, AneFamily::A16).expect("Failed to build SIR");

    println!("\n=== SIR Graph ===");
    println!("Total nodes: {}", sir.nodes.len());

    // Count op types
    let mut op_counts = std::collections::HashMap::new();
    for node in &sir.nodes {
        let op_name = match &node.op {
            SirOp::LinearProjection { .. } => "LinearProjection",
            SirOp::RMSNorm { .. } => "RMSNorm",
            SirOp::ScaledDotProductAttention { .. } => "SDPA",
            SirOp::Add { .. } => "Add",
            SirOp::Mul { .. } => "Mul",
            SirOp::Silu { .. } => "Silu",
            SirOp::Gather { .. } => "Gather",
            SirOp::Identity { .. } => "Identity",
            SirOp::Tile { .. } => "Tile",
            _ => "Other",
        };
        *op_counts.entry(op_name).or_insert(0) += 1;
    }

    for (op, count) in &op_counts {
        println!("  {}: {}", op, count);
    }

    // Critical checks
    let add_count = sir.nodes.iter().filter(|n| matches!(n.op, SirOp::Add { .. })).count();
    let rms_count = sir.nodes.iter().filter(|n| matches!(n.op, SirOp::RMSNorm { .. })).count();
    let tile_count = sir.nodes.iter().filter(|n| matches!(n.op, SirOp::Tile { .. })).count();

    println!("\n=== Bug Checks ===");
    println!("Residual Adds: {} (need 56)", add_count);
    println!("RMSNorm ops: {} (need ~59 with QK-norm)", rms_count);
    println!("Tile ops: {} (need 56 for GQA)", tile_count);

    // Check first few nodes in detail
    println!("\n=== First 15 nodes ===");
    for node in sir.nodes.iter().take(15) {
        let desc = match &node.op {
            SirOp::LinearProjection { input, weight, bias, .. } => {
                format!("Linear(in={}, w={}, b={:?})", input.0, weight, bias)
            }
            SirOp::RMSNorm { input, weight: _, epsilon, axes: _ } => {
                format!("RMSNorm(in={}, eps={})", input.0, epsilon)
            }
            SirOp::ScaledDotProductAttention { query, key, value, attention_mask, scale } => {
                format!(
                    "SDPA(q={},k={},v={},mask={:?},scale={:?})",
                    query.0, key.0, value.0, attention_mask, scale
                )
            }
            SirOp::Add { x, y } => format!("Add({},{})", x.0, y.0),
            SirOp::Mul { x, y } => format!("Mul({},{})", x.0, y.0),
            SirOp::Silu { input } => format!("Silu({})", input.0),
            SirOp::Tile { input, reps } => format!("Tile({},{:?})", input.0, reps),
            SirOp::Gather { input, indices, axis } => {
                format!("Gather({},{},a={})", input.0, indices.0, axis)
            }
            SirOp::Identity { input } => format!("Id({})", input.0),
            _ => format!("{:?}", node.op),
        };
        println!("  {} [{}] = {}", node.id.0, node.name, desc);
    }
}
