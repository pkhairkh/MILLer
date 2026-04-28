//! Python subprocess interface for torch.fx tracing.
//!
//! Manages the Python subprocess that performs torch.fx tracing of
//! HuggingFace transformers models. The traced graph is exported as
//! JSON and consumed by the Rust-side SIR construction pipeline.

use crate::config::{TraceConfig, TraceTarget};
use crate::graph::TracedGraph;
use anyhow::Result;
use std::io::Write;
use std::process::{Command, Stdio};

/// Trace a HuggingFace model via the Python subprocess.
///
/// This invokes `python/trace_model.py` which:
/// 1. Loads the model using HuggingFace transformers
/// 2. Traces the model using torch.fx
/// 3. Decomposes composite ops (if decompose_at_trace is true)
/// 4. Exports the traced graph as JSON to stdout
///
/// The JSON output is deserialized into a `TracedGraph`.
pub fn trace_model(config: &TraceConfig) -> Result<TracedGraph> {
    match &config.target {
        TraceTarget::PreTraced(path) => load_pre_traced(path),
        _ => run_trace_subprocess(config),
    }
}

/// Load a pre-traced graph from a JSON file (skips the Python subprocess).
fn load_pre_traced(path: &str) -> Result<TracedGraph> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read pre-traced graph '{}': {}", path, e))?;
    let graph: TracedGraph = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("Failed to parse pre-traced graph: {}", e))?;
    Ok(graph)
}

/// Run the Python tracing subprocess.
fn run_trace_subprocess(config: &TraceConfig) -> Result<TracedGraph> {
    let model_id = match &config.target {
        TraceTarget::HuggingFaceId(id) => id.clone(),
        TraceTarget::LocalPath(path) => path.clone(),
        TraceTarget::PreTraced(_) => unreachable!(),
    };

    // Build the JSON payload for the Python script
    let payload = serde_json::json!({
        "model_id": model_id,
        "input_shapes": config.input_shapes,
        "decompose": config.decompose_at_trace,
        "with_kv_cache": config.with_kv_cache,
        "max_seq_len": config.max_seq_len,
        "dtype": config.dtype,
        "model_class": config.model_class,
        "fx_options": {
            "concrete_args": config.fx_options.concrete_args,
            "flatten": config.fx_options.flatten,
            "leaf_modules": config.fx_options.leaf_modules,
            "suppress_shape_assertions": config.fx_options.suppress_shape_assertions,
        },
    });

    let payload_str = serde_json::to_string(&payload)?;

    // Invoke the Python tracing script
    let mut child = Command::new(&config.python_path)
        .arg(&config.trace_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to spawn Python tracer: {}", e))?;

    // Write the payload to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(payload_str.as_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to write to Python tracer stdin: {}", e))?;
    }

    let output = child.wait_with_output()
        .map_err(|e| anyhow::anyhow!("Python tracer process failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Python tracer failed with exit code {:?}: {}",
            output.status.code(),
            stderr
        ));
    }

    // Show Python tracer's stderr output for diagnostic purposes
    // (safetensors discovery warnings, strategy failures, etc.)
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        for line in stderr.lines() {
            eprintln!("  [tracer] {}", line);
        }
    }

    // Parse the JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    let graph: TracedGraph = serde_json::from_str(&stdout)
        .map_err(|e| anyhow::anyhow!("Failed to parse traced graph JSON: {}\nOutput: {}", e, &stdout[..stdout.len().min(500)]))?;

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{TraceConfig, TraceTarget};

    #[test]
    fn test_load_pre_traced() {
        // Create a temporary JSON file with a minimal traced graph
        let tmp_dir = tempfile::tempdir().unwrap();
        let path = tmp_dir.path().join("test_trace.json");
        let graph_json = serde_json::json!({
            "model_id": "test-model",
            "architecture": "TestModel",
            "transformers_version": "4.36.0",
            "torch_version": "2.1.0",
            "model_config": {
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_hidden_layers": 2,
                "intermediate_size": 1024,
                "vocab_size": 32000,
                "max_position_embeddings": 2048,
                "layer_norm_epsilon": 1e-6,
                "hidden_act": "silu",
                "uses_rope": true,
                "uses_rms_norm": true,
                "uses_gqa": false,
                "model_type": "llama",
                "model_class": "causal_lm",
                "is_encoder_decoder": false
            },
            "nodes": [],
            "weights": {},
            "inputs": [],
            "outputs": [],
            "state_declarations": [],
            "trace_metadata": {
                "timestamp": "2026-01-01T00:00:00Z",
                "trace_duration_secs": 0.1,
                "num_nodes": 0,
                "num_parameters": 0,
                "parameter_bytes": 0,
                "decomposed": false,
                "warnings": []
            }
        });
        std::fs::write(&path, serde_json::to_string_pretty(&graph_json).unwrap()).unwrap();

        let config = TraceConfig {
            target: TraceTarget::PreTraced(path.to_str().unwrap().to_string()),
            ..TraceConfig::default()
        };

        let result = trace_model(&config);
        assert!(result.is_ok());
        let graph = result.unwrap();
        assert_eq!(graph.model_id, "test-model");
    }
}
