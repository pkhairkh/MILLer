#!/usr/bin/env python3
"""
HuggingFace Transformers Model Tracer for MILLer.

Traces a transformers model using torch.fx and exports the computation
graph as JSON for consumption by the Rust-side SIR construction pipeline.

Usage:
    echo '{"model_id": "gpt2", "input_shapes": [...]}' | python trace_model.py

The JSON payload is read from stdin. The traced graph is written to stdout.

Requirements:
    pip install torch transformers
"""

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def main():
    """Main entry point: read config from stdin, trace model, write graph to stdout."""
    try:
        config = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        error_exit(f"Failed to parse stdin JSON: {e}")

    model_id = config.get("model_id", "gpt2")
    input_shapes = config.get("input_shapes", [{"batch_size": 1, "seq_len": 32}])
    decompose = config.get("decompose", True)
    with_kv_cache = config.get("with_kv_cache", False)
    max_seq_len = config.get("max_seq_len", 2048)
    dtype_str = config.get("dtype", "fp16")
    fx_options = config.get("fx_options", {})

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError as e:
        error_exit(f"Required packages not installed: {e}\nInstall with: pip install torch transformers")

    start_time = time.time()

    try:
        # Load model configuration
        model_config = AutoConfig.from_pretrained(model_id)

        # Load model with fp16 if requested
        torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()

        # Create dummy input
        batch_size = input_shapes[0].get("batch_size", 1)
        seq_len = input_shapes[0].get("seq_len", 32)
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))

        # Trace the model with torch.fx
        traced_graph = trace_model_fx(
            model=model,
            input_ids=input_ids,
            model_config=model_config,
            model_id=model_id,
            decompose=decompose,
            with_kv_cache=with_kv_cache,
            fx_options=fx_options,
        )

        # Add metadata
        trace_duration = time.time() - start_time
        traced_graph["trace_metadata"]["trace_duration_secs"] = trace_duration
        traced_graph["trace_metadata"]["transformers_version"] = _get_transformers_version()
        traced_graph["trace_metadata"]["torch_version"] = torch.__version__

        # Write to stdout
        json.dump(traced_graph, sys.stdout, indent=2)
        sys.stdout.write("\n")

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        error_exit(f"Tracing failed: {e}")


def trace_model_fx(
    model,
    input_ids,
    model_config,
    model_id: str,
    decompose: bool = True,
    with_kv_cache: bool = False,
    fx_options: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Trace a transformers model using torch.fx and export as a MILLer TracedGraph.

    Args:
        model: The HuggingFace model to trace.
        input_ids: Dummy input tensor for tracing.
        model_config: The model's configuration.
        model_id: HuggingFace model ID.
        decompose: Whether to decompose composite ops during tracing.
        with_kv_cache: Whether to include KV-cache state in the graph.
        fx_options: Additional torch.fx tracing options.

    Returns:
        A dictionary representing a TracedGraph (serializable to JSON).
    """
    import torch
    fx_options = fx_options or {}

    # Perform torch.fx tracing
    concrete_args = fx_options.get("concrete_args", True)
    flatten = fx_options.get("flatten", True)

    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        # Fallback: try with concrete_args
        sys.stderr.write(f"Warning: symbolic_trace failed ({e}), falling back to manual graph construction\n")
        return build_fallback_graph(model_config, model_id, decompose)

    # Extract nodes from the traced graph
    nodes = []
    weights = {}
    state_declarations = []

    for fx_node in traced.graph.nodes:
        traced_op = map_fx_node_to_traced_op(fx_node, model_config, decompose)

        node = {
            "id": fx_node.name,
            "op": traced_op,
            "name": fx_node.name,
            "inputs": [str(arg) for arg in fx_node.args if isinstance(arg, torch.fx.Node)],
            "output_shape": extract_shape(fx_node),
            "is_parameter": fx_node.op == "get_attr",
            "module_path": get_module_path(fx_node),
        }
        nodes.append(node)

    # Build model config section
    config_section = extract_model_config(model_config)

    # Build weight metadata
    for name, param in model.named_parameters():
        weights[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype).replace("torch.", ""),
            "data_path": None,
            "quantized": None,
        }

    # Build KV-cache state declarations if requested
    if with_kv_cache:
        num_layers = getattr(model_config, "num_hidden_layers", 1)
        num_heads = getattr(model_config, "num_attention_heads", 1)
        num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
        head_dim = config_section["hidden_size"] // num_heads

        for layer_idx in range(num_layers):
            state_declarations.append({
                "state_id": f"kv_cache_layer_{layer_idx}_key",
                "shape": [1, num_kv_heads, max_seq_len, head_dim],
                "dtype": dtype_str,
                "layer_idx": layer_idx,
                "is_key": True,
            })
            state_declarations.append({
                "state_id": f"kv_cache_layer_{layer_idx}_value",
                "shape": [1, num_kv_heads, max_seq_len, head_dim],
                "dtype": dtype_str,
                "layer_idx": layer_idx,
                "is_key": False,
            })

    graph = {
        "model_id": model_id,
        "architecture": model.__class__.__name__,
        "transformers_version": _get_transformers_version(),
        "torch_version": "0.0.0",  # Will be filled by main()
        "model_config": config_section,
        "nodes": nodes,
        "weights": weights,
        "inputs": [{"name": "input_ids", "shape": {"dims": list(input_ids.shape), "dtype": "int32"}}],
        "outputs": [{"name": "logits", "shape": {"dims": [0], "dtype": "fp16"}}],
        "state_declarations": state_declarations,
        "trace_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_duration_secs": 0.0,  # Will be filled by main()
            "num_nodes": len(nodes),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "decomposed": decompose,
            "warnings": [],
        },
    }

    return graph


def map_fx_node_to_traced_op(fx_node, model_config, decompose: bool) -> Dict[str, Any]:
    """
    Map a torch.fx node to a MILLer TracedOp representation.

    This is the Python-side mapping that produces a JSON-serializable
    operation descriptor. The Rust-side `sir_build` module will then
    map these to SIR ops.
    """
    op = fx_node.op
    target = str(fx_node.target) if fx_node.target else ""

    # Placeholder (input)
    if op == "placeholder":
        return {"type": "Placeholder"}

    # Output
    if op == "output":
        return {"type": "Output"}

    # GetAttr (parameter/weight access)
    if op == "get_attr":
        return {
            "type": "Unknown",
            "op_name": "get_attr",
            "target": target,
        }

    # CallModule (nn.Module calls)
    if op == "call_module":
        return map_module_call(fx_node, model_config, decompose)

    # CallFunction (torch.nn.functional calls)
    if op == "call_function":
        return map_function_call(fx_node, model_config, decompose)

    # CallMethod (tensor method calls like .view(), .transpose())
    if op == "call_method":
        return map_method_call(fx_node)

    # Fallback
    return {
        "type": "Unknown",
        "op_name": op,
        "target": target,
    }


def map_module_call(fx_node, model_config, decompose: bool) -> Dict[str, Any]:
    """Map a call_module node to a TracedOp."""
    target = str(fx_node.target)

    # Linear layers
    if "linear" in target.lower() or any(proj in target for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        # Try to get the module to extract dimensions
        return {
            "type": "Linear",
            "in_features": 0,  # Will be inferred from weight shape
            "out_features": 0,
            "has_bias": "bias" in target.lower() or any(
                p in target for p in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            "_module_path": target,
        }

    # Embedding layers
    if "embed" in target.lower() or "wte" in target.lower():
        return {
            "type": "Embedding",
            "vocab_size": getattr(model_config, "vocab_size", 32000),
            "embed_dim": getattr(model_config, "hidden_size", 768),
        }

    # LayerNorm / RMSNorm
    if "layernorm" in target.lower() or "ln" in target.lower() or "norm" in target.lower():
        if decompose:
            # Use config-driven norm type detection
            config_section = extract_model_config(model_config)
            if config_section.get("uses_rms_norm", False):
                return {
                    "type": "RmsNorm",
                    "hidden_size": getattr(model_config, "hidden_size", 768),
                    "epsilon": getattr(model_config, "rms_norm_eps", 1e-6),
                }
        return {
            "type": "LayerNorm",
            "normalized_shape": [getattr(model_config, "hidden_size", 768)],
            "epsilon": getattr(model_config, "layer_norm_eps", 1e-5),
        }

    # Attention modules
    if "attention" in target.lower() or "attn" in target.lower():
        if decompose:
            # Return composite — will be decomposed in Rust
            return {
                "type": "AttentionBlock",
                "embed_dim": getattr(model_config, "hidden_size", 768),
                "num_heads": getattr(model_config, "num_attention_heads", 12),
                "head_dim": getattr(model_config, "hidden_size", 768) // getattr(model_config, "num_attention_heads", 12),
                "use_sdpa": True,
            }
        return {
            "type": "AttentionBlock",
            "embed_dim": getattr(model_config, "hidden_size", 768),
            "num_heads": getattr(model_config, "num_attention_heads", 12),
            "head_dim": getattr(model_config, "hidden_size", 768) // getattr(model_config, "num_attention_heads", 12),
            "use_sdpa": True,
        }

    # MLP / feed-forward modules
    if "mlp" in target.lower() or "feed_forward" in target.lower():
        if decompose:
            return {
                "type": "MlpBlock",
                "input_dim": getattr(model_config, "hidden_size", 768),
                "hidden_dim": getattr(model_config, "intermediate_size", 3072),
                "output_dim": getattr(model_config, "hidden_size", 768),
                "activation": getattr(model_config, "hidden_act", "gelu"),
            }
        return {
            "type": "MlpBlock",
            "input_dim": getattr(model_config, "hidden_size", 768),
            "hidden_dim": getattr(model_config, "intermediate_size", 3072),
            "output_dim": getattr(model_config, "hidden_size", 768),
            "activation": getattr(model_config, "hidden_act", "gelu"),
        }

    return {
        "type": "Unknown",
        "op_name": "call_module",
        "target": target,
    }


def map_function_call(fx_node, model_config, decompose: bool) -> Dict[str, Any]:
    """Map a call_function node to a TracedOp."""
    target = str(fx_node.target)

    func_map = {
        "scaled_dot_product_attention": lambda: {
            "type": "ScaledDotProductAttention",
            "scale": 1.0 / ((getattr(model_config, "hidden_size", 768) // getattr(model_config, "num_attention_heads", 12)) ** 0.5),
        },
        "gelu": lambda: {"type": "Gelu", "approximate": "none"},
        "silu": lambda: {"type": "Silu"},
        "relu": lambda: {"type": "Relu"},
        "softmax": lambda: {"type": "Softmax", "axis": -1},
        "matmul": lambda: {"type": "MatMul", "a_shape": {"dims": [], "dtype": "fp16"}, "b_shape": {"dims": [], "dtype": "fp16"}},
        "add": lambda: {"type": "Add"},
        "mul": lambda: {"type": "Mul"},
        "div": lambda: {"type": "Div"},
        "exp": lambda: {"type": "Exp"},
        "cos": lambda: {"type": "Cos"},
        "sin": lambda: {"type": "Sin"},
        "tanh": lambda: {"type": "Tanh"},
        "sigmoid": lambda: {"type": "Sigmoid"},
        "where": lambda: {"type": "Where"},
        "gather": lambda: {"type": "Gather", "axis": 0},
    }

    for key, factory in func_map.items():
        if key in target.lower():
            return factory()

    return {
        "type": "Unknown",
        "op_name": "call_function",
        "target": target,
    }


def map_method_call(fx_node) -> Dict[str, Any]:
    """Map a call_method node to a TracedOp."""
    method = str(fx_node.target)

    method_map = {
        "view": lambda: {"type": "Reshape", "target_shape": []},
        "reshape": lambda: {"type": "Reshape", "target_shape": []},
        "transpose": lambda: {"type": "Transpose", "perm": []},
        "permute": lambda: {"type": "Transpose", "perm": []},
        "contiguous": lambda: {"type": "Identity"},  # No-op on ANE
        "float": lambda: {"type": "Cast", "target_dtype": "fp32"},
        "half": lambda: {"type": "Cast", "target_dtype": "fp16"},
        "to": lambda: {"type": "Cast", "target_dtype": "unknown"},
        "unsqueeze": lambda: {"type": "ExpandDims", "axis": []},
        "squeeze": lambda: {"type": "Squeeze", "axis": []},
        "split": lambda: {"type": "Split", "axis": -1, "num_splits": 2},
        "chunk": lambda: {"type": "Split", "axis": -1, "num_splits": 2},
        "index_select": lambda: {"type": "IndexSelect", "axis": 0},
        "size": lambda: {"type": "Identity"},  # Shape query — resolved during staticize
    }

    for key, factory in method_map.items():
        if method == key:
            return factory()

    return {
        "type": "Unknown",
        "op_name": "call_method",
        "target": method,
    }


def extract_model_config(model_config) -> Dict[str, Any]:
    """Extract model configuration into the TracedGraph format.
    
    This function is fully config-driven: it derives all decomposition
    hints from the model's AutoConfig, without requiring a hardcoded
    model registry. Any HuggingFace model that provides the standard
    config fields will work ad-hoc.
    
    The key decomposition flags are:
    - uses_rms_norm: Whether to decompose norms as RMSNorm (vs LayerNorm)
    - uses_gqa: Whether the model uses Grouped Query Attention
    - uses_rope: Whether the model uses Rotary Position Embeddings
    - hidden_act: The activation function used in MLP blocks
    
    These are derived from AutoConfig fields where possible, with
    model_type heuristics as fallback for configs that don't expose
    these flags directly.
    """
    hidden_size = getattr(model_config, "hidden_size", 768)
    num_heads = getattr(model_config, "num_attention_heads", 12)
    num_kv_heads = getattr(model_config, "num_key_value_heads", num_heads)
    model_type = getattr(model_config, "model_type", "unknown")

    # ─── Derive decomposition hints from config ─────────────────
    #
    # Strategy: check explicit config fields first, then fall back to
    # model_type heuristics. This ensures new architectures work
    # without a registry as long as they follow the standard pattern.

    # RMSNorm vs LayerNorm
    # Explicit: some configs expose norm_type or rms_norm_eps
    # Heuristic: model_type families known to use RMSNorm
    uses_rms_norm = getattr(model_config, "uses_rms_norm", None)
    if uses_rms_norm is None:
        has_rms_norm_eps = hasattr(model_config, "rms_norm_eps")
        has_layer_norm_eps_only = hasattr(model_config, "layer_norm_eps") and not has_rms_norm_eps
        rms_norm_model_types = {
            "llama", "qwen2", "qwen3", "qwen3_moe", "qwen3.5",
            "mistral", "mixtral", "gemma", "gemma2", "phi3",
            "stablelm", "falcon", "starcoder2", "internlm2",
            "deepseek_v2", "deepseek_v3", "llava",
        }
        uses_rms_norm = has_rms_norm_eps or (
            model_type.lower() in rms_norm_model_types and not has_layer_norm_eps_only
        )

    # GQA (Grouped Query Attention)
    # Explicit: num_key_value_heads < num_attention_heads
    # This is automatically correct from AutoConfig — no heuristic needed.
    uses_gqa = num_kv_heads < num_heads

    # RoPE (Rotary Position Embeddings)
    # Explicit: some configs expose rope_scaling or rotary_emb_base
    # Heuristic: model_type families known to use RoPE
    uses_rope = getattr(model_config, "uses_rope", None)
    if uses_rope is None:
        has_rope_scaling = hasattr(model_config, "rope_scaling") or hasattr(model_config, "rope_theta")
        rope_model_types = {
            "llama", "qwen2", "qwen3", "qwen3_moe", "qwen3.5",
            "mistral", "mixtral", "gemma", "gemma2", "phi", "phi3",
            "falcon", "starcoder2", "internlm2",
            "deepseek_v2", "deepseek_v3", "llava",
        }
        uses_rope = has_rope_scaling or model_type.lower() in rope_model_types

    # Activation function
    hidden_act = getattr(model_config, "hidden_act", None)
    if hidden_act is None:
        # Some configs use different field names
        hidden_act = getattr(model_config, "activation_function", "gelu")

    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", 12),
        "intermediate_size": getattr(model_config, "intermediate_size", 3072),
        "vocab_size": getattr(model_config, "vocab_size", 32000),
        "max_position_embeddings": getattr(model_config, "max_position_embeddings", 2048),
        "layer_norm_epsilon": getattr(model_config, "rms_norm_eps",
                                       getattr(model_config, "layer_norm_eps", 1e-5)),
        "hidden_act": hidden_act,
        "uses_rope": uses_rope,
        "uses_rms_norm": uses_rms_norm,
        "uses_gqa": uses_gqa,
        "model_type": model_type,
    }


def extract_shape(fx_node) -> Dict[str, Any]:
    """Extract the output shape from a torch.fx node."""
    try:
        import torch
        if hasattr(fx_node, 'meta') and 'tensor_meta' in fx_node.meta:
            meta = fx_node.meta['tensor_meta']
            if hasattr(meta, 'shape'):
                return {"dims": list(meta.shape), "dtype": "fp16"}
    except Exception:
        pass
    return {"dims": [], "dtype": "fp16"}


def get_module_path(fx_node) -> Optional[str]:
    """Get the module path for a call_module node."""
    if fx_node.op == "call_module":
        return str(fx_node.target)
    return None


def build_fallback_graph(model_config, model_id: str, decompose: bool) -> Dict[str, Any]:
    """
    Build a TracedGraph using structural knowledge when torch.fx tracing fails.

    This constructs a graph based on the model's configuration rather than
    actual tracing, producing the expected layer structure for known architectures.
    """
    config = extract_model_config(model_config)
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_heads

    nodes = []

    # Input
    nodes.append({
        "id": "input_ids",
        "op": {"type": "Placeholder"},
        "name": "input_ids",
        "inputs": [],
        "output_shape": {"dims": [1, 32], "dtype": "int32"},
        "is_parameter": False,
        "module_path": None,
    })

    # Embedding
    nodes.append({
        "id": "embed_tokens",
        "op": {"type": "Embedding", "vocab_size": config["vocab_size"], "embed_dim": hidden_size},
        "name": "embed_tokens",
        "inputs": ["input_ids"],
        "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
        "is_parameter": False,
        "module_path": None,
    })

    prev_id = "embed_tokens"

    # Transformer layers
    for i in range(num_layers):
        layer_prefix = f"layer_{i}"

        # Input norm
        if config["uses_rms_norm"]:
            nodes.append({
                "id": f"{layer_prefix}_input_norm",
                "op": {"type": "RmsNorm", "hidden_size": hidden_size, "epsilon": config["layer_norm_epsilon"]},
                "name": f"{layer_prefix}_input_norm",
                "inputs": [prev_id],
                "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
                "is_parameter": False,
                "module_path": f"model.layers.{i}.input_layernorm",
            })
            prev_id = f"{layer_prefix}_input_norm"

        # Self-attention
        nodes.append({
            "id": f"{layer_prefix}_self_attn",
            "op": {
                "type": "AttentionBlock",
                "embed_dim": hidden_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "use_sdpa": True,
            },
            "name": f"{layer_prefix}_self_attn",
            "inputs": [prev_id],
            "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
            "is_parameter": False,
            "module_path": f"model.layers.{i}.self_attn",
        })
        prev_id = f"{layer_prefix}_self_attn"

        # Post-attention norm
        if config["uses_rms_norm"]:
            nodes.append({
                "id": f"{layer_prefix}_post_attn_norm",
                "op": {"type": "RmsNorm", "hidden_size": hidden_size, "epsilon": config["layer_norm_epsilon"]},
                "name": f"{layer_prefix}_post_attn_norm",
                "inputs": [prev_id],
                "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
                "is_parameter": False,
                "module_path": f"model.layers.{i}.post_attention_layernorm",
            })
            prev_id = f"{layer_prefix}_post_attn_norm"

        # MLP
        nodes.append({
            "id": f"{layer_prefix}_mlp",
            "op": {
                "type": "MlpBlock",
                "input_dim": hidden_size,
                "hidden_dim": intermediate_size,
                "output_dim": hidden_size,
                "activation": config["hidden_act"],
            },
            "name": f"{layer_prefix}_mlp",
            "inputs": [prev_id],
            "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
            "is_parameter": False,
            "module_path": f"model.layers.{i}.mlp",
        })
        prev_id = f"{layer_prefix}_mlp"

    # Final norm
    if config["uses_rms_norm"]:
        nodes.append({
            "id": "final_norm",
            "op": {"type": "RmsNorm", "hidden_size": hidden_size, "epsilon": config["layer_norm_epsilon"]},
            "name": "final_norm",
            "inputs": [prev_id],
            "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
            "is_parameter": False,
            "module_path": "model.norm",
        })
        prev_id = "final_norm"

    # LM head
    nodes.append({
        "id": "lm_head",
        "op": {"type": "Linear", "in_features": hidden_size, "out_features": config["vocab_size"], "has_bias": False},
        "name": "lm_head",
        "inputs": [prev_id],
        "output_shape": {"dims": [1, 32, config["vocab_size"]], "dtype": "fp16"},
        "is_parameter": False,
        "module_path": "lm_head",
    })

    # Output
    nodes.append({
        "id": "output",
        "op": {"type": "Output"},
        "name": "output",
        "inputs": ["lm_head"],
        "output_shape": {"dims": [1, 32, config["vocab_size"]], "dtype": "fp16"},
        "is_parameter": False,
        "module_path": None,
    })

    return {
        "model_id": model_id,
        "architecture": f"{model_config.model_type.capitalize()}ForCausalLM",
        "transformers_version": _get_transformers_version(),
        "torch_version": "0.0.0",
        "model_config": config,
        "nodes": nodes,
        "weights": {},
        "inputs": [{"name": "input_ids", "shape": {"dims": [1, 32], "dtype": "int32"}}],
        "outputs": [{"name": "logits", "shape": {"dims": [1, 32, config["vocab_size"]], "dtype": "fp16"}}],
        "state_declarations": [],
        "trace_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_duration_secs": 0.0,
            "num_nodes": len(nodes),
            "num_parameters": 0,
            "parameter_bytes": 0,
            "decomposed": decompose,
            "warnings": ["Built via structural fallback (torch.fx tracing failed)"],
        },
    }


def _get_transformers_version() -> str:
    """Get the installed transformers library version."""
    try:
        import transformers
        return transformers.__version__
    except ImportError:
        return "unknown"


def error_exit(message: str):
    """Write an error to stderr and exit with non-zero status."""
    sys.stderr.write(f"ERROR: {message}\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
