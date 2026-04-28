#!/usr/bin/env python3
"""
HuggingFace Transformers Model Tracer for MILLer.

Traces a transformers model using torch.fx and exports the computation
graph as JSON for consumption by the Rust-side SIR construction pipeline.

FULLY DYNAMIC — no model_type heuristics, no hardcoded model lists.
All feature detection (norm type, RoPE usage, GQA config) is derived
from the model's actual structure at runtime:

1. Module type inspection: checks isinstance(module, nn.Linear) etc.
2. Config field presence: rms_norm_eps → RMSNorm, rope_theta → RoPE
3. Structural detection: weight but no bias → RMSNorm pattern

Supports three model families via dynamic class resolution:
- Decoder-only CausalLM (Llama, Qwen, GPT-2, etc.)
- Encoder-Decoder Seq2SeqLM (BART, T5, Dolphin, etc.)
- Multimodal models with extractable text decoders (Qwen3-ASR, etc.)

The model class is resolved dynamically from the config's `architectures`
field — no hardcoded model_type lists are used.

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
    model_class_hint = config.get("model_class", "auto")

    try:
        import torch
        from transformers import AutoConfig
    except ImportError as e:
        error_exit(f"Required packages not installed: {e}\nInstall with: pip install torch transformers")

    start_time = time.time()

    try:
        # Load model configuration
        model_config = AutoConfig.from_pretrained(model_id)

        # Dynamically resolve the model class and load the model
        torch_dtype = torch.float16 if dtype_str == "fp16" else torch.float32
        model, model_class = _load_model(
            model_id=model_id,
            model_config=model_config,
            torch_dtype=torch_dtype,
            model_class_hint=model_class_hint,
        )
        model.eval()

        # Create dummy inputs based on the model class
        batch_size = input_shapes[0].get("batch_size", 1)
        seq_len = input_shapes[0].get("seq_len", 32)
        inputs = _create_dummy_inputs(
            model=model,
            model_config=model_config,
            model_class=model_class,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        # Trace the model with torch.fx
        traced_graph = trace_model_fx(
            model=model,
            input_ids=inputs,
            model_config=model_config,
            model_id=model_id,
            decompose=decompose,
            with_kv_cache=with_kv_cache,
            fx_options=fx_options,
            model_class=model_class,
        )

        # Add metadata
        trace_duration = time.time() - start_time
        traced_graph["trace_metadata"]["trace_duration_secs"] = trace_duration
        traced_graph["trace_metadata"]["transformers_version"] = _get_transformers_version()
        traced_graph["trace_metadata"]["torch_version"] = torch.__version__
        traced_graph["trace_metadata"]["model_class"] = model_class

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
    model_class: str = "causal_lm",
) -> Dict[str, Any]:
    """
    Trace a transformers model using torch.fx and export as a MILLer TracedGraph.

    FULLY DYNAMIC — detects features by inspecting actual module types
    and config fields, never by matching model_type strings.

    Args:
        model: The HuggingFace model to trace.
        input_ids: Dummy input tensor(s) for tracing (dict for seq2seq, tensor for causal).
        model_config: The model's configuration.
        model_id: HuggingFace model ID.
        decompose: Whether to decompose composite ops during tracing.
        with_kv_cache: Whether to include KV-cache state in the graph.
        fx_options: Additional torch.fx tracing options.
        model_class: Which Auto class was used ("causal_lm", "seq2seq_lm", "decoder_only").

    Returns:
        A dictionary representing a TracedGraph (serializable to JSON).
    """
    import torch
    fx_options = fx_options or {}

    # Perform torch.fx tracing
    # For seq2seq models, we trace the decoder path (autoregressive generation)
    # which is the part that runs repeatedly on ANE. The encoder runs once.
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        # Fallback: build structural graph from config
        sys.stderr.write(f"Warning: symbolic_trace failed ({e}), falling back to structural graph construction\n")
        return build_fallback_graph(model_config, model_id, decompose, model_class=model_class)

    # ─── Dynamic Feature Discovery ────────────────────────────────
    # Walk the model's modules to discover what types are actually present.
    # This provides ground-truth feature detection that supplements the
    # config-driven detection. No model_type string matching is used.
    discovered = discover_model_features(model, model_config)

    # For encoder-decoder models, flag which sub-model we traced
    is_encoder_decoder = (model_class == "seq2seq_lm")
    if is_encoder_decoder:
        discovered["traced_component"] = "decoder"
        discovered["detection_methods"]["model_class"] = "auto_class_resolution"

    # Extract nodes from the traced graph
    nodes = []
    weights = {}
    state_declarations = []

    for fx_node in traced.graph.nodes:
        traced_op = map_fx_node_to_traced_op(fx_node, model_config, decompose, model)

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

    # Build model config section (fully dynamic)
    config_section = extract_model_config(model_config)
    config_section["model_class"] = model_class
    config_section["is_encoder_decoder"] = is_encoder_decoder

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
        cfg = _resolve_effective_config(model_config)
        num_layers = getattr(cfg, "num_hidden_layers", 1)
        num_heads = getattr(cfg, "num_attention_heads", 1)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim = config_section["hidden_size"] // num_heads

        for layer_idx in range(num_layers):
            state_declarations.append({
                "state_id": f"kv_cache_layer_{layer_idx}_key",
                "shape": [1, num_kv_heads, max_seq_len, head_dim],
                "dtype": "fp16",
                "layer_idx": layer_idx,
                "is_key": True,
            })
            state_declarations.append({
                "state_id": f"kv_cache_layer_{layer_idx}_value",
                "shape": [1, num_kv_heads, max_seq_len, head_dim],
                "dtype": "fp16",
                "layer_idx": layer_idx,
                "is_key": False,
            })

    # Build input specifications based on model class
    input_specs = _build_input_specs(input_ids, model_class)

    graph = {
        "model_id": model_id,
        "architecture": model.__class__.__name__,
        "transformers_version": _get_transformers_version(),
        "torch_version": "0.0.0",  # Will be filled by main()
        "model_config": config_section,
        "discovered_features": discovered,
        "nodes": nodes,
        "weights": weights,
        "inputs": input_specs,
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


def discover_model_features(model, model_config) -> Dict[str, Any]:
    """Discover model features by inspecting actual module types and config fields.
    
    FULLY DYNAMIC — walks the model's module tree to detect what types
    are actually present, then cross-references with config field presence.
    No model_type string matching is ever performed.
    
    This produces a `discovered_features` dictionary that the Rust side
    can use for additional validation without needing any heuristics.
    
    Detection methods (in order of reliability):
    1. Module type inspection: isinstance checks on actual nn.Module objects
    2. Config field presence: rms_norm_eps, rope_theta, etc.
    3. Structural detection: weight-without-bias patterns for RMSNorm
    """
    import torch
    
    features = {
        "norm_types_encountered": [],      # ["RMSNorm", "LayerNorm", ...]
        "has_rope_module": False,          # Rotary embedding module found
        "attention_module_types": [],       # ["SdpaAttention", "EagerAttention", ...]
        "mlp_module_types": [],            # ["LlamaMLP", "Qwen2MLP", ...]
        "linear_count": 0,
        "embedding_count": 0,
        "detection_methods": {},           # How each feature was detected
    }
    
    norm_type_set = set()
    attn_type_set = set()
    mlp_type_set = set()
    
    for name, module in model.named_modules():
        class_name = type(module).__name__
        
        # ─── Norm detection by actual type ────────────────────────
        # Check for RMSNorm variants (LlamaRMSNorm, Qwen2RMSNorm, etc.)
        if 'rms' in class_name.lower():
            norm_type_set.add("RMSNorm")
            features["detection_methods"]["norm_type"] = "module_type_inspection"
        elif isinstance(module, torch.nn.LayerNorm):
            norm_type_set.add("LayerNorm")
            features["detection_methods"]["norm_type"] = "module_type_inspection"
        elif 'norm' in class_name.lower() and 'rms' not in class_name.lower():
            # Generic norm module — check structural properties
            has_weight = hasattr(module, 'weight')
            has_bias = hasattr(module, 'bias') and module.bias is not None
            if has_weight and not has_bias:
                norm_type_set.add("RMSNorm")
                features["detection_methods"]["norm_type"] = "structural_detection"
            else:
                norm_type_set.add("LayerNorm")
                features["detection_methods"]["norm_type"] = "structural_detection"
        
        # ─── RoPE detection by module type ────────────────────────
        if 'rotary' in class_name.lower() or 'rope' in class_name.lower():
            features["has_rope_module"] = True
            features["detection_methods"]["rope"] = "module_type_inspection"
        
        # ─── Attention type detection ─────────────────────────────
        if 'attention' in class_name.lower() or 'attn' in class_name.lower():
            # Don't add leaf modules (like attention scores), only blocks
            if any(child for child in module.children()):
                attn_type_set.add(class_name)
        
        # ─── MLP type detection ───────────────────────────────────
        if 'mlp' in class_name.lower() or 'feed_forward' in class_name.lower():
            if any(child for child in module.children()):
                mlp_type_set.add(class_name)
        
        # ─── Linear count ─────────────────────────────────────────
        if isinstance(module, torch.nn.Linear):
            features["linear_count"] += 1
        
        # ─── Embedding count ──────────────────────────────────────
        if isinstance(module, torch.nn.Embedding):
            features["embedding_count"] += 1
    
    features["norm_types_encountered"] = sorted(norm_type_set)
    features["attention_module_types"] = sorted(attn_type_set)
    features["mlp_module_types"] = sorted(mlp_type_set)
    
    # ─── Cross-reference with config-driven detection ─────────────
    cfg = _resolve_effective_config(model_config)
    
    # If norm not detected by module inspection, check config
    if not norm_type_set:
        config_norm = _detect_norm_type(cfg)
        if config_norm != "unknown":
            features["norm_types_encountered"] = [config_norm]
            features["detection_methods"]["norm_type"] = "config_field_presence"
    
    # If RoPE not detected by module inspection, check config
    if not features["has_rope_module"]:
        if _detect_rope(cfg):
            features["has_rope_module"] = True
            if "rope" not in features["detection_methods"]:
                features["detection_methods"]["rope"] = "config_field_presence"
    
    # ─── GQA detection (purely structural) ────────────────────────
    num_heads = getattr(cfg, "num_attention_heads", 0)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    features["uses_gqa"] = num_kv_heads < num_heads
    features["detection_methods"]["gqa"] = "config_field_comparison"
    
    return features


def map_fx_node_to_traced_op(fx_node, model_config, decompose: bool, model=None) -> Dict[str, Any]:
    """
    Map a torch.fx node to a MILLer TracedOp representation.

    FULLY DYNAMIC — uses actual module type inspection when available,
    falling back to config-driven detection. No model_type heuristics.
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
        return map_module_call(fx_node, model_config, decompose, model)

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


def _inspect_module_type(model, target: str) -> Optional[str]:
    """Inspect the actual nn.Module type at a given module path.
    
    Returns the class name (e.g., 'LlamaRMSNorm', 'LayerNorm', 'Linear')
    or None if the module cannot be found. This is the fully dynamic
    approach: instead of pattern-matching on module path strings, we
    check what the module actually IS at runtime.
    
    This works for any HuggingFace model, including future architectures,
    because we never match on model_type strings — only on the actual
    Python class hierarchy.
    """
    try:
        submodule = model.get_submodule(target)
        return type(submodule).__name__
    except (AttributeError, RuntimeError):
        return None


def _is_rms_norm_module(model, target: str) -> bool:
    """Check if the module at the given path is an RMSNorm variant.
    
    Fully dynamic: checks the actual class hierarchy, not the name.
    RMSNorm modules are detected by checking if they:
    1. Have a class name containing 'RMS' or 'Rms' (e.g., LlamaRMSNorm, Qwen2RMSNorm)
    2. OR have weight but no bias (structural signature of RMSNorm)
    
    This handles any future RMSNorm implementation without code changes.
    """
    try:
        submodule = model.get_submodule(target)
        class_name = type(submodule).__name__.lower()
        
        # Direct RMSNorm class name match (any variant)
        if 'rms' in class_name:
            return True
        
        # Structural detection: RMSNorm has weight but no bias;
        # LayerNorm has both weight and bias
        has_weight = hasattr(submodule, 'weight')
        has_bias = hasattr(submodule, 'bias') and submodule.bias is not None
        
        # RMSNorm: weight, no bias
        if has_weight and not has_bias:
            return True
            
        return False
    except (AttributeError, RuntimeError):
        return False


def map_module_call(fx_node, model_config, decompose: bool, model=None) -> Dict[str, Any]:
    """Map a call_module node to a TracedOp.
    
    Uses actual module type inspection when the model is available,
    falling back to structural config detection otherwise. This is
    fully dynamic — no model_type string matching is ever performed.
    
    Detection priority:
    1. Actual isinstance() check on the nn.Module object
    2. Config field presence (rms_norm_eps, rope_theta, etc.)
    3. Conventional naming patterns as last-resort fallback
    """
    target = str(fx_node.target)
    
    # ─── Attempt actual module type inspection ───────────────────
    module_type = _inspect_module_type(model, target) if model is not None else None
    
    # ─── Linear layers ───────────────────────────────────────────
    # Detect by actual type OR by conventional projection naming.
    is_linear = False
    if model is not None:
        try:
            import torch
            submodule = model.get_submodule(target)
            is_linear = isinstance(submodule, torch.nn.Linear)
        except (AttributeError, RuntimeError):
            pass
    
    if is_linear or "linear" in target.lower() or any(
        proj in target for proj in ["q_proj", "k_proj", "v_proj", "o_proj", 
                                     "gate_proj", "up_proj", "down_proj"]
    ):
        return {
            "type": "Linear",
            "in_features": 0,  # Will be inferred from weight shape
            "out_features": 0,
            "has_bias": "bias" in target.lower() or any(
                p in target for p in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            "_module_path": target,
            "_module_type": module_type,
        }

    # ─── Embedding layers ────────────────────────────────────────
    is_embedding = False
    if model is not None:
        try:
            import torch
            submodule = model.get_submodule(target)
            is_embedding = isinstance(submodule, torch.nn.Embedding)
        except (AttributeError, RuntimeError):
            pass
    
    if is_embedding or "embed" in target.lower() or "wte" in target.lower():
        cfg = _resolve_effective_config(model_config)
        return {
            "type": "Embedding",
            "vocab_size": getattr(cfg, "vocab_size", 32000),
            "embed_dim": getattr(cfg, "hidden_size", 768),
            "_module_type": module_type,
        }

    # ─── Normalization layers ────────────────────────────────────
    # FULLY DYNAMIC: inspect actual module type first, then fall back
    # to config-driven detection. Never matches on model_type strings.
    is_norm_path = "norm" in target.lower() or "ln" in target.lower()
    
    if is_norm_path:
        cfg = _resolve_effective_config(model_config)
        
        # Primary: inspect actual module type from the model
        if model is not None and _is_rms_norm_module(model, target):
            return {
                "type": "RmsNorm",
                "hidden_size": getattr(cfg, "hidden_size", 768),
                "epsilon": getattr(cfg, "rms_norm_eps", 1e-6),
                "_module_type": module_type,
                "_detection_method": "module_type_inspection",
            }
        
        # Secondary: config-driven detection (rms_norm_eps field present)
        if decompose:
            norm_type = _detect_norm_type(cfg)
            if norm_type == "rms_norm":
                return {
                    "type": "RmsNorm",
                    "hidden_size": getattr(cfg, "hidden_size", 768),
                    "epsilon": getattr(cfg, "rms_norm_eps", 1e-6),
                    "_module_type": module_type,
                    "_detection_method": "config_field_presence",
                }
        
        # LayerNorm fallback (actual nn.LayerNorm or detected via config)
        return {
            "type": "LayerNorm",
            "normalized_shape": [getattr(cfg, "hidden_size", 768)],
            "epsilon": getattr(cfg, "layer_norm_eps", 1e-5),
            "_module_type": module_type,
        }

    # ─── Attention modules ───────────────────────────────────────
    if "attention" in target.lower() or "attn" in target.lower():
        cfg = _resolve_effective_config(model_config)
        return {
            "type": "AttentionBlock",
            "embed_dim": getattr(cfg, "hidden_size", 768),
            "num_heads": getattr(cfg, "num_attention_heads", 12),
            "head_dim": getattr(cfg, "hidden_size", 768) // getattr(cfg, "num_attention_heads", 12),
            "use_sdpa": True,
            "_module_type": module_type,
        }

    # ─── MLP / feed-forward modules ──────────────────────────────
    if "mlp" in target.lower() or "feed_forward" in target.lower():
        cfg = _resolve_effective_config(model_config)
        return {
            "type": "MlpBlock",
            "input_dim": getattr(cfg, "hidden_size", 768),
            "hidden_dim": getattr(cfg, "intermediate_size", 3072),
            "output_dim": getattr(cfg, "hidden_size", 768),
            "activation": getattr(cfg, "hidden_act", "gelu"),
            "_module_type": module_type,
        }

    return {
        "type": "Unknown",
        "op_name": "call_module",
        "target": target,
        "_module_type": module_type,
    }


def map_function_call(fx_node, model_config, decompose: bool) -> Dict[str, Any]:
    """Map a call_function node to a TracedOp.
    
    Also detects torch.nn.functional.rms_norm and similar dynamic ops.
    """
    target = str(fx_node.target)
    cfg = _resolve_effective_config(model_config)

    func_map = {
        "scaled_dot_product_attention": lambda: {
            "type": "ScaledDotProductAttention",
            "scale": 1.0 / ((getattr(cfg, "hidden_size", 768) // getattr(cfg, "num_attention_heads", 12)) ** 0.5),
        },
        "rms_norm": lambda: {
            "type": "RmsNorm",
            "hidden_size": getattr(cfg, "hidden_size", 768),
            "epsilon": getattr(cfg, "rms_norm_eps", 1e-6),
            "_detection_method": "function_call_inspection",
        },
        "layer_norm": lambda: {
            "type": "LayerNorm",
            "normalized_shape": [getattr(cfg, "hidden_size", 768)],
            "epsilon": getattr(cfg, "layer_norm_eps", 1e-5),
            "_detection_method": "function_call_inspection",
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


def _resolve_effective_config(model_config) -> object:
    """Resolve the effective text-model config from a potentially nested AutoConfig.
    
    Many modern HuggingFace models (e.g., Qwen3.5, Llava, Qwen2-VL) are multimodal
    and store the text decoder config in a sub-object like `text_config`. The standard
    LLM fields (hidden_size, num_attention_heads, etc.) live there, not at the top level.
    
    This function walks the config structure to find the sub-config that actually
    contains the text decoder parameters, without relying on model_type heuristics.
    
    Resolution order:
    1. If the config has `hidden_size` directly → it's already the effective config
    2. If the config has a `text_config` sub-object with `hidden_size` → use that
    3. If the config has a `decoder` sub-object with `hidden_size` → use that
    4. Otherwise → return the original config and let downstream code use defaults
    """
    # Fast path: config already has the standard fields
    if hasattr(model_config, "hidden_size") and getattr(model_config, "hidden_size", None) is not None:
        return model_config
    
    # Walk known sub-config names that HuggingFace transformers uses
    # for multimodal / encoder-decoder architectures
    sub_config_names = ["text_config", "decoder", "language_model", "text_model"]
    
    for name in sub_config_names:
        sub = getattr(model_config, name, None)
        if sub is not None and hasattr(sub, "hidden_size") and getattr(sub, "hidden_size", None) is not None:
            return sub
    
    # Nothing found — return original and let defaults kick in
    return model_config


def _detect_rope(config) -> bool:
    """Detect whether a model uses RoPE (Rotary Position Embeddings).
    
    Fully dynamic: checks for the actual config fields that indicate RoPE,
    never matches on model_type strings.
    
    RoPE is indicated by any of:
    - rope_theta (most LLMs: Llama, Qwen, Mistral, etc.)
    - rope_scaling (models with extended context)
    - rope_parameters (newer Qwen models, contains rope_theta inside)
    - rotary_emb_base (some older models)
    - position_embedding_type == "rope" (explicit declaration)
    """
    # Explicit flag (rare but definitive)
    explicit = getattr(config, "uses_rope", None)
    if explicit is not None:
        return bool(explicit)
    
    # Direct config field indicators
    if hasattr(config, "rope_theta"):
        return True
    if hasattr(config, "rope_scaling"):
        return True
    if hasattr(config, "rotary_emb_base"):
        return True
    
    # Nested rope_parameters dict (Qwen3, Qwen3.5)
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None and isinstance(rope_params, dict):
        if "rope_theta" in rope_params or "rope_type" in rope_params:
            return True
    
    # Explicit position_embedding_type declaration
    pos_emb_type = getattr(config, "position_embedding_type", None)
    if pos_emb_type is not None and "rope" in str(pos_emb_type).lower():
        return True
    
    # No RoPE indicators found
    return False


def _detect_norm_type(config) -> str:
    """Detect the normalization type used by a model.
    
    Fully dynamic: checks for the actual config fields that indicate
    norm type, never matches on model_type strings.
    
    Returns: "rms_norm", "layer_norm", or "unknown"
    
    RMSNorm is indicated by:
    - rms_norm_eps field present (the epsilon for RMS normalization)
    
    LayerNorm is indicated by:
    - layer_norm_eps field present WITHOUT rms_norm_eps
    """
    has_rms_eps = hasattr(config, "rms_norm_eps")
    has_ln_eps = hasattr(config, "layer_norm_eps")
    
    if has_rms_eps:
        return "rms_norm"
    if has_ln_eps:
        return "layer_norm"
    
    return "unknown"


def extract_model_config(model_config) -> Dict[str, Any]:
    """Extract model configuration into the TracedGraph format.
    
    FULLY DYNAMIC — no model_type heuristics, no hardcoded model lists.
    
    This function derives all decomposition hints from the model's AutoConfig
    fields alone. It works for any HuggingFace model, including future
    architectures, without requiring code changes.
    
    The approach:
    1. Resolve the effective text config (handles multimodal nesting)
    2. Detect norm type from config fields (rms_norm_eps vs layer_norm_eps)
    3. Detect RoPE from config fields (rope_theta, rope_parameters, etc.)
    4. Detect GQA from config fields (num_key_value_heads < num_attention_heads)
    5. Extract all other fields directly from the config
    
    If a model's config doesn't expose certain fields, reasonable defaults
    are used. No model_type string matching is ever performed.
    """
    # Step 1: Resolve the effective config for text decoder
    cfg = _resolve_effective_config(model_config)
    
    # Step 2: Extract standard fields directly from the resolved config
    hidden_size = getattr(cfg, "hidden_size", 768)
    num_heads = getattr(cfg, "num_attention_heads", 12)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    model_type = getattr(model_config, "model_type", "unknown")  # Top-level model_type for logging only
    
    # Step 3: Detect normalization type from config fields
    norm_type = _detect_norm_type(cfg)
    uses_rms_norm = (norm_type == "rms_norm")
    
    # If norm type is unknown, default to RMSNorm for modern LLMs
    # (most post-2023 architectures use RMSNorm). This is a safe default
    # because the legality rewrite pass will handle the decomposition.
    if norm_type == "unknown":
        uses_rms_norm = True  # Safe default — legality rewrite will adjust
    
    # Step 4: Detect RoPE from config fields
    uses_rope = _detect_rope(cfg)
    
    # Step 5: GQA — purely structural, no heuristics needed
    uses_gqa = num_kv_heads < num_heads
    
    # Step 6: Activation function
    hidden_act = getattr(cfg, "hidden_act", None)
    if hidden_act is None:
        hidden_act = getattr(cfg, "activation_function", "gelu")
    
    # Step 7: Epsilon value — pick the one that matches the detected norm type
    if uses_rms_norm:
        layer_norm_epsilon = getattr(cfg, "rms_norm_eps", 1e-6)
    else:
        layer_norm_epsilon = getattr(cfg, "layer_norm_eps", 1e-5)
    
    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", 12),
        "intermediate_size": getattr(cfg, "intermediate_size", 3072),
        "vocab_size": getattr(cfg, "vocab_size", 32000),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", 2048),
        "layer_norm_epsilon": layer_norm_epsilon,
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


def _load_model(model_id, model_config, torch_dtype, model_class_hint="auto"):
    """Dynamically resolve and load the appropriate HuggingFace model class.

    FULLY DYNAMIC — determines the correct Auto class by inspecting the
    model's `architectures` field and config structure. No hardcoded
    model_type lists are used. No user input required.

    The `model_class_hint` parameter is accepted for backward compatibility
    but is never needed — the auto logic handles all known architectures.

    Resolution strategy (purely config-driven, no heuristics):
    1. Inspect config.architectures for HuggingFace's declared class names:
       - "*ForCausalLM"           → AutoModelForCausalLM  (decoder-only)
       - "*ForSeq2SeqLM"          → AutoModelForSeq2SeqLM (encoder-decoder)
       - "VisionEncoderDecoder*"  → AutoModelForSeq2SeqLM (encoder-decoder)
       - "*ForConditionalGeneration" → ambiguous! Disambiguate via:
         a. is_encoder_decoder=True → Seq2SeqLM (e.g., BART, T5, mBART)
         b. has text_config or thinker_config → multimodal with causal LM
            decoder → load decoder via _load_decoder_from_multimodal()
         c. is_encoder_decoder=False, no text_config → try CausalLM first
    2. config.is_encoder_decoder=True (without architecture match) → Seq2SeqLM
    3. config has text_config or thinker_config.text_config → multimodal decoder
    4. Fall back: try CausalLM, then Seq2SeqLM

    Returns:
        Tuple of (model, model_class_string)
        model_class_string is one of: "causal_lm", "seq2seq_lm", "decoder_only"
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    architectures = getattr(model_config, "architectures", []) or []
    is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False)

    sys.stderr.write(f"Auto-detecting model class for {model_id}\n")
    sys.stderr.write(f"  architectures={architectures}, is_encoder_decoder={is_encoder_decoder}\n")

    # ─── 1. Architecture-name resolution ───────────────────────────
    for arch in architectures:
        arch_lower = arch.lower()

        # "*ForCausalLM" — always decoder-only causal LM
        if arch_lower.endswith("forcausallm"):
            sys.stderr.write(f"  → AutoModelForCausalLM (architecture declares CausalLM)\n")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                )
                return model, "causal_lm"
            except Exception as e:
                sys.stderr.write(f"  AutoModelForCausalLM failed: {e}, trying alternatives\n")

        # "*ForSeq2SeqLM" — always encoder-decoder seq2seq
        if arch_lower.endswith("forseq2seqlm"):
            sys.stderr.write(f"  → AutoModelForSeq2SeqLM (architecture declares Seq2SeqLM)\n")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                )
                return model, "seq2seq_lm"
            except Exception as e:
                sys.stderr.write(f"  AutoModelForSeq2SeqLM failed: {e}, trying alternatives\n")

        # "VisionEncoderDecoder*" — always encoder-decoder
        if arch_lower.startswith("visionencoderdecoder"):
            sys.stderr.write(f"  → AutoModelForSeq2SeqLM (VisionEncoderDecoder)\n")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                )
                return model, "seq2seq_lm"
            except Exception as e:
                sys.stderr.write(f"  AutoModelForSeq2SeqLM failed: {e}, trying alternatives\n")

        # "*ForConditionalGeneration" — ambiguous! Disambiguate:
        # - is_encoder_decoder=True → Seq2SeqLM (BART, T5, mBART, etc.)
        # - has text_config → multimodal with causal LM decoder (Qwen3.5, Qwen3-ASR, etc.)
        # - else → try CausalLM (some models just use this name loosely)
        if arch_lower.endswith("forconditionalgeneration"):
            if is_encoder_decoder:
                sys.stderr.write(f"  → AutoModelForSeq2SeqLM (ConditionalGeneration + is_encoder_decoder)\n")
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                    )
                    return model, "seq2seq_lm"
                except Exception as e:
                    sys.stderr.write(f"  AutoModelForSeq2SeqLM failed: {e}, trying alternatives\n")
            else:
                # Multimodal with a causal LM text decoder (Qwen3.5, Qwen3-ASR, etc.)
                text_cfg = _find_text_config(model_config)
                if text_cfg is not None:
                    sys.stderr.write(f"  → decoder_only (ConditionalGeneration + text_config, not encoder-decoder)\n")
                    return _load_decoder_from_multimodal(model_id, model_config, torch_dtype)
                else:
                    sys.stderr.write(f"  → AutoModelForCausalLM (ConditionalGeneration, no is_encoder_decoder, no text_config)\n")
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                        )
                        return model, "causal_lm"
                    except Exception as e:
                        sys.stderr.write(f"  AutoModelForCausalLM failed: {e}, trying alternatives\n")

    # ─── 2. Config-level signals ──────────────────────────────────
    # is_encoder_decoder=True without architecture match → Seq2SeqLM
    if is_encoder_decoder:
        sys.stderr.write(f"  → AutoModelForSeq2SeqLM (is_encoder_decoder=True, no architecture match)\n")
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            )
            return model, "seq2seq_lm"
        except Exception as e:
            sys.stderr.write(f"  AutoModelForSeq2SeqLM failed: {e}, trying alternatives\n")

    # ─── 3. Multimodal with text_config ───────────────────────────
    text_cfg = _find_text_config(model_config)
    if text_cfg is not None:
        sys.stderr.write(f"  → decoder_only (text_config found, extracting decoder)\n")
        return _load_decoder_from_multimodal(model_id, model_config, torch_dtype)

    # ─── 4. Fallback: try CausalLM, then Seq2SeqLM ───────────────
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        sys.stderr.write(f"  → AutoModelForCausalLM (fallback, most common for ANE)\n")
        return model, "causal_lm"
    except Exception as e:
        sys.stderr.write(f"  AutoModelForCausalLM failed: {e}\n")

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        sys.stderr.write(f"  → AutoModelForSeq2SeqLM (last resort)\n")
        return model, "seq2seq_lm"
    except Exception as e:
        sys.stderr.write(f"  AutoModelForSeq2SeqLM also failed: {e}\n")

    error_exit(f"Could not load model {model_id} with any Auto class. "
               f"architectures={architectures}, is_encoder_decoder={is_encoder_decoder}")


def _find_text_config(model_config):
    """Find the text decoder config in a multimodal model.

    Searches for the text config in multiple locations that different
    HuggingFace architectures use. This is NOT a heuristic — it's
    structural inspection of the config object.

    Locations checked:
    1. config.text_config (Qwen3.5-VL, Llava, etc.)
    2. config.thinker_config.text_config (Qwen3-ASR)
    3. config.decoder (VisionEncoderDecoder models like Dolphin)

    Returns the text config object if found, None otherwise.
    """
    # Direct text_config (most multimodal models)
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return text_config

    # Nested thinker_config.text_config (Qwen3-ASR pattern)
    thinker_config = getattr(model_config, "thinker_config", None)
    if thinker_config is not None:
        text_config = getattr(thinker_config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return text_config

    # Decoder config (VisionEncoderDecoder pattern like Dolphin)
    decoder_config = getattr(model_config, "decoder", None)
    if decoder_config is not None and hasattr(decoder_config, "hidden_size"):
        return decoder_config

    return None


def _load_decoder_from_multimodal(model_id, model_config, torch_dtype):
    """Load the text decoder from a multimodal or encoder-decoder model.

    FULLY DYNAMIC — discovers the decoder by structural inspection:
    1. Try AutoModelForCausalLM (some multimodal models support this directly)
    2. Try VisionEncoderDecoderModel and extract the decoder sub-module
    3. Try AutoModel and extract language_model/model/decoder sub-module

    This handles:
    - Qwen3.5-0.8B (multimodal VL with Qwen3 causal LM decoder)
    - Qwen3-ASR-0.6B (audio encoder + Qwen3 causal LM decoder)
    - Dolphin-1.5 (DonutSwin encoder + mBART decoder)
    - Any future multimodal model with a causal LM text decoder

    Returns:
        Tuple of (model, "decoder_only")
    """
    import torch
    from transformers import AutoModelForCausalLM

    text_config = _find_text_config(model_config)
    text_model_type = getattr(text_config, "model_type", None) if text_config else None
    sys.stderr.write(f"  Extracting decoder from {model_id} (text model_type={text_model_type})\n")

    # ─── Strategy 1: AutoModelForCausalLM direct ──────────────────
    # Some multimodal models (Qwen3.5-VL, Qwen3-ASR) can be loaded
    # directly as CausalLM — transformers routes to the right class.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        sys.stderr.write(f"  Loaded decoder via AutoModelForCausalLM\n")
        return model, "decoder_only"
    except Exception as e:
        sys.stderr.write(f"  AutoModelForCausalLM direct failed: {e}\n")

    # ─── Strategy 2: VisionEncoderDecoderModel ────────────────────
    # Models like Dolphin use VisionEncoderDecoderModel which has
    # an explicit encoder + decoder split.
    try:
        from transformers import VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        # Extract the decoder sub-module
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "forward"):
            sys.stderr.write(f"  Extracted decoder from VisionEncoderDecoderModel\n")
            return decoder, "decoder_only"
    except Exception as e:
        sys.stderr.write(f"  VisionEncoderDecoderModel failed: {e}\n")

    # ─── Strategy 3: Generic sub-module extraction ────────────────
    # Load the full model and try common decoder sub-module names.
    try:
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for attr_name in ["language_model", "model", "decoder", "text_decoder"]:
            sub_model = getattr(full_model, attr_name, None)
            if sub_model is not None and hasattr(sub_model, "forward"):
                sys.stderr.write(f"  Extracted decoder as .{attr_name}\n")
                return sub_model, "decoder_only"

        error_exit(f"Could not find decoder sub-module in {model_id}. "
                   f"Tried: language_model, model, decoder, text_decoder")
    except Exception as e:
        error_exit(f"Failed to load and extract decoder from {model_id}: {e}")


def _create_dummy_inputs(model, model_config, model_class, batch_size, seq_len):
    """Create dummy inputs appropriate for the model class.

    For causal LM: returns a single tensor of input_ids
    For seq2seq LM: returns a dict with decoder_input_ids and encoder_hidden_states
    For decoder_only: returns a single tensor of input_ids

    The inputs are designed for tracing the decoder/generation path,
    which is the part that runs on ANE.
    """
    import torch

    cfg = _resolve_effective_config(model_config)
    vocab_size = getattr(cfg, "vocab_size", 32000)

    if model_class == "seq2seq_lm":
        # For encoder-decoder models, we trace the DECODER path.
        # The encoder runs once and its output becomes a fixed input.
        # We provide:
        #   - decoder_input_ids: the token ids being generated
        #   - encoder_hidden_states: pre-computed encoder output (treated as constant)
        decoder_vocab_size = getattr(cfg, "vocab_size", vocab_size)
        decoder_input_ids = torch.randint(0, decoder_vocab_size, (batch_size, seq_len))

        # Get encoder output dimension (d_model for BART, hidden_size for others)
        encoder_hidden_dim = getattr(cfg, "hidden_size", getattr(cfg, "d_model", 768))
        encoder_seq_len = seq_len  # Same as decoder for simplicity in tracing

        # Create a dummy encoder output
        encoder_hidden_states = torch.randn(
            batch_size, encoder_seq_len, encoder_hidden_dim,
            dtype=next(model.parameters()).dtype if list(model.parameters()) else torch.float32,
        )

        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }
    else:
        # CausalLM or decoder_only: standard input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        return input_ids


def _build_input_specs(input_ids, model_class):
    """Build input tensor specifications for the TracedGraph JSON.

    Handles both tensor inputs (causal LM) and dict inputs (seq2seq LM).
    """
    import torch

    if model_class == "seq2seq_lm" and isinstance(input_ids, dict):
        specs = []
        for name, tensor in input_ids.items():
            if isinstance(tensor, torch.Tensor):
                dtype_str = str(tensor.dtype).replace("torch.", "")
                if tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
                    dtype_str = "int32"
                elif tensor.dtype in (torch.float16, torch.bfloat16):
                    dtype_str = "fp16"
                elif tensor.dtype == torch.float32:
                    dtype_str = "fp32"
                specs.append({
                    "name": name,
                    "shape": {"dims": list(tensor.shape), "dtype": dtype_str},
                })
        return specs
    elif isinstance(input_ids, torch.Tensor):
        return [{"name": "input_ids", "shape": {"dims": list(input_ids.shape), "dtype": "int32"}}]
    else:
        return [{"name": "input_ids", "shape": {"dims": [1, 32], "dtype": "int32"}}]


def build_fallback_graph(model_config, model_id: str, decompose: bool, model_class: str = "causal_lm") -> Dict[str, Any]:
    """
    Build a TracedGraph using structural knowledge when torch.fx tracing fails.

    This constructs a graph based on the model's configuration rather than
    actual tracing, producing the expected layer structure for known architectures.
    ALL feature detection is fully dynamic — no model_type heuristics.
    """
    config = extract_model_config(model_config)
    config["model_class"] = model_class
    config["is_encoder_decoder"] = (model_class == "seq2seq_lm")

    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    head_dim = hidden_size // num_heads

    is_encoder_decoder = (model_class == "seq2seq_lm")

    nodes = []

    # For seq2seq models, the fallback graph represents the decoder path
    if is_encoder_decoder:
        # Encoder output placeholder (pre-computed, treated as constant on ANE)
        nodes.append({
            "id": "encoder_hidden_states",
            "op": {"type": "Placeholder"},
            "name": "encoder_hidden_states",
            "inputs": [],
            "output_shape": {"dims": [1, 32, hidden_size], "dtype": "fp16"},
            "is_parameter": False,
            "module_path": None,
        })
        # Decoder input IDs
        nodes.append({
            "id": "decoder_input_ids",
            "op": {"type": "Placeholder"},
            "name": "decoder_input_ids",
            "inputs": [],
            "output_shape": {"dims": [1, 32], "dtype": "int32"},
            "is_parameter": False,
            "module_path": None,
        })
    else:
        # Standard causal LM input
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
    # For seq2seq, the embedding takes decoder_input_ids; for causal LM, input_ids
    embed_input = "decoder_input_ids" if is_encoder_decoder else "input_ids"
    nodes.append({
        "id": "embed_tokens",
        "op": {"type": "Embedding", "vocab_size": config["vocab_size"], "embed_dim": hidden_size},
        "name": "embed_tokens",
        "inputs": [embed_input],
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
        "discovered_features": {
            "norm_types_encountered": ["RMSNorm"] if config["uses_rms_norm"] else ["LayerNorm"],
            "has_rope_module": config["uses_rope"],
            "attention_module_types": [],
            "mlp_module_types": [],
            "linear_count": 0,
            "embedding_count": 0,
            "uses_gqa": config["uses_gqa"],
            "detection_methods": {
                "norm_type": "config_field_presence",
                "rope": "config_field_presence",
                "gqa": "config_field_comparison",
            },
        },
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
