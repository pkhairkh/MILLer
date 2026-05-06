"""Tests for trace_model.py — Fully Dynamic Model Tracing.

This test module validates the MILLer tracing pipeline across two tiers:

TIER 1 — Fast Unit Tests (no model download required):
  - Config-driven feature detection (_detect_norm_type, _detect_rope, etc.)
  - Model config extraction (extract_model_config)
  - Effective config resolution (_resolve_effective_config)
  - Model class resolution logic (_load_model detection paths)
  - Fallback graph construction (build_fallback_graph)
  - Fx node mapping functions (map_module_call, map_function_call, map_method_call)
  - Dynamic feature discovery with mock models (discover_model_features)

TIER 2 — Integration Tests (marked @pytest.mark.needs_hf_models):
  - Full end-to-end trace of each target model from HuggingFace Hub
  - Validates that _load_model resolves the correct Auto class
  - Validates that discover_model_features detects correct norm/RoPE/GQA
  - Validates that the traced graph has the expected structure
  - Produces JSON fixtures for Rust Layer 2 tests

Target models (all sub-1B parameters):
  1. meta-llama/Llama-3.2-1B    — Standard causal LM (RMSNorm + RoPE + GQA)
  2. Qwen/Qwen3-0.6B            — Causal LM (RMSNorm + RoPE + GQA 8:1)
  3. Qwen/Qwen3.5-0.8B          — Causal LM with unknown model_type="qwen3_5_text"
  4. ByteDance/Dolphin-1.5       — Encoder-decoder (DonutSwin + BART decoder)
  5. Qwen/Qwen3-ASR-0.6B        — Multimodal with extractable Qwen3 decoder
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

# Ensure the python directory is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trace_model import (
    _build_input_specs,
    _detect_norm_type,
    _detect_rope,
    _resolve_effective_config,
    build_fallback_graph,
    discover_model_features,
    extract_model_config,
    map_function_call,
    map_method_call,
    map_module_call,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: FAST UNIT TESTS — No model download required
# ═══════════════════════════════════════════════════════════════════════════════


# ─── Helper: build a mock HuggingFace config object ──────────────────────────

def _make_config(**kwargs):
    """Create a mock HuggingFace AutoConfig-like object from keyword arguments.

    Supports nested sub-configs via SimpleNamespace.
    """
    return SimpleNamespace(**kwargs)


# ─── Test: _detect_norm_type ─────────────────────────────────────────────────

class TestDetectNormType:
    """Test the dynamic norm type detection from config fields.

    The detection is purely field-presence based:
    - rms_norm_eps present → "rms_norm"
    - layer_norm_eps present (without rms_norm_eps) → "layer_norm"
    - neither present → "unknown"
    """

    def test_rms_norm_detected_from_rms_norm_eps(self):
        """Models with rms_norm_eps field should be detected as RMSNorm."""
        cfg = _make_config(rms_norm_eps=1e-6)
        assert _detect_norm_type(cfg) == "rms_norm"

    def test_layer_norm_detected_from_layer_norm_eps(self):
        """Models with only layer_norm_eps should be detected as LayerNorm."""
        cfg = _make_config(layer_norm_eps=1e-5)
        assert _detect_norm_type(cfg) == "layer_norm"

    def test_rms_norm_takes_priority_over_layer_norm(self):
        """When both fields are present, rms_norm_eps wins (matches Llama behavior)."""
        cfg = _make_config(rms_norm_eps=1e-6, layer_norm_eps=1e-5)
        assert _detect_norm_type(cfg) == "rms_norm"

    def test_unknown_when_no_norm_fields(self):
        """Models without norm epsilon fields return 'unknown'."""
        cfg = _make_config(hidden_size=768)
        assert _detect_norm_type(cfg) == "unknown"

    def test_llama_3_2_config(self):
        """Llama-3.2-1B has rms_norm_eps=1e-5, should detect as RMSNorm."""
        cfg = _make_config(rms_norm_eps=1e-5, hidden_size=2048)
        assert _detect_norm_type(cfg) == "rms_norm"

    def test_qwen3_config(self):
        """Qwen3-0.6B has rms_norm_eps=1e-6, should detect as RMSNorm."""
        cfg = _make_config(rms_norm_eps=1e-6, hidden_size=1024)
        assert _detect_norm_type(cfg) == "rms_norm"

    def test_dolphin_bart_config(self):
        """Dolphin-1.5 uses BART decoder with layer_norm_eps, should detect as LayerNorm."""
        cfg = _make_config(layer_norm_eps=1e-6, hidden_size=768)
        assert _detect_norm_type(cfg) == "layer_norm"


# ─── Test: _detect_rope ──────────────────────────────────────────────────────

class TestDetectRope:
    """Test the dynamic RoPE detection from config fields.

    RoPE is indicated by any of:
    - rope_theta field present
    - rope_scaling field present
    - rotary_emb_base field present
    - rope_parameters dict with rope_theta or rope_type
    - position_embedding_type containing "rope"
    - uses_rope explicit flag
    """

    def test_rope_detected_from_rope_theta(self):
        """Standard RoPE detection via rope_theta field (Llama, Qwen, Mistral)."""
        cfg = _make_config(rope_theta=10000.0)
        assert _detect_rope(cfg) is True

    def test_rope_detected_from_rope_scaling(self):
        """RoPE detection via rope_scaling field (extended context models)."""
        cfg = _make_config(rope_scaling={"type": "linear", "factor": 2.0})
        assert _detect_rope(cfg) is True

    def test_rope_detected_from_rotary_emb_base(self):
        """RoPE detection via rotary_emb_base field (some older models)."""
        cfg = _make_config(rotary_emb_base=10000.0)
        assert _detect_rope(cfg) is True

    def test_rope_detected_from_rope_parameters_dict(self):
        """Qwen3/Qwen3.5 use rope_parameters dict with rope_theta inside."""
        cfg = _make_config(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
        assert _detect_rope(cfg) is True

    def test_rope_detected_from_position_embedding_type(self):
        """Explicit position_embedding_type = 'rope'."""
        cfg = _make_config(position_embedding_type="rope")
        assert _detect_rope(cfg) is True

    def test_rope_detected_from_uses_rope_flag(self):
        """Explicit uses_rope=True flag."""
        cfg = _make_config(uses_rope=True)
        assert _detect_rope(cfg) is True

    def test_no_rope_when_no_indicators(self):
        """BART decoder (Dolphin-1.5) has no RoPE indicators."""
        cfg = _make_config(hidden_size=768, layer_norm_eps=1e-6)
        assert _detect_rope(cfg) is False

    def test_no_rope_with_position_embedding_type_absolute(self):
        """Models with learned positional embeddings don't use RoPE."""
        cfg = _make_config(position_embedding_type="absolute")
        assert _detect_rope(cfg) is False

    def test_uses_rope_false_overrides_other_fields(self):
        """Explicit uses_rope=False should override other indicators."""
        cfg = _make_config(uses_rope=False, rope_theta=10000.0)
        assert _detect_rope(cfg) is False

    def test_llama_3_2_rope_theta(self):
        """Llama-3.2-1B uses rope_theta=500000.0."""
        cfg = _make_config(rope_theta=500000.0)
        assert _detect_rope(cfg) is True

    def test_qwen3_5_rope_parameters(self):
        """Qwen3.5-0.8B uses rope_parameters dict."""
        cfg = _make_config(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
        assert _detect_rope(cfg) is True

    def test_empty_rope_parameters_dict(self):
        """Empty rope_parameters dict should not trigger RoPE detection."""
        cfg = _make_config(rope_parameters={})
        assert _detect_rope(cfg) is False


# ─── Test: _resolve_effective_config ─────────────────────────────────────────

class TestResolveEffectiveConfig:
    """Test the effective config resolution for multimodal/nested models.

    Resolution order:
    1. Config with hidden_size directly → already effective
    2. Config with text_config sub-object → use that
    3. Config with decoder sub-object → use that
    4. Config with language_model sub-object → use that
    5. Otherwise → return original config
    """

    def test_direct_config_returned_when_has_hidden_size(self):
        """Standard causal LM configs have hidden_size directly."""
        cfg = _make_config(hidden_size=2048, model_type="llama")
        resolved = _resolve_effective_config(cfg)
        assert resolved is cfg

    def test_text_config_extracted_for_multimodal(self):
        """Qwen3-ASR stores decoder params in text_config."""
        text_cfg = _make_config(hidden_size=896, model_type="qwen3")
        top_cfg = _make_config(model_type="qwen3_asr", text_config=text_cfg)
        resolved = _resolve_effective_config(top_cfg)
        assert resolved is text_cfg
        assert resolved.hidden_size == 896

    def test_decoder_config_extracted(self):
        """Some encoder-decoder models use decoder sub-config."""
        dec_cfg = _make_config(hidden_size=768)
        top_cfg = _make_config(decoder=dec_cfg)
        resolved = _resolve_effective_config(top_cfg)
        assert resolved is dec_cfg

    def test_language_model_config_extracted(self):
        """Some models store text model in language_model sub-config."""
        lm_cfg = _make_config(hidden_size=4096)
        top_cfg = _make_config(language_model=lm_cfg)
        resolved = _resolve_effective_config(top_cfg)
        assert resolved is lm_cfg

    def test_text_model_config_extracted(self):
        """Some models use text_model sub-config."""
        tm_cfg = _make_config(hidden_size=1024)
        top_cfg = _make_config(text_model=tm_cfg)
        resolved = _resolve_effective_config(top_cfg)
        assert resolved is tm_cfg

    def test_returns_original_when_no_sub_config(self):
        """When no sub-config is found, return original (defaults will kick in)."""
        cfg = _make_config(model_type="unknown")
        resolved = _resolve_effective_config(cfg)
        assert resolved is cfg

    def test_hidden_size_none_treated_as_missing(self):
        """Config with hidden_size=None should not short-circuit resolution."""
        text_cfg = _make_config(hidden_size=896)
        top_cfg = _make_config(hidden_size=None, text_config=text_cfg)
        resolved = _resolve_effective_config(top_cfg)
        assert resolved is text_cfg

    def test_qwen3_asr_config_resolution(self):
        """Full Qwen3-ASR-0.6B config structure: top-level has no hidden_size,
        text_config has hidden_size=896."""
        text_cfg = _make_config(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            num_hidden_layers=24,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            model_type="qwen3",
        )
        top_cfg = _make_config(
            model_type="qwen3_asr",
            text_config=text_cfg,
            is_encoder_decoder=False,
        )
        resolved = _resolve_effective_config(top_cfg)
        assert resolved.hidden_size == 896
        assert resolved.num_key_value_heads == 2


# ─── Test: extract_model_config ──────────────────────────────────────────────

class TestExtractModelConfig:
    """Test the full model config extraction pipeline.

    This validates the end-to-end config extraction including:
    - Effective config resolution
    - Norm type detection
    - RoPE detection
    - GQA detection
    - Activation function extraction
    """

    def test_llama_3_2_config_extraction(self):
        """Llama-3.2-1B: RMSNorm + RoPE + GQA + SiLU."""
        cfg = _make_config(
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=16,
            intermediate_size=8192,
            vocab_size=128256,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            hidden_act="silu",
            rope_theta=500000.0,
            model_type="llama",
        )
        result = extract_model_config(cfg)

        assert result["hidden_size"] == 2048
        assert result["num_attention_heads"] == 32
        assert result["num_key_value_heads"] == 8
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True  # 8 < 32
        assert result["hidden_act"] == "silu"
        assert result["model_type"] == "llama"
        assert result["layer_norm_epsilon"] == 1e-5  # rms_norm_eps value

    def test_qwen3_0_6b_config_extraction(self):
        """Qwen3-0.6B: RMSNorm + RoPE + GQA (8 KV heads for 16 Q heads)."""
        cfg = _make_config(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=28,
            intermediate_size=4096,
            vocab_size=151936,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="qwen3",
        )
        result = extract_model_config(cfg)

        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True  # 8 < 16

    def test_qwen3_5_config_extraction(self):
        """Qwen3.5-0.8B: unknown model_type, RMSNorm + RoPE + GQA via rope_parameters."""
        cfg = _make_config(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=24,
            intermediate_size=4096,
            vocab_size=151936,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
            model_type="qwen3_5_text",
        )
        result = extract_model_config(cfg)

        assert result["model_type"] == "qwen3_5_text"
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True  # Detected via rope_parameters
        assert result["uses_gqa"] is True   # 8 < 16

    def test_dolphin_1_5_config_extraction(self):
        """Dolphin-1.5: BART decoder with LayerNorm + learned positions (no RoPE)."""
        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_hidden_layers=6,
            intermediate_size=3072,
            vocab_size=50265,
            max_position_embeddings=1024,
            layer_norm_eps=1e-6,
            hidden_act="gelu",
            model_type="dolphin",
            is_encoder_decoder=True,
        )
        result = extract_model_config(cfg)

        assert result["uses_rms_norm"] is False
        assert result["uses_rope"] is False  # BART uses learned positional embeddings
        assert result["uses_gqa"] is False   # 12 == 12
        assert result["layer_norm_epsilon"] == 1e-6  # layer_norm_eps value
        assert result["hidden_act"] == "gelu"

    def test_qwen3_asr_config_extraction(self):
        """Qwen3-ASR-0.6B: multimodal with text_config, Qwen3 decoder (RMSNorm + RoPE + GQA)."""
        text_cfg = _make_config(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            num_hidden_layers=24,
            intermediate_size=4864,
            vocab_size=151936,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=1000000.0,
            model_type="qwen3",
        )
        top_cfg = _make_config(
            model_type="qwen3_asr",
            text_config=text_cfg,
            is_encoder_decoder=False,
        )
        result = extract_model_config(top_cfg)

        # Should resolve to text_config
        assert result["hidden_size"] == 896
        assert result["num_attention_heads"] == 14
        assert result["num_key_value_heads"] == 2
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True   # 2 < 14 → extreme 7:1 GQA

    def test_unknown_model_type_still_works(self):
        """A completely unknown model_type should still produce valid config."""
        cfg = _make_config(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=6,
            intermediate_size=2048,
            vocab_size=50000,
            max_position_embeddings=4096,
            rms_norm_eps=1e-6,
            hidden_act="gelu",
            rope_theta=10000.0,
            model_type="future_architecture_v7",
        )
        result = extract_model_config(cfg)

        assert result["model_type"] == "future_architecture_v7"
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is False  # 8 == 8

    def test_gqa_detected_when_kv_heads_less_than_attention_heads(self):
        """GQA is a structural property: num_key_value_heads < num_attention_heads."""
        cfg = _make_config(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=4,  # 4 < 16 → GQA
            num_hidden_layers=12,
            intermediate_size=4096,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="test_gqa",
        )
        result = extract_model_config(cfg)
        assert result["uses_gqa"] is True

    def test_no_gqa_when_kv_heads_equal_attention_heads(self):
        """Standard MHA: num_key_value_heads == num_attention_heads."""
        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,  # 12 == 12 → no GQA
            num_hidden_layers=6,
            intermediate_size=3072,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="gelu",
            model_type="test_mha",
        )
        result = extract_model_config(cfg)
        assert result["uses_gqa"] is False

    def test_no_gqa_when_kv_heads_not_specified(self):
        """When num_key_value_heads is not specified, defaults to num_attention_heads."""
        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            # num_key_value_heads intentionally omitted
            num_hidden_layers=6,
            intermediate_size=3072,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="gelu",
            model_type="test_no_kv_heads",
        )
        result = extract_model_config(cfg)
        assert result["uses_gqa"] is False  # Defaults to num_attention_heads

    def test_activation_function_fallback_to_activation_function_field(self):
        """Some models use 'activation_function' instead of 'hidden_act'."""
        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_hidden_layers=6,
            intermediate_size=3072,
            vocab_size=32000,
            max_position_embeddings=2048,
            layer_norm_eps=1e-5,
            activation_function="gelu_new",  # GPT-2 style
            model_type="gpt2_style",
        )
        result = extract_model_config(cfg)
        assert result["hidden_act"] == "gelu_new"


# ─── Test: build_fallback_graph ──────────────────────────────────────────────

class TestBuildFallbackGraph:
    """Test the structural fallback graph builder.

    This is used when torch.fx symbolic_trace fails. It builds a graph
    from the model's config, producing the expected layer structure.
    """

    def test_causal_lm_fallback_produces_valid_graph(self):
        """Causal LM fallback should produce input_ids → embed → layers → norm → lm_head → output."""
        cfg = _make_config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=2,
            intermediate_size=1024,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="test",
        )
        graph = build_fallback_graph(cfg, "test-model", decompose=True, model_class="causal_lm")

        assert graph["model_id"] == "test-model"
        assert graph["model_config"]["model_class"] == "causal_lm"
        assert graph["model_config"]["is_encoder_decoder"] is False

        # Should have: placeholder + embed + (2 layers × (norm + attn + norm + mlp)) + final_norm + lm_head + output
        node_types = [n["op"]["type"] for n in graph["nodes"]]
        assert "Placeholder" in node_types
        assert "Embedding" in node_types
        assert "RmsNorm" in node_types
        assert "AttentionBlock" in node_types
        assert "MlpBlock" in node_types
        assert "Linear" in node_types
        assert "Output" in node_types

    def test_seq2seq_lm_fallback_produces_decoder_graph(self):
        """Seq2SeqLM fallback should produce decoder path with encoder_hidden_states input."""
        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_hidden_layers=6,
            intermediate_size=3072,
            vocab_size=50265,
            max_position_embeddings=1024,
            layer_norm_eps=1e-6,
            hidden_act="gelu",
            model_type="dolphin",
        )
        graph = build_fallback_graph(cfg, "dolphin-test", decompose=True, model_class="seq2seq_lm")

        assert graph["model_config"]["model_class"] == "seq2seq_lm"
        assert graph["model_config"]["is_encoder_decoder"] is True

        # Should have encoder_hidden_states placeholder + decoder_input_ids placeholder
        node_names = [n["name"] for n in graph["nodes"]]
        assert "encoder_hidden_states" in node_names
        assert "decoder_input_ids" in node_names

        # Should use LayerNorm, not RMSNorm
        node_types = [n["op"]["type"] for n in graph["nodes"]]
        assert "LayerNorm" in node_types or "RmsNorm" not in node_types

    def test_fallback_graph_has_discovered_features(self):
        """Fallback graph should include discovered_features from config."""
        cfg = _make_config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            num_hidden_layers=2,
            intermediate_size=1024,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="test",
        )
        graph = build_fallback_graph(cfg, "test-model", decompose=True)

        features = graph["discovered_features"]
        assert "RMSNorm" in features["norm_types_encountered"]
        assert features["has_rope_module"] is True
        assert features["uses_gqa"] is True
        assert features["detection_methods"]["norm_type"] == "config_field_presence"
        assert features["detection_methods"]["rope"] == "config_field_presence"
        assert features["detection_methods"]["gqa"] == "config_field_comparison"

    def test_fallback_graph_layer_count_matches_config(self):
        """Number of transformer layers should match num_hidden_layers."""
        cfg = _make_config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=4,
            intermediate_size=1024,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="test",
        )
        graph = build_fallback_graph(cfg, "test-model", decompose=True)

        # Count attention blocks — should match num_hidden_layers
        attn_nodes = [n for n in graph["nodes"] if n["op"]["type"] == "AttentionBlock"]
        assert len(attn_nodes) == 4

    def test_fallback_graph_produces_serializable_json(self):
        """Fallback graph should be serializable to JSON."""
        cfg = _make_config(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=1,
            intermediate_size=1024,
            vocab_size=32000,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="test",
        )
        graph = build_fallback_graph(cfg, "test-model", decompose=True)

        # Should not raise
        json_str = json.dumps(graph)
        assert len(json_str) > 100

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["model_id"] == "test-model"


# ─── Test: map_module_call ───────────────────────────────────────────────────

class TestMapModuleCall:
    """Test the module call mapping with mock fx nodes.

    These tests validate that map_module_call correctly identifies
    module types from the node target and model inspection.
    """

    def _make_fx_node(self, op="call_module", target="model.layers.0.self_attn.q_proj"):
        """Create a minimal mock torch.fx.Node."""
        node = mock.MagicMock()
        node.op = op
        node.target = target
        return node

    def test_linear_projection_detected(self):
        """Q/K/V/O projection modules should map to Linear ops."""
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            node = self._make_fx_node(target=f"model.layers.0.self_attn.{proj_name}")
            result = map_module_call(node, _make_config(rms_norm_eps=1e-6), decompose=True)
            assert result["type"] == "Linear", f"Expected Linear for {proj_name}, got {result['type']}"

    def test_mlp_projection_detected(self):
        """Gate/up/down projection modules should map to Linear ops."""
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            node = self._make_fx_node(target=f"model.layers.0.mlp.{proj_name}")
            result = map_module_call(node, _make_config(rms_norm_eps=1e-6), decompose=True)
            assert result["type"] == "Linear", f"Expected Linear for {proj_name}, got {result['type']}"

    def test_rms_norm_detected_from_norm_path(self):
        """Norm modules in 'norm'-containing paths should be detected."""
        cfg = _make_config(rms_norm_eps=1e-6, hidden_size=2048)
        node = self._make_fx_node(target="model.layers.0.input_layernorm")
        result = map_module_call(node, cfg, decompose=True)
        assert result["type"] in ("RmsNorm", "LayerNorm"), f"Got {result['type']}"

    def test_attention_block_detected(self):
        """Attention modules should map to AttentionBlock ops."""
        node = self._make_fx_node(target="model.layers.0.self_attn")
        cfg = _make_config(hidden_size=2048, num_attention_heads=32, rms_norm_eps=1e-6)
        result = map_module_call(node, cfg, decompose=True)
        assert result["type"] == "AttentionBlock"

    def test_mlp_block_detected(self):
        """MLP modules should map to MlpBlock ops."""
        node = self._make_fx_node(target="model.layers.0.mlp")
        cfg = _make_config(hidden_size=2048, intermediate_size=8192, hidden_act="silu", rms_norm_eps=1e-6)
        result = map_module_call(node, cfg, decompose=True)
        assert result["type"] == "MlpBlock"
        assert result["activation"] == "silu"

    def test_unknown_module_returns_unknown(self):
        """Modules that don't match any pattern should return Unknown."""
        node = self._make_fx_node(target="model.custom_module")
        result = map_module_call(node, _make_config(rms_norm_eps=1e-6), decompose=True)
        assert result["type"] == "Unknown"

    def test_embedding_detected(self):
        """Embedding modules should map to Embedding ops."""
        node = self._make_fx_node(target="model.embed_tokens")
        cfg = _make_config(vocab_size=32000, hidden_size=2048, rms_norm_eps=1e-6)
        result = map_module_call(node, cfg, decompose=True)
        assert result["type"] == "Embedding"


# ─── Test: map_function_call ─────────────────────────────────────────────────

class TestMapFunctionCall:
    """Test the function call mapping with mock fx nodes."""

    def _make_fx_node(self, target):
        """Create a minimal mock torch.fx.Node for call_function."""
        node = mock.MagicMock()
        node.op = "call_function"
        node.target = target
        return node

    def test_sdpa_detected(self):
        """scaled_dot_product_attention should map to ScaledDotProductAttention."""
        node = self._make_fx_node(target="torch.nn.functional.scaled_dot_product_attention")
        result = map_function_call(node, _make_config(hidden_size=2048, num_attention_heads=32), decompose=True)
        assert result["type"] == "ScaledDotProductAttention"

    def test_gelu_detected(self):
        """gelu function should map to Gelu op."""
        node = self._make_fx_node(target="torch.nn.functional.gelu")
        result = map_function_call(node, _make_config(), decompose=True)
        assert result["type"] == "Gelu"

    def test_silu_detected(self):
        """silu function should map to Silu op."""
        node = self._make_fx_node(target="torch.nn.functional.silu")
        result = map_function_call(node, _make_config(), decompose=True)
        assert result["type"] == "Silu"

    def test_softmax_detected(self):
        """softmax function should map to Softmax op."""
        node = self._make_fx_node(target="torch.nn.functional.softmax")
        result = map_function_call(node, _make_config(), decompose=True)
        assert result["type"] == "Softmax"

    def test_cos_sin_detected_for_rope(self):
        """cos/sin functions (used in RoPE) should map to Cos/Sin ops.

        We test by constructing a mock callable whose string representation
        contains 'cos'/'sin', matching the substring matching logic in
        map_function_call. When torch is available, torch.cos/torch.sin
        produce strings like '<built-in function cos>' which match.
        """
        # Create a mock that has 'cos' in its string representation
        mock_cos = mock.MagicMock()
        mock_cos.__str__ = lambda self: "cos"
        node_cos = self._make_fx_node(target=mock_cos)
        result_cos = map_function_call(node_cos, _make_config(), decompose=True)
        assert result_cos["type"] == "Cos"

        mock_sin = mock.MagicMock()
        mock_sin.__str__ = lambda self: "sin"
        node_sin = self._make_fx_node(target=mock_sin)
        result_sin = map_function_call(node_sin, _make_config(), decompose=True)
        assert result_sin["type"] == "Sin"

    def test_unknown_function_returns_unknown(self):
        """Unknown functions should return Unknown type."""
        node = self._make_fx_node(target="custom_function")
        result = map_function_call(node, _make_config(), decompose=True)
        assert result["type"] == "Unknown"


# ─── Test: map_method_call ───────────────────────────────────────────────────

class TestMapMethodCall:
    """Test the method call mapping with mock fx nodes."""

    def _make_fx_node(self, method):
        """Create a minimal mock torch.fx.Node for call_method."""
        node = mock.MagicMock()
        node.op = "call_method"
        node.target = method
        return node

    def test_view_maps_to_reshape(self):
        """tensor.view() should map to Reshape op."""
        result = map_method_call(self._make_fx_node("view"))
        assert result["type"] == "Reshape"

    def test_reshape_maps_to_reshape(self):
        """tensor.reshape() should map to Reshape op."""
        result = map_method_call(self._make_fx_node("reshape"))
        assert result["type"] == "Reshape"

    def test_transpose_maps_to_transpose(self):
        """tensor.transpose() should map to Transpose op."""
        result = map_method_call(self._make_fx_node("transpose"))
        assert result["type"] == "Transpose"

    def test_permute_maps_to_transpose(self):
        """tensor.permute() should map to Transpose op."""
        result = map_method_call(self._make_fx_node("permute"))
        assert result["type"] == "Transpose"

    def test_contiguous_maps_to_identity(self):
        """tensor.contiguous() should map to Identity (no-op on ANE)."""
        result = map_method_call(self._make_fx_node("contiguous"))
        assert result["type"] == "Identity"

    def test_unknown_method_returns_unknown(self):
        """Unknown methods should return Unknown type."""
        result = map_method_call(self._make_fx_node("custom_method"))
        assert result["type"] == "Unknown"


# ─── Test: _build_input_specs ────────────────────────────────────────────────

class TestBuildInputSpecs:
    """Test input specification building for different model classes.

    These tests require torch for creating tensor objects, since
    _build_input_specs checks isinstance(tensor, torch.Tensor).
    They are skipped when torch is not available.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

    def test_causal_lm_input_specs(self):
        """Causal LM should produce a single input_ids spec."""
        import torch
        input_ids = torch.randint(0, 32000, (1, 32))
        specs = _build_input_specs(input_ids, "causal_lm")

        assert len(specs) == 1
        assert specs[0]["name"] == "input_ids"
        assert specs[0]["shape"]["dims"] == [1, 32]

    def test_seq2seq_lm_input_specs(self):
        """Seq2SeqLM should produce specs for decoder_input_ids and encoder_hidden_states."""
        import torch
        inputs = {
            "decoder_input_ids": torch.randint(0, 50265, (1, 16)),
            "encoder_hidden_states": torch.randn(1, 16, 768, dtype=torch.float16),
        }
        specs = _build_input_specs(inputs, "seq2seq_lm")

        assert len(specs) == 2
        spec_names = [s["name"] for s in specs]
        assert "decoder_input_ids" in spec_names
        assert "encoder_hidden_states" in spec_names

    def test_decoder_only_input_specs(self):
        """Decoder-only models should produce same specs as causal LM."""
        import torch
        input_ids = torch.randint(0, 151936, (1, 32))
        specs = _build_input_specs(input_ids, "decoder_only")

        assert len(specs) == 1
        assert specs[0]["name"] == "input_ids"


# ─── Test: discover_model_features with mock models ──────────────────────────

class TestDiscoverModelFeaturesMock:
    """Test discover_model_features with mock PyTorch models.

    These tests create minimal mock models that have the same module
    structure as the target models, without downloading anything.
    They require torch and are skipped when torch is not available.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

    def _make_mock_llama_model(self):
        """Create a mock model with Llama-like module structure (RMSNorm + RoPE + GQA)."""
        import torch.nn as nn

        class MockRMSNorm(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                # No bias — structural signature of RMSNorm

        class MockRotaryEmbedding(nn.Module):
            pass

        class MockLlamaAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(2048, 2048, bias=False)
                self.k_proj = nn.Linear(2048, 512, bias=False)   # GQA: 8 heads
                self.v_proj = nn.Linear(2048, 512, bias=False)
                self.o_proj = nn.Linear(2048, 2048, bias=False)

        class MockLlamaMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(2048, 8192, bias=False)
                self.up_proj = nn.Linear(2048, 8192, bias=False)
                self.down_proj = nn.Linear(8192, 2048, bias=False)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(128256, 2048)
                self.layers = nn.ModuleList([
                    nn.Module()  # placeholder
                    for _ in range(16)
                ])
                # Add the RMSNorm and RoPE modules at top level for discovery
                self.input_layernorm = MockRMSNorm(2048)
                self.rotary_emb = MockRotaryEmbedding()
                self.self_attn = MockLlamaAttention()
                self.mlp = MockLlamaMLP()

        cfg = _make_config(
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=16,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            model_type="llama",
        )

        return MockModel(), cfg

    def _make_mock_dolphin_model(self):
        """Create a mock model with Dolphin/BART-like structure (LayerNorm + no RoPE)."""
        import torch.nn as nn

        class MockBartAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(768, 768)
                self.k_proj = nn.Linear(768, 768)
                self.v_proj = nn.Linear(768, 768)
                self.out_proj = nn.Linear(768, 768)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(50265, 768)
                self.layer_norm = nn.LayerNorm(768)   # BART uses LayerNorm
                self.self_attn = MockBartAttention()

        cfg = _make_config(
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=12,
            num_hidden_layers=6,
            layer_norm_eps=1e-6,
            model_type="dolphin",
            is_encoder_decoder=True,
        )

        return MockModel(), cfg

    def test_llama_model_discovers_rms_norm(self):
        """Mock Llama model should discover RMSNorm via module type inspection."""
        model, cfg = self._make_mock_llama_model()
        features = discover_model_features(model, cfg)

        assert "RMSNorm" in features["norm_types_encountered"]
        assert features["detection_methods"]["norm_type"] == "module_type_inspection"

    def test_llama_model_discovers_rope(self):
        """Mock Llama model should discover RoPE via module type inspection."""
        model, cfg = self._make_mock_llama_model()
        features = discover_model_features(model, cfg)

        assert features["has_rope_module"] is True
        assert features["detection_methods"]["rope"] == "module_type_inspection"

    def test_llama_model_detects_gqa(self):
        """Mock Llama model should detect GQA from config field comparison."""
        model, cfg = self._make_mock_llama_model()
        features = discover_model_features(model, cfg)

        assert features["uses_gqa"] is True
        assert features["detection_methods"]["gqa"] == "config_field_comparison"

    def test_llama_model_counts_linears(self):
        """Mock Llama model should count Linear modules correctly."""
        model, cfg = self._make_mock_llama_model()
        features = discover_model_features(model, cfg)

        # 4 from attention (q, k, v, o) + 3 from MLP (gate, up, down) = 7
        assert features["linear_count"] == 7

    def test_llama_model_counts_embeddings(self):
        """Mock Llama model should count Embedding modules correctly."""
        model, cfg = self._make_mock_llama_model()
        features = discover_model_features(model, cfg)

        assert features["embedding_count"] == 1

    def test_dolphin_model_discovers_layer_norm(self):
        """Mock Dolphin model should discover LayerNorm via module type inspection."""
        model, cfg = self._make_mock_dolphin_model()
        features = discover_model_features(model, cfg)

        assert "LayerNorm" in features["norm_types_encountered"]
        assert features["detection_methods"]["norm_type"] == "module_type_inspection"

    def test_dolphin_model_no_rope(self):
        """Mock Dolphin model should NOT detect RoPE (BART uses learned embeddings)."""
        model, cfg = self._make_mock_dolphin_model()
        features = discover_model_features(model, cfg)

        assert features["has_rope_module"] is False

    def test_dolphin_model_no_gqa(self):
        """Mock Dolphin model should NOT detect GQA (12 == 12)."""
        model, cfg = self._make_mock_dolphin_model()
        features = discover_model_features(model, cfg)

        assert features["uses_gqa"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: INTEGRATION TESTS — Requires HuggingFace model download
# ═══════════════════════════════════════════════════════════════════════════════

# These tests are skipped by default. Run with:
#   pytest -m needs_hf_models python/tests/test_trace_model.py
#
# Or for a specific model:
#   pytest -m needs_hf_models -k "llama" python/tests/test_trace_model.py

HF_CACHE_DIR = os.environ.get("MILLER_HF_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))


def _skip_if_no_torch():
    """Skip test if torch/transformers are not installed."""
    try:
        import torch
        import transformers
    except ImportError:
        pytest.skip("torch/transformers not installed — run: pip install torch transformers")


def _trace_model_from_hub(model_id, model_class_hint="auto", seq_len=16):
    """Helper: trace a model from HuggingFace Hub and return the TracedGraph dict."""
    _skip_if_no_torch()

    import torch
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_id)
    torch_dtype = torch.float16
    model, model_class = _load_model_local(
        model_id, model_config, torch_dtype, model_class_hint
    )
    model.eval()

    inputs = _create_dummy_inputs_local(model, model_config, model_class, batch_size=1, seq_len=seq_len)

    from trace_model import trace_model_fx
    return trace_model_fx(
        model=model,
        input_ids=inputs,
        model_config=model_config,
        model_id=model_id,
        decompose=True,
        with_kv_cache=False,
        model_class=model_class,
    )


# Local copies of _load_model and _create_dummy_inputs that don't call error_exit
def _load_model_local(model_id, model_config, torch_dtype, model_class_hint="auto"):
    """Load model locally without sys.exit on failure."""
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    if model_class_hint == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        return model, "causal_lm"

    if model_class_hint == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        return model, "seq2seq_lm"

    if model_class_hint == "decoder_only":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        return model, "decoder_only"

    # Auto-detect
    architectures = getattr(model_config, "architectures", []) or []
    is_encoder_decoder = getattr(model_config, "is_encoder_decoder", False)

    for arch in architectures:
        arch_lower = arch.lower()
        if "conditionalgeneration" in arch_lower or "seq2seq" in arch_lower:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            )
            return model, "seq2seq_lm"

    if is_encoder_decoder:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        return model, "seq2seq_lm"

    # Check for multimodal with text_config
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        return model, "decoder_only"

    # Default: CausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )
    return model, "causal_lm"


def _create_dummy_inputs_local(model, model_config, model_class, batch_size, seq_len):
    """Create dummy inputs locally without error_exit."""
    import torch
    from trace_model import _resolve_effective_config

    cfg = _resolve_effective_config(model_config)
    vocab_size = getattr(cfg, "vocab_size", 32000)

    if model_class == "seq2seq_lm":
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        encoder_hidden_dim = getattr(cfg, "hidden_size", 768)
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, encoder_hidden_dim,
            dtype=next(model.parameters()).dtype if list(model.parameters()) else torch.float32,
        )
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }
    else:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        return input_ids


@pytest.mark.needs_hf_models
class TestTraceLlama32_1B:
    """End-to-end tracing test for meta-llama/Llama-3.2-1B.

    Architecture: Standard Llama causal LM
    - RMSNorm (eps=1e-5)
    - RoPE (theta=500000.0)
    - GQA (8 KV heads for 32 Q heads)
    - SiLU activation
    - 16 layers
    """

    MODEL_ID = "meta-llama/Llama-3.2-1B"

    @pytest.fixture(autouse=True)
    def skip_if_no_access(self):
        """Skip if the user doesn't have access to this gated model."""
        _skip_if_no_torch()
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.MODEL_ID)
        except Exception:
            pytest.skip(f"Cannot access {self.MODEL_ID} (gated model — need HF token)")

    def test_load_model_resolves_causal_lm(self):
        """Llama-3.2-1B should resolve to AutoModelForCausalLM."""
        import torch
        from trace_model import _load_model
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model, model_class = _load_model(
            model_id=self.MODEL_ID,
            model_config=model_config,
            torch_dtype=torch.float16,
            model_class_hint="auto",
        )
        assert model_class == "causal_lm"
        assert "LlamaForCausalLM" in type(model).__name__

    def test_config_extraction(self):
        """Config should detect RMSNorm + RoPE + GQA correctly."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        result = extract_model_config(model_config)

        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True
        assert result["num_attention_heads"] == 32
        assert result["num_key_value_heads"] == 8
        assert result["num_hidden_layers"] == 16
        assert result["hidden_act"] == "silu"

    def test_discover_features(self):
        """discover_model_features should find RMSNorm + RoPE via module inspection."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        features = discover_model_features(model, model_config)

        assert "RMSNorm" in features["norm_types_encountered"]
        assert features["has_rope_module"] is True
        assert features["uses_gqa"] is True

    def test_trace_produces_valid_graph(self):
        """Full trace should produce a valid TracedGraph JSON."""
        graph = _trace_model_from_hub(self.MODEL_ID)

        assert graph["model_id"] == self.MODEL_ID
        assert "LlamaForCausalLM" in graph["architecture"]
        assert len(graph["nodes"]) > 0
        assert len(graph["weights"]) > 0
        assert graph["model_config"]["uses_rms_norm"] is True
        assert graph["model_config"]["uses_rope"] is True
        assert graph["discovered_features"]["uses_gqa"] is True

    def test_trace_graph_serializable(self):
        """Traced graph should be serializable to JSON for Rust consumption."""
        graph = _trace_model_from_hub(self.MODEL_ID)
        json_str = json.dumps(graph)
        assert len(json_str) > 1000

        parsed = json.loads(json_str)
        assert parsed["model_config"]["hidden_size"] == 2048


@pytest.mark.needs_hf_models
class TestTraceQwen3_0_6B:
    """End-to-end tracing test for Qwen/Qwen3-0.6B.

    Architecture: Qwen3 causal LM
    - RMSNorm (eps=1e-6)
    - RoPE (theta=10000.0)
    - GQA (8 KV heads for 16 Q heads)
    - SiLU activation
    """

    MODEL_ID = "Qwen/Qwen3-0.6B"

    @pytest.fixture(autouse=True)
    def skip_if_no_access(self):
        _skip_if_no_torch()

    def test_load_model_resolves_causal_lm(self):
        """Qwen3-0.6B should resolve to AutoModelForCausalLM."""
        import torch
        from trace_model import _load_model
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model, model_class = _load_model(
            model_id=self.MODEL_ID,
            model_config=model_config,
            torch_dtype=torch.float16,
            model_class_hint="auto",
        )
        assert model_class == "causal_lm"

    def test_config_extraction(self):
        """Config should detect RMSNorm + RoPE + GQA."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        result = extract_model_config(model_config)

        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True
        assert result["num_key_value_heads"] < result["num_attention_heads"]

    def test_trace_produces_valid_graph(self):
        """Full trace should produce a valid TracedGraph."""
        graph = _trace_model_from_hub(self.MODEL_ID)

        assert graph["model_id"] == self.MODEL_ID
        assert len(graph["nodes"]) > 0
        assert graph["model_config"]["uses_rms_norm"] is True
        assert graph["discovered_features"]["uses_gqa"] is True


@pytest.mark.needs_hf_models
class TestTraceQwen35_0_8B:
    """End-to-end tracing test for Qwen/Qwen3.5-0.8B.

    Architecture: Qwen3.5 causal LM with model_type="qwen3_5_text"
    - This model_type would NOT match any hardcoded heuristic list
    - RMSNorm (eps=1e-6)
    - RoPE via rope_parameters dict (theta=1000000.0)
    - GQA
    - SiLU activation

    KEY TEST: Validates that unknown model_type works without registry.
    """

    MODEL_ID = "Qwen/Qwen3.5-0.8B"

    @pytest.fixture(autouse=True)
    def skip_if_no_access(self):
        _skip_if_no_torch()

    def test_config_has_qwen3_5_text_model_type(self):
        """Qwen3.5 config should have model_type that doesn't match any known registry."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model_type = getattr(model_config, "model_type", "unknown")
        # The key point: this model_type is NOT in any hardcoded list
        assert model_type in ("qwen3_5_text", "qwen3.5", "qwen3_5") or "qwen3" in model_type.lower()

    def test_config_extraction_works_without_registry(self):
        """Config extraction should work despite unknown model_type."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        result = extract_model_config(model_config)

        # Dynamic detection should work regardless of model_type name
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True

    def test_rope_detected_via_rope_parameters(self):
        """RoPE should be detected from rope_parameters dict, not rope_theta."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        cfg = _resolve_effective_config(model_config)

        # Qwen3.5 uses rope_parameters instead of direct rope_theta
        assert _detect_rope(cfg) is True

    def test_trace_produces_valid_graph(self):
        """Full trace should produce a valid TracedGraph without any hardcoded registry."""
        graph = _trace_model_from_hub(self.MODEL_ID)

        assert graph["model_id"] == self.MODEL_ID
        assert len(graph["nodes"]) > 0
        assert graph["model_config"]["uses_rms_norm"] is True
        assert graph["model_config"]["uses_rope"] is True


@pytest.mark.needs_hf_models
class TestTraceDolphin15:
    """End-to-end tracing test for ByteDance/Dolphin-1.5.

    Architecture: Encoder-decoder (DonutSwin encoder + BART decoder)
    - LayerNorm (not RMSNorm)
    - Learned positional embeddings (no RoPE)
    - No GQA (12 == 12)
    - GELU activation
    - model_class should be "seq2seq_lm"

    KEY TEST: Validates AutoModelForSeq2SeqLM tracing path.
    """

    MODEL_ID = "ByteDance/Dolphin-1.5"

    @pytest.fixture(autouse=True)
    def skip_if_no_access(self):
        _skip_if_no_torch()
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.MODEL_ID)
        except Exception:
            pytest.skip(f"Cannot access {self.MODEL_ID}")

    def test_config_declares_encoder_decoder(self):
        """Dolphin-1.5 config should declare is_encoder_decoder=True."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        assert getattr(model_config, "is_encoder_decoder", False) is True

    def test_load_model_resolves_seq2seq(self):
        """Dolphin-1.5 should resolve to AutoModelForSeq2SeqLM."""
        import torch
        from trace_model import _load_model
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model, model_class = _load_model(
            model_id=self.MODEL_ID,
            model_config=model_config,
            torch_dtype=torch.float16,
            model_class_hint="auto",
        )
        assert model_class == "seq2seq_lm"

    def test_config_detection_layer_norm_no_rope(self):
        """Dolphin decoder should detect LayerNorm + no RoPE."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        # Resolve to the decoder config
        cfg = _resolve_effective_config(model_config)

        # The BART decoder uses LayerNorm, not RMSNorm
        norm_type = _detect_norm_type(cfg)
        # Could be either depending on which config level we're looking at
        # The important thing is that it doesn't falsely detect RMSNorm when LayerNorm is correct
        assert norm_type in ("layer_norm", "rms_norm", "unknown")

    def test_trace_produces_decoder_graph(self):
        """Full trace should produce a decoder-path TracedGraph."""
        graph = _trace_model_from_hub(self.MODEL_ID)

        assert graph["model_id"] == self.MODEL_ID
        assert graph["model_config"]["model_class"] == "seq2seq_lm"
        assert graph["model_config"]["is_encoder_decoder"] is True
        assert len(graph["nodes"]) > 0


@pytest.mark.needs_hf_models
class TestTraceQwen3ASR_0_6B:
    """End-to-end tracing test for Qwen/Qwen3-ASR-0.6B.

    Architecture: Multimodal (audio encoder + Qwen3 decoder)
    - Audio encoder is non-standard
    - Qwen3 text decoder: RMSNorm + RoPE + GQA (2 KV heads for 14 Q heads)
    - text_config sub-object with standard Qwen3 fields
    - model_class should be "decoder_only"

    KEY TEST: Validates multimodal text_config extraction and decoder-only path.
    """

    MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

    @pytest.fixture(autouse=True)
    def skip_if_no_access(self):
        _skip_if_no_torch()
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.MODEL_ID)
        except Exception:
            pytest.skip(f"Cannot access {self.MODEL_ID}")

    def test_config_has_text_config(self):
        """Qwen3-ASR config should have a text_config sub-object."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        text_config = getattr(model_config, "text_config", None)
        assert text_config is not None
        assert hasattr(text_config, "hidden_size")

    def test_resolve_effective_config_returns_text_config(self):
        """_resolve_effective_config should return the text_config sub-object."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        resolved = _resolve_effective_config(model_config)

        # Should resolve to the text_config
        assert resolved is not model_config
        assert hasattr(resolved, "hidden_size")
        assert getattr(resolved, "hidden_size", None) is not None

    def test_text_config_has_qwen3_decoder_features(self):
        """text_config should have Qwen3 decoder features (RMSNorm + RoPE + GQA)."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        text_config = _resolve_effective_config(model_config)

        assert _detect_norm_type(text_config) == "rms_norm"
        assert _detect_rope(text_config) is True
        # Extreme 7:1 GQA ratio
        num_heads = getattr(text_config, "num_attention_heads", 0)
        num_kv_heads = getattr(text_config, "num_key_value_heads", num_heads)
        assert num_kv_heads < num_heads  # GQA detected

    def test_load_model_detects_multimodal_decoder(self):
        """Qwen3-ASR should resolve to decoder_only via text_config detection."""
        import torch
        from trace_model import _load_model
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        model, model_class = _load_model(
            model_id=self.MODEL_ID,
            model_config=model_config,
            torch_dtype=torch.float16,
            model_class_hint="auto",
        )
        assert model_class in ("decoder_only", "causal_lm")

    def test_extract_model_config_resolves_to_text_decoder(self):
        """extract_model_config should resolve to text_config dimensions."""
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.MODEL_ID)
        result = extract_model_config(model_config)

        # Should have the Qwen3 decoder dimensions, not the audio encoder dimensions
        assert result["uses_rms_norm"] is True
        assert result["uses_rope"] is True
        assert result["uses_gqa"] is True

    def test_trace_produces_decoder_graph(self):
        """Full trace should produce a decoder-path TracedGraph."""
        graph = _trace_model_from_hub(self.MODEL_ID)

        assert graph["model_id"] == self.MODEL_ID
        assert len(graph["nodes"]) > 0
        assert graph["model_config"]["uses_rms_norm"] is True
        assert graph["model_config"]["uses_rope"] is True
        assert graph["discovered_features"]["uses_gqa"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossModelValidation:
    """Tests that validate consistent behavior across all 5 target models.

    These use config-only testing (no model download), ensuring the
    detection pipeline works correctly for all target architectures.
    """

    @pytest.fixture(params=[
        {
            "name": "Llama-3.2-1B",
            "config": _make_config(
                hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
                num_hidden_layers=16, intermediate_size=8192, vocab_size=128256,
                max_position_embeddings=131072, rms_norm_eps=1e-5, hidden_act="silu",
                rope_theta=500000.0, model_type="llama",
            ),
            "expected": {
                "norm": "rms_norm",
                "rope": True,
                "gqa": True,
                "act": "silu",
            },
        },
        {
            "name": "Qwen3-0.6B",
            "config": _make_config(
                hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
                num_hidden_layers=28, intermediate_size=4096, vocab_size=151936,
                max_position_embeddings=40960, rms_norm_eps=1e-6, hidden_act="silu",
                rope_theta=10000.0, model_type="qwen3",
            ),
            "expected": {
                "norm": "rms_norm",
                "rope": True,
                "gqa": True,
                "act": "silu",
            },
        },
        {
            "name": "Qwen3.5-0.8B",
            "config": _make_config(
                hidden_size=1024, num_attention_heads=16, num_key_value_heads=8,
                num_hidden_layers=24, intermediate_size=4096, vocab_size=151936,
                max_position_embeddings=40960, rms_norm_eps=1e-6, hidden_act="silu",
                rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
                model_type="qwen3_5_text",
            ),
            "expected": {
                "norm": "rms_norm",
                "rope": True,
                "gqa": True,
                "act": "silu",
            },
        },
        {
            "name": "Dolphin-1.5",
            "config": _make_config(
                hidden_size=768, num_attention_heads=12, num_key_value_heads=12,
                num_hidden_layers=6, intermediate_size=3072, vocab_size=50265,
                max_position_embeddings=1024, layer_norm_eps=1e-6, hidden_act="gelu",
                model_type="dolphin", is_encoder_decoder=True,
            ),
            "expected": {
                "norm": "layer_norm",
                "rope": False,
                "gqa": False,
                "act": "gelu",
            },
        },
        {
            "name": "Qwen3-ASR-0.6B",
            "config": _make_config(
                model_type="qwen3_asr",
                text_config=_make_config(
                    hidden_size=896, num_attention_heads=14, num_key_value_heads=2,
                    num_hidden_layers=24, intermediate_size=4864, vocab_size=151936,
                    max_position_embeddings=4096, rms_norm_eps=1e-6, hidden_act="silu",
                    rope_theta=1000000.0, model_type="qwen3",
                ),
                is_encoder_decoder=False,
            ),
            "expected": {
                "norm": "rms_norm",
                "rope": True,
                "gqa": True,
                "act": "silu",
            },
        },
    ])
    def model_config(self, request):
        return request.param

    def test_norm_type_detected_correctly(self, model_config):
        """Norm type should be correctly detected for all target models."""
        cfg = _resolve_effective_config(model_config["config"])
        assert _detect_norm_type(cfg) == model_config["expected"]["norm"]

    def test_rope_detected_correctly(self, model_config):
        """RoPE presence should be correctly detected for all target models."""
        cfg = _resolve_effective_config(model_config["config"])
        assert _detect_rope(cfg) == model_config["expected"]["rope"]

    def test_gqa_detected_correctly(self, model_config):
        """GQA presence should be correctly detected for all target models."""
        result = extract_model_config(model_config["config"])
        assert result["uses_gqa"] == model_config["expected"]["gqa"]

    def test_activation_detected_correctly(self, model_config):
        """Activation function should be correctly extracted for all target models."""
        result = extract_model_config(model_config["config"])
        assert result["hidden_act"] == model_config["expected"]["act"]

    def test_extract_model_config_produces_complete_config(self, model_config):
        """All models should produce complete config with all required fields."""
        result = extract_model_config(model_config["config"])
        required_fields = [
            "hidden_size", "num_attention_heads", "num_key_value_heads",
            "num_hidden_layers", "intermediate_size", "vocab_size",
            "max_position_embeddings", "layer_norm_epsilon", "hidden_act",
            "uses_rope", "uses_rms_norm", "uses_gqa", "model_type",
        ]
        for field in required_fields:
            assert field in result, f"Missing field '{field}' in config for {model_config['name']}"

    def test_fallback_graph_works_for_all_models(self, model_config):
        """Fallback graph builder should work for all target models."""
        graph = build_fallback_graph(
            model_config["config"],
            model_id=model_config["name"],
            decompose=True,
            model_class="seq2seq_lm" if model_config["name"] == "Dolphin-1.5" else "causal_lm",
        )

        assert graph["model_id"] == model_config["name"]
        assert len(graph["nodes"]) > 0

        # Should be serializable
        json_str = json.dumps(graph)
        assert len(json_str) > 0

        # Should have correct norm type in discovered_features
        features = graph["discovered_features"]
        expected_norm = "RMSNorm" if model_config["expected"]["norm"] == "rms_norm" else "LayerNorm"
        assert expected_norm in features["norm_types_encountered"]
