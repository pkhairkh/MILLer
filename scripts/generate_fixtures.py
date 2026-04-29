#!/usr/bin/env python3
"""Generate pre-traced JSON fixture files for the MILLer project.

Creates one TracedGraph JSON per target model by calling
build_fallback_graph() with SimpleNamespace config objects,
then writes the results to crates/trace/test_fixtures/.
"""

import json
import os
import sys
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Make trace_model.py importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from trace_model import build_fallback_graph  # noqa: E402

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "crates", "trace", "test_fixtures"
)
os.makedirs(FIXTURE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    {
        "filename": "llama_3_2_1b.json",
        "model_id": "meta-llama/Llama-3.2-1B",
        "model_class": "causal_lm",
        "config": SimpleNamespace(
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
        ),
    },
    {
        "filename": "qwen3_0_6b.json",
        "model_id": "Qwen/Qwen3-0.6B",
        "model_class": "causal_lm",
        "config": SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=28,
            intermediate_size=3072,
            vocab_size=151936,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_theta=10000.0,
            model_type="qwen3",
            head_dim=128,
            has_qk_norm=True,
        ),
    },
    {
        "filename": "qwen3_5_0_8b.json",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "model_class": "causal_lm",
        "config": SimpleNamespace(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=24,
            intermediate_size=3072,
            vocab_size=151936,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
            model_type="qwen3_5_text",
            head_dim=128,
            has_qk_norm=True,
        ),
    },
    {
        "filename": "dolphin_1_5.json",
        "model_id": "ByteDance/Dolphin-1.5",
        "model_class": "seq2seq_lm",
        "config": SimpleNamespace(
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
        ),
    },
    {
        "filename": "qwen3_asr_0_6b.json",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "model_class": "decoder_only",
        "config": SimpleNamespace(
            model_type="qwen3_asr",
            is_encoder_decoder=False,
            text_config=SimpleNamespace(
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
                head_dim=128,
                has_qk_norm=True,
            ),
        ),
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    created = []
    for spec in MODELS:
        filename = spec["filename"]
        model_id = spec["model_id"]
        model_class = spec["model_class"]
        config = spec["config"]

        print(f"Generating {filename} ({model_id}) ...")

        graph = build_fallback_graph(
            config, model_id, decompose=True, model_class=model_class
        )

        out_path = os.path.join(FIXTURE_DIR, filename)
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
            f.write("\n")

        size = os.path.getsize(out_path)
        created.append((filename, model_id, size))
        print(f"  -> {out_path}  ({size} bytes)")

    # ------------------------------------------------------------------
    # Verification pass
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    all_ok = True
    for filename, model_id, size in created:
        path = os.path.join(FIXTURE_DIR, filename)
        try:
            with open(path) as f:
                data = json.load(f)
            got_id = data.get("model_id", "<missing>")
            ok = got_id == model_id
            status = "OK" if ok else "MISMATCH"
            print(f"  {filename}: model_id={got_id!r}  size={size}  [{status}]")
            if not ok:
                all_ok = False
        except Exception as exc:
            print(f"  {filename}: FAILED to read/parse: {exc}")
            all_ok = False

    if all_ok:
        print("\nAll 5 fixtures generated and verified successfully.")
    else:
        print("\nSome fixtures failed verification!", file=sys.stderr)
        sys.exit(1)

    return created


if __name__ == "__main__":
    main()
