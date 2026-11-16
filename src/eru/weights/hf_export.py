"""
Export Eru GPTModel checkpoints to standard Hugging Face Transformers format
(config.json, pytorch_model.bin, generation_config.json).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import torch
except ImportError:
    torch = None


def eru_to_hf_state_dict(model: Any) -> Dict[str, "torch.Tensor"]:
    """
    Map Eru internal parameter names and shapes to Hugging Face GPT-2 parameter names and Conv1D shapes.
    """
    state_dict = model.state_dict()
    hf_dict = {}

    # Embeddings
    if "tok_emb.weight" in state_dict:
        hf_dict["transformer.wte.weight"] = state_dict["tok_emb.weight"]
    if "pos_emb.weight" in state_dict and state_dict["pos_emb.weight"] is not None:
        hf_dict["transformer.wpe.weight"] = state_dict["pos_emb.weight"]

    # Transformer Blocks
    num_layers = len(model.trf_blocks)
    for b in range(num_layers):
        prefix_eru = f"trf_blocks.{b}."
        prefix_hf = f"transformer.h.{b}."

        # Layer Norm 1
        hf_dict[f"{prefix_hf}ln_1.weight"] = state_dict[f"{prefix_eru}norm1.scale"]
        hf_dict[f"{prefix_hf}ln_1.bias"] = state_dict[f"{prefix_eru}norm1.shift"]

        # Attention Q, K, V -> merged c_attn (Conv1D weights are transposed)
        q_w = state_dict[f"{prefix_eru}att.W_query.weight"]
        k_w = state_dict[f"{prefix_eru}att.W_key.weight"]
        v_w = state_dict[f"{prefix_eru}att.W_value.weight"]
        c_attn_w = torch.cat([q_w, k_w, v_w], dim=0).t()
        hf_dict[f"{prefix_hf}attn.c_attn.weight"] = c_attn_w

        if f"{prefix_eru}att.W_query.bias" in state_dict and state_dict[f"{prefix_eru}att.W_query.bias"] is not None:
            q_b = state_dict[f"{prefix_eru}att.W_query.bias"]
            k_b = state_dict[f"{prefix_eru}att.W_key.bias"]
            v_b = state_dict[f"{prefix_eru}att.W_value.bias"]
            hf_dict[f"{prefix_hf}attn.c_attn.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        # Attention output projection
        out_w = state_dict[f"{prefix_eru}att.out_proj.weight"].t()
        hf_dict[f"{prefix_hf}attn.c_proj.weight"] = out_w
        if f"{prefix_eru}att.out_proj.bias" in state_dict and state_dict[f"{prefix_eru}att.out_proj.bias"] is not None:
            hf_dict[f"{prefix_hf}attn.c_proj.bias"] = state_dict[f"{prefix_eru}att.out_proj.bias"]

        # Layer Norm 2
        hf_dict[f"{prefix_hf}ln_2.weight"] = state_dict[f"{prefix_eru}norm2.scale"]
        hf_dict[f"{prefix_hf}ln_2.bias"] = state_dict[f"{prefix_eru}norm2.shift"]

        # FeedForward (MLP)
        if hasattr(model.trf_blocks[b].ff, "layers"):
            mlp_fc_w = model.trf_blocks[b].ff.layers[0].weight.t()
            mlp_fc_b = model.trf_blocks[b].ff.layers[0].bias
            mlp_proj_w = model.trf_blocks[b].ff.layers[2].weight.t()
            mlp_proj_b = model.trf_blocks[b].ff.layers[2].bias

            hf_dict[f"{prefix_hf}mlp.c_fc.weight"] = mlp_fc_w
            hf_dict[f"{prefix_hf}mlp.c_fc.bias"] = mlp_fc_b
            hf_dict[f"{prefix_hf}mlp.c_proj.weight"] = mlp_proj_w
            hf_dict[f"{prefix_hf}mlp.c_proj.bias"] = mlp_proj_b

    # Final LayerNorm
    hf_dict["transformer.ln_f.weight"] = state_dict["final_norm.scale"]
    hf_dict["transformer.ln_f.bias"] = state_dict["final_norm.shift"]

    # Language modeling head
    hf_dict["lm_head.weight"] = state_dict["out_head.weight"]

    return hf_dict


def export_to_huggingface(
    model: Any,
    output_dir: Union[str, Path],
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export an Eru GPTModel to standard Hugging Face directory structure.

    Args:
        model: GPTModel instance.
        output_dir: Target directory path to save model files.
        cfg: Optional model configuration dict (defaults to model.cfg).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = cfg or getattr(model, "cfg", {})

    # Generate Hugging Face config.json
    hf_config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": cfg.get("drop_rate", 0.1),
        "embd_pdrop": cfg.get("drop_rate", 0.1),
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": cfg.get("context_length", 1024),
        "n_embd": cfg.get("emb_dim", 768),
        "n_head": cfg.get("n_heads", 12),
        "n_layer": cfg.get("n_layers", 12),
        "n_positions": cfg.get("context_length", 1024),
        "resid_pdrop": cfg.get("drop_rate", 0.1),
        "vocab_size": cfg.get("vocab_size", 50257),
    }

    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # Generate generation_config.json
    gen_config = {
        "_from_model_config": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }
    with open(out_path / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)

    # Save state dict
    hf_state_dict = eru_to_hf_state_dict(model)
    torch.save(hf_state_dict, out_path / "pytorch_model.bin")

    print(f"Successfully exported Hugging Face model -> {out_path}")
