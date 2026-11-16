"""
Canonical GPTModel implementation and standard configuration definitions,
supporting both classic GPT-2 (learned position embeddings, LayerNorm, GELU)
and modern LLM architectures (RoPE, RMSNorm, SwiGLU, GQA).
"""

from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

from eru.model.normalization import LayerNorm, RMSNorm
from eru.model.transformer_block import TransformerBlock
from eru.model.kv_cache import KVCache
from eru.model.rope import RotaryEmbedding

# Canonical GPT-2 124M configuration
GPT_CONFIG_124M: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "pos_emb_type": "learned",
    "norm_type": "layernorm",
    "ffn_type": "gelu",
}

# Small lightweight configuration for fast CPU training & tests
GPT_CONFIG_SMALL: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "pos_emb_type": "learned",
    "norm_type": "layernorm",
    "ffn_type": "gelu",
}

# Modern LLaMA-style small configuration (RoPE, RMSNorm, SwiGLU, GQA)
MODERN_CONFIG_SMALL: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 128,
    "n_heads": 4,
    "n_kv_heads": 2,  # 2x GQA
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "pos_emb_type": "rope",
    "norm_type": "rmsnorm",
    "ffn_type": "swiglu",
    "use_sdpa": True,
}

# GPT-2 standard model presets
GPT_MODEL_CONFIGS: Dict[str, Dict[str, int]] = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


class GPTModel(nn.Module):
    """
    GPT Language Model architecture with modular support for both classic GPT-2
    and modern LLaMA/Mistral/Gemma design choices.

    Args:
        cfg: Model configuration dictionary containing keys:
             - vocab_size: Size of vocabulary (e.g. 50257)
             - context_length: Maximum sequence context length (e.g. 1024)
             - emb_dim: Hidden representation dimension (e.g. 768)
             - n_layers: Number of transformer blocks (e.g. 12)
             - n_heads: Number of attention heads (e.g. 12)
             - n_kv_heads: Number of key-value heads for GQA (optional, defaults to n_heads)
             - drop_rate: Dropout probability (e.g. 0.1)
             - qkv_bias: Whether linear QKV projections include bias terms (default: False)
             - pos_emb_type: 'learned' (default GPT-2) or 'rope' (Rotary Position Embeddings)
             - norm_type: 'layernorm' (default) or 'rmsnorm'
             - ffn_type: 'gelu' (default) or 'swiglu'
             - use_sdpa: If True, uses PyTorch scaled_dot_product_attention
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        emb_dim = cfg["emb_dim"]
        n_heads = cfg["n_heads"]
        head_dim = emb_dim // n_heads

        self.tok_emb = nn.Embedding(cfg["vocab_size"], emb_dim)

        pos_emb_type = cfg.get("pos_emb_type", "learned").lower()
        if pos_emb_type == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_seq_len=cfg["context_length"],
                base=cfg.get("rope_base", 10000.0),
            )
            self.pos_emb = None
        else:
            self.rotary_emb = None
            self.pos_emb = nn.Embedding(cfg["context_length"], emb_dim)

        self.drop_emb = nn.Dropout(cfg.get("drop_rate", 0.0))

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx=i) for i in range(cfg["n_layers"])]
        )

        norm_cls = RMSNorm if cfg.get("norm_type", "layernorm").lower() == "rmsnorm" else LayerNorm
        self.final_norm = norm_cls(emb_dim)
        self.out_head = nn.Linear(emb_dim, cfg["vocab_size"], bias=False)

    def forward(
        self,
        in_idx: "torch.Tensor",
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
    ) -> "torch.Tensor":
        """
        Forward pass producing raw unnormalized next-token logits.

        Args:
            in_idx: Input token ID tensor of shape (batch_size, seq_len).
            kv_cache: Optional KVCache instance for caching keys and values during generation.
            start_pos: Offset for positional embeddings when using KV-cache.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        if self.pos_emb is not None:
            pos_ids = torch.arange(start_pos, start_pos + seq_len, device=in_idx.device)
            x = tok_embeds + self.pos_emb(pos_ids)
        else:
            x = tok_embeds

        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(
                x,
                kv_cache=kv_cache,
                rotary_emb=self.rotary_emb,
                start_pos=start_pos,
            )

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
