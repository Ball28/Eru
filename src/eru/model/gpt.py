"""
Canonical GPTModel implementation and standard configuration definitions.
"""

from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

from eru.model.normalization import LayerNorm
from eru.model.transformer_block import TransformerBlock

# Canonical GPT-2 124M configuration
GPT_CONFIG_124M: Dict[str, Any] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
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
    GPT Language Model architecture with stacked transformer blocks and token/position embeddings.

    Args:
        cfg: Model configuration dictionary containing keys:
             - vocab_size: Size of vocabulary (e.g. 50257)
             - context_length: Maximum sequence context length (e.g. 1024)
             - emb_dim: Hidden representation dimension (e.g. 768)
             - n_layers: Number of transformer blocks (e.g. 12)
             - n_heads: Number of attention heads (e.g. 12)
             - drop_rate: Dropout probability (e.g. 0.1)
             - qkv_bias: Whether linear QKV projections include bias terms (default: False)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg.get("drop_rate", 0.0))

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass producing raw unnormalized next-token logits.

        Args:
            in_idx: Input token ID tensor of shape (batch_size, seq_len).

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
