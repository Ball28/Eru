"""
Transformer block with Pre-LayerNorm and residual shortcut connections (Chapter 4, Listing 4.3).
"""

from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

from eru.model.attention import MultiHeadAttention
from eru.model.normalization import LayerNorm
from eru.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer Block.
    x -> LayerNorm -> MultiHeadAttention -> Dropout -> (+) x -> LayerNorm -> FeedForward -> Dropout -> (+) x
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        drop_rate = cfg.get("drop_rate", 0.0)

        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=cfg["context_length"],
            dropout=drop_rate,
            num_heads=cfg["n_heads"],
            qkv_bias=cfg.get("qkv_bias", False),
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Pre-norm multi-head attention shortcut
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Pre-norm feedforward shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
