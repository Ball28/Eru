"""
Transformer block with Pre-Normalization, residual shortcut connections,
and support for LayerNorm / RMSNorm and RoPE / GQA.
"""

from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

from eru.model.attention import MultiHeadAttention
from eru.model.normalization import LayerNorm, RMSNorm
from eru.model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer Block.
    x -> Norm1 -> MultiHeadAttention -> Dropout -> (+) x -> Norm2 -> FeedForward -> Dropout -> (+) x
    """

    def __init__(self, cfg: Dict[str, Any], layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        emb_dim = cfg["emb_dim"]
        drop_rate = cfg.get("drop_rate", 0.0)

        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=cfg["context_length"],
            dropout=drop_rate,
            num_heads=cfg["n_heads"],
            num_kv_heads=cfg.get("n_kv_heads", None),
            qkv_bias=cfg.get("qkv_bias", False),
            use_sdpa=cfg.get("use_sdpa", False),
        )
        self.ff = FeedForward(cfg)

        norm_cls = RMSNorm if cfg.get("norm_type", "layernorm").lower() == "rmsnorm" else LayerNorm
        self.norm1 = norm_cls(emb_dim)
        self.norm2 = norm_cls(emb_dim)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[Any] = None,
        rotary_emb: Optional[Any] = None,
        start_pos: int = 0,
    ) -> "torch.Tensor":
        # Pre-norm multi-head attention shortcut
        shortcut = x
        x = self.norm1(x)
        x = self.att(
            x,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            rotary_emb=rotary_emb,
            start_pos=start_pos,
        )
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Pre-norm feedforward shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
