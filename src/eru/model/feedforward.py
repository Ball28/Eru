"""
FeedForward network for transformer block (Chapter 4).
"""

from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})

from eru.model.activations import GELU


class FeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with 4x expansion and GELU activation.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.layers(x)
