"""
Layer normalization implementation (Chapter 4).
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})


class LayerNorm(nn.Module):
    """
    Layer Normalization over embedding dimension with learnable scale (gamma) and shift (beta).
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
