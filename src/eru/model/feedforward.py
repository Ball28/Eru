"""
Feed-Forward Networks for transformer blocks: standard GELU MLP and modern SwiGLU.
"""

from typing import Any, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    F = None

from eru.model.activations import GELU


class GELUFeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network with 4x expansion and GELU activation.
    Matching standard GPT-2.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        hidden_dim = cfg.get("hidden_dim", 4 * emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.layers(x)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network as used in modern architectures (LLaMA, Mistral, Gemma).
    Computes: Linear2(SiLU(LinearGate(x)) * LinearUp(x))
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        # Standard LLaMA expansion: ~2.67x (8/3) or custom hidden_dim
        hidden_dim = cfg.get("hidden_dim", int(8 * emb_dim / 3))

        self.w_gate = nn.Linear(emb_dim, hidden_dim, bias=cfg.get("ffn_bias", False))
        self.w_up = nn.Linear(emb_dim, hidden_dim, bias=cfg.get("ffn_bias", False))
        self.w_down = nn.Linear(hidden_dim, emb_dim, bias=cfg.get("ffn_bias", False))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class FeedForward(nn.Module):
    """
    Configurable Feed-Forward network dispatcher.
    Supports 'gelu' (default) and 'swiglu'.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        ffn_type = cfg.get("ffn_type", "gelu").lower()
        if ffn_type == "swiglu":
            self.net = SwiGLUFeedForward(cfg)
        else:
            self.net = GELUFeedForward(cfg)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)

    @property
    def layers(self):
        """Backward-compatibility property for legacy weight loaders accessing .layers."""
        if hasattr(self.net, "layers"):
            return self.net.layers
        raise AttributeError("SwiGLUFeedForward does not have sequential .layers attribute.")
