"""
Rotary Position Embedding (RoPE) implementation.
Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
"""

from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for rotating Query and Key representations.
    Encodes relative positional distances directly into attention dot-products
    without requiring learned absolute position embeddings.

    Args:
        dim: Dimension of each attention head (head_dim).
        max_seq_len: Precomputed maximum sequence length cache limit.
        base: Frequency scaling base (default: 10000.0).
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequencies: theta_i = 1 / (base ** (2i / dim))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: "torch.Tensor",
        seq_len: int,
        start_pos: int = 0,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Retrieve cosine and sine rotary components for the requested sequence span.

        Args:
            x: Input tensor to derive device and dtype from.
            seq_len: Number of sequence tokens.
            start_pos: Offset index for position embeddings (used with KV-cache).

        Returns:
            Tuple of (cos, sin) tensors shaped (1, 1, seq_len, dim).
        """
        end_pos = start_pos + seq_len
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(max(end_pos, self.max_seq_len * 2))

        cos = self.cos_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[start_pos:end_pos].unsqueeze(0).unsqueeze(0)
        return cos.to(device=x.device, dtype=x.dtype), sin.to(device=x.device, dtype=x.dtype)


def rotate_half(x: "torch.Tensor") -> "torch.Tensor":
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: "torch.Tensor",
    k: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Apply rotary embeddings to Query and Key tensors.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        cos: Cosine tensor of shape (1, 1, seq_len, head_dim).
        sin: Sine tensor of shape (1, 1, seq_len, head_dim).

    Returns:
        Tuple of (q_rotated, k_rotated).
    """
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
