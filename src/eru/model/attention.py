"""
Vectorized Multi-Head Self-Attention with causal masking and dropout.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})


class MultiHeadAttention(nn.Module):
    """
    Canonical vectorized Multi-Head Attention module with causal masking and optional QKV projection bias.

    Args:
        d_in: Input embedding dimension.
        d_out: Desired total output dimension (must be divisible by num_heads).
        context_length: Maximum context sequence length for causal masking.
        dropout: Attention dropout probability.
        num_heads: Number of parallel attention heads.
        qkv_bias: If True, adds a learnable bias to Q, K, and V linear layers (default: False).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 12,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask buffer (values above diagonal are set to -inf)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Compute multi-head causal attention for input batch.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_in).

        Returns:
            Context vector of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, _ = x.shape

        # Linear projections for query, key, value
        keys = self.W_key(x)        # (b, num_tokens, d_out)
        queries = self.W_query(x)    # (b, num_tokens, d_out)
        values = self.W_value(x)    # (b, num_tokens, d_out)

        # Split d_out into num_heads * head_dim: (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted combination of values
        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
