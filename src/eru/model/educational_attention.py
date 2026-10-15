"""
Educational self-attention variants from Chapter 3 of 'Build a Large Language Model (From Scratch)'.
Preserved for instructional clarity and characterization testing.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})


class SelfAttention_v1(nn.Module):
    """Initial self-attention mechanism using raw Parameter weights (Listing 3.1)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """Self-attention mechanism using Linear layers (Listing 3.2)."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """Single-head causal attention with dropout and upper-triangular mask (Listing 3.3)."""

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """Multi-head attention implemented as a list of independent CausalAttention heads (Listing 3.4)."""

    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.cat([head(x) for head in self.heads], dim=-1)
