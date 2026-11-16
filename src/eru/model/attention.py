"""
Vectorized Multi-Head, Grouped-Query (GQA), and Multi-Query Attention (MQA)
with causal masking, PyTorch SDPA, RoPE, and KV-cache support.
"""

from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    F = None

from eru.model.rope import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Vectorized Attention module supporting:
    - Multi-Head Attention (MHA)
    - Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
    - Rotary Position Embeddings (RoPE)
    - Key-Value (KV) Caching
    - PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float = 0.0,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_sdpa: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.use_sdpa = use_sdpa

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Upper-triangular causal mask buffer
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(
        self,
        x: "torch.Tensor",
        kv_cache: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        rotary_emb: Optional[Any] = None,
        start_pos: int = 0,
    ) -> "torch.Tensor":
        """
        Compute attention with optional RoPE, GQA, and KV-cache.

        Args:
            x: Input tensor of shape (batch_size, num_tokens, d_in).
            kv_cache: Optional KVCache instance.
            layer_idx: Layer index for caching keys/values.
            rotary_emb: Optional RotaryEmbedding module.
            start_pos: Sequence position offset for RoPE and caching.

        Returns:
            Context vector of shape (batch_size, num_tokens, d_out).
        """
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape to (b, num_tokens, heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rotary_emb is not None:
            cos, sin = rotary_emb(x, seq_len=num_tokens, start_pos=start_pos)
            queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        # Update and retrieve cached keys/values
        if kv_cache is not None and layer_idx is not None:
            keys, values = kv_cache.update(layer_idx, keys, values)

        total_keys = keys.shape[2]

        # Expand KV heads for GQA/MQA if num_queries_per_kv > 1
        if self.num_queries_per_kv > 1:
            keys_for_attn = torch.repeat_interleave(keys, repeats=self.num_queries_per_kv, dim=1)
            values_for_attn = torch.repeat_interleave(values, repeats=self.num_queries_per_kv, dim=1)
        else:
            keys_for_attn = keys
            values_for_attn = values

        if self.use_sdpa and F is not None:
            is_causal = (num_tokens > 1 and num_tokens == total_keys)
            context_vec = F.scaled_dot_product_attention(
                queries,
                keys_for_attn,
                values_for_attn,
                is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn_scores = queries @ keys_for_attn.transpose(2, 3)

            if num_tokens == total_keys:
                mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
                attn_scores.masked_fill_(mask_bool, -torch.inf)
            elif num_tokens > 1:
                mask_bool = self.mask.bool()[:num_tokens, :total_keys]
                attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values_for_attn

        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
