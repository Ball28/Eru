"""
Key-Value (KV) Cache for efficient autoregressive generation in Transformer models.
"""

from typing import List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


class KVCache:
    """
    Maintains cached Key and Value states across all transformer blocks.
    Transforms generation time complexity from O(T^2) to O(T) by avoiding
    redundant recalculation of past token representations.
    """

    def __init__(self, num_layers: int = 0):
        self.k_cache: List[Optional["torch.Tensor"]] = [None] * num_layers if num_layers > 0 else []
        self.v_cache: List[Optional["torch.Tensor"]] = [None] * num_layers if num_layers > 0 else []

    def update(
        self,
        layer_idx: int,
        k: "torch.Tensor",
        v: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Append new key and value projections to the cache for the specified layer.

        Args:
            layer_idx: Zero-based transformer block index.
            k: New key tensor of shape (batch_size, num_heads, num_new_tokens, head_dim).
            v: New value tensor of shape (batch_size, num_heads, num_new_tokens, head_dim).

        Returns:
            Tuple of concatenated full (keys, values) spanning all past and current tokens.
        """
        # Dynamically extend lists if needed
        while len(self.k_cache) <= layer_idx:
            self.k_cache.append(None)
            self.v_cache.append(None)

        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self) -> None:
        """Clear all cached tensors."""
        self.k_cache = [None] * len(self.k_cache)
        self.v_cache = [None] * len(self.v_cache)

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """Return total sequence length currently stored in the cache."""
        if layer_idx < len(self.k_cache) and self.k_cache[layer_idx] is not None:
            return self.k_cache[layer_idx].shape[2]
        return 0

    def __len__(self) -> int:
        return self.get_seq_len(0)
