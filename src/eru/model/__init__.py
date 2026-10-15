"""
Neural network model components for Eru LLM.
"""

from eru.model.activations import GELU
from eru.model.normalization import LayerNorm
from eru.model.attention import MultiHeadAttention
from eru.model.feedforward import FeedForward
from eru.model.transformer_block import TransformerBlock
from eru.model.gpt import (
    GPTModel,
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    GPT_MODEL_CONFIGS,
)
from eru.model.educational_attention import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
)

__all__ = [
    "GELU",
    "LayerNorm",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "GPTModel",
    "GPT_CONFIG_124M",
    "GPT_CONFIG_SMALL",
    "GPT_MODEL_CONFIGS",
    "SelfAttention_v1",
    "SelfAttention_v2",
    "CausalAttention",
    "MultiHeadAttentionWrapper",
]
