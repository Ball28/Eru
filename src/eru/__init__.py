"""
Eru: A clean, modular LLM framework built from scratch.
"""

from eru.tokenizer.bpe import BPETokenizer
from eru.tokenizer.simple import SimpleTokenizerV1, SimpleTokenizerV2
from eru.model.gpt import GPTModel, GPT_CONFIG_124M, GPT_CONFIG_SMALL
from eru.model.attention import MultiHeadAttention
from eru.generation.sampling import generate
from eru.generation.greedy import generate_greedy

__version__ = "0.1.0"

__all__ = [
    "BPETokenizer",
    "SimpleTokenizerV1",
    "SimpleTokenizerV2",
    "GPTModel",
    "GPT_CONFIG_124M",
    "GPT_CONFIG_SMALL",
    "MultiHeadAttention",
    "generate",
    "generate_greedy",
    "__version__",
]
