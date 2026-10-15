"""
Pretrained weight loading and conversion utilities for Eru LLM.
"""

from eru.weights.loader import assign, load_weights_into_gpt
from eru.weights.tf_download import download_and_load_gpt2

__all__ = [
    "assign",
    "load_weights_into_gpt",
    "download_and_load_gpt2",
]
