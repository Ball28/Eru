"""
Text generation strategies for Eru LLM.
"""

from eru.generation.sampling import (
    generate,
    generate_stream,
    apply_sampling_filters,
    sample_next_token,
)
from eru.generation.greedy import generate_greedy, generate_text_simple

__all__ = [
    "generate",
    "generate_stream",
    "apply_sampling_filters",
    "sample_next_token",
    "generate_greedy",
    "generate_text_simple",
]
