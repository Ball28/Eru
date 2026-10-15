"""
Text generation strategies for Eru LLM.
"""

from eru.generation.sampling import generate
from eru.generation.greedy import generate_greedy, generate_text_simple

__all__ = [
    "generate",
    "generate_greedy",
    "generate_text_simple",
]
