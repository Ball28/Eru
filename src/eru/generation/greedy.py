"""
Greedy next-token text generation.
"""

from typing import Any

try:
    import torch
except ImportError:
    torch = None


def generate_greedy(
    model: Any,
    idx: Any,
    max_new_tokens: int,
    context_size: int,
) -> Any:
    """
    Greedy deterministic autoregressive generation via next-token argmax.

    Args:
        model: GPTModel instance.
        idx: Input tensor of shape (batch_size, num_tokens).
        max_new_tokens: Number of tokens to generate.
        context_size: Supported context length.

    Returns:
        Tensor of shape (batch_size, num_tokens + max_new_tokens).
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Backward-compatible alias matching original naming
generate_text_simple = generate_greedy
