"""
Sampling-based and temperature-scaled text generation (Chapter 5, Listing 5.3).
"""

from typing import Any, Optional

try:
    import torch
except ImportError:
    torch = None


def generate(
    model: Any,
    idx: Any,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Any:
    """
    Generate new token IDs auto-regressively from input prompt tensor `idx`.

    Args:
        model: GPTModel or compatible language model module.
        idx: Input token IDs tensor of shape (batch_size, num_tokens).
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Maximum context length supported by the model.
        temperature: Temperature scaling factor (>0 scales logits, ==0 uses argmax greedy decoding).
        top_k: Optional top-k filtering threshold.
        eos_id: Optional end-of-sequence token ID that halts generation early.

    Returns:
        Tensor of shape (batch_size, num_tokens + num_generated_tokens).
    """
    for _ in range(max_new_tokens):
        # Crop context window if sequence exceeds context_size
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step: shape becomes (batch_size, vocab_size)
        logits = logits[:, -1]

        # Apply top-k filtering if requested
        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # Apply temperature scaling or greedy argmax
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Early stopping on end of sequence
        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
