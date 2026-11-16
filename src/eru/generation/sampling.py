"""
Sampling-based, temperature-scaled, top-k/top-p/min-p, and streaming text generation
with optional Key-Value (KV) cache acceleration.
"""

from typing import Any, Generator, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from eru.model.kv_cache import KVCache


def apply_sampling_filters(
    logits: "torch.Tensor",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """
    Apply temperature scaling, repetition penalty, top-k, top-p, and min-p filtering to logits.

    Args:
        logits: Unnormalized next-token logits of shape (batch_size, vocab_size).
        temperature: Temperature scaling factor (>0 scales logits, ==0 uses argmax).
        top_k: Truncates distribution to top K most probable tokens.
        top_p: Nucleus sampling probability cutoff (0.0 to 1.0).
        min_p: Min-P sampling threshold relative to the top token's probability.
        repetition_penalty: Penalty factor applied to previously seen token logits.
        generated_tokens: Tensor of already generated token IDs (batch_size, seq_len).

    Returns:
        Filtered logits tensor of shape (batch_size, vocab_size).
    """
    # Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None:
        for b in range(logits.shape[0]):
            unique_tokens = torch.unique(generated_tokens[b])
            for token_id in unique_tokens:
                if logits[b, token_id] > 0:
                    logits[b, token_id] /= repetition_penalty
                else:
                    logits[b, token_id] *= repetition_penalty

    # Temperature scaling
    if temperature > 0.0:
        logits = logits / temperature

    # Top-K filtering
    if top_k is not None and top_k > 0:
        top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        min_val = top_logits[:, -1, None]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    # Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # Min-P filtering
    if min_p is not None and min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        top_probs, _ = torch.max(probs, dim=-1, keepdim=True)
        min_p_threshold = min_p * top_probs
        logits = torch.where(
            probs < min_p_threshold,
            torch.tensor(float("-inf"), device=logits.device),
            logits,
        )

    return logits


def sample_next_token(
    logits: "torch.Tensor",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """
    Sample or pick argmax next token from filtered logits.
    """
    if temperature > 0.0:
        probs = torch.softmax(logits, dim=-1)
        # Numerical guard: fallback to uniform if NaN / all -inf
        if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    return idx_next


def generate(
    model: Any,
    idx: Any,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
    use_cache: bool = True,
) -> Any:
    """
    Generate new token IDs autoregressively from input prompt tensor `idx`.

    Args:
        model: GPTModel or compatible language model module.
        idx: Input token IDs tensor of shape (batch_size, num_tokens).
        max_new_tokens: Maximum number of tokens to generate.
        context_size: Maximum context length supported by the model.
        temperature: Temperature scaling factor (>0 scales logits, ==0 uses argmax greedy decoding).
        top_k: Optional top-k filtering threshold.
        top_p: Optional nucleus sampling threshold (0.0 to 1.0).
        min_p: Optional min-p sampling threshold relative to top token probability.
        repetition_penalty: Factor penalizing tokens that have already appeared (default: 1.0).
        eos_id: Optional end-of-sequence token ID that halts generation early.
        use_cache: If True, uses KV-cache to avoid recomputing past tokens (O(T) time complexity).

    Returns:
        Tensor of shape (batch_size, num_tokens + num_generated_tokens).
    """
    cache = KVCache() if use_cache else None

    # Prefill step
    idx_cond = idx[:, -context_size:] if not use_cache else idx
    with torch.no_grad():
        logits = model(idx_cond, kv_cache=cache, start_pos=0)

    for _ in range(max_new_tokens):
        # Focus on the last token's logits
        step_logits = logits[:, -1].clone()

        step_logits = apply_sampling_filters(
            logits=step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=idx,
        )

        idx_next = sample_next_token(step_logits, temperature=temperature)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

        # Context boundary check
        if use_cache and cache.get_seq_len() >= context_size:
            # Fall back to sliding window if cache exceeds supported context
            cache.reset()
            with torch.no_grad():
                logits = model(idx[:, -context_size:])
        else:
            with torch.no_grad():
                if use_cache:
                    logits = model(idx_next, kv_cache=cache, start_pos=cache.get_seq_len())
                else:
                    logits = model(idx[:, -context_size:])

    return idx


def generate_stream(
    model: Any,
    idx: Any,
    max_new_tokens: int,
    context_size: int,
    tokenizer: Optional[Any] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
    use_cache: bool = True,
) -> Generator[Tuple[int, Optional[str]], None, None]:
    """
    Autoregressively stream newly generated tokens one-by-one.

    Yields:
        Tuple of (token_id, decoded_token_str or None).
    """
    cache = KVCache() if use_cache else None

    idx_cond = idx[:, -context_size:] if not use_cache else idx
    with torch.no_grad():
        logits = model(idx_cond, kv_cache=cache, start_pos=0)

    for _ in range(max_new_tokens):
        step_logits = logits[:, -1].clone()

        step_logits = apply_sampling_filters(
            logits=step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=idx,
        )

        idx_next = sample_next_token(step_logits, temperature=temperature)
        token_id = int(idx_next[0, 0].item())

        if eos_id is not None and token_id == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

        decoded_str = tokenizer.decode([token_id]) if tokenizer is not None else None
        yield token_id, decoded_str

        if use_cache and cache.get_seq_len() >= context_size:
            cache.reset()
            with torch.no_grad():
                logits = model(idx[:, -context_size:])
        else:
            with torch.no_grad():
                if use_cache:
                    logits = model(idx_next, kv_cache=cache, start_pos=cache.get_seq_len())
                else:
                    logits = model(idx[:, -context_size:])
