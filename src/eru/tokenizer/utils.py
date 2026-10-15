"""
Token conversion helpers for converting between text, token IDs, and tensors.
"""

from typing import Any, List, Optional, Set, Union


def text_to_token_ids(
    text: str,
    tokenizer: Any,
    allowed_special: Optional[Set[str]] = None,
    device: Optional[Any] = None,
) -> Any:
    """
    Encode text to token IDs and return a 2D tensor of shape (1, num_tokens).

    Args:
        text: Prompt string to encode.
        tokenizer: Tokenizer instance with an .encode() method.
        allowed_special: Set of allowed special tokens.
        device: Optional target device for the returned tensor.

    Returns:
        Tensor of shape (1, num_tokens) on the requested device.
    """
    if allowed_special is None:
        allowed_special = {"<|endoftext|>"}

    try:
        encoded = tokenizer.encode(text, allowed_special=allowed_special)
    except TypeError:
        # Fallback if tokenizer.encode doesn't accept allowed_special (e.g. SimpleTokenizer)
        encoded = tokenizer.encode(text)

    try:
        import torch
        tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
    except ImportError:
        # Fallback if torch is not installed
        return [encoded]


def token_ids_to_text(token_ids: Any, tokenizer: Any) -> str:
    """
    Decode a 1D or 2D tensor/list of token IDs back into text.

    Args:
        token_ids: Tensor or nested list of token IDs.
        tokenizer: Tokenizer instance with a .decode() method.

    Returns:
        Decoded text string.
    """
    if hasattr(token_ids, "squeeze"):
        # Squeeze batch dimension if present
        flat = token_ids.squeeze(0) if token_ids.ndim > 1 else token_ids
        token_list = flat.tolist()
    elif isinstance(token_ids, list):
        if len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_list = token_ids[0]
        else:
            token_list = token_ids
    else:
        token_list = list(token_ids)

    return tokenizer.decode(token_list)
