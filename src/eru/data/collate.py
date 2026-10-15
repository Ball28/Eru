"""
Custom collate functions for dynamic batch padding and cross-entropy ignore index masking (Chapter 7).
"""

from typing import Any, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


def custom_collate_fn(
    batch: List[List[int]],
    pad_token_id: int = 50256,
    ignore_index: int = -100,
    allowed_max_length: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Collate variable-length token sequences into padded input and target tensors.

    Replaces all but the first padding token in targets with `ignore_index` (-100)
    so loss is only computed over valid tokens and the first end-of-text marker.

    Args:
        batch: List of integer token lists.
        pad_token_id: Token ID used for padding (default: 50256).
        ignore_index: Cross-entropy ignore index (default: -100).
        allowed_max_length: Optional truncation length.
        device: Target tensor device string or torch.device.

    Returns:
        Tuple of (inputs_tensor, targets_tensor).
    """
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = list(item)
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1], dtype=torch.long)
        targets = torch.tensor(padded[1:], dtype=torch.long)

        # Replace all but the first padding token in targets with ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
