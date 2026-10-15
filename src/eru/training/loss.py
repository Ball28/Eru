"""
Loss calculation and evaluation routines for language model pretraining.
"""

from typing import Any, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def calc_loss_batch(input_batch: Any, target_batch: Any, model: Any, device: Any) -> Any:
    """
    Calculate cross-entropy loss for a single batch of sequences.

    Args:
        input_batch: Tensor of token IDs (batch_size, seq_len).
        target_batch: Target token IDs (batch_size, seq_len).
        model: GPTModel.
        device: Device to transfer tensors to.

    Returns:
        Scalar loss tensor.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader: Any, model: Any, device: Any, num_batches: Optional[int] = None) -> float:
    """
    Compute average cross-entropy loss across batches in a dataloader.

    Args:
        data_loader: DataLoader instance.
        model: GPTModel.
        device: Execution device.
        num_batches: Maximum number of batches to evaluate (evaluates all if None).

    Returns:
        Average loss as a float.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    total_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= total_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / total_batches


def evaluate_model(model: Any, train_loader: Any, val_loader: Any, device: Any, eval_iter: int) -> Tuple[float, float]:
    """
    Evaluate model on train and validation subsets without tracking gradients.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
