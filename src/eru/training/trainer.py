"""
Standard and optimized pretraining training loops with AMP, gradient clipping,
and cosine learning rate scheduling.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

from eru.training.loss import calc_loss_batch, evaluate_model


def get_cosine_schedule_with_warmup(
    optimizer: Any,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> Any:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of initial linear warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum learning rate floor as a fraction of initial lr.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model_simple(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    device: Any,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    epoch_callback: Optional[Callable[[int, float, float], None]] = None,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = 1.0,
    scheduler: Optional[Any] = None,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Train a GPT model using standard cross-entropy loss, with optional AMP,
    gradient clipping, and learning rate scheduling.

    Args:
        model: GPTModel instance.
        train_loader: Pretraining training DataLoader.
        val_loader: Pretraining validation DataLoader.
        optimizer: PyTorch optimizer (e.g. AdamW).
        device: Execution device (e.g. 'cuda' or 'cpu').
        num_epochs: Number of training epochs.
        eval_freq: Step frequency for evaluating and logging loss.
        eval_iter: Number of batches to evaluate during each evaluation step.
        start_context: Optional text prompt to generate sample completions.
        tokenizer: Tokenizer for decoding sample completions.
        epoch_callback: Optional callback receiving (epoch, train_loss, val_loss).
        use_amp: If True, uses Automatic Mixed Precision (bfloat16/float16).
        max_grad_norm: Optional maximum gradient norm for gradient clipping.
        scheduler: Optional learning rate scheduler updated after each step.

    Returns:
        Tuple of (train_losses, val_losses, tokens_seen_history).
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    device_type = "cuda" if "cuda" in str(device) else ("mps" if "mps" in str(device) else "cpu")
    amp_dtype = torch.bfloat16 if (device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device_type == "cuda"))

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            if use_amp and device_type in ("cuda", "cpu"):
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                scaler.scale(loss).backward()
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        if epoch_callback is not None and len(train_losses) > 0:
            epoch_callback(epoch + 1, train_losses[-1], val_losses[-1])

    return train_losses, val_losses, track_tokens_seen
