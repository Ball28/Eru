"""
Standard pretraining training loop (Chapter 5, Listing 5.2).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from eru.training.loss import calc_loss_batch, evaluate_model


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
) -> Tuple[List[float], List[float], List[int]]:
    """
    Train a GPT model using standard cross-entropy language modeling loss.

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

    Returns:
        Tuple of (train_losses, val_losses, tokens_seen_history).
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
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
