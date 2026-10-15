"""
Spam classification fine-tuning routines (Chapter 6).
"""

from typing import Any, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = type("nn", (), {"Linear": object})
    F = None


def setup_spam_classifier_head(model: Any, num_classes: int = 2) -> None:
    """
    Freeze all base transformer layers and replace the language modeling head
    with a binary classification linear layer (Chapter 6, Listing 6.3).
    Only the last transformer block, final norm, and new out_head are trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    emb_dim = model.out_head.in_features
    model.out_head = nn.Linear(in_features=emb_dim, out_features=num_classes)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True


def calc_accuracy_loader(data_loader: Any, model: Any, device: Any, num_batches: Optional[int] = None) -> float:
    """
    Compute accuracy across classification batches using the last output token.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0
    if len(data_loader) == 0:
        return 0.0

    total_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= total_batches:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples if num_examples > 0 else 0.0


def calc_classification_loss_batch(input_batch: Any, target_batch: Any, model: Any, device: Any) -> Any:
    """
    Cross-entropy loss computed strictly over the logits of the last token.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return F.cross_entropy(logits, target_batch)


def train_classifier_simple(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    optimizer: Any,
    device: Any,
    num_epochs: int,
    eval_freq: int = 50,
    eval_iter: int = 5,
    tokenizer: Optional[Any] = None,
) -> Tuple[List[float], List[float], List[float], List[float], int]:
    """
    Fine-tuning loop for sequence classification.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    # Evaluate partial batch loss
                    train_loss = 0.0
                    for i, (xb, yb) in enumerate(train_loader):
                        if i >= eval_iter:
                            break
                        train_loss += calc_classification_loss_batch(xb, yb, model, device).item()
                    train_loss /= min(eval_iter, len(train_loader))

                    val_loss = 0.0
                    for i, (xb, yb) in enumerate(val_loader):
                        if i >= eval_iter:
                            break
                        val_loss += calc_classification_loss_batch(xb, yb, model, device).item()
                    val_loss /= min(eval_iter, len(val_loader))

                model.train()
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Compute accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}: Train Acc {train_accuracy * 100:.2f}% | "
            f"Val Acc {val_accuracy * 100:.2f}%"
        )

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(
    text: str,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_length: Optional[int] = None,
    pad_token_id: int = 50256,
) -> str:
    """
    Classify a single raw text string as 'spam' or 'not spam' (Chapter 6).
    """
    model.eval()
    input_ids = tokenizer.encode(text)

    # Correctly retrieve max context length from position embedding shape[0]
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is not None:
        target_len = min(max_length, supported_context_length)
    else:
        target_len = supported_context_length

    input_ids = input_ids[:target_len]
    if len(input_ids) < target_len:
        input_ids = input_ids + [pad_token_id] * (target_len - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
