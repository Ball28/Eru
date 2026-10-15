"""
Plotting utilities for loss curves and evaluation metrics.
"""

from typing import List, Optional


def plot_losses(
    epochs_seen: List[float],
    tokens_seen: List[int],
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation losses against epochs and tokens seen."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs_seen, train_losses, label="Training loss", color="blue")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss", color="red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    if tokens_seen and len(tokens_seen) == len(train_losses):
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()
    plt.close()
