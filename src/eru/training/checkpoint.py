"""
Checkpoint saving and loading utilities for models and optimizers (Chapter 5, Listing 5.4).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None


def save_checkpoint(
    checkpoint_path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    epoch: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state, optimizer state, and configuration to disk."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        state["epoch"] = epoch
    if cfg is not None:
        state["cfg"] = cfg

    torch.save(state, str(path))
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Any,
    optimizer: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load model weights (and optional optimizer state) from disk."""
    if device is None:
        map_location = torch.device("cpu")
    else:
        map_location = device

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Bare state_dict
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
