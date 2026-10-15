"""
Device selection helper with cross-platform support for CUDA, Apple Silicon MPS, and CPU.
"""

from typing import Optional


def get_device(preference: Optional[str] = None):
    """
    Select best available compute device (CUDA -> MPS -> CPU), or respect user preference.

    Fixes typos from legacy scripts (e.g. 'cude' -> 'cuda').
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if preference is not None:
        cleaned = preference.lower().strip()
        if cleaned in ("cuda", "cude") and torch.cuda.is_available():
            return torch.device("cuda")
        if cleaned == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if cleaned == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
