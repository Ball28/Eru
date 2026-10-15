"""
Random seed management for reproducible runs across PyTorch, NumPy, and random.
"""

import random
from typing import Optional


def set_seed(seed: int = 123) -> None:
    """Set seeds across Python's random module, NumPy, and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
