"""
Activation functions for transformer architectures (Chapter 4).
"""

import math
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) with exact tanh approximation matching GPT-2 (Chapter 4).
    formula: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )
