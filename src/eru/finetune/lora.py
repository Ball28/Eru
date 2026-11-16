"""
From-scratch Low-Rank Adaptation (LoRA) implementation for parameter-efficient fine-tuning (PEFT).
Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object, "Linear": object, "Parameter": object})


class LoRALinear(nn.Module):
    """
    LoRA adapter layer wrapping an existing nn.Linear module.
    Computes: y = x @ W^T + (dropout(x) @ A^T @ B^T) * (alpha / rank)

    Args:
        base_layer: Existing frozen nn.Linear module.
        rank: Low-rank dimension (r).
        alpha: Scaling parameter (alpha).
        dropout: LoRA input dropout probability.
    """

    def __init__(
        self,
        base_layer: Any,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank adapters: A is initialized randomly, B is initialized to 0
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.merged:
            return self.base_layer(x)
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def merge(self) -> None:
        """Fold adapter weights directly into the base layer for zero-overhead inference."""
        if not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self) -> None:
        """Subtract adapter weights from base layer to restore original weights."""
        if self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False


def apply_lora_to_model(
    model: Any,
    target_module_names: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> Any:
    """
    Freeze entire model and wrap specified linear projections with LoRALinear adapters.

    Args:
        model: GPTModel instance.
        target_module_names: List of module name substrings to adapt (default: ["W_query", "W_value"]).
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling factor.
        dropout: LoRA dropout probability.

    Returns:
        Adapted model instance.
    """
    if target_module_names is None:
        target_module_names = ["W_query", "W_value"]

    # 1. Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # 2. Replace target linear layers
    for block in model.trf_blocks:
        for name, module in block.named_modules():
            for target in target_module_names:
                if hasattr(block.att, target):
                    base_layer = getattr(block.att, target)
                    if isinstance(base_layer, nn.Linear):
                        setattr(
                            block.att,
                            target,
                            LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout),
                        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Applied LoRA (r={rank}, alpha={alpha}). "
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)"
    )
    return model


def get_lora_state_dict(model: Any) -> Dict[str, "torch.Tensor"]:
    """Extract only the trainable LoRA adapter weights."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def save_lora_checkpoint(path: Union[str, Path], model: Any, cfg: Optional[Dict[str, Any]] = None) -> None:
    """Save LoRA weights and configuration."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lora_weights": get_lora_state_dict(model),
        "cfg": cfg or getattr(model, "cfg", {}),
    }
    torch.save(payload, save_path)
    print(f"Saved LoRA adapter checkpoint -> {save_path}")


def load_lora_checkpoint(path: Union[str, Path], model: Any) -> None:
    """Load LoRA weights into an adapted model."""
    checkpoint = torch.load(path, map_location="cpu")
    lora_weights = checkpoint.get("lora_weights", checkpoint)
    model.load_state_dict(lora_weights, strict=False)
    print(f"Loaded LoRA adapter checkpoint from {path}")
