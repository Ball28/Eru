#!/usr/bin/env python3
"""
Thin wrapper script to fine-tune GPT on instructions (SFT).
Run from the Eru/ project root:
    python scripts/finetune_instruction.py
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eru.utils.device import get_device
from eru.utils.seed import set_seed
from eru.utils.config import load_config
from eru.tokenizer.bpe import BPETokenizer
from eru.model.gpt import GPTModel, GPT_CONFIG_124M
from eru.data.download import download_instruction_data
from eru.data.instruction_dataset import InstructionDataset
from eru.data.collate import custom_collate_fn
from eru.training.trainer import train_model_simple
from eru.training.checkpoint import save_checkpoint


def main():
    set_seed(123)
    device = get_device()
    print(f"Instruction fine-tuning using device: {device}")

    data = download_instruction_data(dest_path="data/instruction-data.json")
    train_portion = int(len(data) * 0.85)
    train_data = data[:train_portion]
    val_data = data[train_portion:]

    tokenizer = BPETokenizer()
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    from functools import partial
    from torch.utils.data import DataLoader
    import torch

    collate = partial(custom_collate_fn, device=str(device), allowed_max_length=1024)
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate, shuffle=False)

    cfg = GPT_CONFIG_124M.copy()
    cfg["drop_rate"] = 0.0
    cfg["qkv_bias"] = True
    model = GPTModel(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
    )
    save_checkpoint("checkpoints/instruction_sft.pth", model, optimizer=optimizer, cfg=cfg)


if __name__ == "__main__":
    main()
