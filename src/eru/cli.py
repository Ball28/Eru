"""
Command-line interface (CLI) entry point for Eru LLM.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from eru.tokenizer.bpe import BPETokenizer
from eru.tokenizer.utils import text_to_token_ids, token_ids_to_text
from eru.model.gpt import GPTModel, GPT_CONFIG_124M, GPT_CONFIG_SMALL
from eru.generation.sampling import generate
from eru.utils.device import get_device
from eru.utils.seed import set_seed
from eru.utils.config import load_config
from eru.training.trainer import train_model_simple
from eru.training.checkpoint import save_checkpoint, load_checkpoint
from eru.data.pretrain_dataset import create_pretrain_dataloader


def cli_train(args):
    """Pretrain a GPT model on text."""
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load configuration
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = GPT_CONFIG_SMALL if args.small else GPT_CONFIG_124M

    # Load text corpus
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    split_idx = int(0.9 * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = BPETokenizer()
    train_loader = create_pretrain_dataloader(
        train_data,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=True,
        shuffle=True,
        tokenizer=tokenizer,
    )
    val_loader = create_pretrain_dataloader(
        val_data,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=False,
        shuffle=False,
        tokenizer=tokenizer,
    )

    import torch
    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training model ({sum(p.numel() for p in model.parameters()):,} parameters) for {args.epochs} epochs...")
    train_losses, val_losses, _ = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
    )

    if args.output:
        save_checkpoint(args.output, model, optimizer=optimizer, cfg=cfg)


def cli_generate(args):
    """Generate text from a model checkpoint."""
    set_seed(args.seed)
    device = get_device(args.device)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = GPT_CONFIG_SMALL if args.small else GPT_CONFIG_124M

    model = GPTModel(cfg).to(device)
    if args.weights:
        load_checkpoint(args.weights, model=model, device=device)
    model.eval()

    tokenizer = BPETokenizer()
    input_ids = text_to_token_ids(args.prompt, tokenizer, device=device)

    print(f"\n--- Prompt ---\n{args.prompt}\n--- Generation ---")
    out_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=args.max_tokens,
        context_size=cfg["context_length"],
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_text = token_ids_to_text(out_ids, tokenizer)
    print(output_text)


def cli_finetune_spam(args):
    """Fine-tune GPT-2 on the SMS spam classification dataset."""
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    import torch
    from eru.data.spam_dataset import create_spam_dataloader
    from eru.finetune.classification import setup_spam_classifier_head, train_classifier_simple

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = GPT_CONFIG_124M

    tokenizer = BPETokenizer()
    train_loader = create_spam_dataloader(
        args.train_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        shuffle=True,
    )
    val_loader = create_spam_dataloader(
        args.val_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=cfg["context_length"],
        shuffle=False,
    )

    model = GPTModel(cfg).to(device)
    if args.weights:
        load_checkpoint(args.weights, model=model, device=device)

    setup_spam_classifier_head(model, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Fine-tuning classifier for {args.epochs} epochs...")
    train_classifier_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        tokenizer=tokenizer,
    )

    if args.output:
        save_checkpoint(args.output, model, optimizer=optimizer, cfg=cfg)


def main():
    parser = argparse.ArgumentParser(
        prog="eru",
        description="Eru LLM: Train, fine-tune, and generate text with GPT models built from scratch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train sub-command
    p_train = subparsers.add_parser("train", help="Pretrain a GPT model on text.")
    p_train.add_argument("--data", default="data/the-verdict.txt", help="Path to text corpus.")
    p_train.add_argument("--config", default=None, help="Path to YAML configuration file.")
    p_train.add_argument("--small", action="store_true", help="Use small fast CPU configuration.")
    p_train.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    p_train.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    p_train.add_argument("--lr", type=float, default=0.0004, help="Learning rate.")
    p_train.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay.")
    p_train.add_argument("--eval-freq", type=int, default=5, help="Step evaluation frequency.")
    p_train.add_argument("--eval-iter", type=int, default=1, help="Batches per evaluation.")
    p_train.add_argument("--device", default=None, help="Execution device (cpu, cuda, mps).")
    p_train.add_argument("--seed", type=int, default=123, help="Random seed.")
    p_train.add_argument("--output", default="checkpoints/model.pth", help="Checkpoint save path.")
    p_train.set_defaults(func=cli_train)

    # Generate sub-command
    p_gen = subparsers.add_parser("generate", help="Generate text from prompt.")
    p_gen.add_argument("--prompt", required=True, help="Input prompt string.")
    p_gen.add_argument("--weights", default=None, help="Path to checkpoint weights.")
    p_gen.add_argument("--config", default=None, help="Path to YAML configuration file.")
    p_gen.add_argument("--small", action="store_true", help="Use small fast CPU configuration.")
    p_gen.add_argument("--max-tokens", type=int, default=25, help="Maximum new tokens.")
    p_gen.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p_gen.add_argument("--top-k", type=int, default=25, help="Top-k filtering threshold.")
    p_gen.add_argument("--device", default=None, help="Execution device (cpu, cuda, mps).")
    p_gen.add_argument("--seed", type=int, default=123, help="Random seed.")
    p_gen.set_defaults(func=cli_generate)

    # Finetune spam sub-command
    p_ft = subparsers.add_parser("finetune-spam", help="Fine-tune GPT-2 for spam classification.")
    p_ft.add_argument("--train-csv", default="data/spam/train.csv", help="Training CSV.")
    p_ft.add_argument("--val-csv", default="data/spam/validation.csv", help="Validation CSV.")
    p_ft.add_argument("--weights", default=None, help="Initial pretrained weights.")
    p_ft.add_argument("--config", default=None, help="Path to YAML config.")
    p_ft.add_argument("--epochs", type=int, default=3, help="Epochs.")
    p_ft.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    p_ft.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    p_ft.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay.")
    p_ft.add_argument("--eval-freq", type=int, default=50, help="Evaluation frequency.")
    p_ft.add_argument("--device", default=None, help="Execution device.")
    p_ft.add_argument("--seed", type=int, default=123, help="Random seed.")
    p_ft.add_argument("--output", default="checkpoints/spam_classifier.pth", help="Output checkpoint.")
    p_ft.set_defaults(func=cli_finetune_spam)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
