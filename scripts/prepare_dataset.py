#!/usr/bin/env python3
"""
Pre-tokenize raw text corpora into binary memory-mapped datasets (train.bin, val.bin)
for fast, zero-overhead pretraining.
"""

import argparse
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
from tqdm import tqdm
from eru.tokenizer.bpe import BPETokenizer


def tokenize_corpus(
    input_path: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    split_char: str = "<|endoftext|>",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer()

    print(f"Reading corpus: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Total characters: {len(text):,}")
    print("Tokenizing corpus...")
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)
    print(f"Total tokens: {total_tokens:,}")

    split_idx = int(total_tokens * (1.0 - val_ratio))
    train_tokens = np.array(tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(tokens[split_idx:], dtype=np.uint16)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    print(f"Saving {len(train_tokens):,} train tokens -> {train_path}")
    train_tokens.tofile(train_path)

    print(f"Saving {len(val_tokens):,} val tokens -> {val_path}")
    val_tokens.tofile(val_path)

    print("Pre-tokenization complete!")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize text corpus into binary .bin files.")
    parser.add_argument("--input", default="data/the-verdict.txt", help="Input text file path.")
    parser.add_argument("--output-dir", default="data/bin", help="Output directory for train.bin/val.bin.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of tokens for validation.")
    args = parser.parse_args()

    tokenize_corpus(Path(args.input), Path(args.output_dir), val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
