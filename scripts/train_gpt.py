#!/usr/bin/env python3
"""
Thin wrapper script to pretrain a GPT model.
Run from the Eru/ project root:
    python scripts/train_gpt.py --config configs/gpt_small.yaml
"""

import sys
from pathlib import Path

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eru.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["train", "--small"])
    elif sys.argv[1] != "train":
        sys.argv.insert(1, "train")
    main()
