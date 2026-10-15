#!/usr/bin/env python3
"""
Thin wrapper script to fine-tune GPT for SMS spam classification.
Run from the Eru/ project root:
    python scripts/finetune_spam.py --train-csv data/spam/train.csv --val-csv data/spam/validation.csv
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eru.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["finetune-spam"])
    elif sys.argv[1] != "finetune-spam":
        sys.argv.insert(1, "finetune-spam")
    main()
