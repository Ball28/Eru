#!/usr/bin/env python3
"""
Thin wrapper script to generate text completions from a model.
Run from the Eru/ project root:
    python scripts/generate_text.py --prompt "Every effort moves you" --small
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from eru.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["generate", "--prompt", "Every effort moves you", "--small"])
    elif sys.argv[1] != "generate":
        sys.argv.insert(1, "generate")
    main()
