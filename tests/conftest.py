"""
Test configuration and fixtures for Eru LLM test suite.
"""

import os
import sys
from pathlib import Path

# Add Eru/src to python sys.path
ERU_SRC = Path(__file__).resolve().parent.parent / "src"
if str(ERU_SRC) not in sys.path:
    sys.path.insert(0, str(ERU_SRC))
