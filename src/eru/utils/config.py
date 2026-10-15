"""
YAML configuration loader with standard dictionary parsing.
"""

from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a Python dictionary.
    Falls back to a basic key-value parser if PyYAML is not installed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        # Fallback simple parser for simple flat YAML configurations
        data: Dict[str, Any] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.split("#")[0].strip()
                if val.lower() == "true":
                    data[key] = True
                elif val.lower() == "false":
                    data[key] = False
                elif val.isdigit():
                    data[key] = int(val)
                else:
                    try:
                        data[key] = float(val)
                    except ValueError:
                        data[key] = val.strip("\"'")
        return data
