"""
Download helpers for external datasets.
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def download_file(url: str, destination: str, backup_url: Optional[str] = None) -> None:
    """Download a file with optional backup URL fallback."""
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"File already exists: {destination}")
        return

    try:
        print(f"Downloading {url} -> {destination}...")
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        print(f"Downloaded: {destination}")
    except Exception as e:
        if backup_url:
            print(f"Primary download failed ({e}). Trying backup URL: {backup_url}...")
            with urllib.request.urlopen(backup_url) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            print(f"Downloaded from backup: {destination}")
        else:
            raise e


def download_spam_dataset(dest_dir: str = "data/spam") -> Path:
    """Download and extract SMS Spam Collection TSV."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    tsv_path = dest_path / "SMSSpamCollection.tsv"
    if tsv_path.exists():
        print(f"Spam TSV already exists at {tsv_path}")
        return tsv_path

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = dest_path / "sms_spam_collection.zip"
    download_file(url, str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    extracted_original = dest_path / "SMSSpamCollection"
    if extracted_original.exists():
        extracted_original.rename(tsv_path)

    if zip_path.exists():
        zip_path.unlink()

    return tsv_path


def download_instruction_data(
    dest_path: str = "data/instruction-data.json",
    url: str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json",
) -> List[Dict[str, str]]:
    """Download and return Alpaca-style instruction data."""
    path = Path(dest_path)
    if not path.exists():
        download_file(url, str(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
