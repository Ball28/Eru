"""
TensorFlow GPT-2 checkpoint downloader and parameter extractor.
"""

import json
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple


def download_file(url: str, destination: str, backup_url: Optional[str] = None) -> bool:
    """Download checkpoint file with progress reporting and backup URL fallback."""
    if os.path.exists(destination):
        local_size = os.path.getsize(destination)
        if local_size > 0:
            print(f"File already exists: {destination}")
            return True

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    urls_to_try = [url]
    if backup_url:
        urls_to_try.append(backup_url)

    for attempt_url in urls_to_try:
        try:
            print(f"Downloading {attempt_url} -> {destination}...")
            urllib.request.urlretrieve(attempt_url, destination)
            return True
        except Exception as e:
            print(f"Failed to download from {attempt_url}: {e}")

    return False


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Extract GPT-2 parameters from a TensorFlow checkpoint directory into a nested dictionary."""
    import numpy as np
    import tensorflow as tf

    params: Dict[str, Any] = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # Skip 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def download_and_load_gpt2(model_size: str = "124M", models_dir: str = "gpt2") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Download OpenAI GPT-2 TensorFlow checkpoint and extract its parameters."""
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}, got: {model_size}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        backup_url = f"{backup_base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    import tensorflow as tf
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    with open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params
