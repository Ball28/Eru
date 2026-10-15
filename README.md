# Eru: Build a Large Language Model (From Scratch)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**Eru** is a clean, cohesive, fully tested PyTorch implementation of a Generative Pretrained Transformer (GPT) built from scratch
Every component—tokenization, multi-head causal attention, pre-norm transformer blocks, pretraining loops, classification fine-tuning, instruction tuning (SFT), and text generation—is modular, type-hinted, and accompanied by automated tests.

---

## Table of Contents
- [Quickstart (30 Seconds)](#quickstart-30-seconds)
- [Installation Guide](#installation-guide)
  - [Linux & macOS](#linux--macos)
  - [Windows](#windows)
  - [GPU Acceleration (CUDA / Apple Silicon MPS)](#gpu-acceleration-cuda--apple-silicon-mps)
- [Core Workflows](#core-workflows)
  - [1. Pretrain a Small GPT Model](#1-pretrain-a-small-gpt-model)
  - [2. Generate Text Completions](#2-generate-text-completions)
  - [3. Fine-Tune for SMS Spam Classification](#3-fine-tune-for-sms-spam-classification)
  - [4. Instruction Fine-Tuning (SFT)](#4-instruction-fine-tuning-sft)
- [Project Architecture](#project-architecture)
- [Running Tests](#running-tests)
- [Configuration Reference](#configuration-reference)
- [Attribution & License](#attribution--license)

---

## Quickstart (30 Seconds)

Train a small GPT model on your CPU and generate text immediately:

```bash
# Option A — clone the full repo and enter the Eru project root:
git clone https://github.com/Ball28/Eru.git
cd Eru

# Option B — download only the Eru/ folder (it is fully self-contained):
# cd /path/to/downloaded/Eru

# Install package
pip install -e .

# Train tiny GPT model on sample text (runs on CPU in ~15 seconds)
python scripts/train_gpt.py --config configs/gpt_small.yaml --output checkpoints/gpt_small.pth

# Generate text completion
python scripts/generate_text.py --prompt "Every effort moves you" --weights checkpoints/gpt_small.pth --config configs/gpt_small.yaml
```

---

## Installation Guide

Eru works cross-platform on Linux, macOS, and Windows with Python 3.9 through 3.12.

### Linux & macOS

1. Open your terminal and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Upgrade `pip` and install Eru in editable development mode:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

### Windows

1. Open PowerShell or Command Prompt:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Upgrade `pip` and install:
   ```powershell
   python -m pip install --upgrade pip
   pip install -e .
   ```

### GPU Acceleration (CUDA / Apple Silicon MPS)

- **NVIDIA GPU (CUDA):** If you have an NVIDIA GPU, install the official CUDA-enabled PyTorch build:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -e .
  ```
- **Apple Silicon (M1/M2/M3/M4):** Eru automatically detects and utilizes Metal Performance Shaders (`mps`). No extra configuration needed.
- **CPU:** Out of the box, Eru automatically runs on CPU if no GPU is available.

---

## Core Workflows

### 1. Pretrain a Small GPT Model

Pretrain a language model using sliding-window causal chunking on Edith Wharton's "The Verdict" (`data/the-verdict.txt`):

```bash
# Using CLI
eru train --config configs/gpt_small.yaml --output checkpoints/model.pth

# Or using the script wrapper
python scripts/train_gpt.py --config configs/gpt_small.yaml --output checkpoints/model.pth
```

To train the full 124M parameter GPT-2 architecture:
```bash
eru train --config configs/gpt_124m.yaml --output checkpoints/gpt_124m.pth
```

### 2. Generate Text Completions

Generate text with temperature scaling and top-$k$ sampling:

```bash
# Using CLI
eru generate --prompt "Every effort moves you" --weights checkpoints/model.pth --config configs/gpt_small.yaml --temperature 0.8 --top-k 25

# Or using script
python scripts/generate_text.py --prompt "The secret was" --weights checkpoints/model.pth --config configs/gpt_small.yaml
```

### 3. Fine-Tune for SMS Spam Classification

Fine-tune a GPT model to classify SMS messages as spam or not spam:

```bash
# Using CLI
eru finetune-spam --config configs/finetune_classification.yaml --output checkpoints/spam_classifier.pth

# Or using script
python scripts/finetune_spam.py --train-csv data/spam/train.csv --val-csv data/spam/validation.csv
```

### 4. Instruction Fine-Tuning (SFT)

Fine-tune a model on Alpaca-style instruction pairs:

```bash
python scripts/finetune_instruction.py
```

---

## Project Architecture

```
Eru/
├── configs/                             # Declarative YAML configurations
│   ├── gpt_small.yaml                   # Fast CPU-friendly model (64 ctx, 128 emb, 2 layers)
│   ├── gpt_124m.yaml                    # Full GPT-2 (1024 ctx, 768 emb, 12 layers)
│   ├── finetune_classification.yaml     # Classification fine-tuning settings
│   └── finetune_instruction.yaml        # Instruction tuning settings
├── src/eru/                             # Main Python package
│   ├── tokenizer/                       # Tokenizers (SimpleTokenizerV1/V2, BPETokenizer)
│   ├── data/                            # Datasets, dataloaders, collators, downloaders
│   ├── model/                           # Attention, LayerNorm, GELU, TransformerBlock, GPTModel
│   ├── training/                        # Loss, trainer loop, checkpointing
│   ├── finetune/                        # Spam classifier & instruction tuning
│   ├── generation/                      # Temperature/top-k sampling & greedy generation
│   ├── weights/                         # Pure PyTorch weight loading & TF checkpoint converters
│   ├── utils/                           # Config, device detection, seeds, plotting
│   └── cli.py                           # Unified CLI interface (`eru`)
├── scripts/                             # Executable wrapper scripts
├── tests/                               # Automated unit & golden parity test suite
├── data/                                # Local pretraining text & classification CSVs
└── checkpoints/                         # Gitignored output directory for saved weights
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed architectural diagrams.

---

## Running Tests

Run the full test suite using Python's built-in test discovery:

```bash
python -m unittest discover tests
```

Or with `pytest`:
```bash
pip install pytest
pytest tests/
```

All golden characterization tests guarantee exact parity with the reference outputs from Sebastian Raschka's book.

---

## Configuration Reference

Eru uses declarative YAML files in `configs/`. Key parameters:

| Parameter | Default (Small) | Default (124M) | Description |
| :--- | :--- | :--- | :--- |
| `vocab_size` | 50257 | 50257 | Vocabulary size (BPE). |
| `context_length` | 64 | 1024 | Maximum sequence context window. |
| `emb_dim` | 128 | 768 | Hidden embedding dimension. |
| `n_heads` | 4 | 12 | Number of multi-head attention heads. |
| `n_layers` | 2 | 12 | Number of stacked transformer blocks. |
| `drop_rate` | 0.0 | 0.1 | Dropout rate. |
| `qkv_bias` | false | false | Query-Key-Value projection bias. |

---

## Attribution & License

- Licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
