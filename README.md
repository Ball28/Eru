# Eru: Large Language Model (Built from Scratch)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Tests: 29 Passing](https://img.shields.io/badge/tests-29%20passing-brightgreen.svg)](tests/)

**Eru** is a clean, modular, fully tested PyTorch implementation of a generative causal language model built from first principles. It supports both classic GPT-2 architectures and modern state-of-the-art designs (RoPE, RMSNorm, SwiGLU, Grouped-Query Attention, and FlashAttention/SDPA), along with $O(T)$ Key-Value caching, real-time streaming, LoRA fine-tuning, and Direct Preference Optimization (DPO).

---

## Table of Contents
- [Quickstart (30 Seconds)](#quickstart-30-seconds)
- [Installation Guide](#installation-guide)
  - [Linux & macOS](#linux--macos)
  - [Windows](#windows)
  - [GPU Acceleration (CUDA / Apple Silicon MPS)](#gpu-acceleration-cuda--apple-silicon-mps)
- [Feature Matrix](#feature-matrix)
- [Core Workflows](#core-workflows)
  - [1. Real-Time Streaming Generation](#1-real-time-streaming-generation)
  - [2. Interactive Chat REPL (`eru chat`)](#2-interactive-chat-repl-eru-chat)
  - [3. Interactive Web Chat Interface](#3-interactive-web-chat-interface)
  - [4. Pretrain on Custom Text or Memory-Mapped Binary Datasets](#4-pretrain-on-custom-text-or-memory-mapped-binary-datasets)
  - [5. Modern Architecture (RoPE, RMSNorm, SwiGLU, GQA)](#5-modern-architecture-rope-rmsnorm-swiglu-gqa)
  - [6. Parameter-Efficient Fine-Tuning (LoRA from Scratch)](#6-parameter-efficient-fine-tuning-lora-from-scratch)
  - [7. Alignment with Direct Preference Optimization (DPO)](#7-alignment-with-direct-preference-optimization-dpo)
  - [8. Export Checkpoints to Hugging Face Format](#8-export-checkpoints-to-hugging-face-format)
- [Project Architecture](#project-architecture)
- [Running Tests](#running-tests)
- [Configuration Reference](#configuration-reference)
- [Attribution & License](#attribution--license)

---

## Quickstart (30 Seconds)

Clone the repo, install dependencies, and run text generation immediately:

```bash
# 1. Clone repository
git clone https://github.com/Ball28/Eru.git
cd Eru

# 2. Create and activate virtual environment
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 3. Install in editable mode
pip install -e .[dev]

# 4. Generate text with real-time streaming and KV-cache acceleration
eru generate --small --prompt "The secret to" --max-tokens 30 --stream
```

To launch an interactive streaming chat session:
```bash
eru chat --small
```

---

## Installation Guide

Eru works cross-platform on Linux, macOS, and Windows with Python 3.9 through 3.12.

### Linux & macOS

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install Eru with development tools
pip install --upgrade pip
pip install -e .[dev]
```

### Windows

```powershell
# 1. Open PowerShell and create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Upgrade pip and install Eru
python -m pip install --upgrade pip
pip install -e .[dev]
```

### GPU Acceleration (CUDA / Apple Silicon MPS)

- **NVIDIA GPU (CUDA):** Install official CUDA-enabled PyTorch:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -e .[dev]
  ```
- **Apple Silicon (M1/M2/M3/M4):** Eru automatically detects and utilizes Metal Performance Shaders (`mps`). No extra configuration needed.
- **CPU:** Out of the box, Eru automatically runs on CPU if no GPU is available. To install lightweight CPU PyTorch wheels directly:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -e .[dev]
  ```

---

## Feature Matrix

| Capability | Eru Implementation | Benefits |
| :--- | :--- | :--- |
| **Inference Acceleration** | Key-Value Cache (`KVCache`) | Reduces generation complexity from $O(T^2)$ to $O(T)$, giving a 5–20× speedup. |
| **Attention Optimization** | PyTorch Flash / SDPA | Uses `F.scaled_dot_product_attention` for $O(N)$ memory and fast kernel fusion. |
| **Positional Encoding** | RoPE (Rotary Position Embeddings) | Preserves relative distances and extrapolates to longer context windows. |
| **Normalization** | RMSNorm & LayerNorm | Fast mean-free root mean square normalization used in LLaMA / Mistral. |
| **Feed-Forward Networks** | SwiGLU & GELU | Gated linear units with SiLU activations for increased representation capacity. |
| **Attention Architectures** | GQA, MQA, and MHA | Grouped-Query Attention minimizes KV cache memory footprints. |
| **Advanced Sampling** | Top-$p$, Min-$P$, Repetition Penalty | Nucleus and dynamic threshold sampling prevent repetitive generation loops. |
| **Streaming Output** | `generate_stream` | Real-time token streaming for CLI and Web UIs. |
| **Scalable Data Pipeline**| Binary Memory-Mapped Datasets | Zero-RAM streaming of pre-tokenized corpora (`uint16` shards) via `numpy.memmap`. |
| **Parameter-Efficient Tuning** | LoRA from scratch | Fine-tune large base models with $<1\%$ trainable parameters and zero-latency merging. |
| **Preference Alignment** | Direct Preference Optimization (DPO)| Align models to human preference pairs without training a separate reward model. |
| **Ecosystem Export** | Hugging Face Exporter | Export directly to `config.json` and `pytorch_model.bin` for `transformers`. |

---

## Core Workflows

### 1. Real-Time Streaming Generation

Generate text autoregressively with token-by-token streaming, KV-cache acceleration, top-$p$, and repetition penalty:

```bash
eru generate --small \
  --prompt "Every effort moves you" \
  --max-tokens 50 \
  --temperature 0.8 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --stream
```

### 2. Interactive Chat REPL (`eru chat`)

Chat with your model in real time straight from your terminal:

```bash
eru chat --small --temperature 0.7 --top-p 0.9
```

### 3. Interactive Web Chat Interface

Launch a browser-based chat interface with real-time Server-Sent Events (SSE) token streaming. No external web frameworks required (uses standard library `http.server`):

```bash
python scripts/web_chat.py --port 8000 --small
```
Then navigate to `http://localhost:8000` in your browser.

### 4. Pretrain on Custom Text or Memory-Mapped Binary Datasets

#### Standard Pretraining:
Pretrain on text using causal sliding-window chunking:
```bash
eru train --data data/the-verdict.txt --config configs/gpt_small.yaml --epochs 5 --output checkpoints/model.pth
```

#### Scalable Memory-Mapped Pretraining:
For multi-gigabyte corpora (OpenWebText, FineWeb-Edu, TinyStories), pre-tokenize into binary `.bin` shards:
```bash
# 1. Convert text to binary token shards (uint16)
python scripts/prepare_dataset.py --input data/the-verdict.txt --output-dir data/bin

# 2. Train with zero memory overhead using MMapTokenDataset
python -c "
from eru.data.mmap_dataset import MMapTokenDataset
dataset = MMapTokenDataset('data/bin/train.bin', context_length=64)
print(f'Ready to stream {len(dataset):,} batches directly from disk.')
"
```

### 5. Modern Architecture (RoPE, RMSNorm, SwiGLU, GQA)

Train or evaluate modern architectures using declarative YAML configurations:

```bash
# configs/modern_small.yaml specifies:
# - pos_emb_type: "rope"
# - norm_type: "rmsnorm"
# - ffn_type: "swiglu"
# - n_kv_heads: 2 (Grouped-Query Attention)
# - use_sdpa: true
eru train --config configs/modern_small.yaml --data data/the-verdict.txt --epochs 3
```

### 6. Parameter-Efficient Fine-Tuning (LoRA from Scratch)

Wrap attention projection layers with low-rank adapters ($W_0 + \frac{\alpha}{r} BA$) and train $<1\%$ of model weights:

```python
from eru.model.gpt import GPTModel, GPT_CONFIG_124M
from eru.finetune.lora import apply_lora_to_model, save_lora_checkpoint

model = GPTModel(GPT_CONFIG_124M)
# Freeze base model and inject LoRA adapters into Q and V projections:
apply_lora_to_model(model, target_module_names=["W_query", "W_value"], rank=8, alpha=16.0)

# Save lightweight adapter weights:
save_lora_checkpoint("checkpoints/lora_adapter.pt", model)
```

### 7. Alignment with Direct Preference Optimization (DPO)

Align your instruction-tuned model using paired preference triplets `(prompt, chosen, rejected)` without training a separate reward model:

```python
import torch
from eru.finetune.dpo import dpo_loss

# Given token log probabilities under the active policy and reference model:
policy_chosen = torch.tensor([-1.2, -0.8])
policy_rejected = torch.tensor([-2.5, -2.1])
ref_chosen = torch.tensor([-1.5, -1.0])
ref_rejected = torch.tensor([-1.6, -1.2])

loss, chosen_rewards, rejected_rewards, accuracy = dpo_loss(
    policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
)
print(f"DPO Loss: {loss.item():.4f} | Accuracy: {accuracy.item() * 100:.1f}%")
```

### 8. Export Checkpoints to Hugging Face Format

Export any trained Eru checkpoint to standard Hugging Face `transformers` directory format (`config.json`, `pytorch_model.bin`, `generation_config.json`):

```python
from eru.model.gpt import GPTModel, GPT_CONFIG_124M
from eru.weights.hf_export import export_to_huggingface

model = GPTModel(GPT_CONFIG_124M)
export_to_huggingface(model, output_dir="checkpoints/hf_gpt2")
```

The exported model can now be loaded directly with `transformers`:
```python
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("checkpoints/hf_gpt2")
```

---

## Project Architecture

```
Eru/
├── configs/                             # Declarative YAML configurations
│   ├── gpt_small.yaml                   # Fast CPU-friendly model (64 ctx, 128 emb, 2 layers)
│   ├── gpt_124m.yaml                    # Full GPT-2 (1024 ctx, 768 emb, 12 layers)
│   ├── modern_small.yaml                # Modern LLaMA-style (RoPE, RMSNorm, SwiGLU, GQA)
│   ├── finetune_classification.yaml     # Classification fine-tuning settings
│   └── finetune_instruction.yaml        # Instruction tuning settings
├── src/eru/                             # Main Python package
│   ├── model/                           # Attention (SDPA/GQA), RoPE, Norm, SwiGLU, KVCache, GPTModel
│   ├── data/                            # Datasets (in-memory & binary memory-mapped), collators
│   ├── generation/                      # Sampling (top-k, top-p, min-p, repetition penalty), streaming
│   ├── finetune/                        # Classification, SFT, LoRA PEFT, DPO alignment
│   ├── training/                        # Loss, trainer loop with AMP, gradient clipping, cosine scheduler
│   ├── weights/                         # Weight loaders, TF converter, Hugging Face exporter
│   ├── tokenizer/                       # Tokenizers (SimpleTokenizerV1/V2, BPETokenizer)
│   ├── utils/                           # Device detection, random seed, config loader
│   └── cli.py                           # Unified CLI interface (`eru`)
├── scripts/                             # Executable wrapper scripts and web chat server
│   ├── web_chat.py                      # Standalone streaming web chat server
│   ├── prepare_dataset.py               # Pre-tokenization for memory-mapped datasets
│   ├── train_gpt.py                     # Training script wrapper
│   ├── generate_text.py                 # Text generation wrapper
│   ├── finetune_spam.py                 # Classification fine-tuning wrapper
│   └── finetune_instruction.py          # Instruction SFT wrapper
├── tests/                               # Comprehensive unit, parity, and regression test suite
└── data/                                # Sample datasets and tokenized binary data
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed architectural diagrams.

---

## Running Tests

Run the complete 29-test test suite using `pytest`:

```bash
pytest tests/
```

All characterization tests guarantee exact parity with the reference baseline outputs while covering all Stage 1, Stage 2, and Stage 3 extensions.

---

## Configuration Reference

Eru uses declarative YAML files in `configs/`. Key parameters:

| Parameter | Default (Small) | Default (124M) | Default (Modern) | Description |
| :--- | :--- | :--- | :--- | :--- |
| `vocab_size` | 50257 | 50257 | 50257 | Vocabulary size (BPE). |
| `context_length` | 64 | 1024 | 128 | Maximum sequence context window. |
| `emb_dim` | 128 | 768 | 128 | Hidden embedding dimension. |
| `n_heads` | 4 | 12 | 4 | Number of query attention heads. |
| `n_kv_heads` | null | null | 2 | Number of key/value heads for GQA (null = MHA). |
| `n_layers` | 2 | 12 | 2 | Number of stacked transformer blocks. |
| `pos_emb_type` | "learned" | "learned" | "rope" | Position embeddings: `"learned"` or `"rope"`. |
| `norm_type` | "layernorm" | "layernorm" | "rmsnorm" | Normalization: `"layernorm"` or `"rmsnorm"`. |
| `ffn_type` | "gelu" | "gelu" | "swiglu" | Feed-forward MLP: `"gelu"` or `"swiglu"`. |
| `use_sdpa` | false | false | true | Use PyTorch Flash / Scaled Dot-Product Attention. |
| `drop_rate` | 0.0 | 0.1 | 0.0 | Dropout rate. |
| `qkv_bias` | false | false | false | Query-Key-Value projection bias. |

---

## Attribution & License

- Licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
