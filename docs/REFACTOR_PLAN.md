# Refactoring Plan: Eru LLM Project

**Project Name:** Eru (LLM From Scratch)  
**Branch:** `refactor/cohesive-structure`  
**Reference:** Sebastian Raschka, *Build a Large Language Model (From Scratch)*  
**Date:** October 2026  
**Status:** Awaiting User Approval (Phase 2 Checkpoint)

---

## 1. Executive Overview

This plan details the reorganization of the experimental codebase into a cohesive, installable, tested Python package called **`eru`**, located within a dedicated project directory **`Eru/`**. The refactoring preserves all existing mathematical behaviors, architectures, and hyperparameters while consolidating duplicate implementations, fixing confirmed syntax/regex bugs, eliminating global import side effects, and adding standard packaging (`pyproject.toml`, `requirements.txt`, CLI commands, and automated regression tests).

No files will be permanently deleted: all superseded files will be moved to `archive/` at the repository root and cataloged in `archive/README.md`.

---

## 2. Proposed Target Structure

```
SimpleTokenizerV1/                           # Git Repository Root
├── archive/                                 # Superseded originals (preserved, not deleted)
│   ├── README.md                            # Inventory and history of all archived files
│   ├── BPETokenizer.py
│   ├── DSDownload.py
│   ├── DummyGPTModel.py
│   ├── GPTDatasetV1.py
│   ├── GPTModel.py
│   ├── SelfAttention_V1.py
│   ├── SelfAttention_v2.py
│   ├── SimpleTokenizerV1.py
│   ├── SimpleTokenizerV2.py
│   ├── TextGen.py
│   ├── dataset.py
│   ├── finetuning.py
│   ├── gpt_download.py
│   ├── main.py
│   ├── self_attention.py
│   └── test.py
├── docs/                                    # Repository-level documentation & audits
│   ├── AUDIT.md                             # Phase 1 audit
│   ├── REFACTOR_PLAN.md                     # Phase 2 plan (this document)
│   └── DECISIONS.md                         # Architectural decisions & bug log
├── Eru/                                     # Clean, installable project root ("Eru")
│   ├── README.md                            # Complete quickstart, install, and usage guide
│   ├── LICENSE                              # Project license (Apache 2.0 / MIT)
│   ├── NOTICE.md                            # Attribution to Sebastian Raschka & OpenAI
│   ├── pyproject.toml                       # Modern PEP 621 package specification
│   ├── requirements.txt                     # Pinned direct dependencies
│   ├── .gitignore                           # Clean gitignore excluding __pycache__, checkpoints
│   ├── .gitattributes                       # Line ending normalization
│   ├── configs/                             # Declarative YAML configurations
│   │   ├── gpt_small.yaml                   # Tiny model (fast CPU training & testing in seconds)
│   │   ├── gpt_124m.yaml                    # Full GPT-2 (124M) configuration
│   │   ├── finetune_classification.yaml     # SMS spam classification fine-tuning config
│   │   └── finetune_instruction.yaml        # Alpaca-style instruction fine-tuning config
│   ├── src/
│   │   └── eru/                             # Core Python package namespace
│   │       ├── __init__.py                  # Package exports and version
│   │       ├── tokenizer/                   # Tokenization modules
│   │       │   ├── __init__.py
│   │       │   ├── simple.py                # SimpleTokenizerV1 & SimpleTokenizerV2 (educational)
│   │       │   ├── bpe.py                   # Production BPE tokenizer wrapper (tiktoken)
│   │       │   └── utils.py                 # text_to_token_ids, token_ids_to_text
│   │       ├── data/                        # Datasets & DataLoader helpers
│   │       │   ├── __init__.py
│   │       │   ├── pretrain_dataset.py      # GPTDatasetV1 & sliding-window causal loader
│   │       │   ├── spam_dataset.py          # Balanced SMS Spam classification dataset
│   │       │   ├── instruction_dataset.py   # Alpaca-format instruction dataset
│   │       │   ├── collate.py               # Dynamic padding & cross-entropy ignore index collator
│   │       │   └── download.py              # Download utilities for datasets & text
│   │       ├── model/                       # Neural network architecture components
│   │       │   ├── __init__.py
│   │       │   ├── activations.py           # GELU activation (exact tanh approximation)
│   │       │   ├── normalization.py         # LayerNorm implementation
│   │       │   ├── attention.py             # Canonical MultiHeadAttention (vectorized)
│   │       │   ├── feedforward.py           # FeedForward module (4x expansion)
│   │       │   ├── transformer_block.py     # TransformerBlock (residual pre-norm)
│   │       │   ├── gpt.py                   # Canonical GPTModel architecture
│   │       │   └── educational_attention.py # SelfAttention_v1, SelfAttention_v2, CausalAttention
│   │       ├── training/                    # Pretraining & general training engine
│   │       │   ├── __init__.py
│   │       │   ├── loss.py                  # calc_loss_batch, calc_loss_loader
│   │       │   ├── trainer.py               # train_model_simple, evaluate_model
│   │       │   └── checkpoint.py            # Checkpoint save/load helpers
│   │       ├── finetune/                    # Fine-tuning pipelines
│   │       │   ├── __init__.py
│   │       │   ├── classification.py        # Spam classifier fine-tuner & accuracy evaluation
│   │       │   └── instruction.py           # Instruction SFT trainer & prompt formatting
│   │       ├── generation/                  # Generation & decoding strategies
│   │       │   ├── __init__.py
│   │       │   ├── sampling.py              # Canonical generate() (temperature, top-k, eos_id)
│   │       │   └── greedy.py                # generate_greedy() (fast argmax decoding)
│   │       ├── weights/                     # Pretrained GPT-2 weight loading
│   │       │   ├── __init__.py
│   │       │   ├── loader.py                # Pure PyTorch weight assignment into GPTModel
│   │       │   └── tf_download.py           # Optional TF checkpoint downloader & extractor
│   │       ├── utils/                       # Shared utilities
│   │       │   ├── __init__.py
│   │       │   ├── config.py                # YAML configuration loader & dataclass validation
│   │       │   ├── device.py                # Robust device selection (CUDA, MPS, CPU)
│   │       │   ├── plotting.py              # Loss and metric visualization
│   │       │   └── seed.py                  # Seed management for reproducibility
│   │       └── cli.py                       # Unified CLI: eru train, finetune, generate
│   ├── scripts/                             # Executable entry-point scripts
│   │   ├── train_gpt.py                     # Pretrain a GPT model
│   │   ├── finetune_spam.py                 # Fine-tune classifier on SMS spam
│   │   ├── finetune_instruction.py          # Fine-tune model on instructions
│   │   └── generate_text.py                 # Generate text completions from prompt/checkpoint
│   ├── notebooks/                           # Educational Jupyter notebooks
│   │   ├── 01_tokenization_demo.ipynb       # Cleaned Chapter 2 walkthrough
│   │   ├── 02_attention_mechanics.ipynb     # Cleaned Chapter 3 walkthrough
│   │   └── 03_gpt_training_demo.ipynb       # Cleaned Chapter 4–5 walkthrough
│   ├── tests/                               # Comprehensive automated test suite
│   │   ├── conftest.py                      # Test fixtures & seed utilities
│   │   ├── test_tokenizer.py                # SimpleTokenizer roundtrips & BPE tests
│   │   ├── test_attention.py                # Attention shapes, masking & numeric outputs
│   │   ├── test_model.py                    # GPTModel forward pass, parameter counts, shapes
│   │   ├── test_training.py                 # Few-step deterministic loss decrease test
│   │   ├── test_generation.py               # Deterministic text generation tests
│   │   └── test_finetune.py                 # Classification & instruction collate tests
│   ├── data/                                # Active data directory
│   │   ├── README.md                        # Data guide and download instructions
│   │   ├── the-verdict.txt                  # Pretraining sample text
│   │   └── spam/                            # Processed spam classification datasets
│   │       ├── train.csv
│   │       ├── validation.csv
│   │       └── test.csv
│   ├── checkpoints/                         # Gitignored output directory for models
│   │   └── .gitkeep
│   └── docs/                                # Project documentation
│       └── ARCHITECTURE.md                  # Component diagrams & dataflow
└── README.md                                # Root README pointing to Eru/ and archive/
```

---

## 3. Rename & Move Map Table

| Original Path | Target Path | Action | Rationale |
| :--- | :--- | :--- | :--- |
| `the-verdict.txt` | `Eru/data/the-verdict.txt` | Copied to Eru, original moved to `archive/` | Canonical pretraining data for Edith Wharton sample text. |
| `train.csv` | `Eru/data/spam/train.csv` | Copied to Eru, original moved to `archive/` | Pre-split balanced training data for spam classification. |
| `validation.csv` | `Eru/data/spam/validation.csv` | Copied to Eru, original moved to `archive/` | Pre-split validation data for spam classification. |
| `test.csv` | `Eru/data/spam/test.csv` | Copied to Eru, original moved to `archive/` | Pre-split test data for spam classification. |
| `SimpleTokenizerV1.py` | `archive/SimpleTokenizerV1.py` | Moved to archive | Superseded by canonical `Eru/src/eru/tokenizer/simple.py`. |
| `SimpleTokenizerV2.py` | `archive/SimpleTokenizerV2.py` | Moved to archive | Superseded by canonical `Eru/src/eru/tokenizer/simple.py` (with decode bug fixed). |
| `BPETokenizer.py` | `archive/BPETokenizer.py` | Moved to archive | Superseded by canonical `Eru/src/eru/tokenizer/bpe.py`. |
| `self_attention.py` | `archive/self_attention.py` | Moved to archive | Scratchpad script superseded by `Eru/src/eru/model/attention.py`. |
| `SelfAttention_V1.py` | `archive/SelfAttention_V1.py` | Moved to archive | Superseded by canonical `Eru/src/eru/model/attention.py`. |
| `SelfAttention_v2.py` | `archive/SelfAttention_v2.py` | Moved to archive | Superseded by canonical `Eru/src/eru/model/attention.py`. |
| `DummyGPTModel.py` | `archive/DummyGPTModel.py` | Moved to archive | Modularized into `activations.py`, `normalization.py`, `feedforward.py`, and `transformer_block.py`. |
| `GPTDatasetV1.py` | `archive/GPTDatasetV1.py` | Moved to archive | Superseded by canonical `Eru/src/eru/data/pretrain_dataset.py`. |
| `GPTModel.py` | `archive/GPTModel.py` | Moved to archive | Superseded by canonical `Eru/src/eru/model/gpt.py` and `Eru/src/eru/weights/loader.py`. |
| `TextGen.py` | `archive/TextGen.py` | Moved to archive | Modularized into `trainer.py`, `sampling.py`, and CLI scripts. |
| `test.py` | `archive/test.py` | Moved to archive | Converted into unit tests (`Eru/tests/`) and canonical trainer/sampling modules. |
| `main.py` | `archive/main.py` | Moved to archive | Broken script superseded by `Eru/scripts/generate_text.py` and CLI. |
| `dataset.py` | `archive/dataset.py` | Moved to archive | Superseded by `Eru/src/eru/data/download.py`. |
| `finetuning.py` | `archive/finetuning.py` | Moved to archive | Modularized into `Eru/src/eru/finetune/classification.py` and `Eru/scripts/finetune_spam.py`. |
| `DSDownload.py` | `archive/DSDownload.py` | Moved to archive | Modularized into `Eru/src/eru/finetune/instruction.py` and `Eru/scripts/finetune_instruction.py`. |
| `gpt_download.py` | `archive/gpt_download.py` | Moved to archive | Modularized into `Eru/src/eru/weights/tf_download.py` without duplication. |
| `__pycache__/*` | Removed from git tracking | Removed | Bytecode should never be tracked in source control. |
| `requirements.txt` | `Eru/requirements.txt` | Replaced with clean pinned dependencies | Clean dependencies without platform conflicts. |
| `README.md` (root) | `README.md` (root) | Rewritten | Root guide linking to `Eru/` and explaining repo layout. |

---

## 4. Duplicate Resolution Table

| Component | Duplicate Locations | Canonical Implementation & File | Rationale |
| :--- | :--- | :--- | :--- |
| **Tokenizers** | `SimpleTokenizerV1.py`, `SimpleTokenizerV2.py`, `BPETokenizer.py` | `Eru/src/eru/tokenizer/simple.py` (`SimpleTokenizerV1`, `SimpleTokenizerV2`) & `bpe.py` (`BPETokenizer`) | Consolidates both educational custom tokenizers with bug-fixed decoding, alongside a robust `tiktoken` wrapper. |
| **Multi-Head Attention** | `SelfAttention_v2.py:103-146`, `MultiHeadAttentionWrapper` (`SelfAttention_v2:95`), `SelfAttention_v1` (`SelfAttention_V1`) | `Eru/src/eru/model/attention.py` (`MultiHeadAttention`) | Vectorized multi-head implementation matching GPT-2 weight loading. Educational variants moved to `educational_attention.py`. |
| **LayerNorm** | `DummyGPTModel.py:81-91` (`LayerNorm`), `DummyGPTModel.py:41` (`DummyLayerNorm`) | `Eru/src/eru/model/normalization.py` (`LayerNorm`) | Standard PyTorch module with trainable scale ($\gamma$) and shift ($\beta$), unbiased variance computation, and epsilon $1\times 10^{-5}$. |
| **Activation Function** | `DummyGPTModel.py:100-104` (`GELU`) | `Eru/src/eru/model/activations.py` (`GELU`) | Exact tanh approximation matching GPT-2 specification ($0.5 x [1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3))]$). |
| **FeedForward** | `DummyGPTModel.py:123-132` (`FeedForward`) | `Eru/src/eru/model/feedforward.py` (`FeedForward`) | Clean $4\times$ expansion with GELU and linear projection. |
| **Transformer Block** | `DummyGPTModel.py:192-221` (`TransformerBlock`), `DummyTransformerBlock` | `Eru/src/eru/model/transformer_block.py` (`TransformerBlock`) | Real pre-norm transformer block with residual connections and shortcut dropout. |
| **GPT Model** | `GPTModel.py:17-41` (`GPTModel`), `DummyGPTModel.py:15` (`DummyGPTModel`) | `Eru/src/eru/model/gpt.py` (`GPTModel`) | Production-ready GPT-2 implementation with modular configuration dictionary support. |
| **Text Generation** | `GPTModel.py:87-96` (`generate_text_simple`), `TextGen.py:27-51` (`generate`), `test.py:82-109` (`generate`), `finetuning.py:175` | `Eru/src/eru/generation/sampling.py` (`generate`) & `greedy.py` (`generate_greedy`) | `generate` from `test.py` is the most complete (temperature, top-$k$, `eos_id`), automatically reducing to greedy argmax when `temperature=0.0`. |
| **Token Helpers** | `text_to_token_ids`, `token_ids_to_text` (in `GPTModel.py`, `TextGen.py`, `test.py`, `main.py`) | `Eru/src/eru/tokenizer/utils.py` | Single shared implementation with optional batch dimension and special token handling. |
| **Pretraining Loss & Eval** | `TextGen.py:140-163`, `test.py:27-52` | `Eru/src/eru/training/loss.py` (`calc_loss_batch`, `calc_loss_loader`, `evaluate_model`) | Canonical loss evaluation based on `test.py` with support for partial validation iterations. |
| **Pretraining Loop** | `TextGen.py:172-195`, `test.py:54-79` (`train_model_simple`) | `Eru/src/eru/training/trainer.py` (`train_model_simple`) | Clean, side-effect-free training function returning metrics history. |
| **Classification Fine-Tuning** | `finetuning.py:213-308` | `Eru/src/eru/finetune/classification.py` (`SpamClassifier`, `train_classifier_simple`, `calc_accuracy_loader`) | Eliminates hardcoded device strings and shapes, providing modular classification training and evaluation. |
| **Instruction SFT** | `DSDownload.py:117-149`, `190-282` | `Eru/src/eru/finetune/instruction.py` | Fixes redundant nested loop, implements dynamic padding with $-100$ cross-entropy mask, and modularizes prompt formatting. |
| **Weight Loading** | `GPTModel.py:116-147`, `main.py:5-36`, `gpt_download.py` | `Eru/src/eru/weights/loader.py` (`load_weights_into_gpt`, `assign`) | Pure weight loading decoupled from text generation and side-effects. |

---

## 5. Coding & Naming Conventions

To ensure uniform code quality and maintainability:
- **Module Names:** Strict `snake_case` (e.g. `pretrain_dataset.py`, `transformer_block.py`, `train_gpt.py`). No camelCase file names, numbers without context, or vague names (`test.py`, `main.py`, `DSDownload.py` are deprecated).
- **Class Names:** Strict `PascalCase` (e.g. `GPTModel`, `MultiHeadAttention`, `LayerNorm`, `SpamDataset`).
- **Function & Variable Names:** Strict `snake_case` (e.g. `calc_loss_batch`, `generate_greedy`, `custom_collate_fn`).
- **Constants & Defaults:** `UPPER_SNAKE_CASE` (e.g. `DEFAULT_CONFIG_124M`, `PAD_TOKEN_ID = 50256`).
- **Type Annotations:** Full type hints on all public functions, classes, and methods (e.g. `torch.Tensor`, `nn.Module`, `Optional[int]`, `Dict[str, Any]`).
- **Docstrings:** Google-style or standard Sphinx docstrings explaining inputs, outputs, tensor shapes, and book references.
- **Device Handling:** Centralized through `get_device(device_name=None)` returning `torch.device("cuda" | "mps" | "cpu")`.
- **Reproducibility:** Seed setting helper `set_seed(seed: int = 123)` configuring PyTorch, NumPy, and random.

---

## 6. Execution Plan & Phase Sequence

### Step 3.1: Golden Characterization Tests (Phase 3 Prep)
Before moving or modifying any files, write and run golden characterization tests against the **original unrefactored code** in the current branch:
1. `SimpleTokenizerV1` and `SimpleTokenizerV2` encode/decode test on fixed sample sentences.
2. `GPTDatasetV1` batch tensor shape test on `the-verdict.txt`.
3. `MultiHeadAttention` seeded forward pass output tensor values and shape `(2, 4, 2)`.
4. `TransformerBlock` and `GPTModel` seeded forward pass logits on sample token indices.
5. `train_model_simple` seeded 2-step loss decrease verification.
6. `generate` deterministic output token sequence with seed 123.

### Step 3.2: Create Directory Layout & Scaffolding
Create `Eru/`, `archive/`, `Eru/src/eru/`, `Eru/configs/`, `Eru/scripts/`, `Eru/tests/`, `Eru/data/`.

### Step 3.3: Build Canonical Modules
Extract, consolidate, type-hint, and document each component into `Eru/src/eru/`:
- `tokenizer/`: `simple.py`, `bpe.py`, `utils.py`
- `data/`: `pretrain_dataset.py`, `spam_dataset.py`, `instruction_dataset.py`, `collate.py`, `download.py`
- `model/`: `activations.py`, `normalization.py`, `attention.py`, `feedforward.py`, `transformer_block.py`, `gpt.py`, `educational_attention.py`
- `training/`: `loss.py`, `trainer.py`, `checkpoint.py`
- `finetune/`: `classification.py`, `instruction.py`
- `generation/`: `sampling.py`, `greedy.py`
- `weights/`: `loader.py`, `tf_download.py`
- `utils/`: `config.py`, `device.py`, `plotting.py`, `seed.py`
- `cli.py`: Unified command line interface

### Step 3.4: Create Configurations & Scripts
- Create `configs/gpt_small.yaml` (fast CPU config: 2 layers, 4 heads, emb_dim 128, context 64).
- Create `configs/gpt_124m.yaml` (standard GPT-2 124M: 12 layers, 12 heads, emb_dim 768, context 1024).
- Create `configs/finetune_classification.yaml` and `configs/finetune_instruction.yaml`.
- Create executable scripts in `Eru/scripts/`.

### Step 3.5: Move Superseded Files to Archive
Move original unrefactored files to `archive/`, keeping their exact original names, and populate `archive/README.md`.

### Step 3.6: Verify Tests
Run the golden characterization tests against the refactored `eru` package, verifying 100% pass rate and exact numerical parity.

### Step 3.7: Packaging & Documentation
- Author `Eru/pyproject.toml` and `Eru/requirements.txt`.
- Author `Eru/README.md` with foolproof, step-by-step instructions for installation, training, fine-tuning, and text generation.
- Author `NOTICE.md` providing full attribution to Sebastian Raschka and Manning Publications.

---

## 7. Risks, Bug Confirmations & Items for User Decision

### 7.1 Confirmed Bugs to be Fixed in Canonical Implementation
The following bugs will be fixed in the new canonical code while leaving the originals archived:
1. **Device typo in `finetuning.py:234`:** `"cude"` replaced with `"cuda"`.
2. **Context length dimension in `finetuning.py:361`:** `pos_emb.weight.shape[1]` replaced with `pos_emb.weight.shape[0]`.
3. **Punctuation regex in `SimpleTokenizerV2.py:33`:** replaced with `r'\s([,.:;?_!"()\'])'`.
4. **Hardcoded `num_workers=0` in `GPTDatasetV1.py:31`:** replaced with parameter `num_workers`.
5. **Redundant nested loop in `DSDownload.py:306-328`:** unindented to execute once.
6. **Syntax & recursion errors in `main.py`:** superseded by clean CLI and scripts.

### 7.2 Items for User Decision / Clarification
1. **Folder Placement Confirmation:**
   - Proposal: Place the new project root inside `Eru/` (with package `src/eru/` and data in `Eru/data/`), while `archive/` resides at the repo root.
   - *Alternative:* Place `src/eru/` at the repository root and store only new data files in `Eru/`.
   - *Recommendation:* Approve Proposal (Option B) for the cleanest separation.
2. **Pretrained Weights & TensorFlow Dependency:**
   - OpenAI GPT-2 weights in Raschka's original code require `tensorflow` to read TF v1 checkpoints.
   - Proposal: Keep the TF checkpoint loader in `src/eru/weights/tf_download.py` as an optional extra (`pip install -e ".[tensorflow]"`), while providing a CPU-friendly small model (`gpt_small.yaml`) that requires only PyTorch and runs out-of-the-box on any machine.
   - *Recommendation:* Approve Proposal.
3. **Data Files Strategy:**
   - `the-verdict.txt` (20.5 KB) and the pre-split spam CSVs (~160 KB total) are small and convenient to include in `Eru/data/`.
   - Proposal: Include them directly in `Eru/data/` for immediate out-of-the-box execution, while providing `download.py` for automated re-download if desired.
   - *Recommendation:* Approve Proposal.

---

> [!IMPORTANT]
> **STOPPING FOR USER APPROVAL:**  
> In accordance with the prompt rules:  
> *"Stop and wait for my approval at the end of Phase 2 (the plan). Do not start Phase 3 until I say 'approved'."*  
> No Phase 3 refactoring or file movements will proceed until explicit approval is received.
