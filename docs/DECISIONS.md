# Architectural & Refactoring Decisions Log

**Project:** SimpleTokenizerV1 / Eru LLM  
**Branch:** `refactor/cohesive-structure`  
**Status:** Phase 2 (Planning)

This document records all architectural choices, duplicate resolutions, and safe defaults selected during the repository reorganization in accordance with the project rules.

---

## Decision 1: Directory Structure and Folder "Eru"

### Context
The user prompt specifies: *"turn this fragmented repository into a clean, cohesive, installable, well-documented Python project... and place all the new files and data in a new Folder called 'Eru'"*.

### Options Considered
- **Option A (Subpackage only):** Place the Python package inside `src/eru/`, keeping root folders (`configs/`, `data/`, `tests/`) at the repository root.
- **Option B (Complete Project in `Eru/` - RECOMMENDED):** Place the clean, self-contained, installable project root inside `Eru/`:
  - `Eru/pyproject.toml`, `Eru/requirements.txt`, `Eru/README.md`, `Eru/configs/`, `Eru/src/eru/`, `Eru/data/`, `Eru/scripts/`, `Eru/tests/`, `Eru/docs/`.
  - The repository root retains `archive/` for all superseded original files and root documentation linking into `Eru/`.
- **Option C (Project root is `Eru`, top-level repo renamed):** Flatten everything into the repository root and rename the package to `eru`.

### Decision
Implement **Option B** as the primary proposal: `Eru/` houses the complete, self-contained project, ensuring that existing files in the repository can be moved to `archive/` without collision, while all new code and data reside strictly in `Eru/`. In addition, create top-level root redirect documentation and convenience links so users entering the repository root immediately find the clean project.

---

## Decision 2: Canonical Multi-Head Self-Attention Implementation

### Context
The repository contains four implementations of attention:
1. `self_attention.py`: Step-by-step unvectorized script.
2. `SelfAttention_V1.py`: `SelfAttention_v1` using raw `nn.Parameter` tensors.
3. `SelfAttention_v2.py`: `SelfAttention_v2` (single-head `nn.Linear`), `CausalAttention` (causal mask + dropout), `MultiHeadAttentionWrapper` (sequential list of `CausalAttention`), and `MultiHeadAttention` (vectorized projection across heads).

### Decision
- **Canonical:** `MultiHeadAttention` from `SelfAttention_v2.py:103-146` is selected as the canonical implementation in `Eru/src/eru/model/attention.py`.
- **Educational modules:** `SelfAttention_v1`, `SelfAttention_v2`, and `CausalAttention` are preserved in `Eru/src/eru/model/educational_attention.py` for instructional value and regression testing.
- **Superseded files moved to archive:** `self_attention.py`, `SelfAttention_V1.py`, `SelfAttention_v2.py`.

---

## Decision 3: Canonical Transformer Block, LayerNorm, and GELU

### Context
`DummyGPTModel.py` contains placeholder classes (`DummyGPTModel`, `DummyTransformerBlock`, `DummyLayerNorm`) along with the true implementations of `LayerNorm`, `GELU`, `FeedForward`, and `TransformerBlock`. `GPTModel.py` imported these components from `DummyGPTModel.py`.

### Decision
- Consolidate the real neural network modules into canonical files:
  - `Eru/src/eru/model/normalization.py`: `LayerNorm` (with epsilon $1\times 10^{-5}$, trainable scale and shift).
  - `Eru/src/eru/model/activations.py`: `GELU`
  - `Eru/src/eru/model/feedforward.py`: `FeedForward` (expansion factor $4\times$, `GELU`, linear projection).
  - `Eru/src/eru/model/transformer_block.py`: `TransformerBlock` (pre-LayerNorm, residual connections, dropout, `MultiHeadAttention`, and `FeedForward`).
- `DummyGPTModel.py` is archived to `archive/DummyGPTModel.py`.

---

## Decision 4: Canonical GPTModel Architecture

### Context
`GPTModel.py` defines the standard GPT-2 architecture with token embeddings, positional embeddings, embedding dropout, stacked `TransformerBlock` modules, final `LayerNorm`, and output linear head.

### Decision
- The canonical `GPTModel` class will reside in `Eru/src/eru/model/gpt.py`.
- Support standard configurations (`gpt2-small (124M)`, `gpt2-medium (355M)`, `gpt2-large (774M)`, `gpt2-xl (1558M)`, and `gpt-small` CPU tiny test configuration).
- Strip all top-level side effects and execution logic.

---

## Decision 5: Canonical Text Generation and Decoding Strategies

### Context
Two versions of text generation exist:
1. `generate_text_simple` (in `GPTModel.py` and `finetuning.py`): Pure greedy argmax decoding.
2. `generate` (in `TextGen.py`, `test.py`, `DSDownload.py`): Supports temperature scaling, top-$k$ filtering, and `eos_id` stopping criteria.

### Decision
- Canonical implementation: `generate` in `Eru/src/eru/generation/sampling.py` (adapted from `test.py:82-109`).
- When `temperature=0.0`, `generate` automatically falls back to greedy argmax decoding.
- Also provide `generate_greedy` as an explicit convenience alias matching Raschka's `generate_text_simple` for zero-ambiguity backward compatibility.
- Token conversion utilities (`text_to_token_ids`, `token_ids_to_text`) are placed in `Eru/src/eru/tokenizer/utils.py`.

---

## Decision 6: Canonical Pretraining and Fine-Tuning Pipelines

### Context
The repository has three separate training domains:
1. **Language Model Pretraining:** `TextGen.py` and `test.py`.
2. **Classification Fine-Tuning:** `finetuning.py`.
3. **Instruction Fine-Tuning / SFT:** `DSDownload.py`.

### Decision
- Cleanly separate each pipeline into reusable library modules and executable CLI subcommands:
  - **Pretraining Trainer:** `Eru/src/eru/training/trainer.py` (canonicalized from `test.py`).
  - **Classification Fine-Tuning:** `Eru/src/eru/finetune/classification.py` (model head replacement, freezing backbone, accuracy evaluation).
  - **Instruction Fine-Tuning:** `Eru/src/eru/finetune/instruction.py` (instruction prompt formatting, padding with $-100$ cross-entropy mask, training).
  - **CLI Entry Point:** `Eru/src/eru/cli.py` exposing subcommands `train`, `finetune-spam`, `finetune-instruction`, and `generate`.

---

## Decision 7: Bug Handling Policy

### Context
Hard rule: *"Do NOT change model math, hyperparameter defaults, or tokenizer behavior unless it is an obvious bug. Report any suspected bug and ask me; do not silently fix it."*

### Decision & Bug Inventory
The following 6 bugs have been formally identified and will be resolved in the canonical implementation while being reported to the user for explicit approval at the Phase 2 checkpoint:
1. **`finetuning.py:234`**: Device typo `"cude"` $\to$ fixed to `"cuda"`.
2. **`finetuning.py:361`**: Context length dimension `model.pos_emb.weight.shape[1]` $\to$ fixed to `model.pos_emb.weight.shape[0]`.
3. **`SimpleTokenizerV2.py:33`**: Decode punctuation regex $\to$ fixed to `r'\s([,.:;?_!"()\'])'`.
4. **`GPTDatasetV1.py:31`**: `num_workers=0` hardcoded in `DataLoader` initialization $\to$ fixed to respect passed `num_workers` parameter.
5. **`DSDownload.py:306-328`**: Redundant loop nesting evaluating test set 3 times $\to$ unindented to run once.
6. **`main.py`**: Broken imports, recursion bug, and syntax errors $\to$ archived; canonical CLI provides working functionality.

---

## Decision 8: Pretrained Weights Loading Dependency Decoupling

### Context
`gpt_download.py` uses `tensorflow` to load OpenAI checkpoint variables. Installing TensorFlow is problematic on certain OS/architecture configurations (e.g. Intel macOS or machines lacking prebuilt wheels for recent Python versions).

### Decision
- Keep `Eru/src/eru/weights/tf_loader.py` for compatibility with the book's original workflow, but mark TensorFlow as an optional extra dependency in `pyproject.toml` (`pip install -e ".[tensorflow]"`).
- The core package (`torch`, `tiktoken`, `pyyaml`, `pandas`, `tqdm`) can be installed and used without TensorFlow.
- Provide a small, fast CPU model configuration (`configs/gpt_small.yaml`) that trains and generates in seconds on any machine without downloading weights or needing TensorFlow.

---

## Decision 9: Data File Organization

### Context
The repository currently has data files at root (`the-verdict.txt`, `train.csv`, `validation.csv`, `test.csv`).

### Decision
- Move active datasets into `Eru/data/`:
  - `Eru/data/the-verdict.txt` (sample pretraining text)
  - `Eru/data/spam/train.csv`, `Eru/data/spam/validation.csv`, `Eru/data/spam/test.csv` (classification datasets)
  - Provide download/preparation scripts in `Eru/src/eru/data/download.py` to fetch SMS Spam Collection and `instruction-data.json` on demand.
