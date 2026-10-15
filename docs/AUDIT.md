# Codebase Audit: SimpleTokenizerV1 / LLM From Scratch Refactor

**Date:** October 2026  
**Auditor:** Senior ML Engineer & Technical Writer  
**Branch:** `refactor/cohesive-structure`  

---

## 1. Executive Summary
1. Custom tokenization and byte pair encoding (tiktoken).
2. Data preparation and causal sliding-window dataloaders.
3. Multi-head self-attention and GPT-2 architectural components in PyTorch.
4. Pretraining loops with cross-entropy loss, evaluation, and temperature/top-$k$ generation.
5. Transfer learning / loading official OpenAI GPT-2 weights.
6. Classification fine-tuning on SMS spam detection.
7. Instruction fine-tuning (Supervised Fine-Tuning / SFT) on Alpaca-style instruction datasets.

However, the current repository state is highly fragmented and exhibits common symptoms of an incremental learning notebook-to-script migration:
- **Global execution side effects:** Almost every file executes computationally intensive code (downloading large files, initializing models, running 5 to 10 training epochs, writing checkpoints to disk) immediately upon being imported or run.
- **Widespread code duplication:** At least 3 different attention implementations, 3 generation functions, multiple training and evaluation routines, and 3 weight-loading implementations exist with subtle variations and discrepancies.
- **Untracked / polluted git tracking:** 11 compiled `.pyc` files are tracked in git under `__pycache__/`.
- **Several critical bugs:** A typo in device string (`"cude"` instead of `"cuda"` in `finetuning.py:234`), dimension bug in context length check (`pos_emb.weight.shape[1]` instead of `shape[0]` in `finetuning.py:361`), decoding regex flaw in `SimpleTokenizerV2.py:33`, infinite recursion and syntax bugs in `main.py`, and a nested loop bug in `DSDownload.py:306-328` that causes the test evaluation to run repeatedly.
- **Lack of modularity & packaging:** No `pyproject.toml`, no package namespace, hardcoded relative paths for datasets and weights, and no automated tests.

---

## 2. File Inventory

| Path | Type | Book Chapter | Description |
| :--- | :--- | :--- | :--- |
| `README.md` | Doc | General | Minimal placeholder (`# README`). |
| `.gitignore` | Config | General | Git ignore file with rules for `.pth`, `gpt2/`, `.venv`, `.tsv`, `*.zip`, `*.pdf`. Missing standard Python ignores (e.g. `__pycache__`, build artifacts). |
| `.gitattributes` | Config | General | Git attributes file configuring `* text=auto` LF normalization. |
| `requirements.txt` | Config | General | Environment dependencies adapted directly from Raschka's book repository (torch, tiktoken, tensorflow, matplotlib, pandas, etc.). |
| `prompt.md` | Doc / Task | N/A | Original refactoring prompt template instructions. |
| `the-verdict.txt` | Data | Ch 2 | Edith Wharton's short story (20,479 characters, ~5,145 BPE tokens) used for pretraining demonstrations. |
| `SimpleTokenizerV1.py` | Script / Module | Ch 2 | Implements `SimpleTokenizationV1` vocabulary mapping based on split regex without handling unknown tokens. Executes on module load. |
| `SimpleTokenizerV2.py` | Script / Module | Ch 2 | Implements `SimpleTokenizationV2` adding `<|unk|>` and `<|endoftext|>` special tokens. Contains a regex decoding bug. Executes on module load. |
| `BPETokenizer.py` | Script | Ch 2 | Introductory demonstration script using OpenAI `tiktoken` (`gpt2` encoding) on `the-verdict.txt`. |
| `GPTDatasetV1.py` | Script / Module | Ch 2 | Defines `GPTDatasetV1` and `create_dataloader_v1` for pretraining sliding-window chunking. Executes token embedding demo on import. |
| `self_attention.py` | Script | Ch 3 | Step-by-step mathematical demonstration of self-attention mechanics (dot product, naive softmax, context vectors). |
| `SelfAttention_V1.py` | Script / Module | Ch 3 | Implements `SelfAttention_v1` using raw trainable `nn.Parameter` tensors. Executes test pass on import. |
| `SelfAttention_v2.py` | Script / Module | Ch 3 | Implements `SelfAttention_v2` (using `nn.Linear`), causal masking, dropout, `CausalAttention`, `MultiHeadAttentionWrapper`, and `MultiHeadAttention`. Executes test passes on import. |
| `DummyGPTModel.py` | Script / Module | Ch 4 | Implements dummy placeholders (`DummyGPTModel`, `DummyTransformerBlock`, `DummyLayerNorm`) alongside real components: `LayerNorm`, `GELU`, `FeedForward`, `ExampleDeepNeuralNetwork`, and `TransformerBlock`. Imports `SelfAttention_v2` (triggering side effects) and executes on load. |
| `GPTModel.py` | Script / Module | Ch 4 & 5 | Defines `GPTModel`, configuration dictionaries, `generate_text_simple`, weight assignment `assign`, and weight loader `load_weights_into_gpt`. Imports from `test.py` and `DummyGPTModel.py`, executing side-effects on load. |
| `TextGen.py` | Script | Ch 5 | Large script with pretraining loop (`train_model_simple`), evaluation (`calc_loss_loader`), top-$k$ + temperature sampling (`generate`), and checkpoint saving. Executes 10 training epochs on module load. |
| `test.py` | Script | Ch 5 | Refined testbed extracting training functions (`calc_loss_batch`, `calc_loss_loader`, `evaluate_model`, `train_model_simple`, `generate`) from `TextGen.py`. Contains a `main()` function, but still imports `GPTDatasetV1` at top-level. |
| `main.py` | Script | Ch 5 | Attempt to tie together GPT-2 weight loading and text generation. Contains broken imports, syntax bugs, and infinite recursion. |
| `gpt_download.py` | Script / Utility | Ch 5 | Helper module adapted from Raschka's repo to download TensorFlow GPT-2 checkpoint files from OpenAI / Backblaze and extract numpy arrays. Contains duplicate function blocks. |
| `dataset.py` | Script / Utility | Ch 6 | Downloads and extracts the SMS Spam Collection zip archive into `sms_spam_collection/SMSSpamCollection.tsv`. Executes immediately upon load. |
| `train.csv` | Data | Ch 6 | Pre-split, balanced training set (1,046 lines) for spam classification. |
| `validation.csv` | Data | Ch 6 | Pre-split validation set (150 lines) for spam classification. |
| `test.csv` | Data | Ch 6 | Pre-split test set (301 lines) for spam classification. |
| `finetuning.py` | Script | Ch 6 | End-to-end spam classification fine-tuning script: loads TSV, balances dataset, exports CSVs, creates `SpamDataset`, reconfigures GPT-2 head, freezes backbone, trains classifier, and saves `review_classifier.pth`. Contains multiple bugs and executes everything on module load. |
| `DSDownload.py` | Script | Ch 7 | End-to-end instruction fine-tuning script: downloads `instruction-data.json`, defines `InstructionDataset`, `custom_collate_fn`, fine-tunes GPT-2 with cross-entropy ignoring padding, and evaluates responses. Contains loop bug and executes on load. |
| `__pycache__/*` | Binary | N/A | 11 tracked CPython bytecode files (`cpython-312.pyc` and `cpython-314.pyc`) committed to git. |

---

## 3. Duplicated and Near-Duplicated Components

### 3.1 Tokenizers
- **Implementations:**
  - `SimpleTokenizerV1.py`: `SimpleTokenizationV1` (basic word + punctuation splitting).
  - `SimpleTokenizerV2.py`: `SimpleTokenizationV2` (adds `<|unk|>` and `<|endoftext|>`).
  - `BPETokenizer.py`: Direct script usage of `tiktoken.get_encoding("gpt2")`.
- **Assessment & Canonical Choice:**
  - `SimpleTokenizerV1` and `SimpleTokenizerV2` are educational implementations. In `SimpleTokenizerV2.py`, line 33 contains a bug:
    ```python
    text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)  # Bug: does not remove space before punctuation
    ```
    whereas `SimpleTokenizerV1.py` line 35 has the correct regex:
    ```python
    text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
    ```
  - **Canonical:** Provide both educational tokenizers in `src/eru/tokenizer/simple_tokenizer.py` (with the decode bug fixed), plus a production-grade wrapper around `tiktoken` in `src/eru/tokenizer/bpe_tokenizer.py`.

### 3.2 Attention Mechanisms
- **Implementations:**
  - `self_attention.py`: Standalone scripts computing attention scores via for-loops and dot products.
  - `SelfAttention_V1.py`: `SelfAttention_v1` using raw parameters `self.W_query = nn.Parameter(torch.rand(d_in, d_out))`.
  - `SelfAttention_v2.py`: `SelfAttention_v2` (single head with `nn.Linear`), `CausalAttention` (with lower-triangular causal mask buffer and dropout), `MultiHeadAttentionWrapper` (module list of sequential causal attention heads), and `MultiHeadAttention` (consolidated multi-head projection splitting query, key, value matrices).
- **Assessment & Canonical Choice:**
  - `MultiHeadAttention` in `SelfAttention_v2.py` (lines 103–146) is the canonical, efficient, vectorized multi-head causal attention implementation used by all subsequent chapters and compatible with pretrained GPT-2 weights.
  - `CausalAttention`, `SelfAttention_v1`, and `SelfAttention_v2` should be preserved in modular educational files or tests for completeness, with `MultiHeadAttention` as the canonical production module.

### 3.3 Transformer Block, Normalization, and Architecture
- **Implementations:**
  - `DummyGPTModel.py`: Contains `DummyGPTModel`, `DummyTransformerBlock`, `DummyLayerNorm`, `LayerNorm`, `GELU`, `FeedForward`, `ExampleDeepNeuralNetwork`, and `TransformerBlock`.
  - `GPTModel.py`: Imports `TransformerBlock` and `LayerNorm` from `DummyGPTModel.py` and defines `GPTModel`.
- **Assessment & Canonical Choice:**
  - `LayerNorm`, `GELU`, `FeedForward`, and `TransformerBlock` should be cleanly decoupled into dedicated model modules (`src/eru/model/normalization.py`, `src/eru/model/activations.py`, `src/eru/model/feedforward.py`, `src/eru/model/transformer_block.py`).
  - `GPTModel` in `GPTModel.py` (lines 17–41) is the canonical full GPT model architecture. Dummy placeholders belong in test fixtures or educational archives.

### 3.4 Text Generation and Decoding
- **Implementations:**
  - `GPTModel.py`: `generate_text_simple` (greedy next-token argmax).
  - `finetuning.py`: `generate_text_simple` (identical greedy decoding).
  - `TextGen.py`: `generate` (supports `temperature`, `top_k`, and `eos_id`).
  - `test.py`: `generate` (duplicate of `TextGen.py`'s `generate`).
  - `DSDownload.py`: Imports `generate` from `GPTModel` (which actually came from `test.py`).
- **Assessment & Canonical Choice:**
  - `generate` in `test.py` (lines 82–109) is the most complete implementation, supporting greedy decoding (when `temperature=0.0`), top-$k$ truncation, temperature sampling, and early stopping on `eos_id`.
  - Canonical location: `src/eru/generation/sampling.py`. Support both `generate_greedy` (or `generate_text_simple`) and full `generate` as clean, tested library functions.

### 3.5 Token Conversion Helpers
- **Implementations:**
  - `text_to_token_ids` & `token_ids_to_text`:
    - Defined in `GPTModel.py:160-167`
    - Defined in `TextGen.py:19-25`
    - Defined in `test.py:17-25`
    - Imported in `main.py`, `DSDownload.py`, `finetuning.py`
- **Assessment & Canonical Choice:**
  - Canonical location: `src/eru/tokenizer/utils.py` or methods on the BPE tokenizer wrapper class.

### 3.6 Loss Computation and Evaluation
- **Implementations:**
  - `calc_loss_batch`:
    - `TextGen.py:140-146`: Language modeling loss (flattens `(batch * seq_len, vocab_size)`).
    - `test.py:27-30`: Identical language modeling loss.
    - `finetuning.py:246-250`: Classification loss (uses last token logits `logits[:, -1, :]`).
  - `calc_loss_loader`:
    - `TextGen.py:148-162` vs `test.py:33-42` (handles empty loader with `nan`, averages over `num_batches`).
    - `finetuning.py:252-266` (handles classification evaluation).
  - `evaluate_model`:
    - Duplicated across `TextGen.py`, `test.py`, and `finetuning.py`.
  - `calc_accuracy_loader`:
    - Defined in `finetuning.py:213-232` for classification accuracy.
- **Assessment & Canonical Choice:**
  - Disentangle language modeling loss (`src/eru/training/loss.py`) from classification loss and accuracy (`src/eru/finetune/classification.py`). Both should accept clean inputs and avoid hardcoded globals.

### 3.7 Training Loops
- **Implementations:**
  - `train_model_simple`: Duplicated in `TextGen.py:172-195` and `test.py:54-79` for pretraining / language modeling.
  - `train_classifier_simple`: Defined in `finetuning.py:277-308` for classification fine-tuning.
  - Fine-tuning loop in `DSDownload.py:278-282`: Directly calls `train_model_simple` for instruction SFT.
- **Assessment & Canonical Choice:**
  - Pretraining / SFT loop: `src/eru/training/trainer.py` (canonicalized from `test.py` with callback/logging support).
  - Classification fine-tuning loop: `src/eru/finetune/classification_trainer.py`.

### 3.8 Weight Loading & Checkpoint Porting
- **Implementations:**
  - `gpt_download.py`: `download_and_load_gpt2` (downloads checkpoint files, parses TF v1 checkpoint into numpy dict).
  - `GPTModel.py`: `assign` (lines 57–60) and `load_weights_into_gpt` (lines 116–158).
  - `main.py`: Broken duplicate of `load_weights_into_gpt` with typos and recursion bugs.
  - `finetuning.py` & `DSDownload.py`: Import `load_weights_into_gpt` from `GPTModel`.
- **Assessment & Canonical Choice:**
  - `load_weights_into_gpt` and `assign` in `GPTModel.py` are correct, but lines 149–158 in `GPTModel.py` accidentally execute text generation inside the loader function!
  - Canonical location: `src/eru/weights/gpt2_loader.py` with pure weight transposition and mapping, completely decoupled from inference generation.

### 3.9 Datasets and Data Loaders
- **Implementations:**
  - Pretraining: `GPTDatasetV1` and `create_dataloader_v1` in `GPTDatasetV1.py`.
  - Classification: `SpamDataset` in `finetuning.py`.
  - Instruction SFT: `InstructionDataset` and `custom_collate_fn` in `DSDownload.py`.
- **Assessment & Canonical Choice:**
  - All three represent distinct tasks across Chapters 2, 6, and 7. They should be cleanly modularized:
    - Pretraining dataset: `src/eru/data/pretrain_dataset.py`
    - Classification dataset: `src/eru/data/spam_dataset.py`
    - Instruction dataset: `src/eru/data/instruction_dataset.py`
    - Collate functions: `src/eru/data/collate.py`

---

## 4. Code Quality, Bugs, and Technical Debt

### 4.1 Confirmed Bugs
1. **Device Typo in `finetuning.py` (Line 234):**
   ```python
   device = torch.device("cude" if torch.cuda.is_available() else "cpu")
   ```
   If CUDA is present, `torch.device("cude")` throws `RuntimeError: Expected one of cpu, cuda, ... device type at start of device string: cude`.
2. **Context Length Shape Bug in `finetuning.py` (Line 361):**
   ```python
   supported_context_length = model.pos_emb.weight.shape[1]
   ```
   `pos_emb` is `nn.Embedding(num_embeddings=context_length, embedding_dim=emb_dim)`. `weight.shape[0]` is context length (1024), while `weight.shape[1]` is embedding dimension (768). This erroneously truncates input prompts to 768 tokens instead of 1024.
3. **Punctuation Regex Bug in `SimpleTokenizerV2.py` (Line 33):**
   ```python
   text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
   ```
   This replaces matched punctuation with itself without removing preceding whitespace. The correct regex from V1 is `r'\s([,.:;?_!"()\'])'`.
4. **Infinite Recursion & Syntax Bugs in `main.py`:**
   - Line 3: `from TextGen import ..., NEW_CONFIG, tokenizer` fails because `NEW_CONFIG` is not defined in `TextGen.py`.
   - Line 11: `gpt.trf_blocks[b].attW_query.weight` has a missing dot (`att.W_query`).
   - Line 37: Calls `load_weights_into_gpt(gpt, params)` recursively inside itself.
   - Line 38: `gpt.to(torch.device)` crashes because `torch.device` is a type, not an instance.
5. **Redundant Nested Loop in `DSDownload.py` (Lines 306–328):**
   ```python
   for entry in test_data[:3]:
       ...
       for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
           ...
       with open("instruction-data-with-response.json", "w") as file:
           json.dump(test_data, file, indent=4)
       torch.save(model.state_dict(), file_name)
   ```
   The entire test set loop and file saving are indented inside the 3-element sample loop, causing the full dataset evaluation and saving to execute 3 consecutive times.
6. **Dead Code after Return in `DSDownload.py` (Lines 151–154):**
   Calls `inputs, targets = custom_collate_fn(batch)` indented directly under the return statement of `custom_collate_fn`.
7. **Duplicated Code Blocks in `gpt_download.py`:**
   Lines 1–45 and lines 46–90 repeat identical definitions of `download_and_load_gpt2`. Lines 280–306 duplicate `load_gpt2_params_from_tf_ckpt`.

### 4.2 Top-Level Execution Side Effects (Import Hazards)
Importing almost any file executes top-level side effects:
- `GPTDatasetV1.py`: Reads `the-verdict.txt`, creates dataloaders, instantiates embedding layers.
- `DummyGPTModel.py`: Imports `SelfAttention_v2` (running its side effects), runs dummy forward passes, computes gradients.
- `GPTModel.py`: Instantiates models, runs inference, creates dummy tensors.
- `TextGen.py`: Trains a model for 10 epochs, writes `model.pth` and `model_and_optimizer.pth`.
- `finetuning.py`: Reads `sms_spam_collection/SMSSpamCollection.tsv`, rewrites CSV files, downloads GPT-2 weights, trains for 5 epochs, writes `review_classifier.pth`.
- `DSDownload.py`: Downloads `instruction-data.json`, downloads GPT-2 weights, runs fine-tuning, writes checkpoints.
- `dataset.py`: Immediately downloads and extracts spam data on import.

### 4.3 Hardcoded Absolute / Relative Paths
- Relative paths assume the current working directory is the repository root:
  - `"the-verdict.txt"`
  - `"sms_spam_collection/SMSSpamCollection.tsv"`
  - `"train.csv"`, `"validation.csv"`, `"test.csv"`
  - `"instruction-data.json"`
  - `"instruction-data-with-response.json"`
  - `"model.pth"`, `"model_and_optimizer.pth"`, `"review_classifier.pth"`, `"<model>-sft.pth"`
  - `"gpt2/"` directory

### 4.4 Hardcoded Hyperparameters and Magic Numbers
- Vocabulary size `50257` hardcoded across models and tests.
- End-of-text / padding token `50256` hardcoded in collate functions and datasets.
- Context length varies between `256` (in `TextGen.py` and `test.py`) and `1024` (in `GPTModel.py`, `finetuning.py`, `DSDownload.py`).
- Learning rates (`0.0004`, `5e-5`, `5e-4`), batch sizes (`2`, `4`, `8`), dropout rates (`0.0`, `0.01`, `0.1`, `0.5`) scattered without centralized configuration files.

---

## 5. Repository Assets and Tracked Files Audit

### 5.1 Large Files and Bytecode Committed to Git
- **Committed Python Bytecode (`__pycache__`):**
  11 `.pyc` files are actively tracked in git:
  - `__pycache__/DummyGPTModel.cpython-312.pyc`
  - `__pycache__/DummyGPTModel.cpython-314.pyc`
  - `__pycache__/GPTDatasetV1.cpython-312.pyc`
  - `__pycache__/GPTDatasetV1.cpython-314.pyc`
  - `__pycache__/GPTModel.cpython-312.pyc`
  - `__pycache__/GPTModel.cpython-314.pyc`
  - `__pycache__/SelfAttention_v2.cpython-312.pyc`
  - `__pycache__/SelfAttention_v2.cpython-314.pyc`
  - `__pycache__/TextGen.cpython-312.pyc`
  - `__pycache__/gpt_download.cpython-312.pyc`
  - `__pycache__/test.cpython-312.pyc`
  *Action:* Remove bytecode files from git index in Phase 3 and enforce ignoring via `.gitignore`.
- **Committed Datasets:**
  - `the-verdict.txt`: 20,479 bytes (~20.5 KB). Safe to keep as standard sample data for pretraining.
  - `train.csv`: 112,762 bytes (~113 KB).
  - `test.csv`: 31,786 bytes (~32 KB).
  - `validation.csv`: 17,770 bytes (~18 KB).
  *Action:* Retain text/CSV data under `Eru/data/` or provide automated download/generation scripts.
- **Model Checkpoints:**
  - No large `.pth` or TensorFlow model binary checkpoints are committed to git (`.gitignore` successfully kept `model.pth`, `model_optimizer.pth`, and `gpt2/` untracked).

### 5.2 Secrets, Credentials, and Personal Information
- **Scan results:** Full regex search across all files and git commit history:
  - **API Keys / Secrets:** None found. No OpenAI API keys, tokens, or credentials exist in the repository.
  - **Spam Text False Positives:** The spam CSVs (`train.csv`, `test.csv`, `validation.csv`) contain lines with spam trigger words (e.g. `secret admirer`, `password "MIX"`), which are purely test dataset strings.
  - **Personal Info in Commit History:** Commit author is `Evan <balleva28@gmail.com>`. This is standard git metadata and does not contain sensitive secrets.

---

## 6. Pipeline Gap Analysis (Train $\to$ Finetune $\to$ Generate)

To achieve the primary goal ("A stranger can clone the repo and, by following the README exactly, install it, train a small GPT model, fine-tune it, and generate text, on Windows, macOS, or Linux, with or without a GPU"), the following pieces are currently missing:

1. **Standard Python Project Packaging:**
   - No `pyproject.toml` or setup configuration. The repo cannot currently be installed via `pip install -e .`.
2. **Unified Command-Line Interface (CLI):**
   - No unified CLI or entry points. Running any pipeline step requires running an entire script that executes all steps including data downloads and hardcoded training epochs.
3. **Config-Driven Architecture:**
   - Hyperparameters, model architecture configurations (tiny CPU-friendly model vs. full 124M GPT-2), paths, and hardware device options need to be externalized into YAML config files (`configs/gpt_small.yaml`, `configs/gpt_124m.yaml`, `configs/finetune_classification.yaml`, `configs/finetune_instruction.yaml`).
4. **Pure PyTorch Pretrained Weight Loading:**
   - Currently, loading pretrained GPT-2 weights requires TensorFlow (`tensorflow >= 2.18.0`) to read old TensorFlow v1 checkpoints. On modern systems (especially Python 3.12+ or Apple Silicon/ARM), TensorFlow installation is heavy and error-prone. Providing an optional Hugging Face / PyTorch state_dict converter or clear isolation ensures CPU/cross-platform portability.
5. **Automated Verification & Golden Tests:**
   - Zero formal unit tests or regression tests currently exist.

---

## 7. Python Environment & Library Audit

### 7.1 Inferred Python Version
- Compiled bytecode files in git indicate development occurred across **CPython 3.12** and **CPython 3.14**.
- `tensorflow >= 2.18.0` has official wheel support for Python 3.9 through 3.12.
- **Target Recommended Version:** Python 3.10 – 3.12 (ensures compatibility with PyTorch, TensorFlow, and tiktoken across Linux, macOS, and Windows).

### 7.2 Inferred Direct Dependencies
- `torch >= 2.2.0` (Core neural network operations, autograd, optimizers)
- `tiktoken >= 0.5.1` (GPT-2 BPE tokenizer)
- `pandas >= 2.2.0` (Dataframe manipulation for classification datasets)
- `numpy >= 1.26.0` (Array manipulations during weight loading)
- `matplotlib >= 3.7.0` (Loss and metric plotting)
- `tqdm >= 4.66.0` (Progress indicators)
- `pyyaml >= 6.0` (Required for YAML configuration loading)
- *Optional / Extra:* `tensorflow >= 2.16.0` (Only needed if extracting weights from legacy TensorFlow checkpoints)
- *Test / Dev:* `pytest >= 8.0.0`
