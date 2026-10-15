# Datasets Guide

This directory holds dataset text and CSV files used for pretraining, classification fine-tuning, and instruction fine-tuning.

## 1. Edith Wharton's "The Verdict" (`the-verdict.txt`)
- **Use:** Pretraining sample corpus (Chapters 2–5).
- **Source:** Public domain short story by Edith Wharton (1908).
- **Tokens:** ~5,145 BPE tokens (20,479 characters).
- **Status:** Bundled directly in `Eru/data/the-verdict.txt`.

## 2. SMS Spam Collection (`spam/`)
- **Use:** Classification fine-tuning (Chapter 6).
- **Files:**
  - `spam/train.csv`: 1,046 balanced samples (50% ham, 50% spam).
  - `spam/validation.csv`: 150 balanced validation samples.
  - `spam/test.csv`: 301 balanced test samples.
- **Automated Download:**
  To download and prepare fresh data from the UCI repository:
  ```bash
  python -c "from eru.data.download import download_spam_dataset; download_spam_dataset('data/spam')"
  ```

## 3. Instruction Dataset (`instruction-data.json`)
- **Use:** Supervised Fine-Tuning / SFT (Chapter 7).
- **Format:** Alpaca-style JSON entries with `instruction`, `input`, and `output` keys (1,100 synthetic instruction pairs).
- **Automated Download:**
  ```bash
  python -c "from eru.data.download import download_instruction_data; download_instruction_data('data/instruction-data.json')"
  ```
