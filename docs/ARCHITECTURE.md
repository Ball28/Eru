# Eru LLM Architecture Guide

```mermaid
flowchart TD
    subgraph Data ["Data & Tokenization"]
        RawText["Raw Text Corpus"] --> Tokenizer["BPETokenizer / SimpleTokenizer"]
        Tokenizer --> Dataset["Sliding Window / MMapTokenDataset (.bin)"]
        Dataset --> DataLoader["DataLoader (batch_size, seq_len)"]
    end

    subgraph GPT ["GPTModel Architecture"]
        DataLoader --> TokEmb["Token Embeddings (vocab_size, emb_dim)"]
        DataLoader -. PosEmb .-> PosEmb["Positional Embeddings (Learned or RoPE)"]
        TokEmb --> AddEmbeds["Embedding Layer"]
        PosEmb -.-> AddEmbeds

        AddEmbeds --> Block1["TransformerBlock 1"]
        Block1 --> BlockN["TransformerBlock ... N"]
        BlockN --> FinalNorm["Normalization (LayerNorm or RMSNorm)"]
        FinalNorm --> Head["Linear Output Head (emb_dim -> vocab_size / num_classes)"]
    end

    subgraph TrfBlock ["Inside TransformerBlock"]
        X["Input x"] --> N1["Norm 1 (LayerNorm or RMSNorm)"]
        N1 --> Attn["Attention (MHA / GQA / MQA + RoPE + SDPA + KV-Cache)"]
        Attn --> Drop1["Dropout"]
        Drop1 --> Res1["(+) Residual Shortcut (x)"]
        Res1 --> N2["Norm 2 (LayerNorm or RMSNorm)"]
        N2 --> FF["FeedForward (GELU or SwiGLU)"]
        FF --> Drop2["Dropout"]
        Drop2 --> Res2["(+) Residual Shortcut"]
    end

    subgraph Pipelines ["Downstream Tasks & Alignment"]
        Head --> Pretrain["Pretraining (Cross-Entropy + Cosine Warmup + AMP)"]
        Head --> Classify["Classification (SMS Spam Detection)"]
        Head --> SFT["Instruction Tuning (Alpaca Prompt with -100 Target Masking)"]
        Head --> LoRA["LoRA Fine-Tuning (Parameter-Efficient Low-Rank Adapters)"]
        Head --> DPO["DPO Alignment (Direct Preference Optimization)"]
        Head --> Generate["Generation (KV-Cache, Top-p, Min-p, Repetition Penalty, Streaming)"]
        Head --> Export["Hugging Face Export (config.json, pytorch_model.bin)"]
    end
```

## Component Overview

1. **`eru.tokenizer`**: Tokenization implementations:
   - `SimpleTokenizerV1`: Basic regex-based splitting.
   - `SimpleTokenizerV2`: Adds `<|unk|>` and `<|endoftext|>`.
   - `BPETokenizer`: Production Byte-Pair Encoding wrapping OpenAI `tiktoken`.
2. **`eru.model`**:
   - `GELU`: Gaussian Error Linear Unit activation with exact tanh approximation.
   - `LayerNorm`: Normalization over embedding dimension with learnable scale and shift.
   - `RMSNorm`: Root Mean Square Normalization omitting mean centering for modern speed and stability.
   - `RotaryEmbedding (RoPE)`: Rotary positional embeddings directly rotating Query and Key vectors in complex conjugate pairs.
   - `MultiHeadAttention`: Vectorized causal attention supporting:
     - Classic Multi-Head Attention (MHA).
     - Grouped-Query Attention (GQA) and Multi-Query Attention (MQA).
     - PyTorch Flash / Scaled Dot-Product Attention (`F.scaled_dot_product_attention`).
     - Key-Value Caching (`KVCache`) turning generation complexity from $O(T^2)$ to $O(T)$.
   - `FeedForward`: Position-wise MLP supporting standard GPT-2 GELU and modern LLaMA SwiGLU.
   - `TransformerBlock`: Pre-norm residual transformer layer.
   - `GPTModel`: Unified causal language model supporting classic GPT-2 and modern LLaMA architectures.
3. **`eru.data`**:
   - `GPTDatasetV1`: In-memory sliding-window causal chunking.
   - `MMapTokenDataset`: Zero-RAM memory-mapped binary dataset reading pre-tokenized `uint16` shards from disk for arbitrarily large corpora.
   - `InstructionDataset`: SFT dataset with dynamic batch padding and ignore-index target masking.
   - `SpamDataset`: SMS classification dataset loader.
4. **`eru.training`**:
   - `train_model_simple`: Pretraining loop with Automatic Mixed Precision (AMP), gradient norm clipping, and cosine learning rate scheduling with linear warmup.
   - `save_checkpoint` / `load_checkpoint`: State serialization.
5. **`eru.generation`**:
   - `generate`: Autoregressive generation with KV-cache, temperature scaling, top-$k$, nucleus top-$p$, min-$p$, and repetition penalties.
   - `generate_stream`: Real-time token streaming generator yielding tokens dynamically.
6. **`eru.finetune`**:
   - `classification`: Spam classifier with linear head adaptation, freezing base layers, and accuracy evaluation.
   - `instruction`: Supervised Fine-Tuning (SFT) engine with Alpaca formatting.
   - `lora`: Parameter-Efficient Fine-Tuning (PEFT) implementing low-rank $A$ and $B$ adapter matrices with runtime merging.
   - `dpo`: Direct Preference Optimization loss and preference dataset handler.
7. **`eru.weights`**:
   - `loader`: Pure PyTorch assignment of weights into `GPTModel`.
   - `tf_download`: Downloader for OpenAI TensorFlow v1 checkpoint files.
   - `hf_export`: Converter exporting Eru models directly into standard Hugging Face `transformers` checkpoints.
