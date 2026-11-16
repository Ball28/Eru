"""
Unit tests for Stage 2 features:
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed-Forward Networks
- Grouped-Query Attention (GQA)
- Modern architecture GPTModel integration
- Memory-Mapped Binary Token Dataset
"""

import unittest
from pathlib import Path
import torch

from eru.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from eru.model.normalization import RMSNorm
from eru.model.feedforward import SwiGLUFeedForward, FeedForward
from eru.model.attention import MultiHeadAttention
from eru.model.gpt import GPTModel, MODERN_CONFIG_SMALL
from eru.data.mmap_dataset import MMapTokenDataset
from eru.generation.sampling import generate


class TestStage2Features(unittest.TestCase):

    def test_rope(self):
        dim = 16
        rope = RotaryEmbedding(dim=dim, max_seq_len=64)
        x = torch.randn(2, 4, 10, dim)
        cos, sin = rope(x, seq_len=10, start_pos=0)
        self.assertEqual(cos.shape, (1, 1, 10, dim))
        self.assertEqual(sin.shape, (1, 1, 10, dim))

        q = torch.randn(2, 4, 10, dim)
        k = torch.randn(2, 4, 10, dim)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_rmsnorm(self):
        norm = RMSNorm(emb_dim=16)
        x = torch.randn(4, 8, 16)
        out = norm(x)
        self.assertEqual(out.shape, (4, 8, 16))
        # Variance along last dimension should be ~1.0
        rms_val = torch.sqrt(out.pow(2).mean(dim=-1))
        self.assertTrue(torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-3))

    def test_swiglu(self):
        cfg = {"emb_dim": 32, "hidden_dim": 64, "ffn_bias": False}
        swiglu = SwiGLUFeedForward(cfg)
        x = torch.randn(2, 5, 32)
        out = swiglu(x)
        self.assertEqual(out.shape, (2, 5, 32))

        # Test via FeedForward dispatcher
        ff = FeedForward({"emb_dim": 32, "ffn_type": "swiglu"})
        out_ff = ff(x)
        self.assertEqual(out_ff.shape, (2, 5, 32))

    def test_gqa_attention(self):
        # 4 query heads, 2 KV heads (2x GQA)
        mha = MultiHeadAttention(
            d_in=32,
            d_out=32,
            context_length=16,
            num_heads=4,
            num_kv_heads=2,
            use_sdpa=True,
        )
        x = torch.randn(2, 8, 32)
        out = mha(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_modern_gpt_model(self):
        cfg = MODERN_CONFIG_SMALL.copy()
        model = GPTModel(cfg)
        model.eval()

        in_tokens = torch.randint(0, cfg["vocab_size"], (2, 16))
        logits = model(in_tokens)
        self.assertEqual(logits.shape, (2, 16, cfg["vocab_size"]))

        # Autoregressive generation with modern architecture + KV-cache
        gen_tokens = generate(
            model=model,
            idx=in_tokens[:1, :4],
            max_new_tokens=6,
            context_size=cfg["context_length"],
            temperature=0.0,
            use_cache=True,
        )
        self.assertEqual(gen_tokens.shape, (1, 10))

    def test_mmap_dataset(self):
        bin_file = Path("data/bin/train.bin")
        if bin_file.exists():
            dataset = MMapTokenDataset(bin_file, context_length=16, stride=16)
            self.assertTrue(len(dataset) > 0)
            x, y = dataset[0]
            self.assertEqual(x.shape, (16,))
            self.assertEqual(y.shape, (16,))
            self.assertEqual(x.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
