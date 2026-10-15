"""
Unit tests for neural network model architecture, shapes, and parameters.
"""

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from eru.model.gpt import GPTModel, GPT_CONFIG_124M, GPT_CONFIG_SMALL
from eru.model.attention import MultiHeadAttention
from eru.model.transformer_block import TransformerBlock
from eru.model.normalization import LayerNorm
from eru.model.activations import GELU


@unittest.skipUnless(HAS_TORCH, "PyTorch required for model tests")
class TestModelArchitecture(unittest.TestCase):

    def test_gelu(self):
        gelu = GELU()
        x = torch.tensor([0.0, 1.0, -1.0])
        out = gelu(x)
        self.assertEqual(out.shape, x.shape)
        self.assertAlmostEqual(out[0].item(), 0.0, places=4)
        self.assertTrue(out[1].item() > 0.8)

    def test_layer_norm(self):
        ln = LayerNorm(emb_dim=5)
        x = torch.randn(2, 5)
        out = ln(x)
        self.assertEqual(out.shape, (2, 5))
        # Normalized mean should be ~0 and variance ~1
        self.assertAlmostEqual(out.mean(dim=-1)[0].item(), 0.0, places=4)
        self.assertAlmostEqual(out.var(dim=-1, unbiased=False)[0].item(), 1.0, places=3)

    def test_multi_head_attention_shape(self):
        batch = torch.randn(2, 4, 16)
        mha = MultiHeadAttention(d_in=16, d_out=16, context_length=4, num_heads=2)
        out = mha(batch)
        self.assertEqual(out.shape, (2, 4, 16))

    def test_transformer_block_shape(self):
        cfg = {"emb_dim": 16, "context_length": 4, "n_heads": 2, "drop_rate": 0.0, "qkv_bias": False}
        block = TransformerBlock(cfg)
        x = torch.randn(2, 4, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 4, 16))

    def test_gpt_model_forward_shape(self):
        cfg = GPT_CONFIG_SMALL
        model = GPTModel(cfg)
        in_tokens = torch.randint(0, cfg["vocab_size"], (2, 8))
        logits = model(in_tokens)
        self.assertEqual(logits.shape, (2, 8, cfg["vocab_size"]))


if __name__ == "__main__":
    unittest.main()
