"""
Unit tests for Stage 1 features:
- KV-Cache correctness & parity
- PyTorch SDPA attention
- Advanced sampling (Top-p, Min-p, Repetition Penalty)
- Streaming text generation
- Cosine LR scheduler with warmup
"""

import unittest
import torch

from eru.model.gpt import GPTModel, GPT_CONFIG_SMALL
from eru.model.attention import MultiHeadAttention
from eru.model.kv_cache import KVCache
from eru.generation.sampling import (
    generate,
    generate_stream,
    apply_sampling_filters,
    sample_next_token,
)
from eru.training.trainer import get_cosine_schedule_with_warmup


class TestStage1Features(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = GPT_CONFIG_SMALL.copy()
        self.cfg["vocab_size"] = 100
        self.cfg["context_length"] = 32
        self.cfg["emb_dim"] = 32
        self.cfg["n_heads"] = 4
        self.cfg["n_layers"] = 2

    def test_kv_cache_basic(self):
        cache = KVCache(num_layers=2)
        self.assertEqual(len(cache), 0)

        k1 = torch.randn(2, 4, 5, 8)
        v1 = torch.randn(2, 4, 5, 8)
        out_k, out_v = cache.update(0, k1, v1)
        self.assertEqual(out_k.shape, (2, 4, 5, 8))
        self.assertEqual(len(cache), 5)

        # Append next single token
        k2 = torch.randn(2, 4, 1, 8)
        v2 = torch.randn(2, 4, 1, 8)
        out_k2, out_v2 = cache.update(0, k2, v2)
        self.assertEqual(out_k2.shape, (2, 4, 6, 8))
        self.assertEqual(len(cache), 6)

        cache.reset()
        self.assertEqual(len(cache), 0)

    def test_sdpa_attention(self):
        mha = MultiHeadAttention(
            d_in=32,
            d_out=32,
            context_length=16,
            num_heads=4,
            use_sdpa=True,
        )
        x = torch.randn(2, 8, 32)
        out = mha(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_kv_cache_generation_parity(self):
        """
        Verify that greedy autoregressive generation with KV-cache produces
        the EXACT same sequence as standard generation without cache.
        """
        model = GPTModel(self.cfg)
        model.eval()

        prompt = torch.randint(0, self.cfg["vocab_size"], (1, 6))

        # Generate without cache
        torch.manual_seed(123)
        out_no_cache = generate(
            model=model,
            idx=prompt.clone(),
            max_new_tokens=10,
            context_size=self.cfg["context_length"],
            temperature=0.0,
            use_cache=False,
        )

        # Generate with cache
        torch.manual_seed(123)
        out_with_cache = generate(
            model=model,
            idx=prompt.clone(),
            max_new_tokens=10,
            context_size=self.cfg["context_length"],
            temperature=0.0,
            use_cache=True,
        )

        self.assertTrue(torch.equal(out_no_cache, out_with_cache))

    def test_sampling_filters(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Top-k
        filtered_topk = apply_sampling_filters(logits.clone(), top_k=2)
        # Lowest 3 should be -inf
        self.assertTrue(torch.isneginf(filtered_topk[0, 0]))
        self.assertTrue(torch.isneginf(filtered_topk[0, 1]))
        self.assertTrue(torch.isneginf(filtered_topk[0, 2]))
        self.assertFalse(torch.isneginf(filtered_topk[0, 3]))
        self.assertFalse(torch.isneginf(filtered_topk[0, 4]))

        # Top-p (Nucleus)
        filtered_topp = apply_sampling_filters(logits.clone(), top_p=0.5)
        # The largest logit (index 4) has prob > 0.6, so it alone should be kept
        self.assertFalse(torch.isneginf(filtered_topp[0, 4]))

        # Min-p
        filtered_minp = apply_sampling_filters(logits.clone(), min_p=0.8)
        self.assertFalse(torch.isneginf(filtered_minp[0, 4]))

        # Repetition penalty
        seen = torch.tensor([[4]])
        penalized = apply_sampling_filters(logits.clone(), repetition_penalty=2.0, generated_tokens=seen)
        # Token 4 logit was 5.0, with penalty 2.0 it should become 2.5
        self.assertAlmostEqual(penalized[0, 4].item(), 2.5, places=4)

    def test_streaming_generation(self):
        model = GPTModel(self.cfg)
        model.eval()
        prompt = torch.randint(0, self.cfg["vocab_size"], (1, 4))

        stream_tokens = []
        for token_id, _ in generate_stream(
            model=model,
            idx=prompt.clone(),
            max_new_tokens=5,
            context_size=self.cfg["context_length"],
            temperature=0.0,
            use_cache=True,
        ):
            stream_tokens.append(token_id)

        self.assertEqual(len(stream_tokens), 5)

    def test_cosine_schedule_with_warmup(self):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=1.0)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)

        # Step 0: Warmup start should be 0.0
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)

        for _ in range(10):
            optimizer.step()
            scheduler.step()
        # Step 10: Warmup peak should reach 1.0
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1.0, places=4)

        for _ in range(90):
            optimizer.step()
            scheduler.step()
        # Step 100: Should reach min_lr_ratio (0.1)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
