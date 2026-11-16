"""
Unit tests for Stage 3 features:
- Low-Rank Adaptation (LoRA from scratch)
- Direct Preference Optimization (DPO loss & dataset)
- Hugging Face Transformers export
"""

import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn

from eru.model.gpt import GPTModel, GPT_CONFIG_SMALL
from eru.finetune.lora import (
    LoRALinear,
    apply_lora_to_model,
    save_lora_checkpoint,
    load_lora_checkpoint,
    get_lora_state_dict,
)
from eru.finetune.dpo import dpo_loss, compute_sequence_log_probs, DPODataset
from eru.weights.hf_export import eru_to_hf_state_dict, export_to_huggingface
from eru.tokenizer.simple import SimpleTokenizerV2


class TestStage3Features(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = GPT_CONFIG_SMALL.copy()
        self.cfg["vocab_size"] = 100
        self.cfg["context_length"] = 32
        self.cfg["emb_dim"] = 32
        self.cfg["n_heads"] = 4
        self.cfg["n_layers"] = 2

    def test_lora_linear(self):
        base = nn.Linear(32, 64)
        lora = LoRALinear(base, rank=4, alpha=8.0)

        # Base layer must be frozen
        self.assertFalse(lora.base_layer.weight.requires_grad)

        # Initial forward pass: lora_B is initialized to 0, so initial output must match base layer output exactly
        x = torch.randn(2, 5, 32)
        out_base = base(x)
        out_lora = lora(x)
        self.assertTrue(torch.allclose(out_base, out_lora, atol=1e-5))

        # Merge and unmerge
        lora.lora_B.data.normal_()
        out_before_merge = lora(x)
        lora.merge()
        out_merged = lora(x)
        self.assertTrue(torch.allclose(out_before_merge, out_merged, atol=1e-5))

        lora.unmerge()
        out_unmerged = lora(x)
        self.assertTrue(torch.allclose(out_before_merge, out_unmerged, atol=1e-5))

    def test_apply_lora_to_model(self):
        model = GPTModel(self.cfg)
        apply_lora_to_model(model, rank=4, alpha=8.0)

        # Trainable params should only be LoRA parameters
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertTrue(len(trainable) > 0)

        # Save and load checkpoint
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "lora.pt"
            save_lora_checkpoint(save_path, model)
            self.assertTrue(save_path.exists())

            new_model = GPTModel(self.cfg)
            apply_lora_to_model(new_model, rank=4, alpha=8.0)
            load_lora_checkpoint(save_path, new_model)

    def test_dpo_loss(self):
        # 2 examples
        policy_chosen = torch.tensor([-1.2, -0.8])
        policy_rejected = torch.tensor([-2.5, -2.1])
        ref_chosen = torch.tensor([-1.5, -1.0])
        ref_rejected = torch.tensor([-1.6, -1.2])

        loss, chosen_r, rejected_r, acc = dpo_loss(
            policy_chosen_log_probs=policy_chosen,
            policy_rejected_log_probs=policy_rejected,
            ref_chosen_log_probs=ref_chosen,
            ref_rejected_log_probs=ref_rejected,
            beta=0.1,
        )

        self.assertTrue(loss.item() > 0.0)
        self.assertEqual(acc.item(), 1.0)
        self.assertTrue((chosen_r > rejected_r).all())

    def test_dpo_log_probs(self):
        logits = torch.randn(2, 6, 20)
        labels = torch.tensor([
            [-100, -100, 5, 8, 12, 18],
            [-100, 3, 7, 9, -100, -100],
        ])
        log_probs = compute_sequence_log_probs(logits, labels, ignore_index=-100)
        self.assertEqual(log_probs.shape, (2,))

    def test_huggingface_export(self):
        model = GPTModel(self.cfg)
        hf_dict = eru_to_hf_state_dict(model)

        self.assertIn("transformer.wte.weight", hf_dict)
        self.assertIn("transformer.wpe.weight", hf_dict)
        self.assertIn("transformer.h.0.attn.c_attn.weight", hf_dict)
        self.assertIn("transformer.ln_f.weight", hf_dict)
        self.assertIn("lm_head.weight", hf_dict)

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_to_huggingface(model, tmp_dir, cfg=self.cfg)
            out_path = Path(tmp_dir)
            self.assertTrue((out_path / "config.json").exists())
            self.assertTrue((out_path / "pytorch_model.bin").exists())
            self.assertTrue((out_path / "generation_config.json").exists())


if __name__ == "__main__":
    unittest.main()
