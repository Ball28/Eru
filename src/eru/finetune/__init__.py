"""
Fine-tuning and alignment engines for Eru LLM:
- Sequence classification (Spam detection)
- Instruction fine-tuning (SFT)
- Low-Rank Adaptation (LoRA PEFT)
- Direct Preference Optimization (DPO alignment)
"""

from eru.finetune.classification import (
    setup_spam_classifier_head,
    train_classifier_simple,
    classify_review,
)
from eru.finetune.instruction import generate_instruction_response
from eru.finetune.lora import (
    LoRALinear,
    apply_lora_to_model,
    save_lora_checkpoint,
    load_lora_checkpoint,
    get_lora_state_dict,
)
from eru.finetune.dpo import dpo_loss, DPODataset, compute_sequence_log_probs

__all__ = [
    "setup_spam_classifier_head",
    "train_classifier_simple",
    "classify_review",
    "generate_instruction_response",
    "LoRALinear",
    "apply_lora_to_model",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "get_lora_state_dict",
    "dpo_loss",
    "DPODataset",
    "compute_sequence_log_probs",
]
