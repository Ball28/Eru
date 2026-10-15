"""
Fine-tuning pipelines for classification and instruction tuning.
"""

from eru.finetune.classification import (
    setup_spam_classifier_head,
    calc_accuracy_loader,
    calc_classification_loss_batch,
    train_classifier_simple,
    classify_review,
)
from eru.finetune.instruction import (
    generate_instruction_response,
)

__all__ = [
    "setup_spam_classifier_head",
    "calc_accuracy_loader",
    "calc_classification_loss_batch",
    "train_classifier_simple",
    "classify_review",
    "generate_instruction_response",
]
