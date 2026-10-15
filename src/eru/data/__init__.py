"""
Data loading, processing, and collate utilities for Eru LLM.
"""

from eru.data.pretrain_dataset import (
    GPTDatasetV1,
    create_pretrain_dataloader,
    create_dataloader_v1,
)
from eru.data.spam_dataset import (
    SpamDataset,
    create_spam_dataloader,
    create_balanced_dataset,
    random_split,
)
from eru.data.instruction_dataset import (
    InstructionDataset,
    format_input,
)
from eru.data.collate import custom_collate_fn
from eru.data.download import (
    download_file,
    download_spam_dataset,
    download_instruction_data,
)

__all__ = [
    "GPTDatasetV1",
    "create_pretrain_dataloader",
    "create_dataloader_v1",
    "SpamDataset",
    "create_spam_dataloader",
    "create_balanced_dataset",
    "random_split",
    "InstructionDataset",
    "format_input",
    "custom_collate_fn",
    "download_file",
    "download_spam_dataset",
    "download_instruction_data",
]
