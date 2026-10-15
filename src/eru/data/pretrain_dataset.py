"""
Pretraining sliding-window dataset and dataloader (Chapter 2, Listing 2.5-2.6).
"""

from typing import Any, Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object

from eru.tokenizer.bpe import BPETokenizer


class GPTDatasetV1(Dataset):
    """
    Chunks a continuous tokenized text corpus into overlapping input/target sequence pairs of fixed length.
    """

    def __init__(self, txt: str, tokenizer: Any, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def create_pretrain_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: Optional[Any] = None,
) -> Any:
    """
    Convenience factory to create a DataLoader for pretraining with sliding-window chunking.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer()
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


# Backward-compatible alias
create_dataloader_v1 = create_pretrain_dataloader
