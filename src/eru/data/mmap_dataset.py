"""
High-throughput memory-mapped binary token dataset for scalable pretraining.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
except ImportError:
    np = None
    torch = None
    Dataset = object


class MMapTokenDataset(Dataset):
    """
    Zero-RAM, memory-mapped token dataset reading directly from pre-tokenized binary files.
    Allows pretraining on massive multi-gigabyte corpora (FineWeb-Edu, OpenWebText, TinyStories)
    with instant startup and zero memory overhead.

    Args:
        bin_path: Path to the binary file containing uint16 or uint32 token IDs.
        context_length: Number of tokens per training input sequence.
        stride: Step stride between consecutive training windows (defaults to context_length).
        dtype: Numpy dtype of the binary file (default: np.uint16).
    """

    def __init__(
        self,
        bin_path: Union[str, Path],
        context_length: int,
        stride: Optional[int] = None,
        dtype=None,
    ):
        super().__init__()
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary token dataset not found: {self.bin_path}")

        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        self.dtype = dtype if dtype is not None else (np.uint16 if np else None)

        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        total_tokens = len(self.data)

        if total_tokens <= context_length:
            raise ValueError(
                f"Binary dataset contains {total_tokens} tokens, but context_length is {context_length}."
            )

        self.num_samples = (total_tokens - context_length - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length + 1
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))

        input_tokens = chunk[:-1]
        target_tokens = chunk[1:]
        return input_tokens, target_tokens
