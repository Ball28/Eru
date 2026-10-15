"""
Spam classification dataset and balancing utilities (Chapter 6).
"""

from typing import Any, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    Dataset = object
    DataLoader = object

try:
    import pandas as pd
except ImportError:
    pd = None


def create_balanced_dataset(df: Any, random_state: int = 123) -> Any:
    """
    Subsample majority 'ham' class to match count of minority 'spam' class.
    """
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(n=num_spam, random_state=random_state)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]]).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df


def random_split(
    df: Any,
    train_frac: float = 0.7,
    validation_frac: float = 0.1,
    random_state: int = 123,
) -> Tuple[Any, Any, Any]:
    """
    Randomly partition dataframe into train, validation, and test subsets.
    """
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_end = int(len(shuffled) * train_frac)
    val_end = train_end + int(len(shuffled) * validation_frac)
    train_df = shuffled.iloc[:train_end]
    val_df = shuffled.iloc[train_end:val_end]
    test_df = shuffled.iloc[val_end:]
    return train_df, val_df, test_df


class SpamDataset(Dataset):
    """
    PyTorch Dataset loading SMS Spam text and binary labels from CSV (Chapter 6, Listing 6.1).
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer: Any,
        max_length: Optional[int] = None,
        pad_token_id: int = 50256,
    ):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded[:self.max_length] for encoded in self.encoded_texts
            ]

        # Pad all sequences to max_length
        self.encoded_texts = [
            encoded + [pad_token_id] * (self.max_length - len(encoded))
            for encoded in self.encoded_texts
        ]

    def __getitem__(self, index: int):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        max_len = 0
        for encoded in self.encoded_texts:
            if len(encoded) > max_len:
                max_len = len(encoded)
        return max_len


def create_spam_dataloader(
    csv_file: str,
    tokenizer: Any,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Any:
    """Convenience helper to create a DataLoader for spam classification."""
    dataset = SpamDataset(csv_file=csv_file, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
