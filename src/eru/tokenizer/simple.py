import re
from typing import Dict, List, Optional


def build_vocab_from_text(raw_text: str, add_special_tokens: bool = False) -> Dict[str, int]:
    """
    Build a vocabulary dictionary mapping tokens to integer IDs from raw text.

    Args:
        raw_text: Source text to extract vocabulary from.
        add_special_tokens: If True, adds '<|endoftext|>' and '<|unk|>'.

    Returns:
        Dictionary mapping each token string to its unique integer ID.
    """
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    unique_tokens = sorted(list(set(preprocessed)))
    if add_special_tokens:
        unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(unique_tokens)}
    return vocab


class SimpleTokenizerV1:
    """
    Basic vocabulary-based tokenizer without unknown-token handling.
    """

    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = dict(vocab)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

    def encode(self, text: str) -> List[int]:
        """Convert input text into token IDs."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back into text, cleaning whitespace before punctuation."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    """
    Enhanced tokenizer handling unknown tokens with <|unk|> and end-of-text markers.
    """

    def __init__(self, vocab: Dict[str, int], unk_token: str = "<|unk|>"):
        self.str_to_int = dict(vocab)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}
        self.unk_token = unk_token

    def encode(self, text: str) -> List[int]:
        """Convert input text into token IDs, mapping unknown words to <|unk|>."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else self.unk_token for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: List[int], fix_punctuation: bool = True) -> str:
        """
        Convert token IDs back into text.

        Args:
            ids: List of token integers.
            fix_punctuation: If True (default), removes spaces before punctuation.
                             If False, reproduces the original verbatim bug for test parity.
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        if fix_punctuation:
            text = re.sub(r'\s([,.:;?_!"()\'])', r'\1', text)
        else:
            # Preserves original unrefactored SimpleTokenizerV2.py bug behavior
            text = re.sub(r'([,.:;?_!"()\']|--|\s)', r'\1', text)
        return text


# Backward-compatibility aliases matching original naming
SimpleTokenizationV1 = SimpleTokenizerV1
SimpleTokenizationV2 = SimpleTokenizerV2
