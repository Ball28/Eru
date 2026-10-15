"""
Byte Pair Encoding (BPE) wrapper using tiktoken (Chapters 2, 4, 5, 6, 7).
"""

from typing import List, Optional, Set


class BPETokenizer:
    """
    Production-grade Byte Pair Encoding (BPE) tokenizer wrapping OpenAI's tiktoken.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding(self.encoding_name)
        return self._tokenizer

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """Encode string into BPE token IDs."""
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back into string."""
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size of the BPE encoding."""
        return self.tokenizer.n_vocab
