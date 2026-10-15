"""
Unit tests for SimpleTokenizerV1, SimpleTokenizerV2, and vocabulary building.
"""

import sys
import unittest
from pathlib import Path

# Ensure Eru/src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eru.tokenizer.simple import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    build_vocab_from_text,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "the-verdict.txt"


class TestTokenizers(unittest.TestCase):

    def setUp(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.raw_text = f.read()
        self.vocab_v1 = build_vocab_from_text(self.raw_text, add_special_tokens=False)
        self.vocab_v2 = build_vocab_from_text(self.raw_text, add_special_tokens=True)

    def test_vocab_sizes(self):
        self.assertEqual(len(self.vocab_v1), 1130)
        self.assertEqual(len(self.vocab_v2), 1132)
        self.assertEqual(self.vocab_v2["<|endoftext|>"], 1130)
        self.assertEqual(self.vocab_v2["<|unk|>"], 1131)

    def test_simple_tokenizer_v1_encode_decode(self):
        tokenizer = SimpleTokenizerV1(self.vocab_v1)
        text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable \npride."""
        ids = tokenizer.encode(text)
        expected_ids = [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
        self.assertEqual(ids, expected_ids)

        decoded = tokenizer.decode(ids)
        expected_text = '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
        self.assertEqual(decoded, expected_text)

    def test_simple_tokenizer_v2_unknown_tokens(self):
        tokenizer = SimpleTokenizerV2(self.vocab_v2)
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|endoftext|> ".join((text1, text2))

        ids = tokenizer.encode(text)
        expected_ids = [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
        self.assertEqual(ids, expected_ids)

        # Fixed decode behavior (cleans space before punctuation)
        decoded_clean = tokenizer.decode(ids, fix_punctuation=True)
        self.assertEqual(decoded_clean, "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.")

        # Backward compatibility mode reproducing original bug
        decoded_legacy = tokenizer.decode(ids, fix_punctuation=False)
        self.assertEqual(decoded_legacy, "<|unk|> , do you like tea ? <|endoftext|> In the sunlit terraces of the <|unk|> .")


if __name__ == "__main__":
    unittest.main()
