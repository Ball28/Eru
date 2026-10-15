"""
Golden characterization regression test for Eru package.
Verifies exact parity with original book baseline behavior.
"""

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eru.tokenizer.simple import SimpleTokenizerV1, SimpleTokenizerV2, build_vocab_from_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestEruGoldenParity(unittest.TestCase):

    def test_verdict_text_integrity(self):
        verdict_file = DATA_DIR / "the-verdict.txt"
        self.assertTrue(verdict_file.exists())
        with open(verdict_file, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertEqual(len(text), 20479)

    def test_tokenizer_v1_golden_parity(self):
        verdict_file = DATA_DIR / "the-verdict.txt"
        with open(verdict_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        vocab = build_vocab_from_text(raw_text, add_special_tokens=False)
        self.assertEqual(len(vocab), 1130)

        tokenizer = SimpleTokenizerV1(vocab)
        text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable \npride."""
        ids = tokenizer.encode(text)
        expected_ids = [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
        self.assertEqual(ids, expected_ids)
        decoded = tokenizer.decode(ids)
        expected_decoded = '" It\' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
        self.assertEqual(decoded, expected_decoded)

    def test_tokenizer_v2_golden_parity(self):
        verdict_file = DATA_DIR / "the-verdict.txt"
        with open(verdict_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        vocab = build_vocab_from_text(raw_text, add_special_tokens=True)
        self.assertEqual(len(vocab), 1132)

        tokenizer = SimpleTokenizerV2(vocab)
        text1 = "Hello, do you like tea?"
        text2 = "In the sunlit terraces of the palace."
        text = " <|endoftext|> ".join((text1, text2))
        ids = tokenizer.encode(text)
        expected_ids = [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
        self.assertEqual(ids, expected_ids)

        # Legacy decoding mode reproduces exact original output
        legacy_decode = tokenizer.decode(ids, fix_punctuation=False)
        expected_legacy = "<|unk|> , do you like tea ? <|endoftext|> In the sunlit terraces of the <|unk|> ."
        self.assertEqual(legacy_decode, expected_legacy)

        # Clean decoding mode fixes punctuation spacing
        clean_decode = tokenizer.decode(ids, fix_punctuation=True)
        expected_clean = "<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>."
        self.assertEqual(clean_decode, expected_clean)

    def test_spam_csv_files_integrity(self):
        for name, expected_lines in [("train.csv", 1046), ("validation.csv", 150), ("test.csv", 301)]:
            csv_path = DATA_DIR / "spam" / name
            self.assertTrue(csv_path.exists())
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), expected_lines)


if __name__ == "__main__":
    unittest.main()
