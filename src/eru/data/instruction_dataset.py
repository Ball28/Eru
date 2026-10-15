"""
Instruction fine-tuning dataset and prompt formatting.
"""

from typing import Any, Dict, List

try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    Dataset = object
    DataLoader = object


def format_input(entry: Dict[str, str]) -> str:
    """
    Format an instruction dictionary into Alpaca prompt style.

    Args:
        entry: Dict containing keys 'instruction', 'input', and optionally 'output'.
    """
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """
    PyTorch Dataset loading instruction pairs and concatenating formatted prompt with target response.
    """

    def __init__(self, data: List[Dict[str, str]], tokenizer: Any):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry.get('output', '')}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index: int) -> List[int]:
        return self.encoded_texts[index]

    def __len__(self) -> int:
        return len(self.data)
