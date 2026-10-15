"""
Instruction fine-tuning (Supervised Fine-Tuning / SFT) engine (Chapter 7).
"""

from typing import Any, Dict, List, Optional, Tuple

from eru.data.instruction_dataset import format_input, InstructionDataset
from eru.data.collate import custom_collate_fn
from eru.generation.sampling import generate
from eru.tokenizer.utils import text_to_token_ids, token_ids_to_text
from eru.training.trainer import train_model_simple


def generate_instruction_response(
    model: Any,
    tokenizer: Any,
    instruction_entry: Dict[str, str],
    device: Any,
    context_size: int = 1024,
    max_new_tokens: int = 256,
    eos_id: int = 50256,
) -> str:
    """
    Format an instruction entry, generate completion, and strip prompt prefix.
    """
    model.eval()
    formatted_prompt = format_input(instruction_entry)
    input_ids = text_to_token_ids(formatted_prompt, tokenizer).to(device)

    token_ids = generate(
        model=model,
        idx=input_ids,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        eos_id=eos_id,
        temperature=0.0,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(formatted_prompt):].replace("### Response:", "").strip()
    return response_text
