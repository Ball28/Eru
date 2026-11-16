"""
Direct Preference Optimization (DPO) training and loss computation.
Reference: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (NeurIPS 2023).
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})
    F = None


def compute_sequence_log_probs(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    ignore_index: int = -100,
) -> "torch.Tensor":
    """
    Compute log probabilities for each sequence, ignoring masked prompt positions.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size).
        labels: Ground truth token IDs of shape (batch_size, seq_len).
        ignore_index: Index identifying masked prompt or padding tokens.

    Returns:
        Per-example sum of log probabilities of shape (batch_size,).
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_mask = shift_labels != ignore_index
    vocab_size = shift_logits.shape[-1]
    clamped_labels = shift_labels.unsqueeze(2).clamp(min=0, max=vocab_size - 1)

    # Gather log-probabilities of true tokens
    per_token_log_probs = torch.gather(
        shift_logits.log_softmax(dim=-1),
        dim=2,
        index=clamped_labels,
    ).squeeze(2)

    return (per_token_log_probs * loss_mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_log_probs: "torch.Tensor",
    policy_rejected_log_probs: "torch.Tensor",
    ref_chosen_log_probs: "torch.Tensor",
    ref_rejected_log_probs: "torch.Tensor",
    beta: float = 0.1,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Compute the Direct Preference Optimization (DPO) objective.

    Args:
        policy_chosen_log_probs: Log probs of winner completion under active policy pi_theta.
        policy_rejected_log_probs: Log probs of loser completion under active policy pi_theta.
        ref_chosen_log_probs: Log probs of winner completion under frozen reference model pi_ref.
        ref_rejected_log_probs: Log probs of loser completion under frozen reference model pi_ref.
        beta: Temperature scaling parameter controlling deviation from reference model (default: 0.1).

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards, accuracy).
    """
    pi_logratios = policy_chosen_log_probs - policy_rejected_log_probs
    ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs

    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_log_probs - ref_chosen_log_probs).detach()
    rejected_rewards = beta * (policy_rejected_log_probs - ref_rejected_log_probs).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss, chosen_rewards, rejected_rewards, accuracy


class DPODataset:
    """
    Dataset container for preference tuning with (prompt, chosen, rejected) triplets.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any,
        max_length: int = 512,
        pad_token_id: int = 50256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.items = data

    def __len__(self) -> int:
        return len(self.items)

    def _format_and_tokenize(
        self, prompt: str, completion: str
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        prompt_ids = self.tokenizer.encode(prompt)
        completion_ids = self.tokenizer.encode(completion)
        input_ids = prompt_ids + completion_ids + [self.pad_token_id]

        labels = [-100] * len(prompt_ids) + completion_ids + [self.pad_token_id]

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(
        self, idx: int
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        entry = self.items[idx]
        prompt = entry["prompt"]
        chosen_input, chosen_labels = self._format_and_tokenize(prompt, entry["chosen"])
        rejected_input, rejected_labels = self._format_and_tokenize(prompt, entry["rejected"])
        return chosen_input, chosen_labels, rejected_input, rejected_labels
