"""Bradley-Terry reward modeling references and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PairwiseRewardResult:
    """Pairwise loss plus metrics that do not imply generation quality."""

    loss: Tensor
    margins: Tensor
    accuracy: Tensor


def bradley_terry_loss(chosen_rewards: Tensor, rejected_rewards: Tensor) -> PairwiseRewardResult:
    """Compute -log sigmoid(r_chosen-r_rejected) over equally shaped rewards."""
    if chosen_rewards.shape != rejected_rewards.shape or chosen_rewards.numel() == 0:
        raise ValueError("chosen/rejected rewards must have the same non-empty shape")
    margins = chosen_rewards - rejected_rewards
    losses = -F.logsigmoid(margins)
    return PairwiseRewardResult(
        loss=losses.mean(),
        margins=margins,
        accuracy=(margins > 0).to(dtype=chosen_rewards.dtype).mean(),
    )


def pad_byte_sequences(
    texts: list[str], *, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Encode UTF-8 bytes as 1..256 and reserve 0 for padding."""
    if not texts or any(not text for text in texts):
        raise ValueError("tiny reward model requires non-empty texts")
    sequences = [[byte + 1 for byte in text.encode("utf-8")] for text in texts]
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = True
    return input_ids, mask


class TinyRewardModel(nn.Module):
    """Small byte-level scalar reward model used only for the overfit gate."""

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.embedding = nn.Embedding(257, hidden_size, padding_idx=0)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return one scalar per byte sequence."""
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids/mask must be aligned rank-2 tensors")
        if not torch.all(attention_mask.any(dim=1)):
            raise ValueError("every reward-model sequence needs at least one token")
        embedded = self.embedding(input_ids)
        encoded, _ = self.encoder(embedded)
        last_indices = attention_mask.sum(dim=1).to(dtype=torch.long) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_hidden = encoded[rows, last_indices]
        return cast(Tensor, self.reward_head(last_hidden).squeeze(-1))


def pearson_correlation(left: Tensor, right: Tensor) -> float:
    """Compute Pearson correlation or return 0 for a constant/short vector."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("correlation vectors must be aligned and rank 1")
    if left.numel() < 2:
        return 0.0
    centered_left = left.float() - left.float().mean()
    centered_right = right.float() - right.float().mean()
    denominator = torch.sqrt(centered_left.square().sum() * centered_right.square().sum())
    if denominator.item() == 0.0:
        return 0.0
    return float((centered_left * centered_right).sum().item() / denominator.item())
