"""Assistant-only causal loss and small inspectable SFT utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from forgellm.post_training.chat_template import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class SFTLoss:
    """Token-normalized SFT loss with explicit numerator and accuracy."""

    loss: Tensor
    negative_log_likelihood_sum: Tensor
    target_tokens: int
    correct_tokens: int

    @property
    def token_accuracy(self) -> float:
        """Return teacher-forced assistant-token accuracy."""
        return self.correct_tokens / self.target_tokens


def assistant_only_causal_loss(logits: Tensor, labels: Tensor) -> SFTLoss:
    """Shift once and average NLL only across non-ignored assistant targets."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [B, T, V] and labels must be [B, T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits batch/sequence dimensions must match labels")
    if labels.size(1) < 2:
        raise ValueError("causal SFT requires at least two token positions")
    predictions = logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    selected = targets.ne(IGNORE_INDEX)
    target_tokens = int(selected.sum().item())
    if target_tokens == 0:
        raise ValueError("shifted labels contain no supervised assistant targets")
    nll_sum = F.cross_entropy(
        predictions.view(-1, predictions.size(-1)),
        targets.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    predicted_ids = predictions.argmax(dim=-1)
    correct_tokens = int(((predicted_ids == targets) & selected).sum().item())
    loss = nll_sum / target_tokens
    if not torch.isfinite(loss):
        raise FloatingPointError("SFT loss is not finite")
    return SFTLoss(
        loss=loss,
        negative_log_likelihood_sum=nll_sum,
        target_tokens=target_tokens,
        correct_tokens=correct_tokens,
    )


def scale_gradients_by_token_count(parameters: list[Tensor], target_tokens: int) -> None:
    """Normalize accumulated summed-NLL gradients by their true token denominator."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    scale = 1.0 / target_tokens
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
