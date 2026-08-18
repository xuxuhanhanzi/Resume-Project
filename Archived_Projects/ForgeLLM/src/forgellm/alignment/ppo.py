"""Proximal Policy Optimization tensor references."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PPOLossResult:
    """Clipped surrogate details."""

    loss: Tensor
    ratios: Tensor
    unclipped_objective: Tensor
    clipped_objective: Tensor
    selected_objective: Tensor
    clip_fraction: Tensor


def clipped_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> PPOLossResult:
    """Compute PPO's pessimistic clipped surrogate for either advantage sign."""
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != advantages.shape:
        raise ValueError("PPO tensors must have identical shapes")
    if new_log_probs.numel() == 0 or not 0.0 < clip_epsilon < 1.0:
        raise ValueError("PPO needs non-empty tensors and clip epsilon in (0, 1)")
    ratios = torch.exp(new_log_probs - old_log_probs)
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratios * advantages
    clipped = clipped_ratios * advantages
    selected = torch.minimum(unclipped, clipped)
    clipped_items = ratios.ne(clipped_ratios)
    return PPOLossResult(
        loss=-selected.mean(),
        ratios=ratios,
        unclipped_objective=unclipped,
        clipped_objective=clipped,
        selected_objective=selected,
        clip_fraction=clipped_items.float().mean(),
    )


def clipped_value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> Tensor:
    """PPO value loss using the larger of clipped and unclipped squared errors."""
    if new_values.shape != old_values.shape or new_values.shape != returns.shape:
        raise ValueError("value tensors must align")
    clipped = old_values + (new_values - old_values).clamp(-clip_epsilon, clip_epsilon)
    return 0.5 * torch.maximum((new_values - returns).square(), (clipped - returns).square()).mean()


def assert_policy_revision(*, expected: str, actual: str, name: str = "old policy") -> None:
    """Reject updates that mix logits from an unknown behavior policy."""
    if not expected or not actual or expected != actual:
        raise RuntimeError(f"{name} revision mismatch: expected={expected!r}, actual={actual!r}")
