"""Group Relative Policy Optimization references and invariants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GroupAdvantageResult:
    """Group-standardized rewards plus zero-signal group flags."""

    advantages: Tensor
    means: Tensor
    standard_deviations: Tensor
    usable_groups: Tensor


@dataclass(frozen=True, slots=True)
class GRPOLossResult:
    """Token-level clipped policy loss and audit metrics."""

    loss: Tensor
    ratios: Tensor
    selected_objective: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    valid_tokens: int


def group_relative_advantages(rewards: Tensor, *, epsilon: float = 1e-6) -> GroupAdvantageResult:
    """Standardize rewards within each prompt and zero constant groups explicitly."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("GRPO rewards must be [prompts, group_size>=2]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    means = rewards.mean(dim=1, keepdim=True)
    standard_deviations = rewards.std(dim=1, unbiased=False, keepdim=True)
    usable = standard_deviations.squeeze(1) > epsilon
    standardized = (rewards - means) / standard_deviations.clamp_min(epsilon)
    advantages = torch.where(usable.unsqueeze(1), standardized, torch.zeros_like(standardized))
    return GroupAdvantageResult(
        advantages, means.squeeze(1), standard_deviations.squeeze(1), usable
    )


def grpo_policy_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_log_probs: Tensor | None = None,
    kl_beta: float = 0.0,
) -> GRPOLossResult:
    """Apply sequence advantages to token ratios and aggregate over valid response tokens."""
    if new_log_probs.shape != old_log_probs.shape or response_mask.shape != new_log_probs.shape:
        raise ValueError("GRPO token tensors must align")
    if advantages.ndim != 1 or advantages.shape[0] != new_log_probs.shape[0]:
        raise ValueError("GRPO needs one sequence advantage per row")
    if not 0.0 < clip_epsilon < 1.0 or kl_beta < 0:
        raise ValueError("invalid GRPO clip/KL controls")
    mask = response_mask.to(dtype=torch.bool)
    valid_tokens = int(mask.sum().item())
    if valid_tokens == 0:
        raise ValueError("GRPO selected no response tokens")
    ratios = torch.exp(new_log_probs - old_log_probs)
    expanded_advantages = advantages.unsqueeze(1)
    unclipped = ratios * expanded_advantages
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * expanded_advantages
    selected = torch.minimum(unclipped, clipped)
    objective = selected
    if reference_log_probs is not None:
        if reference_log_probs.shape != new_log_probs.shape:
            raise ValueError("GRPO reference log-probs must align")
        log_ratio = reference_log_probs - new_log_probs
        per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
        objective = objective - kl_beta * per_token_kl
    approximate_kl = ((old_log_probs - new_log_probs) * mask).sum() / valid_tokens
    clipped_items = ratios.ne(ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon))
    return GRPOLossResult(
        loss=-(objective * mask).sum() / valid_tokens,
        ratios=ratios,
        selected_objective=selected,
        approximate_kl=approximate_kl,
        clip_fraction=(clipped_items & mask).float().sum() / valid_tokens,
        valid_tokens=valid_tokens,
    )


def flatten_group_advantages(result: GroupAdvantageResult) -> Tensor:
    """Flatten prompt-major group advantages to match rollout rows."""
    return result.advantages.reshape(-1)


def assert_initial_on_policy(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    response_mask: Tensor,
    *,
    absolute_tolerance: float = 1e-5,
) -> None:
    """Reject a first GRPO step whose rollout and update policies do not match."""
    if new_log_probs.shape != old_log_probs.shape or response_mask.shape != new_log_probs.shape:
        raise ValueError("on-policy tensors must align")
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    mask = response_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        raise ValueError("on-policy check selected no response tokens")
    maximum_delta = (new_log_probs - old_log_probs).abs()[mask].max()
    if not torch.isfinite(maximum_delta) or float(maximum_delta.item()) > absolute_tolerance:
        raise RuntimeError(
            "initial GRPO update is not on-policy: "
            f"maximum log-probability delta={float(maximum_delta.item()):.8g}"
        )
