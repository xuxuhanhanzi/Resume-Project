"""Small frozen-tensor references for modern post-training methods."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FrontierLoss:
    """One scalar loss plus the effective per-sequence weights."""

    loss: Tensor
    weights: Tensor


def dr_grpo_advantages(rewards: Tensor) -> Tensor:
    """Center group rewards without per-group standard-deviation scaling."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Dr. GRPO rewards must be [prompts, group]")
    return rewards - rewards.mean(dim=1, keepdim=True)


def dr_grpo_policy_loss(
    token_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    constant_normalizer: int,
) -> Tensor:
    """Use a fixed token normalizer to expose the length-normalization difference."""
    if token_log_probs.shape != response_mask.shape:
        raise ValueError("Dr. GRPO token tensors must align")
    if advantages.shape != (token_log_probs.shape[0],) or constant_normalizer <= 0:
        raise ValueError("invalid Dr. GRPO advantages/normalizer")
    denominator = token_log_probs.shape[0] * constant_normalizer
    return -(token_log_probs * response_mask * advantages.unsqueeze(1)).sum() / denominator


def dapo_clipped_objective(
    ratios: Tensor,
    advantages: Tensor,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.28,
) -> Tensor:
    """DAPO Clip-Higher: decouple the upper clipping bound from the lower one."""
    if ratios.shape != advantages.shape or not 0 < epsilon_low < epsilon_high < 1:
        raise ValueError("invalid DAPO tensors or asymmetric clip bounds")
    clipped = ratios.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    return torch.minimum(ratios * advantages, clipped * advantages)


def dynamic_sampling_mask(group_rewards: Tensor, *, epsilon: float = 1e-6) -> Tensor:
    """Keep only groups containing a non-constant reward signal."""
    if group_rewards.ndim != 2:
        raise ValueError("dynamic sampling expects [prompts, group]")
    return group_rewards.std(dim=1, unbiased=False) > epsilon


def overlong_shaped_reward(
    rewards: Tensor,
    lengths: Tensor,
    *,
    maximum_length: int,
    buffer_length: int,
) -> Tensor:
    """Apply a smooth linear penalty in the final length buffer, then zero beyond max."""
    if (
        rewards.shape != lengths.shape
        or maximum_length <= 0
        or not 0 < buffer_length < maximum_length
    ):
        raise ValueError("invalid overlong reward inputs")
    start = maximum_length - buffer_length
    remaining = (maximum_length - lengths).clamp(min=0, max=buffer_length).to(rewards.dtype)
    factor = torch.where(lengths <= start, torch.ones_like(rewards), remaining / buffer_length)
    return rewards * factor


def gspo_sequence_weights(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    response_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> Tensor:
    """Compute clipped geometric-mean sequence ratios used by GSPO-style studies."""
    if new_log_probs.shape != old_log_probs.shape or response_mask.shape != new_log_probs.shape:
        raise ValueError("GSPO tensors must align")
    lengths = response_mask.sum(dim=1).clamp_min(1)
    mean_log_ratio = ((new_log_probs - old_log_probs) * response_mask).sum(dim=1) / lengths
    return mean_log_ratio.exp().clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)


def vespo_gamma_weights(
    new_log_probs: Tensor,
    behavior_log_probs: Tensor,
    response_mask: Tensor,
    advantages: Tensor,
    *,
    k_positive: float = 2.0,
    lambda_positive: float = 3.0,
    k_negative: float = 3.0,
    lambda_negative: float = 2.0,
) -> Tensor:
    """Compute VESPO's detached gamma kernel on true sequence importance weights."""
    if (
        new_log_probs.shape != behavior_log_probs.shape
        or response_mask.shape != new_log_probs.shape
    ):
        raise ValueError("VESPO token tensors must align")
    if advantages.shape != (new_log_probs.shape[0],):
        raise ValueError("VESPO requires one advantage per sequence")
    if min(k_positive, lambda_positive, k_negative, lambda_negative) <= 0:
        raise ValueError("VESPO gamma parameters must be positive")
    sequence_log_ratio = ((new_log_probs - behavior_log_probs).clamp(-20, 20) * response_mask).sum(
        dim=1
    )
    weights = sequence_log_ratio.clamp(-20, 20).exp().detach()
    positive = advantages >= 0
    k = torch.where(positive, k_positive, k_negative)
    lam = torch.where(positive, lambda_positive, lambda_negative)
    log_phi = lam + k * weights.clamp_min(1e-8).log() - lam * weights
    return torch.nan_to_num(log_phi.exp().detach(), nan=0.0, posinf=0.0, neginf=0.0)


def opd_delta_reward(
    teacher_log_probs: Tensor,
    student_log_probs: Tensor,
    *,
    maximum_reward: float,
) -> Tensor:
    """Kimi K3-style stop-gradient per-token teacher/student log-ratio reward."""
    if teacher_log_probs.shape != student_log_probs.shape or maximum_reward <= 0:
        raise ValueError("invalid OPD tensors or clipping threshold")
    return (teacher_log_probs - student_log_probs).detach().clamp(-maximum_reward, maximum_reward)


def multi_teacher_opd_reward(
    teacher_log_probs: Tensor,
    student_log_probs: Tensor,
    teacher_indices: Tensor,
    *,
    maximum_reward: float,
) -> Tensor:
    """Select one domain/effort teacher per sample before computing dense OPD rewards."""
    if teacher_log_probs.ndim != 3 or student_log_probs.ndim != 2:
        raise ValueError("teachers must be [B,K,T] and student [B,T]")
    if (
        teacher_log_probs.shape[0] != student_log_probs.shape[0]
        or teacher_log_probs.shape[2] != student_log_probs.shape[1]
    ):
        raise ValueError("teacher/student batch and token dimensions must align")
    if teacher_indices.shape != (student_log_probs.shape[0],):
        raise ValueError("one teacher index is required per sequence")
    if torch.any((teacher_indices < 0) | (teacher_indices >= teacher_log_probs.shape[1])):
        raise ValueError("teacher index is out of range")
    rows = torch.arange(student_log_probs.shape[0], device=student_log_probs.device)
    selected = teacher_log_probs[rows, teacher_indices]
    return opd_delta_reward(selected, student_log_probs, maximum_reward=maximum_reward)


def effective_sample_size(weights: Tensor) -> float:
    """Return the importance-weight ESS used to audit staleness variance."""
    if weights.ndim != 1 or weights.numel() == 0 or torch.any(weights < 0):
        raise ValueError("ESS weights must be a non-empty non-negative vector")
    denominator = float(weights.square().sum().item())
    if math.isclose(denominator, 0.0):
        return 0.0
    numerator = float(weights.sum().item()) ** 2
    return numerator / denominator
