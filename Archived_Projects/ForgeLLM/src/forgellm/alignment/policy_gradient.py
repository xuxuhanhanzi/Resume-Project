"""Exact and Monte Carlo policy-gradient references for categorical policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ReinforceBatch:
    """One sampled bandit batch with an auditable baseline."""

    actions: Tensor
    rewards: Tensor
    log_probs: Tensor
    advantages: Tensor
    loss: Tensor


def categorical_expected_return(logits: Tensor, rewards: Tensor) -> Tensor:
    """Compute the exact expected reward for a finite categorical bandit."""
    if logits.ndim != 1 or rewards.shape != logits.shape or logits.numel() < 2:
        raise ValueError("bandit logits/rewards must be aligned rank-1 tensors")
    return (logits.softmax(dim=-1) * rewards).sum()


def exact_policy_loss(logits: Tensor, rewards: Tensor) -> Tensor:
    """Return negative exact expected return for gradient-descent optimizers."""
    return -categorical_expected_return(logits, rewards)


def reinforce_loss(log_probs: Tensor, rewards: Tensor, baseline: Tensor | float = 0.0) -> Tensor:
    """Score-function loss with a detached action-independent baseline."""
    if log_probs.shape != rewards.shape or log_probs.numel() == 0:
        raise ValueError("REINFORCE log-probs/rewards must have the same non-empty shape")
    baseline_tensor = torch.as_tensor(baseline, dtype=rewards.dtype, device=rewards.device)
    if baseline_tensor.numel() not in (1, rewards.numel()):
        raise ValueError("baseline must be scalar or align with rewards")
    advantages = rewards - baseline_tensor.detach()
    return -(advantages.detach() * log_probs).mean()


def sample_reinforce(
    logits: Tensor,
    action_rewards: Tensor,
    *,
    samples: int,
    generator: torch.Generator,
    baseline: float = 0.0,
) -> ReinforceBatch:
    """Sample a bandit batch while keeping actions and log-probs version-consistent."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    probabilities = logits.softmax(dim=-1)
    actions = torch.multinomial(probabilities, samples, replacement=True, generator=generator)
    rewards = action_rewards[actions]
    log_probs = F.log_softmax(logits, dim=-1)[actions]
    advantages = rewards - baseline
    return ReinforceBatch(
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        advantages=advantages,
        loss=reinforce_loss(log_probs, rewards, baseline),
    )


def entropy_from_logits(logits: Tensor) -> Tensor:
    """Categorical entropy with stable log-softmax."""
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def categorical_kl(logits: Tensor, reference_logits: Tensor) -> Tensor:
    """Compute KL(policy || reference) for aligned categorical logits."""
    if logits.shape != reference_logits.shape:
        raise ValueError("policy/reference logits must align")
    log_policy = F.log_softmax(logits, dim=-1)
    log_reference = F.log_softmax(reference_logits, dim=-1)
    policy = log_policy.exp()
    return (policy * (log_policy - log_reference)).sum(dim=-1)
