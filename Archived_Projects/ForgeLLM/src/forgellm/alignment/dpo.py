"""Direct Preference Optimization with explicit response-only accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from forgellm.post_training.chat_template import IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class SequenceLogProb:
    """Token-summed response log-probabilities and their denominators."""

    sums: Tensor
    token_counts: Tensor


@dataclass(frozen=True, slots=True)
class DPOResult:
    """Standard reverse-KL sigmoid DPO outputs."""

    loss: Tensor
    losses: Tensor
    logits: Tensor
    chosen_implicit_rewards: Tensor
    rejected_implicit_rewards: Tensor
    preference_accuracy: Tensor


def response_sequence_log_probs(logits: Tensor, labels: Tensor) -> SequenceLogProb:
    """Sum shifted causal log-probs only where labels are not IGNORE_INDEX."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [B,T,V] and labels [B,T] must align")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("each DPO sequence needs at least one response target")
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = (
        F.log_softmax(shifted_logits, dim=-1)
        .gather(dim=-1, index=safe_labels.unsqueeze(-1))
        .squeeze(-1)
    )
    return SequenceLogProb(
        sums=(token_log_probs * mask).sum(dim=-1),
        token_counts=mask.sum(dim=-1),
    )


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    *,
    beta: float = 0.1,
) -> DPOResult:
    """Compute standard DPO exactly as the TRL sigmoid/reverse-KL branch."""
    shapes = {
        tuple(policy_chosen_logps.shape),
        tuple(policy_rejected_logps.shape),
        tuple(reference_chosen_logps.shape),
        tuple(reference_rejected_logps.shape),
    }
    if len(shapes) != 1 or policy_chosen_logps.numel() == 0:
        raise ValueError("four DPO log-prob tensors must share a non-empty shape")
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    logits = chosen_logratios - rejected_logratios
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * chosen_logratios.detach()
    rejected_rewards = beta * rejected_logratios.detach()
    return DPOResult(
        loss=losses.mean(),
        losses=losses,
        logits=logits,
        chosen_implicit_rewards=chosen_rewards,
        rejected_implicit_rewards=rejected_rewards,
        preference_accuracy=(chosen_rewards > rejected_rewards).float().mean(),
    )


def assert_reference_frozen(model: nn.Module) -> None:
    """Fail if any reference parameter can train or already owns a gradient."""
    offenders = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or parameter.grad is not None
    ]
    if offenders:
        raise RuntimeError(f"reference model is not frozen: {offenders[:3]}")


def log_prob_sum_from_ids(logits: Tensor, token_ids: Tensor, mask: Tensor) -> Tensor:
    """Independent non-shifted helper used for sampled rollout tokens."""
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or mask.shape != token_ids.shape:
        raise ValueError("rollout logits, token IDs and mask must align")
    gathered = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(dim=-1, index=token_ids.unsqueeze(-1))
        .squeeze(-1)
    )
    return (gathered * mask).sum(dim=-1)
