from __future__ import annotations

import pytest
import torch

from forgellm.alignment.frontier import (
    dapo_clipped_objective,
    dr_grpo_advantages,
    dynamic_sampling_mask,
    effective_sample_size,
    gspo_sequence_weights,
    multi_teacher_opd_reward,
    opd_delta_reward,
    overlong_shaped_reward,
    vespo_gamma_weights,
)
from forgellm.alignment.grpo import (
    assert_initial_on_policy,
    group_relative_advantages,
    grpo_policy_loss,
)
from forgellm.alignment.policy_gradient import (
    categorical_expected_return,
    exact_policy_loss,
    reinforce_loss,
)
from forgellm.alignment.ppo import assert_policy_revision, clipped_policy_loss


def test_exact_policy_gradient_matches_score_function_enumeration() -> None:
    logits = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    rewards = torch.tensor([-1.0, 0.5, 2.0])
    exact = exact_policy_loss(logits, rewards)
    exact_gradient = torch.autograd.grad(exact, logits)[0]
    probabilities = logits.softmax(dim=-1)
    manual = torch.zeros_like(logits)
    for action in range(logits.numel()):
        one_hot = torch.nn.functional.one_hot(torch.tensor(action), logits.numel()).float()
        manual -= probabilities[action] * rewards[action] * (one_hot - probabilities)
    torch.testing.assert_close(exact_gradient, manual)
    assert categorical_expected_return(logits, rewards).item() > 0


def test_action_independent_baseline_does_not_change_expected_gradient() -> None:
    logits = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    rewards = torch.tensor([-1.0, 0.5, 2.0])
    probabilities = logits.softmax(dim=-1).detach()
    log_probs = logits.log_softmax(dim=-1)
    no_baseline = -(probabilities * rewards * log_probs).sum()
    centered = -(probabilities * (rewards - 0.75) * log_probs).sum()
    no_baseline_gradient = torch.autograd.grad(no_baseline, logits, retain_graph=True)[0]
    centered_gradient = torch.autograd.grad(centered, logits)[0]
    torch.testing.assert_close(no_baseline_gradient, centered_gradient)
    assert reinforce_loss(log_probs, rewards, 0.75).isfinite()


def test_ppo_clip_branches_for_positive_and_negative_advantage() -> None:
    ratios = torch.tensor([1.5, 1.5, 0.5, 0.5])
    result = clipped_policy_loss(
        ratios.log(), torch.zeros_like(ratios), torch.tensor([1.0, -1.0, 1.0, -1.0])
    )
    expected = torch.tensor([1.2, -1.5, 0.5, -0.8])
    torch.testing.assert_close(result.selected_objective, expected)
    assert result.clip_fraction.item() == 1.0
    with pytest.raises(RuntimeError, match="mismatch"):
        assert_policy_revision(expected="old-v1", actual="new-v2")


def test_grpo_zero_variance_permutation_and_loss() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [7.0, 7.0, 7.0, 7.0]])
    result = group_relative_advantages(rewards)
    assert result.usable_groups.tolist() == [True, False]
    torch.testing.assert_close(result.advantages[1], torch.zeros(4))
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = group_relative_advantages(rewards[:, permutation])
    torch.testing.assert_close(permuted.advantages, result.advantages[:, permutation])
    new = torch.log(torch.tensor([[1.1, 0.9], [1.0, 1.0]]))
    old = torch.zeros_like(new)
    mask = torch.tensor([[True, True], [True, False]])
    loss = grpo_policy_loss(new, old, torch.tensor([1.0, -1.0]), mask)
    assert loss.valid_tokens == 3 and loss.loss.isfinite()


def test_initial_grpo_policy_gate_detects_stale_or_mode_mismatched_log_probs() -> None:
    old = torch.tensor([[0.0, -0.2], [-0.4, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    assert_initial_on_policy(old.clone(), old, mask)
    mismatched = old.clone()
    mismatched[0, 1] += 1e-3
    with pytest.raises(RuntimeError, match="not on-policy"):
        assert_initial_on_policy(mismatched, old, mask)


def test_frontier_single_variable_references() -> None:
    rewards = torch.tensor([[1.0, 2.0, 5.0]])
    torch.testing.assert_close(dr_grpo_advantages(rewards).sum(dim=1), torch.zeros(1))
    ratios = torch.tensor([1.4, 0.7])
    advantages = torch.tensor([1.0, -1.0])
    dapo = dapo_clipped_objective(ratios, advantages)
    torch.testing.assert_close(dapo, torch.tensor([1.28, -0.8]))
    assert dynamic_sampling_mask(torch.tensor([[1.0, 1.0], [0.0, 1.0]])).tolist() == [
        False,
        True,
    ]
    shaped = overlong_shaped_reward(
        torch.ones(4), torch.tensor([5, 8, 9, 11]), maximum_length=10, buffer_length=2
    )
    torch.testing.assert_close(shaped, torch.tensor([1.0, 1.0, 0.5, 0.0]))


def test_gspo_vespo_and_multi_teacher_opd_contracts() -> None:
    new = torch.tensor([[0.1, 0.2], [0.5, -0.3]])
    old = torch.zeros_like(new)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    gspo = gspo_sequence_weights(new, old, mask)
    torch.testing.assert_close(gspo, torch.tensor([float(torch.exp(torch.tensor(0.15))), 1.2]))
    vespo = vespo_gamma_weights(new, old, mask, torch.tensor([1.0, -1.0]))
    assert vespo.shape == (2,) and torch.all(torch.isfinite(vespo))
    teacher = torch.tensor([[[0.0, -0.1], [0.4, 0.3]], [[-0.2, 0.2], [0.1, 0.1]]])
    student = torch.zeros((2, 2))
    selected = multi_teacher_opd_reward(teacher, student, torch.tensor([1, 0]), maximum_reward=0.25)
    expected = torch.stack(
        [
            opd_delta_reward(teacher[0, 1], student[0], maximum_reward=0.25),
            opd_delta_reward(teacher[1, 0], student[1], maximum_reward=0.25),
        ]
    )
    torch.testing.assert_close(selected, expected)
    assert effective_sample_size(torch.tensor([1.0, 1.0, 1.0])) == pytest.approx(3.0)
