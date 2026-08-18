from __future__ import annotations

import torch
from torch.nn import functional as F

from forgellm.alignment.dpo import dpo_loss, response_sequence_log_probs
from forgellm.alignment.reward import bradley_terry_loss
from forgellm.post_training.chat_template import IGNORE_INDEX


def test_bradley_terry_matches_manual_and_swap_reverses_margin() -> None:
    chosen = torch.tensor([2.0, -0.5], requires_grad=True)
    rejected = torch.tensor([0.5, -1.0], requires_grad=True)
    result = bradley_terry_loss(chosen, rejected)
    expected = -torch.log(torch.sigmoid(chosen - rejected)).mean()
    torch.testing.assert_close(result.loss, expected)
    swapped = bradley_terry_loss(rejected, chosen)
    torch.testing.assert_close(swapped.margins, -result.margins)
    result.loss.backward()  # type: ignore[no-untyped-call]
    assert chosen.grad is not None and torch.all(chosen.grad < 0)
    assert rejected.grad is not None and torch.all(rejected.grad > 0)


def test_response_log_probs_ignore_prompt_padding_and_apply_shift() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.0, -1.0],
                [0.0, 3.0, -1.0],
                [0.0, -1.0, 4.0],
                [9.0, -9.0, -9.0],
            ]
        ]
    )
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, IGNORE_INDEX]])
    result = response_sequence_log_probs(logits, labels)
    expected = F.log_softmax(logits[:, 1].float(), dim=-1)[:, 2]
    torch.testing.assert_close(result.sums, expected)
    assert result.token_counts.tolist() == [1]


def test_dpo_matches_independent_trl_sigmoid_branch_and_gradients() -> None:
    policy_chosen = torch.tensor([-1.1, -0.7], requires_grad=True)
    policy_rejected = torch.tensor([-1.4, -1.2], requires_grad=True)
    reference_chosen = torch.tensor([-1.0, -0.9])
    reference_rejected = torch.tensor([-1.2, -1.0])
    beta = 0.1
    result = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )
    trl_delta = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    trl_sigmoid_loss = -F.logsigmoid(beta * trl_delta)
    torch.testing.assert_close(result.logits, trl_delta)
    torch.testing.assert_close(result.losses, trl_sigmoid_loss)
    independent_gradients = torch.autograd.grad(
        trl_sigmoid_loss.mean(), (policy_chosen, policy_rejected)
    )
    actual_gradients = torch.autograd.grad(result.loss, (policy_chosen, policy_rejected))
    for actual, expected in zip(actual_gradients, independent_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_dpo_chosen_rejected_swap_reverses_logit() -> None:
    result = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-2.0]),
        torch.tensor([-1.4]),
        torch.tensor([-1.8]),
    )
    swapped = dpo_loss(
        torch.tensor([-2.0]),
        torch.tensor([-1.0]),
        torch.tensor([-1.8]),
        torch.tensor([-1.4]),
    )
    torch.testing.assert_close(swapped.logits, -result.logits)
