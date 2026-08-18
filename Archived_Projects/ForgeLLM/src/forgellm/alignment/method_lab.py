"""Bounded Stage 5 algorithm experiments with fixed seeds and no large model."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor

from forgellm.alignment.frontier import (
    dapo_clipped_objective,
    dr_grpo_advantages,
    dynamic_sampling_mask,
    effective_sample_size,
    gspo_sequence_weights,
    multi_teacher_opd_reward,
    overlong_shaped_reward,
    vespo_gamma_weights,
)
from forgellm.alignment.grpo import (
    flatten_group_advantages,
    group_relative_advantages,
    grpo_policy_loss,
)
from forgellm.alignment.policy_gradient import categorical_expected_return
from forgellm.alignment.ppo import clipped_policy_loss
from forgellm.alignment.reward import (
    TinyRewardModel,
    bradley_terry_loss,
    pad_byte_sequences,
    pearson_correlation,
)
from forgellm.alignment.schema import PreferenceRecord
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class RewardOverfitResult:
    """One tiny reward-model overfit run."""

    seed: int
    initial_loss: float
    final_loss: float
    pair_accuracy: float
    mean_margin: float
    reward_length_correlation: float


@dataclass(frozen=True, slots=True)
class ToyOptimizationResult:
    """Before/after expected reward for a policy experiment."""

    seed: int
    initial_expected_reward: float
    final_expected_reward: float


def _pair_text(record: PreferenceRecord, response: str) -> str:
    prompt = "\n".join(message.content for message in record.prompt)
    return f"{prompt}\n<response>{response}</response>"


def reward_model_overfit(
    records: list[PreferenceRecord], *, seed: int, steps: int = 160
) -> RewardOverfitResult:
    """Overfit sixteen deterministic pairs as a wiring gate, not a quality benchmark."""
    if len(records) < 16 or steps <= 0:
        raise ValueError("reward overfit requires at least 16 pairs and positive steps")
    selected = records[:16]
    torch.manual_seed(seed)
    model = TinyRewardModel(hidden_size=32)
    chosen_ids, chosen_mask = pad_byte_sequences(
        [_pair_text(record, record.chosen) for record in selected]
    )
    rejected_ids, rejected_mask = pad_byte_sequences(
        [_pair_text(record, record.rejected) for record in selected]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.0)

    def evaluate() -> tuple[Tensor, Tensor, Tensor]:
        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)
        result = bradley_terry_loss(chosen_rewards, rejected_rewards)
        return result.loss, chosen_rewards, rejected_rewards

    initial_loss = float(evaluate()[0].detach().item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = evaluate()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        final_loss, chosen_rewards, rejected_rewards = evaluate()
        margins = chosen_rewards - rejected_rewards
        lengths = torch.tensor([len(record.chosen) for record in selected], dtype=torch.float32)
    return RewardOverfitResult(
        seed=seed,
        initial_loss=initial_loss,
        final_loss=float(final_loss.item()),
        pair_accuracy=float((margins > 0).float().mean().item()),
        mean_margin=float(margins.mean().item()),
        reward_length_correlation=pearson_correlation(chosen_rewards, lengths),
    )


def reinforce_variance_experiment(*, seed: int, samples: int = 4096) -> dict[str, JsonValue]:
    """Compare score-function gradient variance with and without an expected-reward baseline."""
    if samples <= 1:
        raise ValueError("variance experiment needs at least two samples")
    logits = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    probabilities = logits.detach().softmax(dim=-1)
    expected_reward_baseline = float((probabilities * rewards).sum().item())
    action_scores = torch.eye(3, dtype=torch.float64) - probabilities
    score_squared_norm = action_scores.square().sum(dim=1)
    baseline = float(
        (
            (probabilities * rewards * score_squared_norm).sum()
            / (probabilities * score_squared_norm).sum()
        ).item()
    )
    generator = torch.Generator().manual_seed(seed)
    actions = torch.multinomial(probabilities, samples, replacement=True, generator=generator)
    one_hot = torch.nn.functional.one_hot(actions, num_classes=3).to(dtype=torch.float64)
    scores = one_hot - probabilities
    sampled_rewards = rewards[actions]
    no_baseline = -sampled_rewards.unsqueeze(1) * scores
    with_baseline = -(sampled_rewards - baseline).unsqueeze(1) * scores
    exact_gradient = torch.autograd.grad(-categorical_expected_return(logits, rewards), logits)[
        0
    ].detach()
    no_mean = no_baseline.mean(dim=0)
    baseline_mean = with_baseline.mean(dim=0)
    return {
        "seed": seed,
        "samples": samples,
        "expected_reward_baseline": expected_reward_baseline,
        "variance_optimal_baseline": baseline,
        "variance_without_baseline": float(no_baseline.var(dim=0, unbiased=True).sum().item()),
        "variance_with_baseline": float(with_baseline.var(dim=0, unbiased=True).sum().item()),
        "mean_error_without_baseline": float((no_mean - exact_gradient).norm().item()),
        "mean_error_with_baseline": float((baseline_mean - exact_gradient).norm().item()),
        "exact_gradient": cast(JsonValue, exact_gradient.tolist()),
    }


def toy_ppo_training(*, seed: int, iterations: int = 30) -> ToyOptimizationResult:
    """Run true sampled on-policy PPO updates on a three-action bandit."""
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.0]))
    action_rewards = torch.tensor([-1.0, 0.25, 1.5])
    optimizer = torch.optim.Adam([logits], lr=0.08)
    initial = float(categorical_expected_return(logits, action_rewards).detach().item())
    generator = torch.Generator().manual_seed(seed + 1000)
    for _ in range(iterations):
        with torch.no_grad():
            old_logits = logits.detach().clone()
            old_probabilities = old_logits.softmax(dim=-1)
            actions = torch.multinomial(
                old_probabilities, 128, replacement=True, generator=generator
            )
            sampled_rewards = action_rewards[actions]
            advantages = sampled_rewards - sampled_rewards.mean()
            old_log_probs = old_logits.log_softmax(dim=-1)[actions]
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            new_log_probs = logits.log_softmax(dim=-1)[actions]
            loss = clipped_policy_loss(new_log_probs, old_log_probs, advantages).loss
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
    final = float(categorical_expected_return(logits, action_rewards).detach().item())
    return ToyOptimizationResult(seed, initial, final)


def toy_grpo_training(*, seed: int, iterations: int = 40) -> ToyOptimizationResult:
    """Run grouped sampled updates in a tiny verifier environment."""
    torch.manual_seed(seed)
    reward_table = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [0.0, 1.2, -0.5],
            [1.4, -0.4, 0.0],
            [-0.3, 0.2, 1.1],
        ]
    )
    logits = torch.nn.Parameter(torch.zeros_like(reward_table))
    optimizer = torch.optim.Adam([logits], lr=0.08)
    initial = float((logits.softmax(dim=-1) * reward_table).sum(dim=1).mean().detach().item())
    generator = torch.Generator().manual_seed(seed + 2000)
    for _ in range(iterations):
        with torch.no_grad():
            old_logits = logits.detach().clone()
            actions = torch.multinomial(
                old_logits.softmax(dim=-1), 4, replacement=True, generator=generator
            )
            sampled_rewards = reward_table.gather(1, actions)
            group_result = group_relative_advantages(sampled_rewards)
            old_log_probs = old_logits.log_softmax(dim=-1).gather(1, actions).reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        new_log_probs = logits.log_softmax(dim=-1).gather(1, actions).reshape(-1, 1)
        loss = grpo_policy_loss(
            new_log_probs,
            old_log_probs,
            flatten_group_advantages(group_result),
            torch.ones_like(new_log_probs, dtype=torch.bool),
        ).loss
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    final = float((logits.softmax(dim=-1) * reward_table).sum(dim=1).mean().detach().item())
    return ToyOptimizationResult(seed, initial, final)


def frontier_tensor_studies() -> dict[str, JsonValue]:
    """Run the seven pre-registered single-variable modern-method studies."""
    rewards = torch.tensor([[0.0, 1.0, 3.0, 8.0], [2.0, 2.0, 2.0, 2.0]])
    grpo = group_relative_advantages(rewards)
    dr = dr_grpo_advantages(rewards)
    ratios = torch.tensor([0.6, 0.9, 1.1, 1.5])
    advantages = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    standard_clip = torch.minimum(ratios * advantages, ratios.clamp(0.8, 1.2) * advantages)
    dapo_clip = dapo_clipped_objective(ratios, advantages)
    group_signal = dynamic_sampling_mask(rewards)
    lengths = torch.tensor([20, 28, 30, 34])
    hard_truncation = torch.where(lengths <= 32, torch.ones(4), torch.zeros(4))
    shaped = overlong_shaped_reward(torch.ones(4), lengths, maximum_length=32, buffer_length=8)
    new = torch.tensor([[0.0, 0.1, 1.2], [0.2, 0.2, 0.2]])
    old = torch.zeros_like(new)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    token_ratios = (new - old).exp()
    gspo = gspo_sequence_weights(new, old, mask)
    stale_new = torch.tensor([[0.5, 0.4], [-0.6, -0.5], [1.0, 0.8]])
    stale_old = torch.zeros_like(stale_new)
    stale_mask = torch.ones_like(stale_new)
    stale_advantages = torch.tensor([1.0, -1.0, 1.0])
    vespo = vespo_gamma_weights(stale_new, stale_old, stale_mask, stale_advantages)
    hard_gspo = gspo_sequence_weights(stale_new, stale_old, stale_mask)
    teachers = torch.tensor(
        [
            [[-0.4, -0.3], [-0.1, -0.2]],
            [[-0.2, -0.4], [-0.5, -0.1]],
        ]
    )
    student = torch.tensor([[-0.7, -0.6], [-0.6, -0.6]])
    opd = multi_teacher_opd_reward(teachers, student, torch.tensor([1, 0]), maximum_reward=0.4)
    return {
        "grpo_vs_drgrpo": {
            "grpo_advantages": cast(JsonValue, grpo.advantages.tolist()),
            "drgrpo_centered_rewards": cast(JsonValue, dr.tolist()),
            "zero_signal_groups": int((~grpo.usable_groups).sum().item()),
        },
        "dapo_clip_higher": {
            "standard": cast(JsonValue, standard_clip.tolist()),
            "clip_higher": cast(JsonValue, dapo_clip.tolist()),
        },
        "dynamic_sampling": {
            "usable": cast(JsonValue, group_signal.tolist()),
            "utilization": float(group_signal.float().mean().item()),
        },
        "overlong_shaping": {
            "hard": cast(JsonValue, hard_truncation.tolist()),
            "shaped": cast(JsonValue, shaped.tolist()),
        },
        "gspo_sequence_ratio": {
            "token_ratio_max": float(token_ratios.max().item()),
            "sequence_weights": cast(JsonValue, gspo.tolist()),
        },
        "vespo_staleness": {
            "gspo_hard_weights": cast(JsonValue, hard_gspo.tolist()),
            "vespo_soft_weights": cast(JsonValue, vespo.tolist()),
            "gspo_ess": effective_sample_size(hard_gspo),
            "vespo_ess": effective_sample_size(vespo),
        },
        "multi_teacher_opd": {
            "dense_rewards": cast(JsonValue, opd.tolist()),
            "mean": float(opd.mean().item()),
        },
    }


def run_method_lab(records: list[PreferenceRecord]) -> dict[str, JsonValue]:
    """Execute E1/E2/E6–E15 with fixed seeds and conservative pass gates."""
    seeds = (41, 42, 43)
    reward_runs = [reward_model_overfit(records, seed=seed) for seed in seeds]
    variance_runs = [reinforce_variance_experiment(seed=seed) for seed in seeds]
    ppo_runs = [toy_ppo_training(seed=seed) for seed in seeds]
    grpo_runs = [toy_grpo_training(seed=seed) for seed in seeds]
    reward_pass = all(
        run.pair_accuracy == 1.0 and run.final_loss < run.initial_loss * 0.2 for run in reward_runs
    )
    variance_pass = all(
        cast(float, run["variance_with_baseline"]) < cast(float, run["variance_without_baseline"])
        for run in variance_runs
    )
    ppo_pass = all(run.final_expected_reward > run.initial_expected_reward for run in ppo_runs)
    grpo_pass = all(run.final_expected_reward > run.initial_expected_reward for run in grpo_runs)
    improvements = [run.final_expected_reward - run.initial_expected_reward for run in grpo_runs]
    return {
        "schema_version": "forgellm-stage5-method-lab-v1",
        "seeds": list(seeds),
        "reward_model": {
            "runs": [cast(JsonValue, asdict(run)) for run in reward_runs],
            "gate_passed": reward_pass,
        },
        "reinforce_baseline": {
            "runs": cast(JsonValue, variance_runs),
            "gate_passed": variance_pass,
        },
        "toy_ppo": {
            "runs": [cast(JsonValue, asdict(run)) for run in ppo_runs],
            "gate_passed": ppo_pass,
        },
        "toy_grpo": {
            "runs": [cast(JsonValue, asdict(run)) for run in grpo_runs],
            "mean_improvement": statistics.mean(improvements),
            "population_std_improvement": statistics.pstdev(improvements),
            "gate_passed": grpo_pass,
        },
        "frontier_tensor_studies": frontier_tensor_studies(),
        "all_gates_passed": reward_pass and variance_pass and ppo_pass and grpo_pass,
        "claim_boundary": (
            "Tiny and frozen-tensor correctness evidence only; not a large-model alignment result."
        ),
    }
