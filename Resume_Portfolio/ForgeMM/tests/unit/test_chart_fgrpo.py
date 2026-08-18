import numpy as np
import pytest

from forgemm.trainers.chart_fgrpo import compose_chart_fgrpo_advantage, group_standardize


def test_group_standardize_is_group_local_and_zero_mean() -> None:
    result = group_standardize(
        np.array([0.0, 1.0, 2.0, 4.0]),
        np.array([0, 0, 1, 1]),
    )
    assert result[:2].mean() == pytest.approx(0.0)
    assert result[2:].mean() == pytest.approx(0.0)
    assert result[1] > result[0]
    assert result[3] > result[2]


def test_masked_constraint_advantage_does_not_reward_inapplicable_rows() -> None:
    rewards = np.array([0.0, 1.0, 0.0, 1.0])
    groups = np.zeros(4, dtype=int)
    mask = np.array([True, True, False, False])
    result = group_standardize(rewards, groups, mask)
    assert result[0] < 0 < result[1]
    assert np.array_equal(result[2:], np.zeros(2))


def test_chart_fgrpo_combines_independent_advantages() -> None:
    groups = np.zeros(4, dtype=int)
    all_masked = np.ones(4, dtype=bool)
    result = compose_chart_fgrpo_advantage(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 1.0]),
        groups,
        all_masked,
        all_masked,
        lambda_evidence=0.5,
        lambda_operation=0.25,
    )
    expected = result["task"] + 0.5 * result["evidence"] + 0.25 * result["operation"]
    assert np.allclose(result["combined"], expected)
