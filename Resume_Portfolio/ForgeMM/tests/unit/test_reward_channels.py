from __future__ import annotations

import pytest

from forgemm.trainers.reward_channels import reward_channel_indices


class _Reward:
    def __init__(self, channel: str) -> None:
        self.channel = channel


def test_reward_channel_indices_uses_channel_contract_not_class_name() -> None:
    rewards = [_Reward("operation"), _Reward("task"), _Reward("evidence")]

    assert reward_channel_indices(rewards) == {"operation": 0, "task": 1, "evidence": 2}


def test_reward_channel_indices_rejects_missing_channel() -> None:
    with pytest.raises(ValueError, match="operation"):
        reward_channel_indices([_Reward("task"), _Reward("evidence")])
