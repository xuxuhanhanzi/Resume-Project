"""Reward-channel discovery shared by the optional ms-swift trainer."""

from __future__ import annotations

from typing import Any


def reward_channel_indices(reward_funcs: list[Any]) -> dict[str, int]:
    """Resolve the three ForgeMM channels from instantiated reward functions."""

    expected = {"task", "evidence", "operation"}
    result = {
        channel: index
        for index, reward_func in enumerate(reward_funcs)
        if (channel := getattr(reward_func, "channel", None)) in expected
    }
    missing = sorted(expected - result.keys())
    if missing:
        raise ValueError(f"Chart-FGRPO missing reward channels: {missing}")
    return result
