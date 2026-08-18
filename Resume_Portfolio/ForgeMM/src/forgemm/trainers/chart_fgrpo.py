"""Framework-neutral Chart-FGRPO group advantage math."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def group_standardize(
    rewards: NDArray[np.float64],
    group_ids: NDArray[np.int_],
    mask: NDArray[np.bool_] | None = None,
    *,
    epsilon: float = 1e-6,
) -> NDArray[np.float64]:
    rewards = np.asarray(rewards, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    applicable = (
        np.ones(rewards.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    if rewards.ndim != 1 or group_ids.shape != rewards.shape or applicable.shape != rewards.shape:
        raise ValueError("rewards, group_ids and mask must be aligned one-dimensional arrays")
    if not np.isfinite(rewards).all() or epsilon <= 0:
        raise ValueError("rewards must be finite and epsilon must be positive")
    result = np.zeros_like(rewards)
    for group_id in np.unique(group_ids):
        selected = (group_ids == group_id) & applicable
        values = rewards[selected]
        if len(values) < 2:
            continue
        std = float(values.std())
        if std <= epsilon:
            continue
        result[selected] = (values - values.mean()) / (std + epsilon)
    return result


def compose_chart_fgrpo_advantage(
    task_rewards: NDArray[np.float64],
    evidence_rewards: NDArray[np.float64],
    operation_rewards: NDArray[np.float64],
    group_ids: NDArray[np.int_],
    evidence_mask: NDArray[np.bool_],
    operation_mask: NDArray[np.bool_],
    *,
    lambda_evidence: float,
    lambda_operation: float,
) -> dict[str, NDArray[np.float64]]:
    if lambda_evidence < 0 or lambda_operation < 0:
        raise ValueError("dual multipliers must be non-negative")
    task = group_standardize(task_rewards, group_ids)
    evidence = group_standardize(evidence_rewards, group_ids, evidence_mask)
    operation = group_standardize(operation_rewards, group_ids, operation_mask)
    combined = task + lambda_evidence * evidence + lambda_operation * operation
    return {"task": task, "evidence": evidence, "operation": operation, "combined": combined}
