"""Metrics for answer-only ms-swift inference outputs."""

from __future__ import annotations

from typing import Any

from forgemm.rewards.answer import answer_reward


def evaluate_answer_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return relaxed answer accuracy for rows containing response and labels."""

    if not rows:
        raise ValueError("inference result is empty")
    correct = sum(
        answer_reward(str(row["response"]).strip(), str(row["labels"]).strip()) == 1.0
        for row in rows
    )
    return {
        "schema_version": "forgemm-answer-infer-eval-v1",
        "samples": len(rows),
        "correct": correct,
        "task_accuracy": correct / len(rows),
    }
