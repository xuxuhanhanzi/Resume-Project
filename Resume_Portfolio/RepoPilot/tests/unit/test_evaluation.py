from __future__ import annotations

import pytest

from repopilot.core.contracts import RunStatus
from repopilot.evaluation.metrics import EvaluationRecord, retrieval_metrics, summarize
from repopilot.evaluation.report import build_ablation_report


def _record(task_id: str, status: RunStatus) -> EvaluationRecord:
    return EvaluationRecord(task_id, status, status is RunStatus.COMPLETED, 2, 4, 100, 20, 1.5, 1)


def test_outcome_summary() -> None:
    summary = summarize([_record("a", RunStatus.COMPLETED), _record("b", RunStatus.FAILED)])

    assert summary.resolve_rate == 0.5
    assert summary.average_tokens == 120


def test_retrieval_metrics() -> None:
    metrics = retrieval_metrics(
        {"a": ["x.py", "target.py"], "b": ["other.py"]},
        {"a": {"target.py"}, "b": {"other.py"}},
        k=2,
    )

    assert metrics["file_recall@2"] == 1.0
    assert metrics["mrr"] == 0.75


def test_ablation_requires_identical_task_sets() -> None:
    with pytest.raises(ValueError, match="same task IDs"):
        build_ablation_report(
            {
                "base": [_record("a", RunStatus.COMPLETED)],
                "rag": [_record("b", RunStatus.COMPLETED)],
            }
        )
