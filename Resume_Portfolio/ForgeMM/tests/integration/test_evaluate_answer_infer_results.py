from __future__ import annotations

import pytest

from forgemm.evaluation.answer_inference import evaluate_answer_rows


def test_evaluate_answer_rows_uses_relaxed_answer_matching() -> None:
    result = evaluate_answer_rows(
        [
            {"response": "50", "labels": "50.0"},
            {"response": "12", "labels": "10"},
            {"response": "North", "labels": "north"},
        ]
    )

    assert result["samples"] == 3
    assert result["correct"] == 2
    assert result["task_accuracy"] == pytest.approx(2 / 3)


def test_evaluate_answer_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_answer_rows([])
