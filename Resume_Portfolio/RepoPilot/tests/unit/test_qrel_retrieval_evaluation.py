from __future__ import annotations

import pytest

from repopilot.evaluation.retrieval import qrel_retrieval_metrics


def test_qrel_metrics_compute_graded_ndcg_binary_recall_and_mrr() -> None:
    summary = qrel_retrieval_metrics(
        {"q1": ["d2", "d1"], "q2": ["d3", "d4"]},
        {"q1": {"d1": 2, "d2": 1}, "q2": {"d4": 1}},
        cutoffs=(1, 2),
    )

    assert summary.queries == 2
    assert summary.recall_at[1] == pytest.approx(0.25)
    assert summary.recall_at[2] == 1.0
    assert summary.mrr_at[2] == pytest.approx(0.75)
    assert 0 < summary.ndcg_at[1] < summary.ndcg_at[2] <= 1
