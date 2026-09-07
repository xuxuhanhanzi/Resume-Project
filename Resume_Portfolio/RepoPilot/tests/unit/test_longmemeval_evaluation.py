from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.evaluation.longmemeval import (
    build_longmemeval_manifest,
    load_longmemeval,
    session_retrieval_metrics,
)


def _record(
    question_id: str, answer_sessions: list[str], *, session_ids: list[str] | None = None
) -> dict[str, object]:
    session_ids = session_ids or ["s1", "s2"]
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": "Which color?",
        "answer": "blue",
        "question_date": "2024/01/03",
        "haystack_session_ids": session_ids,
        "haystack_dates": ["2024/01/01", "2024/01/02"],
        "haystack_sessions": [
            [{"role": "user", "content": "red"}],
            [{"role": "assistant", "content": "blue", "has_answer": True}],
        ],
        "answer_session_ids": answer_sessions,
    }


def test_longmemeval_adapter_makes_grouped_manifest_and_skips_abstention(tmp_path: Path) -> None:
    path = tmp_path / "longmem.json"
    path.write_text(
        json.dumps(
            [
                _record("q1", ["s2"], session_ids=["s1", "s2"]),
                _record("q2_abs", [], session_ids=["s3", "s4"]),
                _record("q3", ["s6"], session_ids=["s5", "s6"]),
                _record("q4", ["s8"], session_ids=["s7", "s8"]),
                _record("q5", ["s10"], session_ids=["s9", "s10"]),
            ]
        ),
        encoding="utf-8",
    )
    records = load_longmemeval(path)
    manifest = build_longmemeval_manifest(path, dataset_version="unit")
    metrics = session_retrieval_metrics(
        records,
        {
            "q1": ["s1", "s2"],
            "q3": ["s5", "s6"],
            "q4": ["s7", "s8"],
            "q5": ["s9", "s10"],
        },
    )

    assert sum(len(ids) for ids in manifest.assignments.values()) == 5
    assert metrics.evaluated_questions == 4
    assert metrics.skipped_abstentions == 1
    assert metrics.recall_at[3] == 1.0


def test_longmemeval_adapter_refuses_an_unsplittable_leakage_group(tmp_path: Path) -> None:
    path = tmp_path / "connected.json"
    path.write_text(
        json.dumps(
            [
                _record(
                    f"q{index}",
                    [f"answer-{index}"],
                    session_ids=["shared", f"answer-{index}"],
                )
                for index in range(5)
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot produce"):
        build_longmemeval_manifest(path, dataset_version="unit")
