import csv
import json
from pathlib import Path

import pytest

from forgellm.multimodal.evaluation import (
    load_completed_predictions,
    load_evaluation_manifest,
    paired_sign_test_two_sided_p,
    summarize_predictions,
    transition_counts,
)


def _write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence",
                "source_index",
                "image",
                "task_type",
                "answer_type",
                "question",
                "reference",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sequence": 0,
                "source_index": 7,
                "image": "a.png",
                "task_type": "median",
                "answer_type": "numeric",
                "question": "What is the median value?",
                "reference": "40",
            }
        )


def test_manifest_and_completed_prefix_round_trip(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    predictions = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    records = load_evaluation_manifest(manifest)
    predictions.write_text(
        json.dumps(
            {
                "index": 0,
                "image": "a.png",
                "question": "What is the median value?",
                "reference": "40",
                "prediction": "40",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = load_completed_predictions(predictions, records)

    assert len(completed) == 1
    assert completed[0]["prediction"] == "40"


def test_completed_prefix_rejects_manifest_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    predictions = tmp_path / "predictions.jsonl"
    _write_manifest(manifest)
    records = load_evaluation_manifest(manifest)
    predictions.write_text(
        json.dumps(
            {
                "index": 0,
                "image": "wrong.png",
                "question": "What is the median value?",
                "reference": "40",
                "prediction": "40",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_completed_predictions(predictions, records)


def test_summarize_predictions_and_transitions() -> None:
    baseline = [
        {"question": "What is the value?", "reference": "10", "prediction": "9"},
        {"question": "Is it larger?", "reference": "Yes", "prediction": "Yes"},
    ]
    candidate = [
        {"question": "What is the value?", "reference": "10", "prediction": "10"},
        {"question": "Is it larger?", "reference": "Yes", "prediction": "No"},
    ]

    summary = summarize_predictions(candidate)
    transitions = transition_counts(baseline, candidate)

    assert summary["normalized_exact_accuracy"] == pytest.approx(0.5)
    assert transitions == {
        "fixed_baseline_errors": 1,
        "regressed_baseline_correct": 1,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    assert transition_counts(baseline, candidate, metric="relaxed") == transitions
    assert paired_sign_test_two_sided_p(4, 1) == pytest.approx(0.375)
    assert paired_sign_test_two_sided_p(0, 0) == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        paired_sign_test_two_sided_p(-1, 2)
