"""Manifest-driven and resumable multimodal evaluation helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import comb
from pathlib import Path
from typing import Literal, cast

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct
from forgellm.multimodal.data_audit import classify_answer, classify_task

EvaluationRecord = dict[str, object]


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_manifest(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    """Load and validate a deterministic evaluation CSV manifest."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {"sequence", "source_index", "image", "question", "reference"}
    if not raw_rows or not required <= set(raw_rows[0]):
        raise ValueError(f"Manifest is empty or missing required fields: {path}")
    rows = raw_rows if limit is None else raw_rows[:limit]
    if limit is not None and (limit <= 0 or len(rows) != limit):
        raise ValueError("limit must be positive and within the manifest")
    records: list[dict[str, str]] = []
    seen_source_indices: set[int] = set()
    for expected_sequence, row in enumerate(rows):
        sequence = int(row["sequence"])
        source_index = int(row["source_index"])
        if sequence != expected_sequence:
            raise ValueError(f"Manifest sequence is not contiguous at row {expected_sequence}")
        if source_index in seen_source_indices:
            raise ValueError(f"Manifest repeats source index {source_index}")
        seen_source_indices.add(source_index)
        question = row["question"]
        reference = row["reference"]
        expected_task = classify_task(question, reference)
        expected_answer = classify_answer(question, reference)
        if row.get("task_type") not in (None, "", expected_task):
            raise ValueError(f"Manifest task label mismatch at row {expected_sequence}")
        if row.get("answer_type") not in (None, "", expected_answer):
            raise ValueError(f"Manifest answer label mismatch at row {expected_sequence}")
        records.append(
            {
                "sequence": str(sequence),
                "source_index": str(source_index),
                "image": row["image"],
                "question": question,
                "reference": reference,
                "task_type": expected_task,
                "answer_type": expected_answer,
            }
        )
    return records


def load_completed_predictions(
    path: Path, expected_records: Sequence[Mapping[str, str]]
) -> list[EvaluationRecord]:
    """Load a JSONL prefix and verify that it exactly matches the manifest."""
    if not path.exists():
        return []
    completed: list[EvaluationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = cast(EvaluationRecord, json.loads(line))
            index = int(cast(int, row["index"]))
            if index != len(completed) or index >= len(expected_records):
                raise ValueError(f"Prediction sequence mismatch at JSONL line {line_number}")
            expected = expected_records[index]
            observed_key = (row["image"], row["question"], row["reference"])
            expected_key = (expected["image"], expected["question"], expected["reference"])
            if observed_key != expected_key:
                raise ValueError(f"Prediction manifest mismatch at index {index}")
            completed.append(row)
    return completed


def summarize_predictions(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute overall and heuristic-category metrics from prediction rows."""
    if not rows:
        raise ValueError("At least one prediction is required")
    exact_total = 0
    relaxed_total = 0
    task_groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    answer_groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for row in rows:
        prediction = str(row["prediction"])
        reference = str(row["reference"])
        question = str(row["question"])
        exact = normalized_exact_correct(prediction, reference)
        relaxed = chartqa_relaxed_correct(prediction, reference)
        exact_total += int(exact)
        relaxed_total += int(relaxed)
        task_groups[classify_task(question, reference)].append((exact, relaxed))
        answer_groups[classify_answer(question, reference)].append((exact, relaxed))

    def summarize_groups(
        groups: Mapping[str, Sequence[tuple[bool, bool]]],
    ) -> dict[str, object]:
        return {
            name: {
                "count": len(values),
                "normalized_exact_correct_count": sum(int(value[0]) for value in values),
                "normalized_exact_accuracy": sum(int(value[0]) for value in values) / len(values),
                "chartqa_relaxed_correct_count": sum(int(value[1]) for value in values),
                "chartqa_relaxed_accuracy": sum(int(value[1]) for value in values) / len(values),
            }
            for name, values in sorted(groups.items())
        }

    count = len(rows)
    return {
        "count": count,
        "normalized_exact_correct_count": exact_total,
        "normalized_exact_accuracy": exact_total / count,
        "chartqa_relaxed_correct_count": relaxed_total,
        "chartqa_relaxed_accuracy": relaxed_total / count,
        "by_task": summarize_groups(task_groups),
        "by_answer": summarize_groups(answer_groups),
    }


def transition_counts(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    metric: Literal["exact", "relaxed"] = "exact",
) -> dict[str, int]:
    """Count candidate corrections and regressions relative to a baseline."""
    if len(baseline) != len(candidate):
        raise ValueError("Prediction lengths differ")
    if metric not in {"exact", "relaxed"}:
        raise ValueError(f"Unsupported transition metric: {metric}")
    fixed = regressed = unchanged_correct = unchanged_wrong = 0
    scorer = normalized_exact_correct if metric == "exact" else chartqa_relaxed_correct
    for before, after in zip(baseline, candidate, strict=True):
        before_correct = scorer(str(before["prediction"]), str(before["reference"]))
        after_correct = scorer(str(after["prediction"]), str(after["reference"]))
        fixed += int(not before_correct and after_correct)
        regressed += int(before_correct and not after_correct)
        unchanged_correct += int(before_correct and after_correct)
        unchanged_wrong += int(not before_correct and not after_correct)
    return {
        "fixed_baseline_errors": fixed,
        "regressed_baseline_correct": regressed,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }


def paired_sign_test_two_sided_p(fixed: int, regressed: int) -> float:
    """Return the exact two-sided sign-test p-value for paired correctness changes."""
    if fixed < 0 or regressed < 0:
        raise ValueError("Transition counts must be non-negative")
    discordant = fixed + regressed
    if discordant == 0:
        return 1.0
    smaller = min(fixed, regressed)
    lower_tail: float = float(sum(comb(discordant, k) for k in range(smaller + 1))) / float(
        2**discordant
    )
    return min(1.0, 2 * lower_tail)
