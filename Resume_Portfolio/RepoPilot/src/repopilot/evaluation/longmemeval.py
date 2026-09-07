"""Offline LongMemEval manifest and retrieval metrics adapter for R4."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from repopilot.evidence.protocol import (
    DatasetInstance,
    FrozenManifest,
    build_grouped_split_manifest,
    sha256_file,
)

_REQUIRED_FIELDS = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
    "answer_session_ids",
}


@dataclass(frozen=True, slots=True)
class LongMemEvalRecord:
    """Validated public LongMemEval-S record used by retrieval-only code."""

    question_id: str
    question_type: str
    question: str
    question_date: str
    haystack_session_ids: tuple[str, ...]
    haystack_dates: tuple[str, ...]
    haystack_sessions: tuple[tuple[dict[str, str], ...], ...]
    answer_session_ids: tuple[str, ...]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def stratum(self) -> str:
        return f"{self.question_type}|{'abstention' if self.is_abstention else 'answerable'}"


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    evaluated_questions: int
    skipped_abstentions: int
    recall_at: dict[int, float]
    mrr: float


def load_longmemeval(path: Path) -> tuple[LongMemEvalRecord, ...]:
    """Load the official JSON without changing its order or inspecting answers."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load LongMemEval data: {error}") from error
    if not isinstance(raw, list) or not raw:
        raise ValueError("LongMemEval data must be a non-empty JSON array")
    records = tuple(_parse_record(item, index) for index, item in enumerate(raw, start=1))
    if len({record.question_id for record in records}) != len(records):
        raise ValueError("LongMemEval question IDs must be unique")
    return records


def build_longmemeval_manifest(
    data_path: Path,
    *,
    dataset_version: str,
) -> FrozenManifest:
    """Create the required group-isolated 20/40/40 manifest before model runs."""

    records = load_longmemeval(data_path)
    instances = [
        DatasetInstance(record.question_id, record.haystack_session_ids, record.stratum)
        for record in records
    ]
    return build_grouped_split_manifest(
        dataset_name="LongMemEval_S_cleaned",
        dataset_version=dataset_version,
        dataset_sha256=sha256_file(data_path),
        instances=instances,
        notes=(
            "Groups are connected components of shared haystack_session_ids.",
            "Question type and abstention status are recorded as split strata.",
            "This manifest is created before any retrieval or reader model result is inspected.",
        ),
    )


def session_retrieval_metrics(
    records: Iterable[LongMemEvalRecord],
    rankings: dict[str, list[str]],
    *,
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> RetrievalMetrics:
    """Compute official-style session recall and MRR without an LLM judge."""

    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive")
    recalls = {cutoff: [] for cutoff in cutoffs}
    reciprocal_ranks: list[float] = []
    skipped = 0
    evaluated = 0
    for record in records:
        if record.is_abstention:
            skipped += 1
            continue
        ranked = rankings.get(record.question_id)
        if ranked is None:
            raise ValueError(f"missing retrieval ranking for {record.question_id}")
        expected = set(record.answer_session_ids)
        if not expected:
            raise ValueError(f"answerable question has no answer sessions: {record.question_id}")
        evaluated += 1
        for cutoff in cutoffs:
            recalls[cutoff].append(len(set(ranked[:cutoff]) & expected) / len(expected))
        rank = next((index for index, item in enumerate(ranked, start=1) if item in expected), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    if not evaluated:
        raise ValueError("no answerable LongMemEval records were supplied")
    return RetrievalMetrics(
        evaluated_questions=evaluated,
        skipped_abstentions=skipped,
        recall_at={cutoff: sum(values) / evaluated for cutoff, values in recalls.items()},
        mrr=sum(reciprocal_ranks) / evaluated,
    )


def _parse_record(raw: object, index: int) -> LongMemEvalRecord:
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_FIELDS:
        raise ValueError(f"LongMemEval record {index} has an unsupported schema")
    string_fields = ("question_id", "question_type", "question", "question_date")
    if any(not isinstance(raw[name], str) or not raw[name].strip() for name in string_fields):
        raise ValueError(f"LongMemEval record {index} has invalid string metadata")
    if raw["answer"] is None or isinstance(raw["answer"], (dict, list)):
        raise ValueError(f"LongMemEval record {index} has an unsupported answer type")
    ids, dates, sessions, answer_ids = (
        raw["haystack_session_ids"],
        raw["haystack_dates"],
        raw["haystack_sessions"],
        raw["answer_session_ids"],
    )
    if (
        not isinstance(ids, list)
        or not isinstance(dates, list)
        or not isinstance(sessions, list)
        or not isinstance(answer_ids, list)
        or len(ids) != len(dates)
        or len(ids) != len(sessions)
        or not ids
        or not all(isinstance(value, str) and value.strip() for value in ids + dates + answer_ids)
    ):
        raise ValueError(f"LongMemEval record {index} has invalid session metadata")
    normalized_sessions: list[tuple[dict[str, str], ...]] = []
    for session in sessions:
        if not isinstance(session, list):
            raise ValueError(f"LongMemEval record {index} session must be a turn list")
        turns: list[dict[str, str]] = []
        for turn in session:
            if (
                not isinstance(turn, dict)
                or not isinstance(turn.get("role"), str)
                or not isinstance(turn.get("content"), str)
            ):
                raise ValueError(f"LongMemEval record {index} has an invalid turn")
            turns.append({"role": turn["role"], "content": turn["content"]})
        normalized_sessions.append(tuple(turns))
    if not set(answer_ids) <= set(ids):
        raise ValueError(f"LongMemEval record {index} answer session is outside haystack")
    return LongMemEvalRecord(
        question_id=raw["question_id"],
        question_type=raw["question_type"],
        question=raw["question"],
        question_date=raw["question_date"],
        haystack_session_ids=tuple(ids),
        haystack_dates=tuple(dates),
        haystack_sessions=tuple(normalized_sessions),
        answer_session_ids=tuple(answer_ids),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an immutable LongMemEval split manifest.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    args = parser.parse_args(argv)
    manifest = build_longmemeval_manifest(args.data, dataset_version=args.dataset_version)
    manifest.write_once(args.manifest)
    print(
        json.dumps(
            manifest.to_dict() | {"manifest_sha256": manifest.sha256}, indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
