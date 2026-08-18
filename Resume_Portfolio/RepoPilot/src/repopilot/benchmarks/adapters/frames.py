"""Frozen Google FRAMES adapter and deterministic engineering scorer."""

from __future__ import annotations

import ast
import csv
import hashlib
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from repopilot.benchmarks.contracts import (
    AssetRef,
    BenchmarkDomain,
    BenchmarkTask,
    EnvironmentHandle,
    EvaluationOutcome,
    EvaluatorCase,
    NetworkPolicy,
    PreparedTask,
    SecretRef,
    TaskEvaluator,
    TaskOutput,
)


class FramesMode(StrEnum):
    """Keep oracle-document and retrieval results as separate experiments."""

    ORACLE_DOCUMENT = "oracle_document"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True, slots=True)
class _FramesRecord:
    task_id: str
    question: str
    answer: str
    wikipedia_links: tuple[str, ...]
    reasoning_types: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_links(raw: str) -> tuple[str, ...]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise ValueError("FRAMES wiki_links must be a Python-style list") from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("FRAMES wiki_links must contain only strings")
    return tuple(item.strip() for item in value if item.strip())


def _normalize_answer(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in normalized.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def _token_f1(prediction: str, expected: str) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    expected_tokens = _normalize_answer(expected).split()
    if not prediction_tokens or not expected_tokens:
        return float(prediction_tokens == expected_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_url(value: str) -> str:
    parts = urlsplit(value.strip())
    path = unquote(parts.path).rstrip("/")
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))


class FramesAdapter:
    """Load the pinned FRAMES TSV without exposing gold answers to public tasks."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        dataset_revision: str,
        dataset_sha256: str,
        mode: FramesMode = FramesMode.ORACLE_DOCUMENT,
    ) -> None:
        self.dataset_path = dataset_path.resolve(strict=True)
        self.dataset_revision = dataset_revision
        self.dataset_sha256 = dataset_sha256.casefold()
        self.mode = mode
        if _file_sha256(self.dataset_path) != self.dataset_sha256:
            raise ValueError("FRAMES dataset hash does not match the frozen revision")
        self._records = self._read_records()

    @property
    def benchmark_id(self) -> str:
        return f"google_frames_{self.mode.value}"

    def _read_records(self) -> dict[str, _FramesRecord]:
        records: dict[str, _FramesRecord] = {}
        with self.dataset_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            required = {"Prompt", "Answer", "reasoning_types", "wiki_links"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("FRAMES TSV is missing required columns")
            id_column = reader.fieldnames[0]
            for row_number, row in enumerate(reader):
                raw_id = str(row.get(id_column, row_number)).strip()
                task_id = f"frames-test-{int(raw_id):04d}"
                question = str(row.get("Prompt", "")).strip()
                answer = str(row.get("Answer", "")).strip()
                if not question or not answer:
                    raise ValueError(f"FRAMES row {raw_id} has an empty question or answer")
                reasoning_types = tuple(
                    item.strip()
                    for item in str(row.get("reasoning_types", "")).split("|")
                    if item.strip()
                )
                record = _FramesRecord(
                    task_id,
                    question,
                    answer,
                    _parse_links(str(row.get("wiki_links", "[]"))),
                    reasoning_types,
                )
                if task_id in records:
                    raise ValueError(f"duplicate FRAMES task ID: {task_id}")
                records[task_id] = record
        if not records:
            raise ValueError("FRAMES dataset must not be empty")
        return records

    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]:
        if split != "test":
            raise ValueError("FRAMES provides only the test split")
        selected = tuple(task_ids) if task_ids is not None else tuple(sorted(self._records))
        for task_id in selected:
            try:
                record = self._records[task_id]
            except KeyError as error:
                raise KeyError(f"unknown FRAMES task: {task_id}") from error
            assets = (
                tuple(AssetRef(link, media_type="text/html") for link in record.wikipedia_links)
                if self.mode is FramesMode.ORACLE_DOCUMENT
                else ()
            )
            yield BenchmarkTask(
                benchmark_id=self.benchmark_id,
                dataset_revision=self.dataset_revision,
                task_id=record.task_id,
                domain=BenchmarkDomain.RESEARCH,
                instruction=record.question,
                environment_id=f"frames-{self.mode.value}-v1",
                assets=assets,
                allowed_tools=("retrieve_documents", "read_document"),
                network_policy=NetworkPolicy.DENY,
                public_metadata={"evaluation_mode": self.mode.value},
            )

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask:
        if task.task_id not in self._records or task.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to this FRAMES adapter")
        task_dir = run_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return PreparedTask(
            task,
            task_dir,
            public_payload={
                "mode": self.mode.value,
                "document_urls": [asset.uri for asset in task.assets],
            },
        )

    def evaluator_case(self, task_id: str) -> EvaluatorCase:
        try:
            record = self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown FRAMES task: {task_id}") from error
        return EvaluatorCase(
            benchmark_id=self.benchmark_id,
            dataset_revision=self.dataset_revision,
            task_id=task_id,
            evaluator_id="frames-deterministic-v1",
            oracle_ref=SecretRef("frames-gold-answers", task_id),
            evaluator_config={
                "reasoning_types": list(record.reasoning_types),
                "relevant_wikipedia_links": list(record.wikipedia_links),
                "mode": self.mode.value,
            },
        )

    def oracle_answer(self, task_id: str) -> str:
        """Return a gold answer only to trusted evaluator construction code."""
        try:
            return self._records[task_id].answer
        except KeyError as error:
            raise KeyError(f"unknown FRAMES task: {task_id}") from error

    @property
    def task_count(self) -> int:
        return len(self._records)


class FramesEvaluator(TaskEvaluator):
    """Deterministic exact-match, token-F1, and evidence retrieval metrics."""

    def __init__(self, adapter: FramesAdapter) -> None:
        self.adapter = adapter

    async def evaluate(
        self, output: TaskOutput, case: EvaluatorCase, handle: EnvironmentHandle
    ) -> EvaluationOutcome:
        del handle
        if case.benchmark_id != self.adapter.benchmark_id:
            raise ValueError("FRAMES evaluator case belongs to a different adapter")
        prediction = output.final_answer or ""
        expected = self.adapter.oracle_answer(case.task_id)
        exact_match = float(_normalize_answer(prediction) == _normalize_answer(expected))
        relevant_raw = case.evaluator_config.get("relevant_wikipedia_links", [])
        if not isinstance(relevant_raw, list):
            raise ValueError("FRAMES relevant_wikipedia_links must be a list")
        relevant = {_normalize_url(str(link)) for link in relevant_raw}
        cited = {
            _normalize_url(evidence.source_id)
            for evidence in output.evidence
            if re.match(r"^https?://", evidence.source_id, flags=re.IGNORECASE)
        }
        overlap = relevant & cited
        recall = len(overlap) / len(relevant) if relevant else 0.0
        precision = len(overlap) / len(cited) if cited else 0.0
        invalid_rate = len(cited - relevant) / len(cited) if cited else 0.0
        return EvaluationOutcome(
            task_success=bool(exact_match),
            primary_metric_name="normalized_exact_match",
            primary_metric_value=exact_match,
            domain_metrics={
                "answer_token_f1": _token_f1(prediction, expected),
                "evidence_recall": recall,
                "evidence_precision": precision,
                "invalid_citation_rate": invalid_rate,
            },
            shared_metrics=output.run_metrics,
            failure_type=(
                None if exact_match else ("missing_answer" if not prediction else "answer")
            ),
            evaluator_trace_ref=f"frames://{case.dataset_revision}/{case.task_id}",
        )
