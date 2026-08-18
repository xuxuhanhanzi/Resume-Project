"""Frozen InfiAgent-DABench adapter and closed-form evaluator."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

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
from repopilot.core.contracts import JSONValue

_ANSWER_PATTERN = re.compile(r"@([A-Za-z0-9_]+)\[([^\]]*)\]")


@dataclass(frozen=True, slots=True)
class _DABenchRecord:
    task_id: str
    question: str
    concepts: tuple[str, ...]
    constraints: str
    answer_format: str
    file_name: str
    level: str
    answers: tuple[tuple[str, str], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, JSONValue]]:
    records: list[dict[str, JSONValue]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
            records.append(cast(dict[str, JSONValue], raw))
    return records


def _normalized_value(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _values_equal(prediction: str, expected: str) -> bool:
    prediction_normalized = _normalized_value(prediction)
    expected_normalized = _normalized_value(expected)
    try:
        return Decimal(prediction_normalized.replace(",", "")) == Decimal(
            expected_normalized.replace(",", "")
        )
    except InvalidOperation:
        return prediction_normalized == expected_normalized


def _parse_answer_tags(value: str) -> tuple[dict[str, str], bool]:
    answers: dict[str, str] = {}
    duplicate = False
    for raw_key, raw_value in _ANSWER_PATTERN.findall(value):
        key = raw_key.casefold()
        if key in answers:
            duplicate = True
        answers[key] = raw_value.strip()
    return answers, bool(answers) and not duplicate


class DABenchAdapter:
    """Join official question, label, and heterogeneous CSV files by explicit ID."""

    def __init__(
        self,
        questions_path: Path,
        labels_path: Path,
        tables_dir: Path,
        *,
        dataset_revision: str,
        questions_sha256: str,
        labels_sha256: str,
        table_sha256: Mapping[str, str],
    ) -> None:
        self.questions_path = questions_path.resolve(strict=True)
        self.labels_path = labels_path.resolve(strict=True)
        self.tables_dir = tables_dir.resolve(strict=True)
        self.dataset_revision = dataset_revision
        self.table_sha256 = {name: value.casefold() for name, value in table_sha256.items()}
        if _file_sha256(self.questions_path) != questions_sha256.casefold():
            raise ValueError("DABench questions hash does not match the frozen revision")
        if _file_sha256(self.labels_path) != labels_sha256.casefold():
            raise ValueError("DABench labels hash does not match the frozen revision")
        self._records = self._read_records()

    @property
    def benchmark_id(self) -> str:
        return "infiagent_dabench_dev"

    def _read_records(self) -> dict[str, _DABenchRecord]:
        labels: dict[int, tuple[tuple[str, str], ...]] = {}
        for raw in _load_jsonl(self.labels_path):
            question_id = int(raw["id"])
            common_answers = raw.get("common_answers")
            if not isinstance(common_answers, list):
                raise ValueError(f"DABench label {question_id} has invalid common_answers")
            parsed_answers: list[tuple[str, str]] = []
            for answer in common_answers:
                if not isinstance(answer, list) or len(answer) != 2:
                    raise ValueError(f"DABench label {question_id} has an invalid answer pair")
                parsed_answers.append((str(answer[0]), str(answer[1])))
            labels[question_id] = tuple(parsed_answers)

        records: dict[str, _DABenchRecord] = {}
        for raw in _load_jsonl(self.questions_path):
            question_id = int(raw["id"])
            task_id = f"dabench-dev-{question_id:04d}"
            raw_concepts = raw.get("concepts", [])
            if not isinstance(raw_concepts, list):
                raise ValueError(f"DABench question {question_id} has invalid concepts")
            if question_id not in labels:
                raise ValueError(f"DABench question {question_id} has no label")
            records[task_id] = _DABenchRecord(
                task_id=task_id,
                question=str(raw["question"]).strip(),
                concepts=tuple(str(concept) for concept in raw_concepts),
                constraints=str(raw.get("constraints", "")).strip(),
                answer_format=str(raw.get("format", "")).strip(),
                file_name=str(raw["file_name"]).strip(),
                level=str(raw["level"]).strip(),
                answers=labels[question_id],
            )
        if not records:
            raise ValueError("DABench dataset must not be empty")
        return records

    def _table_path(self, record: _DABenchRecord) -> Path:
        expected_hash = self.table_sha256.get(record.file_name)
        if expected_hash is None:
            raise ValueError(f"DABench table is not frozen in this snapshot: {record.file_name}")
        table_path = (self.tables_dir / record.file_name).resolve(strict=True)
        if table_path.parent != self.tables_dir:
            raise ValueError("DABench table path escaped the frozen table directory")
        if _file_sha256(table_path) != expected_hash:
            raise ValueError(f"DABench table hash mismatch: {record.file_name}")
        return table_path

    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]:
        if split not in {"dev", "validation"}:
            raise ValueError("DABench public data supports the dev/validation split")
        selected = tuple(task_ids) if task_ids is not None else tuple(sorted(self._records))
        for task_id in selected:
            try:
                record = self._records[task_id]
            except KeyError as error:
                raise KeyError(f"unknown DABench task: {task_id}") from error
            table_path = self._table_path(record)
            instruction_parts = [record.question]
            if record.constraints:
                instruction_parts.append(f"Constraints: {record.constraints}")
            if record.answer_format:
                instruction_parts.append(f"Required output format: {record.answer_format}")
            yield BenchmarkTask(
                benchmark_id=self.benchmark_id,
                dataset_revision=self.dataset_revision,
                task_id=record.task_id,
                domain=BenchmarkDomain.DATA_ANALYSIS,
                instruction="\n\n".join(instruction_parts),
                environment_id="dabench-python-docker-v1",
                assets=(
                    AssetRef(
                        table_path.as_uri(),
                        sha256=self.table_sha256[record.file_name],
                        mount_path="input.csv",
                        media_type="text/csv",
                    ),
                ),
                allowed_tools=("read_table", "run_python"),
                network_policy=NetworkPolicy.DENY,
                public_metadata={
                    "concepts": list(record.concepts),
                    "level": record.level,
                    "source_file": record.file_name,
                },
            )

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask:
        try:
            record = self._records[task.task_id]
        except KeyError as error:
            raise ValueError("task does not belong to this DABench adapter") from error
        if task.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to this DABench adapter")
        source = self._table_path(record)
        task_dir = run_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        destination = task_dir / "input.csv"
        shutil.copy2(source, destination)
        if _file_sha256(destination) != self.table_sha256[record.file_name]:
            raise ValueError("prepared DABench table failed its content hash check")
        return PreparedTask(
            task,
            task_dir,
            public_payload={"table_path": "input.csv", "source_file": record.file_name},
        )

    def evaluator_case(self, task_id: str) -> EvaluatorCase:
        try:
            record = self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown DABench task: {task_id}") from error
        return EvaluatorCase(
            benchmark_id=self.benchmark_id,
            dataset_revision=self.dataset_revision,
            task_id=task_id,
            evaluator_id="dabench-closed-form-v1",
            oracle_ref=SecretRef("dabench-common-answers", task_id),
            evaluator_config={
                "level": record.level,
                "concepts": list(record.concepts),
                "expected_subquestions": len(record.answers),
            },
        )

    def oracle_answers(self, task_id: str) -> tuple[tuple[str, str], ...]:
        """Return closed-form labels only to trusted evaluator construction code."""
        try:
            return self._records[task_id].answers
        except KeyError as error:
            raise KeyError(f"unknown DABench task: {task_id}") from error

    @property
    def task_count(self) -> int:
        return len(self._records)


class DABenchEvaluator(TaskEvaluator):
    """Score DABench answer tags without an LLM judge."""

    def __init__(self, adapter: DABenchAdapter) -> None:
        self.adapter = adapter

    async def evaluate(
        self, output: TaskOutput, case: EvaluatorCase, handle: EnvironmentHandle
    ) -> EvaluationOutcome:
        del handle
        if case.benchmark_id != self.adapter.benchmark_id:
            raise ValueError("DABench evaluator case belongs to a different adapter")
        parsed, valid_format = _parse_answer_tags(output.final_answer or "")
        structured = output.structured_payload.get("answers")
        if isinstance(structured, dict):
            parsed = {str(key).casefold(): str(value) for key, value in structured.items()}
            valid_format = bool(parsed)
        expected = self.adapter.oracle_answers(case.task_id)
        correct = sum(
            _values_equal(parsed.get(key.casefold(), ""), value) for key, value in expected
        )
        total = len(expected)
        coverage = sum(key.casefold() in parsed for key, _ in expected) / total
        subquestion_accuracy = correct / total
        task_success = bool(valid_format and correct == total)
        if task_success:
            failure_type = None
        elif not parsed:
            failure_type = "missing_answer"
        elif not valid_format:
            failure_type = "format"
        else:
            failure_type = "answer"
        return EvaluationOutcome(
            task_success=task_success,
            primary_metric_name="question_accuracy",
            primary_metric_value=float(task_success),
            domain_metrics={
                "subquestion_accuracy": subquestion_accuracy,
                "answer_coverage": coverage,
                "output_format_valid": float(valid_format),
            },
            shared_metrics=output.run_metrics,
            failure_type=failure_type,
            evaluator_trace_ref=f"dabench://{case.dataset_revision}/{case.task_id}",
        )
