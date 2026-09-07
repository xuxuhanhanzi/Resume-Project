"""Strict, offline JSONL input for reproducible evaluation summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path

from repopilot.core.contracts import RunStatus
from repopilot.evaluation.metrics import EvaluationRecord

_MAX_RECORDS = 50_000
_MAX_LINE_CHARACTERS = 16_000
_REQUIRED_KEYS = {
    "task_id",
    "status",
    "hidden_tests_passed",
    "iterations",
    "tool_calls",
    "input_tokens",
    "output_tokens",
    "wall_seconds",
    "changed_files",
}
_OPTIONAL_KEYS = {"security_blocks"}


def load_evaluation_records(path: Path) -> list[EvaluationRecord]:
    """Load bounded public trial records without contacting a model or evaluator."""
    try:
        lines = path.resolve(strict=True).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read evaluation JSONL: {error}") from error
    records: list[EvaluationRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(line) > _MAX_LINE_CHARACTERS:
            raise ValueError(f"evaluation record {line_number} exceeds the line size limit")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid evaluation JSON at line {line_number}: {error}") from error
        records.append(_record_from_mapping(raw, line_number))
        if len(records) > _MAX_RECORDS:
            raise ValueError(f"evaluation input must contain at most {_MAX_RECORDS} records")
    if not records:
        raise ValueError("evaluation input contains no records")
    return records


def _record_from_mapping(raw: object, line_number: int) -> EvaluationRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation record {line_number} must be an object")
    keys = set(raw)
    if keys - (_REQUIRED_KEYS | _OPTIONAL_KEYS) or not keys >= _REQUIRED_KEYS:
        raise ValueError(f"evaluation record {line_number} has unsupported or missing fields")
    task_id = raw["task_id"]
    status = raw["status"]
    hidden_tests_passed = raw["hidden_tests_passed"]
    if (
        not isinstance(task_id, str)
        or not task_id.strip()
        or not isinstance(status, str)
        or not isinstance(hidden_tests_passed, bool)
    ):
        raise ValueError(f"evaluation record {line_number} has invalid identity fields")
    integer_fields = ("iterations", "tool_calls", "input_tokens", "output_tokens", "changed_files")
    values: dict[str, int] = {}
    for field_name in integer_fields:
        value = raw[field_name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"evaluation record {line_number} field {field_name} must be non-negative"
            )
        values[field_name] = value
    security_blocks = raw.get("security_blocks", 0)
    if (
        not isinstance(security_blocks, int)
        or isinstance(security_blocks, bool)
        or security_blocks < 0
    ):
        raise ValueError(
            f"evaluation record {line_number} field security_blocks must be non-negative"
        )
    wall_seconds = raw["wall_seconds"]
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0
    ):
        raise ValueError(
            f"evaluation record {line_number} field wall_seconds must be finite and non-negative"
        )
    try:
        run_status = RunStatus(status)
    except ValueError as error:
        raise ValueError(
            f"evaluation record {line_number} has unknown status {status!r}"
        ) from error
    return EvaluationRecord(
        task_id=task_id,
        status=run_status,
        hidden_tests_passed=hidden_tests_passed,
        iterations=values["iterations"],
        tool_calls=values["tool_calls"],
        input_tokens=values["input_tokens"],
        output_tokens=values["output_tokens"],
        wall_seconds=float(wall_seconds),
        changed_files=values["changed_files"],
        security_blocks=security_blocks,
    )
