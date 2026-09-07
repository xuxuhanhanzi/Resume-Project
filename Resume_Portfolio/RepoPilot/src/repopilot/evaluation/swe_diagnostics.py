"""Offline, evidence-preserving diagnostics for completed SWE-style runs.

The helper deliberately consumes only the public prediction JSONL plus the
already-produced evaluator result JSON.  It never reads evaluator-only test
patches, starts Docker, or contacts a model.  Its job is to turn a headline
resolved rate into a small, reproducible failure taxonomy.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_MAX_RECORDS = 50_000
_MAX_FILE_BYTES = 16_000_000
_PATCH_APPLY_MARKERS = (
    "patch failed",
    "does not apply",
    "corrupt patch",
    "unrecognized input",
)


@dataclass(frozen=True, slots=True)
class SWETaskDiagnostic:
    """One public task classification without test output or patch content."""

    instance_id: str
    category: str
    patch_bytes: int
    resolved: bool


@dataclass(frozen=True, slots=True)
class SWEDiagnosticSummary:
    """Deterministic aggregate of a prediction file and its evaluator report."""

    tasks: int
    resolved: int
    categories: dict[str, int]
    diagnostics: tuple[SWETaskDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["diagnostics"] = [asdict(item) for item in self.diagnostics]
        return data


def diagnose_swe_run(predictions_path: Path, resolved_results_path: Path) -> SWEDiagnosticSummary:
    """Classify a completed SWE run from two bounded, public result artifacts."""

    predictions = _load_predictions(predictions_path)
    evaluator = _load_evaluator_results(resolved_results_path)
    prediction_ids = set(predictions)
    evaluator_ids = set(evaluator)
    if prediction_ids != evaluator_ids:
        missing_report = sorted(prediction_ids - evaluator_ids)
        missing_prediction = sorted(evaluator_ids - prediction_ids)
        raise ValueError(
            "prediction and evaluator task IDs differ"
            + (
                f"; missing evaluator entries: {', '.join(missing_report[:5])}"
                if missing_report
                else ""
            )
            + (
                f"; missing prediction entries: {', '.join(missing_prediction[:5])}"
                if missing_prediction
                else ""
            )
        )
    diagnostics = tuple(
        _classify(instance_id, predictions[instance_id], evaluator[instance_id])
        for instance_id in sorted(predictions)
    )
    categories = dict(sorted(Counter(item.category for item in diagnostics).items()))
    return SWEDiagnosticSummary(
        tasks=len(diagnostics),
        resolved=sum(item.resolved for item in diagnostics),
        categories=categories,
        diagnostics=diagnostics,
    )


def _load_predictions(path: Path) -> dict[str, str]:
    source = _read_bounded(path, label="SWE prediction JSONL")
    records: dict[str, str] = {}
    for line_number, line in enumerate(source.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid SWE prediction JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError(f"SWE prediction {line_number} must be an object")
        instance_id = raw.get("instance_id")
        patch = raw.get("model_patch", "")
        if (
            not isinstance(instance_id, str)
            or not instance_id.strip()
            or not isinstance(patch, str)
        ):
            raise ValueError(
                f"SWE prediction {line_number} needs string instance_id and model_patch"
            )
        if instance_id in records:
            raise ValueError(f"duplicate SWE prediction instance_id: {instance_id}")
        records[instance_id] = patch
        if len(records) > _MAX_RECORDS:
            raise ValueError(f"SWE predictions must contain at most {_MAX_RECORDS} records")
    if not records:
        raise ValueError("SWE predictions contain no records")
    return records


def _load_evaluator_results(path: Path) -> dict[str, dict[str, Any]]:
    source = _read_bounded(path, label="SWE evaluator result JSON")
    try:
        raw = json.loads(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid SWE evaluator result JSON: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("report"), list):
        raise ValueError("SWE evaluator results must contain a report list")
    records: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw["report"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"SWE evaluator report item {index} must be an object")
        instance_id = item.get("instance_id")
        values = (
            item.get("resolved"),
            item.get("fail_to_pass_pass"),
            item.get("pass_to_pass_pass"),
        )
        if (
            not isinstance(instance_id, str)
            or not instance_id.strip()
            or not all(isinstance(value, bool) for value in values)
        ):
            raise ValueError(
                f"SWE evaluator report item {index} needs instance_id and boolean outcomes"
            )
        if instance_id in records:
            raise ValueError(f"duplicate SWE evaluator instance_id: {instance_id}")
        records[instance_id] = item
        if len(records) > _MAX_RECORDS:
            raise ValueError(f"SWE evaluator report must contain at most {_MAX_RECORDS} records")
    if not records:
        raise ValueError("SWE evaluator report contains no records")
    return records


def _classify(instance_id: str, patch: str, evaluator: dict[str, Any]) -> SWETaskDiagnostic:
    resolved = bool(evaluator["resolved"])
    patch_bytes = len(patch.encode("utf-8"))
    if resolved:
        category = "resolved"
    elif not patch.strip():
        category = "empty_patch"
    else:
        output = "\n".join(
            str(evaluator.get(name, "")) for name in ("ftp_output_tail", "ptp_output_tail")
        ).casefold()
        if any(marker in output for marker in _PATCH_APPLY_MARKERS):
            category = "patch_apply_failed"
        elif bool(evaluator["fail_to_pass_pass"]) and not bool(evaluator["pass_to_pass_pass"]):
            category = "regression"
        elif not bool(evaluator["fail_to_pass_pass"]):
            category = "fail_to_pass_failed"
        else:
            category = "unresolved_other"
    return SWETaskDiagnostic(instance_id, category, patch_bytes, resolved)


def _read_bounded(path: Path, *, label: str) -> str:
    try:
        source = path.resolve(strict=True)
        if source.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"{label} exceeds the {_MAX_FILE_BYTES:,}-byte size limit")
        return source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
