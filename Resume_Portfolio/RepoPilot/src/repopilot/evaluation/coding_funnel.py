"""Offline development-set metrics for a coding-agent repair funnel.

The SWE-bench-Live holdout is intentionally not an input to this module.  A
caller records only public development-task progress at four observable stages:
localization, a non-empty patch, patch application, and verification.  The
module does not run a model, execute a patch, or inspect hidden tests.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_MAX_RECORDS = 10_000
_MAX_FILE_BYTES = 8_000_000
_MAX_LINE_CHARACTERS = 64_000
_REQUIRED_KEYS = {
    "task_id",
    "localized",
    "patch",
    "patch_applied",
    "verification_passed",
}


@dataclass(frozen=True, slots=True)
class CodingFunnelRecord:
    """One redacted development-task outcome; ``patch`` is never emitted."""

    task_id: str
    localized: bool
    nonempty_patch: bool
    patch_applied: bool
    verification_passed: bool
    category: str


@dataclass(frozen=True, slots=True)
class CodingFunnelSummary:
    """Aggregate of the four repair stages with explicit failure categories."""

    tasks: int
    localized: int
    nonempty_patch: int
    patch_applied: int
    verification_passed: int
    localization_rate: float
    nonempty_patch_rate: float
    patch_apply_rate: float
    verification_rate: float
    categories: dict[str, int]
    records: tuple[CodingFunnelRecord, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["records"] = [
            {
                "task_id": record.task_id,
                "localized": record.localized,
                "nonempty_patch": record.nonempty_patch,
                "patch_applied": record.patch_applied,
                "verification_passed": record.verification_passed,
                "category": record.category,
            }
            for record in self.records
        ]
        return payload


def load_coding_funnel_records(path: Path) -> tuple[CodingFunnelRecord, ...]:
    """Load a bounded, strict JSONL development report without external effects."""

    try:
        source = path.resolve(strict=True)
        if source.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"coding funnel JSONL exceeds {_MAX_FILE_BYTES:,} bytes")
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read coding funnel JSONL: {error}") from error
    records: list[CodingFunnelRecord] = []
    task_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(line) > _MAX_LINE_CHARACTERS:
            raise ValueError(f"coding funnel record {line_number} exceeds the line size limit")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid coding funnel JSON at line {line_number}: {error}"
            ) from error
        record = _record_from_mapping(raw, line_number)
        if record.task_id in task_ids:
            raise ValueError(f"duplicate coding funnel task_id: {record.task_id}")
        task_ids.add(record.task_id)
        records.append(record)
        if len(records) > _MAX_RECORDS:
            raise ValueError(f"coding funnel input must contain at most {_MAX_RECORDS} records")
    if not records:
        raise ValueError("coding funnel input contains no records")
    return tuple(records)


def summarize_coding_funnel(records: tuple[CodingFunnelRecord, ...]) -> CodingFunnelSummary:
    """Summarize independently recorded development outcomes."""

    if not records:
        raise ValueError("coding funnel records must not be empty")
    count = len(records)
    categories = dict(sorted(Counter(record.category for record in records).items()))
    localized = sum(record.localized for record in records)
    nonempty_patch = sum(record.nonempty_patch for record in records)
    patch_applied = sum(record.patch_applied for record in records)
    verification_passed = sum(record.verification_passed for record in records)
    return CodingFunnelSummary(
        tasks=count,
        localized=localized,
        nonempty_patch=nonempty_patch,
        patch_applied=patch_applied,
        verification_passed=verification_passed,
        localization_rate=localized / count,
        nonempty_patch_rate=nonempty_patch / count,
        patch_apply_rate=patch_applied / count,
        verification_rate=verification_passed / count,
        categories=categories,
        records=records,
    )


def diagnose_coding_funnel(path: Path) -> CodingFunnelSummary:
    """Convenience entrypoint for CLI use."""

    return summarize_coding_funnel(load_coding_funnel_records(path))


def _record_from_mapping(raw: object, line_number: int) -> CodingFunnelRecord:
    if not isinstance(raw, dict) or set(raw) != _REQUIRED_KEYS:
        raise ValueError(
            f"coding funnel record {line_number} must contain exactly: "
            f"{', '.join(sorted(_REQUIRED_KEYS))}"
        )
    task_id = raw["task_id"]
    localized = raw["localized"]
    patch = raw["patch"]
    patch_applied = raw["patch_applied"]
    verification_passed = raw["verification_passed"]
    if (
        not isinstance(task_id, str)
        or not task_id.strip()
        or not isinstance(localized, bool)
        or not isinstance(patch, str)
        or not isinstance(patch_applied, bool)
        or not isinstance(verification_passed, bool)
    ):
        raise ValueError(f"coding funnel record {line_number} has invalid field types")
    nonempty_patch = bool(patch.strip())
    if nonempty_patch and not localized:
        raise ValueError(f"coding funnel record {line_number} has a patch before localization")
    if patch_applied and not nonempty_patch:
        raise ValueError(f"coding funnel record {line_number} applied an empty patch")
    if verification_passed and not patch_applied:
        raise ValueError(
            f"coding funnel record {line_number} passed verification before patch apply"
        )
    category = _category(localized, nonempty_patch, patch_applied, verification_passed)
    return CodingFunnelRecord(
        task_id.strip(), localized, nonempty_patch, patch_applied, verification_passed, category
    )


def _category(
    localized: bool, nonempty_patch: bool, patch_applied: bool, verification_passed: bool
) -> str:
    if verification_passed:
        return "verified"
    if patch_applied:
        return "verification_failed"
    if nonempty_patch:
        return "patch_apply_failed"
    if localized:
        return "empty_patch"
    return "localization_failed"
