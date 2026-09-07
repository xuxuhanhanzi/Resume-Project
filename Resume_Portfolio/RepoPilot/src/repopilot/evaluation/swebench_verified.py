"""Public-only SWE-bench Verified preparation for the R2 experiment.

The Hugging Face source contains evaluator material (reference patches and
hidden-test metadata).  This module may be used only by an offline preparation
process.  Its emitted JSONL has a strict public-field allowlist and the split
manifest contains identifiers only, so an agent process never receives gold
fields.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from repopilot.evidence.protocol import (
    DatasetInstance,
    FrozenManifest,
    build_grouped_split_manifest,
    canonical_sha256,
)

DATASET_ID = "SWE-bench/SWE-bench_Verified"
DATASET_SPLIT = "test"
PUBLIC_SCHEMA_VERSION = 1
_PUBLIC_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "version",
    "image",
)
_EVALUATOR_ONLY_FIELDS = frozenset(
    {
        "patch",
        "test_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "FAIL_TO_FAIL",
        "PASS_TO_FAIL",
        "eval_script",
        "log_parser",
        "eval_type",
        "image_assets",
    }
)


def extract_public_records(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    """Return a deterministic, allowlisted public projection of source rows."""

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        record: dict[str, str] = {}
        for field in _PUBLIC_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SWE-bench row {index} has no non-empty public {field!r}")
            record[field] = value
        instance_id = record["instance_id"]
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench instance_id: {instance_id}")
        seen.add(instance_id)
        records.append(record)
    if not records:
        raise ValueError("SWE-bench preparation requires at least one source row")
    return sorted(records, key=lambda record: record["instance_id"])


def source_dataset_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    """Hash the complete downloaded source locally without publishing its fields."""

    serialized = [dict(row) for row in rows]
    if not serialized:
        raise ValueError("SWE-bench source digest requires at least one source row")
    return canonical_sha256(serialized)


def public_dataset_sha256(records: Iterable[Mapping[str, str]]) -> str:
    """Hash the exact public agent input independently from the private source."""

    normalized = [dict(record) for record in records]
    if not normalized:
        raise ValueError("SWE-bench public digest requires at least one record")
    return canonical_sha256(sorted(normalized, key=lambda record: record["instance_id"]))


def build_swebench_verified_manifest(
    records: Iterable[Mapping[str, str]],
    *,
    dataset_version: str,
    source_sha256: str,
) -> FrozenManifest:
    """Freeze a repository-isolated 20/40/40 split before agent execution."""

    frozen = [dict(record) for record in records]
    _validate_public_records(frozen)
    return build_grouped_split_manifest(
        dataset_name=DATASET_ID,
        dataset_version=dataset_version,
        dataset_sha256=source_sha256,
        instances=[
            DatasetInstance(
                instance_id=record["instance_id"],
                group_ids=(f"repository:{record['repo']}",),
                stratum=record["repo"],
            )
            for record in frozen
        ],
        notes=(
            "Repository-isolated split: all issues from the same repository belong to one split.",
            "The dataset digest binds the full evaluator-custody source; public agent input has "
            f"SHA-256 {public_dataset_sha256(frozen)}.",
            "Public JSONL is allowlisted to issue/repository/base-commit/environment identifiers; "
            "patches, test patches, test IDs and evaluator scripts are excluded.",
            "Do not execute final_holdout until the candidate is selected once on validation.",
        ),
    )


def write_public_records_once(path: Path, records: Iterable[Mapping[str, str]]) -> None:
    """Write public-only JSONL once and reject accidental replacement."""

    if path.exists():
        raise ValueError(f"refusing to overwrite SWE-bench public records: {path}")
    frozen = [dict(record) for record in records]
    _validate_public_records(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in sorted(frozen, key=lambda record: record["instance_id"])
        ),
        encoding="utf-8",
        newline="\n",
    )


def load_public_records(path: Path) -> list[dict[str, str]]:
    """Load a public projection and fail closed on missing or gold fields."""

    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid SWE-bench public JSON at {path}:{line_number}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"SWE-bench public record {line_number} must be an object")
        records.append(cast(dict[str, str], raw))
    _validate_public_records(records)
    return sorted(records, key=lambda record: record["instance_id"])


def _validate_public_records(records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        raise ValueError("SWE-bench public records must not be empty")
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        keys = set(record)
        leaked = sorted(keys.intersection(_EVALUATOR_ONLY_FIELDS))
        if leaked:
            raise ValueError(
                "SWE-bench public record "
                f"{index} contains evaluator-only field(s): {', '.join(leaked)}"
            )
        if keys != set(_PUBLIC_FIELDS):
            raise ValueError(
                f"SWE-bench public record {index} must contain exactly: {', '.join(_PUBLIC_FIELDS)}"
            )
        for field in _PUBLIC_FIELDS:
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SWE-bench public record {index} has invalid {field!r}")
        instance_id = cast(str, record["instance_id"])
        if instance_id in seen:
            raise ValueError(f"duplicate SWE-bench public instance_id: {instance_id}")
        seen.add(instance_id)
