"""Deterministic JSONL data pipeline with audit artifacts."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from forgellm.data.config import DataConfig, NormalizationForm
from forgellm.data.schema import CandidateDocument, OutputDocument, Rejection
from forgellm.structured_logging import JsonValue

_OUTPUT_SPLITS = ("train", "validation", "test")
_SCHEMA_VERSION = "forgellm-data-manifest-v1"


class DataPipelineError(ValueError):
    """Raised when the pipeline cannot safely process its inputs."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Paths and counts returned by a successful pipeline run."""

    output_dir: Path
    manifest_path: Path
    report_path: Path
    input_records: int
    retained_records: int
    rejected_records: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str, normalization: NormalizationForm) -> str:
    """Apply the documented, deterministic text normalization policy."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_unicode = unicodedata.normalize(normalization, normalized_newlines)
    return "\n".join(line.rstrip() for line in normalized_unicode.split("\n")).strip()


def _contains_disallowed_control(text: str) -> bool:
    return any(
        character not in "\n\t" and unicodedata.category(character) == "Cc" for character in text
    )


def _record_ref(raw_line: bytes) -> str:
    return _sha256_bytes(raw_line.rstrip(b"\r\n"))[:16]


def _parse_records(
    input_path: Path, config: DataConfig
) -> tuple[list[CandidateDocument], list[Rejection], int]:
    candidates: list[CandidateDocument] = []
    rejections: list[Rejection] = []
    input_records = 0

    try:
        stream = input_path.open("rb")
    except OSError as error:
        raise DataPipelineError(f"Could not open input {input_path}: {error}") from error

    with stream:
        for raw_line in stream:
            input_records += 1
            reference = _record_ref(raw_line)
            stripped = raw_line.strip()
            if not stripped:
                rejections.append(Rejection(reference, None, "blank_line"))
                continue
            try:
                decoded = stripped.decode("utf-8")
                parsed: object = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                rejections.append(Rejection(reference, None, "invalid_json"))
                continue
            if not isinstance(parsed, dict) or set(parsed) != {"id", "text"}:
                rejections.append(Rejection(reference, None, "invalid_schema"))
                continue

            values = cast(dict[object, object], parsed)
            document_id = values["id"]
            text = values["text"]
            safe_id = document_id if isinstance(document_id, str) else None
            if (
                not isinstance(document_id, str)
                or not document_id
                or len(document_id) > 256
                or not isinstance(text, str)
            ):
                rejections.append(Rejection(reference, safe_id, "invalid_schema"))
                continue

            normalized = normalize_text(text, config.unicode_normalization)
            content_hash = _sha256_bytes(normalized.encode("utf-8"))
            candidates.append(
                CandidateDocument(
                    document_id=document_id,
                    text=normalized,
                    content_sha256=content_hash,
                    record_ref=reference,
                )
            )
    return candidates, rejections, input_records


def _filter_candidates(
    candidates: Iterable[CandidateDocument], config: DataConfig
) -> tuple[list[CandidateDocument], list[Rejection]]:
    retained: list[CandidateDocument] = []
    rejected: list[Rejection] = []
    for candidate in candidates:
        reason: str | None = None
        if not candidate.text:
            reason = "empty_text"
        elif len(candidate.text) < config.min_chars:
            reason = "too_short"
        elif len(candidate.text) > config.max_chars:
            reason = "too_long"
        elif _contains_disallowed_control(candidate.text):
            reason = "disallowed_control_character"

        if reason:
            rejected.append(Rejection(candidate.record_ref, candidate.document_id, reason))
        else:
            retained.append(candidate)
    return retained, rejected


def _deduplicate(
    candidates: Iterable[CandidateDocument],
) -> tuple[list[CandidateDocument], list[Rejection]]:
    rejected: list[Rejection] = []

    unique_ids: list[CandidateDocument] = []
    seen_ids: set[str] = set()
    for candidate in sorted(
        candidates, key=lambda item: (item.document_id, item.content_sha256, item.record_ref)
    ):
        if candidate.document_id in seen_ids:
            rejected.append(
                Rejection(candidate.record_ref, candidate.document_id, "duplicate_document_id")
            )
        else:
            seen_ids.add(candidate.document_id)
            unique_ids.append(candidate)

    unique_content: list[CandidateDocument] = []
    seen_content: set[str] = set()
    for candidate in sorted(
        unique_ids, key=lambda item: (item.content_sha256, item.document_id, item.record_ref)
    ):
        if candidate.content_sha256 in seen_content:
            rejected.append(
                Rejection(candidate.record_ref, candidate.document_id, "duplicate_exact")
            )
        else:
            seen_content.add(candidate.content_sha256)
            unique_content.append(candidate)
    return unique_content, rejected


def assign_split(content_sha256: str, config: DataConfig) -> str:
    """Assign a document to a stable split using its content hash."""
    split_hash = hashlib.sha256(f"{config.split_seed}:{content_sha256}".encode()).digest()
    bucket = int.from_bytes(split_hash[:8], "big") % 10000
    if bucket < config.train_bps:
        return "train"
    if bucket < config.train_bps + config.validation_bps:
        return "validation"
    return "test"


def _jsonl_bytes(records: Iterable[Mapping[str, JsonValue]]) -> bytes:
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _file_metadata(path: Path, records: int) -> dict[str, JsonValue]:
    return {"bytes": path.stat().st_size, "records": records, "sha256": _sha256_file(path)}


def run_data_pipeline(
    config: DataConfig,
    input_path: Path,
    output_dir: Path,
    *,
    source_name: str,
    source_license: str,
) -> PipelineResult:
    """Run the complete deterministic pipeline and write audit artifacts."""
    if not source_name.strip() or not source_license.strip():
        raise DataPipelineError("source_name and source_license must be non-empty")
    if not input_path.is_file():
        raise DataPipelineError(f"input file does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates, parse_rejections, input_records = _parse_records(input_path, config)
    filtered, filter_rejections = _filter_candidates(candidates, config)
    retained, dedup_rejections = _deduplicate(filtered)
    rejections = sorted(
        [*parse_rejections, *filter_rejections, *dedup_rejections],
        key=lambda item: (item.reason, item.document_id or "", item.record_ref),
    )

    split_documents: dict[str, list[OutputDocument]] = {name: [] for name in _OUTPUT_SPLITS}
    for candidate in retained:
        split = assign_split(candidate.content_sha256, config)
        split_documents[split].append(
            OutputDocument(candidate.document_id, candidate.text, candidate.content_sha256)
        )

    output_metadata: dict[str, JsonValue] = {}
    for split in _OUTPUT_SPLITS:
        records = sorted(
            split_documents[split], key=lambda item: (item.content_sha256, item.document_id)
        )
        path = output_dir / f"{split}.jsonl"
        path.write_bytes(_jsonl_bytes(record.as_dict() for record in records))
        output_metadata[path.name] = _file_metadata(path, len(records))

    rejects_path = output_dir / "rejects.jsonl"
    rejects_path.write_bytes(_jsonl_bytes(rejection.as_dict() for rejection in rejections))
    output_metadata[rejects_path.name] = _file_metadata(rejects_path, len(rejections))

    reason_counts = Counter(rejection.reason for rejection in rejections)
    reason_counts_json: dict[str, JsonValue] = dict(sorted(reason_counts.items()))
    split_counts: dict[str, JsonValue] = {
        name: len(split_documents[name]) for name in _OUTPUT_SPLITS
    }
    report: dict[str, JsonValue] = {
        "input_records": input_records,
        "parsed_records": len(candidates),
        "rejected_records": len(rejections),
        "rejection_reasons": reason_counts_json,
        "retained_bytes": sum(len(item.text.encode("utf-8")) for item in retained),
        "retained_characters": sum(len(item.text) for item in retained),
        "retained_records": len(retained),
        "split_records": split_counts,
    }
    report_path = output_dir / "report.json"
    _write_json(report_path, report)
    output_metadata[report_path.name] = _file_metadata(report_path, 1)

    manifest: dict[str, JsonValue] = {
        "config": {"resolved": config.as_dict(), "sha256": config.fingerprint()},
        "counts": report,
        "input": {
            "bytes": input_path.stat().st_size,
            "file_name": input_path.name,
            "sha256": _sha256_file(input_path),
        },
        "outputs": output_metadata,
        "schema_version": _SCHEMA_VERSION,
        "source": {"license": source_license, "name": source_name},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    return PipelineResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        input_records=input_records,
        retained_records=len(retained),
        rejected_records=len(rejections),
    )
