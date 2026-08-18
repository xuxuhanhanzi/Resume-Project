"""Full dataset integrity audit with attributable reason codes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image

from forgemm.data.chartqa import ChartQALoader
from forgemm.data.chartqapro import ChartQAProLoader
from forgemm.data.schemas import ChartQARecord
from forgemm.reasoning.normalizer import decimal_to_text, normalize_number, normalize_text


def audit_chartqa(root: Path | str) -> dict[str, Any]:
    """Audit all ChartQA splits, sources, assets, leakage, and table/annotation agreement."""

    loader = ChartQALoader(root)
    split_reports: dict[str, dict[str, Any]] = {}
    hashes_by_split: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        records_by_source: dict[str, list[ChartQARecord]] = {}
        for source in ("human", "augmented"):
            records_by_source[source] = list(loader.iter_records(split, source))
        unique_records = _unique_chart_records(records_by_source.values())
        corrupt_images: list[str] = []
        image_hashes: dict[str, str] = {}
        agreement = Counter[str]()
        conflict_examples: list[str] = []
        for record in unique_records:
            try:
                with Image.open(record.image_path) as image:
                    image.verify()
            except (OSError, ValueError):
                corrupt_images.append(record.image_name)
            image_hashes[record.image_name] = _sha256(record.image_path)
            status = compare_table_annotation(record.table_path, record.annotation_path)
            agreement[status] += 1
            if status.startswith("conflict") and len(conflict_examples) < 20:
                conflict_examples.append(record.image_name)
        hashes_by_split[split] = image_hashes
        split_reports[split] = {
            "questions": {source: len(records) for source, records in records_by_source.items()},
            "unique_charts": len(unique_records),
            "corrupt_image_count": len(corrupt_images),
            "corrupt_image_examples": corrupt_images[:20],
            "table_annotation_status": dict(sorted(agreement.items())),
            "conflict_examples": conflict_examples,
        }
    leakage = _cross_split_hashes(hashes_by_split)
    return {
        "dataset": "ChartQA",
        "splits": split_reports,
        "cross_split_duplicate_image_hash_count": len(leakage),
        "cross_split_duplicate_examples": leakage[:20],
    }


def audit_chartqapro(path: Path | str) -> dict[str, Any]:
    """Audit every ChartQAPro row and image while preserving its test-only role."""

    parquet = pq.ParquetFile(Path(path))
    rows = 0
    questions = 0
    valid_questions = 0
    missing_values = Counter[str]()
    invalid_questions = Counter[str]()
    corrupt_image_count = 0
    corrupt_images: list[int] = []
    for batch in parquet.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            row_index = rows
            rows += 1
            row_questions = row.get("Question") or []
            row_answers = row.get("Answer") or []
            questions += len(row_questions)
            for field in ("Question", "Answer", "Question Type", "image", "Year", "Paragraph"):
                if row.get(field) is None:
                    missing_values[field] += 1
            if len(row_questions) != len(row_answers):
                missing_values["Question/Answer length mismatch"] += 1
            for index in range(max(len(row_questions), len(row_answers))):
                question = row_questions[index] if index < len(row_questions) else ""
                answer = row_answers[index] if index < len(row_answers) else ""
                if len(row_questions) != len(row_answers):
                    invalid_questions["question_answer_length_mismatch"] += 1
                elif question is None or not str(question).strip():
                    invalid_questions["empty_question"] += 1
                elif answer is None or not str(answer).strip():
                    invalid_questions["empty_answer"] += 1
                else:
                    valid_questions += 1
            image_bytes = row.get("image")
            try:
                if not isinstance(image_bytes, bytes):
                    raise ValueError("image_not_bytes")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
            except (OSError, ValueError):
                corrupt_image_count += 1
                if len(corrupt_images) < 20:
                    corrupt_images.append(row_index)
    return {
        "dataset": "ChartQAPro",
        "exposed_split": "test",
        "rows": rows,
        "questions": questions,
        "valid_questions": valid_questions,
        "invalid_question_counts": dict(sorted(invalid_questions.items())),
        "missing_value_counts": dict(sorted(missing_values.items())),
        "corrupt_image_count": corrupt_image_count,
        "corrupt_image_examples": corrupt_images,
    }


def audit_question_duplicates(
    chartqa_root: Path | str, chartqapro_path: Path | str
) -> dict[str, Any]:
    """Audit exact QA duplicates without re-reading image payloads."""

    loader = ChartQALoader(chartqa_root)
    keys_by_split: dict[str, set[tuple[str, str, str]]] = {}
    chartqa_report: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        source_keys: dict[str, list[tuple[str, str, str]]] = {}
        for source in ("human", "augmented"):
            source_keys[source] = [
                _qa_key(record.image_name, record.question, record.answer)
                for record in loader.iter_records(split, source)
            ]
        human = set(source_keys["human"])
        augmented = set(source_keys["augmented"])
        keys_by_split[split] = human | augmented
        chartqa_report[split] = {
            "human_duplicate_records": len(source_keys["human"]) - len(human),
            "augmented_duplicate_records": len(source_keys["augmented"]) - len(augmented),
            "human_augmented_overlap": len(human & augmented),
        }
    cross_split: dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        cross_split[f"{left}_{right}"] = len(keys_by_split[left] & keys_by_split[right])

    pro_records = list(ChartQAProLoader(chartqapro_path).iter_records())
    pro_keys = [
        (normalize_text(record.question), normalize_text(record.answer))
        for record in pro_records
        if record.valid
    ]
    return {
        "schema_version": "1.0.0",
        "chartqa": chartqa_report,
        "chartqa_cross_split_exact_qa": cross_split,
        "chartqapro_question_slots": len(pro_records),
        "chartqapro_valid_questions": len(pro_keys),
        "chartqapro_invalid_question_counts": dict(
            sorted(
                Counter(
                    record.exclusion_reason
                    for record in pro_records
                    if not record.valid and record.exclusion_reason is not None
                ).items()
            )
        ),
        "chartqapro_duplicate_records": len(pro_keys) - len(set(pro_keys)),
    }


def compare_table_annotation(table_path: Path, annotation_path: Path) -> str:
    """Return exact/conflict/unverifiable for safely alignable chart cells."""

    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotation_cells = _annotation_cells(annotation)
        table_candidates = _table_candidates(table_path, annotation)
    except (csv.Error, json.JSONDecodeError, OSError, TypeError, ValueError):
        return "unverifiable_parse_error"
    if not annotation_cells or not table_candidates:
        return "unverifiable_layout"
    if any(candidate == annotation_cells for candidate in table_candidates):
        return "exact"
    if any(set(candidate) == set(annotation_cells) for candidate in table_candidates):
        return "conflict_value"
    return "conflict_shape"


def _unique_chart_records(groups: Iterable[list[ChartQARecord]]) -> list[ChartQARecord]:
    records: dict[str, ChartQARecord] = {}
    for group in groups:
        for record in group:
            records.setdefault(record.image_name, record)
    return list(records.values())


def _annotation_cells(annotation: dict[str, Any]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    models = annotation.get("models")
    if not isinstance(models, list):
        return result
    for model in models:
        if not isinstance(model, dict):
            return {}
        name = normalize_text(model.get("name", ""))
        x_values = model.get("x")
        y_values = model.get("y")
        if not name or not isinstance(x_values, list) or not isinstance(y_values, list):
            return {}
        if len(x_values) != len(y_values):
            return {}
        for x_value, y_value in zip(x_values, y_values, strict=True):
            result[(normalize_text(x_value), name)] = _numeric_text(y_value)
    return result


def _table_candidates(
    table_path: Path, annotation: dict[str, Any]
) -> list[dict[tuple[str, str], str]]:
    with table_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 2:
        return []
    headers = rows[0]
    if any(len(row) != len(headers) for row in rows[1:]):
        return []
    models = annotation.get("models")
    if not isinstance(models, list):
        return []
    model_names = [normalize_text(model.get("name", "")) for model in models]
    candidates: list[dict[tuple[str, str], str]] = []
    normalized_headers = [normalize_text(header) for header in headers]
    first_column = [normalize_text(row[0]) for row in rows[1:]]

    if set(model_names).issubset(normalized_headers[1:]):
        by_columns: dict[tuple[str, str], str] = {}
        for row in rows[1:]:
            for column_index in range(1, len(headers)):
                series = normalized_headers[column_index]
                if series in model_names:
                    by_columns[(normalize_text(row[0]), series)] = _numeric_text(row[column_index])
        candidates.append(by_columns)

    if set(model_names).issubset(first_column):
        by_rows: dict[tuple[str, str], str] = {}
        for row in rows[1:]:
            series = normalize_text(row[0])
            if series in model_names:
                for column_index in range(1, len(headers)):
                    by_rows[(normalized_headers[column_index], series)] = _numeric_text(
                        row[column_index]
                    )
        candidates.append(by_rows)

    if len(models) == 1 and model_names == ["bars"] and len(headers) == 2:
        single_series = {
            (normalize_text(row[0]), "bars"): _numeric_text(row[1]) for row in rows[1:]
        }
        candidates.append(single_series)
    return candidates


def _numeric_text(value: object) -> str:
    return decimal_to_text(normalize_number(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cross_split_hashes(hashes_by_split: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    locations: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, mapping in hashes_by_split.items():
        for image_name, digest in mapping.items():
            locations[digest].append((split, image_name))
    duplicates: list[dict[str, str]] = []
    for digest, items in locations.items():
        if len({split for split, _ in items}) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "locations": ",".join(f"{split}/{name}" for split, name in sorted(items)),
                }
            )
    return sorted(duplicates, key=lambda item: item["locations"])


def _qa_key(image_name: str, question: str, answer: str) -> tuple[str, str, str]:
    return image_name, normalize_text(question), normalize_text(answer)
