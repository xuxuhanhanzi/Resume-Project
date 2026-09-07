"""Versioned JSONL persistence and conservative table evidence construction."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from itertools import combinations, permutations
from pathlib import Path

from forgemm.data.audit import compare_table_annotation
from forgemm.data.schemas import (
    ChartQARecord,
    EvidenceCell,
    EvidenceRecord,
    Operation,
    OperationArgument,
)
from forgemm.reasoning.executor import execute
from forgemm.reasoning.normalizer import normalize_number, normalize_text

_ARITHMETIC_CUES = {
    "difference": ("difference", "how much more", "how much less", "by how much"),
    "sum": ("sum", "total", "combined", "together"),
    "average": ("average", "mean"),
    "ratio": ("ratio", "how many times"),
    "product": ("product",),
}
LABELER_VERSION = "rules-1.4.0"


class EvidenceStore:
    """Read and write immutable-style EvidenceStore JSONL artifacts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def write(self, records: Iterable[EvidenceRecord]) -> int:
        """Create a new store; refuse accidental overwrite of experimental evidence."""

        if self.path.exists():
            raise FileExistsError(f"evidence_store_exists:{self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                count += 1
        return count

    def iter_records(self) -> Iterator[EvidenceRecord]:
        """Stream records and fail with an attributable line number."""

        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("record_not_object")
                    yield EvidenceRecord.from_dict(payload)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid_evidence_record:{line_number}:{exc}") from exc


def build_table_evidence(record: ChartQARecord) -> EvidenceRecord:
    """Build unlabelled table cells without inventing a gold reasoning operation."""

    image_hash = _sha256(record.image_path)
    cells: list[EvidenceCell] = []
    exclusion_reason: str | None = None
    with record.table_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration:
            headers = []
        if len(headers) < 2:
            exclusion_reason = "table_missing_headers"
        else:
            for row_index, values in enumerate(reader):
                if len(values) != len(headers):
                    exclusion_reason = "table_ragged_row"
                    cells = []
                    break
                row_label = values[0].strip()
                for column_index in range(1, len(headers)):
                    cells.append(
                        EvidenceCell(
                            evidence_id=f"t{row_index + 1}c{column_index}",
                            row=row_label,
                            column=headers[column_index].strip(),
                            value=values[column_index].strip(),
                        )
                    )
    conflict_status = compare_table_annotation(record.table_path, record.annotation_path)
    if exclusion_reason is None and conflict_status != "exact":
        exclusion_reason = f"table_annotation_{conflict_status}"
    return EvidenceRecord(
        record_id=record.record_id,
        dataset="ChartQA",
        split=record.split,
        source=record.source,
        image_path=str(record.image_path),
        image_sha256=image_hash,
        question=record.question,
        reference_answer=record.answer,
        cells=tuple(cells),
        evidence_mask=False,
        operation_mask=False,
        conflict_status=conflict_status,
        exclusion_reason=exclusion_reason,
    )


def derive_supported_label(record: EvidenceRecord) -> EvidenceRecord:
    """Add only uniquely attributable evidence and executable operation labels."""

    if record.conflict_status != "exact" or record.exclusion_reason is not None:
        return replace(record, labeler_version=LABELER_VERSION)
    question = normalize_text(record.question)
    candidates = _mentioned_cells(record.cells, question)
    operation = _derive_order_operation(record.cells, question, record.reference_answer)
    if operation is None:
        operation = _derive_count_operation(record.cells, question, record.reference_answer)
    if operation is None:
        operation = _derive_arithmetic_operation(candidates, question, record.reference_answer)
    if operation is None:
        operation = _derive_global_reduction_operation(
            record.cells, question, record.reference_answer
        )
    if operation is None and not _is_boolean_answer(record.reference_answer):
        matching = [
            cell for cell in candidates if _lookup_values_match(cell.value, record.reference_answer)
        ]
        if len(matching) == 1:
            operation = Operation(
                "lookup", (OperationArgument("ref", matching[0].evidence_id, True),)
            )
    if operation is None:
        return replace(
            record,
            exclusion_reason="no_unique_supported_label",
            labeler_version=LABELER_VERSION,
        )
    referenced = tuple(argument.value for argument in operation.arguments)
    selected = tuple(cell for cell in record.cells if cell.evidence_id in referenced)
    result = execute(operation, selected)
    if not result.success or not _operation_values_match(result.value, record.reference_answer):
        return replace(
            record,
            exclusion_reason="operation_answer_mismatch",
            labeler_version=LABELER_VERSION,
        )
    return replace(
        record,
        gold_evidence_ids=referenced,
        gold_operation=operation,
        evidence_mask=True,
        operation_mask=True,
        exclusion_reason=None,
        labeler_version=LABELER_VERSION,
    )


def gold_evidence(record: EvidenceRecord) -> tuple[EvidenceCell, ...]:
    """Return the question-level gold cells selected by a trusted label rule."""

    identifiers = set(record.gold_evidence_ids)
    return tuple(cell for cell in record.cells if cell.evidence_id in identifiers)


def _mentioned_cells(cells: tuple[EvidenceCell, ...], question: str) -> tuple[EvidenceCell, ...]:
    result: list[EvidenceCell] = []
    for cell in cells:
        row = normalize_text(cell.row)
        column = normalize_text(cell.column)
        row_mentioned = _contains_term(question, row)
        column_mentioned = _contains_term(question, column)
        if row_mentioned or column_mentioned:
            result.append(cell)
    return tuple(result)


def _derive_arithmetic_operation(
    candidates: tuple[EvidenceCell, ...], question: str, answer: str
) -> Operation | None:
    if len(candidates) < 2 or len(candidates) > 10:
        return None
    for name, cues in _ARITHMETIC_CUES.items():
        if not any(cue in question for cue in cues):
            continue
        pairs = list(combinations(candidates, 2))
        groups: list[tuple[EvidenceCell, ...]] = []
        if name in {"difference", "ratio"}:
            groups.extend(ordered for pair in pairs for ordered in permutations(pair))
        else:
            groups.extend(pairs)
            if len(candidates) > 2:
                groups.append(candidates)
        matches: list[Operation] = []
        for group in groups:
            operation = Operation(
                name,
                tuple(OperationArgument("ref", cell.evidence_id, True) for cell in group),
            )
            result = execute(operation, group)
            if result.success and _operation_values_match(result.value, answer):
                matches.append(operation)
        unique = {tuple(argument.value for argument in item.arguments): item for item in matches}
        return next(iter(unique.values())) if len(unique) == 1 else None
    return None


def _derive_order_operation(
    cells: tuple[EvidenceCell, ...], question: str, answer: str
) -> Operation | None:
    direction: str | None = None
    if any(cue in question for cue in ("highest", "largest", "most", "maximum", "longest", "peak")):
        direction = "argmax"
    elif any(cue in question for cue in ("lowest", "smallest", "least", "minimum", "shortest")):
        direction = "argmin"
    if direction is None:
        return None
    mentioned_columns = {
        normalize_text(cell.column)
        for cell in cells
        if _contains_term(question, normalize_text(cell.column))
    }
    selected = tuple(
        cell
        for cell in cells
        if not mentioned_columns or normalize_text(cell.column) in mentioned_columns
    )
    if len(selected) < 2:
        return None
    try:
        numeric = [normalize_number(cell.value) for cell in selected]
    except ValueError:
        return None
    extreme = max(numeric) if direction == "argmax" else min(numeric)
    if numeric.count(extreme) != 1:
        return None
    operation = Operation(
        direction,
        tuple(OperationArgument("ref", cell.evidence_id, True) for cell in selected),
    )
    result = execute(operation, selected)
    return operation if result.success and _operation_values_match(result.value, answer) else None


def _derive_count_operation(
    cells: tuple[EvidenceCell, ...], question: str, answer: str
) -> Operation | None:
    count_cues = (
        "how many data points",
        "how many values",
        "how many groups",
        "how many bars",
        "how many items",
    )
    if not any(cue in question for cue in count_cues):
        return None
    mentioned_columns = {
        normalize_text(cell.column)
        for cell in cells
        if _contains_term(question, normalize_text(cell.column))
    }
    selected = tuple(
        cell
        for cell in cells
        if not mentioned_columns or normalize_text(cell.column) in mentioned_columns
    )
    operation = Operation(
        "count",
        tuple(OperationArgument("ref", cell.evidence_id, True) for cell in selected),
    )
    result = execute(operation, selected)
    return operation if result.success and _operation_values_match(result.value, answer) else None


def _derive_global_reduction_operation(
    cells: tuple[EvidenceCell, ...], question: str, answer: str
) -> Operation | None:
    """Derive only explicit whole-chart or extremum reductions.

    The question must name the reduction and its scope (all, smallest/largest
    N, or highest-minus-lowest).  This prevents an answer-matching subset search
    from inventing evidence for an otherwise unsupported question.
    """

    selected = _scoped_numeric_cells(cells, question)
    if not selected:
        return None
    if _has_extreme_difference_scope(question):
        extremes = _extreme_cells(selected, 1, highest=True) + _extreme_cells(
            selected, 1, highest=False
        )
        if len(extremes) == 2:
            return _verified_operation("difference", extremes, answer)
        return None

    operation_name = _reduction_name(question)
    if operation_name is None:
        return None
    extreme = _extreme_scope(question)
    if extreme is not None:
        count, highest = extreme
        selected = _extreme_cells(selected, count, highest=highest)
        if len(selected) != count:
            return None
    elif not _has_global_scope(question):
        return None
    return _verified_operation(operation_name, selected, answer)


def _scoped_numeric_cells(
    cells: tuple[EvidenceCell, ...], question: str
) -> tuple[EvidenceCell, ...]:
    mentioned_columns = {
        normalize_text(cell.column)
        for cell in cells
        if _contains_term(question, normalize_text(cell.column))
    }
    candidates = tuple(
        cell
        for cell in cells
        if not mentioned_columns or normalize_text(cell.column) in mentioned_columns
    )
    try:
        for cell in candidates:
            normalize_number(cell.value)
    except ValueError:
        return ()
    return candidates


def _has_extreme_difference_scope(question: str) -> bool:
    difference = any(cue in question for cue in _ARITHMETIC_CUES["difference"])
    high = any(cue in question for cue in ("highest", "largest", "maximum", "max"))
    low = any(cue in question for cue in ("lowest", "smallest", "minimum", "min"))
    return difference and high and low


def _reduction_name(question: str) -> str | None:
    if any(cue in question for cue in _ARITHMETIC_CUES["sum"]):
        return "sum"
    if any(cue in question for cue in _ARITHMETIC_CUES["average"]):
        return "average"
    if any(cue in question for cue in _ARITHMETIC_CUES["product"]):
        return "product"
    return None


def _has_global_scope(question: str) -> bool:
    return any(
        cue in question
        for cue in ("all", "total", "combined", "last ", "first ", "whole chart", "entire")
    )


def _extreme_scope(question: str) -> tuple[int, bool] | None:
    count = _scope_count(question)
    if count is None:
        return None
    if any(cue in question for cue in ("smallest", "lowest", "bottom")):
        return count, False
    if any(cue in question for cue in ("largest", "highest", "top")):
        return count, True
    if "last" in question:
        return count, True
    if "first" in question:
        return count, False
    return None


def _scope_count(question: str) -> int | None:
    word_counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, count in word_counts.items():
        if word in question:
            return count
    match = re.search(r"\b([2-5])(?:st|nd|rd|th)?\b", question)
    return int(match.group(1)) if match is not None else None


def _extreme_cells(
    cells: tuple[EvidenceCell, ...], count: int, *, highest: bool
) -> tuple[EvidenceCell, ...]:
    ordered = sorted(cells, key=lambda cell: normalize_number(cell.value), reverse=highest)
    if len(ordered) < count:
        return ()
    if len(ordered) > count and normalize_number(ordered[count - 1].value) == normalize_number(
        ordered[count].value
    ):
        return ()
    return tuple(ordered[:count])


def _verified_operation(
    name: str, cells: tuple[EvidenceCell, ...], answer: str
) -> Operation | None:
    operation = Operation(
        name,
        tuple(OperationArgument("ref", cell.evidence_id, True) for cell in cells),
    )
    result = execute(operation, cells)
    return operation if result.success and _operation_values_match(result.value, answer) else None


def _contains_term(text: str, term: str) -> bool:
    if not term or term in {"value", "values", "bars"}:
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _is_boolean_answer(answer: str) -> bool:
    return normalize_text(answer) in {"yes", "no", "true", "false"}


def _lookup_values_match(value: str, answer: str) -> bool:
    try:
        return normalize_number(value) == normalize_number(answer)
    except ValueError:
        return normalize_text(value) == normalize_text(answer)


def _operation_values_match(value: str | None, answer: str) -> bool:
    if value is None:
        return False
    try:
        predicted = normalize_number(value)
        reference = normalize_number(answer)
    except ValueError:
        return normalize_text(value) == normalize_text(answer)
    rendered_reference = answer.strip().replace(",", "")
    rendered_reference = rendered_reference.replace("$", "").replace("€", "").replace("£", "")
    if rendered_reference.endswith("%"):
        rendered_reference = rendered_reference[:-1].strip()
    precision = re.fullmatch(r"[+-]?\d+(?:\.(\d+))?", rendered_reference)
    if precision is None:
        return predicted == reference
    decimal_places = len(precision.group(1) or "")
    quantum = Decimal(1).scaleb(-decimal_places)
    return predicted.quantize(quantum, rounding=ROUND_HALF_UP) == reference.quantize(
        quantum, rounding=ROUND_HALF_UP
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
