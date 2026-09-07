"""Strict protocol and independent verifier for controlled visual evidence.

This protocol intentionally lives beside, rather than inside, the legacy
ChartQA table-cell protocol.  A controlled completion must cite a rendered
bounding box; a value-only citation is insufficient for a full pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from forgemm.data.schemas import EvidenceCell, Operation, OperationArgument
from forgemm.reasoning.executor import execute
from forgemm.reasoning.normalizer import (
    answers_match,
    decimal_to_text,
    normalize_number,
    normalize_text,
)

CONTROLLED_PROTOCOL_VERSION = "forgemm-controlled-protocol-1.0.0"
_PROTOCOL = re.compile(
    r"\A\s*<evidence>\s*(?P<evidence>.*?)\s*</evidence>\s*"
    r"<operation>\s*(?P<operation>.*?)\s*</operation>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*\Z",
    re.DOTALL,
)
_EVIDENCE = re.compile(r"^(?P<id>e[1-9]\d*)\s*=\s*cell\((?P<args>.*)\)$")
_OPERATION = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\((?P<args>.*)\)$")
_ARGUMENT = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\s*=\s*(?P<value>.+)$")
_REFERENCE = re.compile(r"e[1-9]\d*")
_ALLOWED_OPERATIONS = {
    "lookup",
    "equal",
    "sum",
    "difference",
    "average",
    "ratio",
    "product",
    "median",
    "count",
    "argmax",
    "argmin",
    "compare",
}


class ControlledParseError(ValueError):
    """Stable parse failure that never contains oracle content."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class VisualEvidence:
    """One generated chart mark and its exact raster-space source rectangle."""

    source_id: str
    row: str
    column: str
    value: str
    bbox: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not all((self.source_id, self.row, self.column, self.value)):
            raise ValueError("invalid_visual_evidence")
        if len(self.bbox) != 4 or any(item < 0 for item in self.bbox):
            raise ValueError("invalid_visual_bbox")
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("empty_visual_bbox")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "row": self.row,
            "column": self.column,
            "value": self.value,
            "bbox": list(self.bbox),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VisualEvidence:
        bbox = payload.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
        ):
            raise ValueError("invalid_visual_bbox")
        return cls(
            source_id=_required_string(payload, "source_id"),
            row=_required_string(payload, "row"),
            column=_required_string(payload, "column"),
            value=_required_string(payload, "value"),
            bbox=tuple(bbox),
        )


@dataclass(frozen=True)
class ControlledRecord:
    """Frozen generated task plus hidden visual oracle fields."""

    record_id: str
    split: str
    image_path: str
    image_sha256: str
    question: str
    reference_answer: str
    chart_type: str
    image_size: tuple[int, int]
    generation_seed: int
    categories: tuple[str, ...]
    values: tuple[int, ...]
    palette_index: int
    evidence: tuple[VisualEvidence, ...]
    gold_evidence_ids: tuple[str, ...]
    gold_operation: Operation
    dataset_version: str = "forgemm-controlled-v1-lite"
    protocol_version: str = CONTROLLED_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.record_id,
                self.split,
                self.image_path,
                self.image_sha256,
                self.question,
                self.reference_answer,
                self.chart_type,
            )
        ):
            raise ValueError("invalid_controlled_record")
        if len(self.image_size) != 2 or any(value <= 0 for value in self.image_size):
            raise ValueError("invalid_image_size")
        if len(self.categories) != len(self.values) or not self.categories:
            raise ValueError("invalid_chart_values")
        ids = {item.source_id for item in self.evidence}
        if len(ids) != len(self.evidence) or set(self.gold_evidence_ids) - ids:
            raise ValueError("invalid_gold_evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "split": self.split,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "chart_type": self.chart_type,
            "image_size": list(self.image_size),
            "generation_seed": self.generation_seed,
            "categories": list(self.categories),
            "values": list(self.values),
            "palette_index": self.palette_index,
            "evidence": [item.to_dict() for item in self.evidence],
            "gold_evidence_ids": list(self.gold_evidence_ids),
            "gold_operation": _operation_to_dict(self.gold_operation),
            "dataset_version": self.dataset_version,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ControlledRecord:
        image_size = cast(tuple[int, int], _int_tuple(payload, "image_size", 2))
        values = _int_tuple(payload, "values", None)
        categories = _string_tuple(payload, "categories")
        evidence_payload = payload.get("evidence")
        if not isinstance(evidence_payload, list) or not all(
            isinstance(item, dict) for item in evidence_payload
        ):
            raise ValueError("invalid_visual_evidence")
        return cls(
            record_id=_required_string(payload, "record_id"),
            split=_required_string(payload, "split"),
            image_path=_required_string(payload, "image_path"),
            image_sha256=_required_string(payload, "image_sha256"),
            question=_required_string(payload, "question"),
            reference_answer=_required_string(payload, "reference_answer"),
            chart_type=_required_string(payload, "chart_type"),
            image_size=image_size,
            generation_seed=_required_int(payload, "generation_seed"),
            categories=categories,
            values=values,
            palette_index=_required_int(payload, "palette_index"),
            evidence=tuple(VisualEvidence.from_dict(item) for item in evidence_payload),
            gold_evidence_ids=_string_tuple(payload, "gold_evidence_ids"),
            gold_operation=_operation_from_dict(payload.get("gold_operation")),
            dataset_version=_required_string(payload, "dataset_version"),
            protocol_version=_required_string(payload, "protocol_version"),
        )


@dataclass(frozen=True)
class PredictedVisualEvidence:
    """A model's evidence assertion, retaining its output-local identifier."""

    evidence_id: str
    row: str
    column: str
    value: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ControlledPrediction:
    evidence: tuple[PredictedVisualEvidence, ...]
    operation: Operation
    answer: str


@dataclass(frozen=True)
class ControlledVerdict:
    answer_ok: bool
    evidence_ok: bool
    operation_ok: bool
    format_ok: bool
    visual_iou_min: float
    errors: tuple[str, ...]

    @property
    def full_pass(self) -> bool:
        return self.answer_ok and self.evidence_ok and self.operation_ok and self.format_ok


def render_controlled_completion(record: ControlledRecord) -> str:
    """Return the canonical gold completion used only for train/dev targets."""

    expected = _expected_evidence(record)
    aliases = {item.source_id: f"e{index}" for index, item in enumerate(expected, start=1)}
    evidence_lines = "\n".join(_render_evidence(item, aliases[item.source_id]) for item in expected)
    operation = _render_operation(record.gold_operation, aliases)
    return (
        f"<evidence>\n{evidence_lines}\n</evidence>\n"
        f"<operation>\n{operation}\n</operation>\n"
        f"<answer>\n{record.reference_answer}\n</answer>"
    )


def parse_controlled_prediction(text: str) -> ControlledPrediction:
    """Parse exactly one controlled visual-evidence completion."""

    match = _PROTOCOL.fullmatch(text)
    if match is None:
        raise ControlledParseError("invalid_protocol")
    if any(match.group(name).count("<") for name in ("evidence", "operation", "answer")):
        raise ControlledParseError("nested_or_extra_tag")
    evidence = _parse_evidence(match.group("evidence"))
    operation = _parse_operation(match.group("operation"))
    identifiers = {item.evidence_id for item in evidence}
    if any(argument.value not in identifiers for argument in operation.arguments):
        raise ControlledParseError("undefined_reference")
    answer = match.group("answer").strip()
    if not answer:
        raise ControlledParseError("empty_answer")
    return ControlledPrediction(evidence=evidence, operation=operation, answer=answer)


def verify_controlled_completion(
    record: ControlledRecord, completion: str, *, min_iou: float = 0.5
) -> ControlledVerdict:
    """Verify values, exact sources, bounding boxes, operation, and answer.

    Error names intentionally reveal neither a gold value nor a gold coordinate.
    """

    try:
        prediction = parse_controlled_prediction(completion)
    except ControlledParseError as exc:
        return ControlledVerdict(False, False, False, False, 0.0, (exc.code.upper(),))

    expected = _expected_evidence(record)
    matched, evidence_errors, ious = _match_evidence(prediction.evidence, expected, min_iou)
    answer_ok = answers_match(prediction.answer, record.reference_answer)
    operation_ok = _operation_matches(record.gold_operation, prediction.operation, matched)
    if operation_ok:
        cells = tuple(_to_cell(item) for item in prediction.evidence)
        execution = execute(prediction.operation, cells)
        operation_ok = execution.success and answers_match(execution.value, prediction.answer)
    errors = list(evidence_errors)
    if not answer_ok:
        errors.append("ANSWER_MISMATCH")
    if not operation_ok:
        errors.append("INVALID_OPERATION")
    return ControlledVerdict(
        answer_ok=answer_ok,
        evidence_ok=not evidence_errors,
        operation_ok=operation_ok,
        format_ok=True,
        visual_iou_min=min(ious) if ious else 0.0,
        errors=tuple(dict.fromkeys(errors)),
    )


def _expected_evidence(record: ControlledRecord) -> tuple[VisualEvidence, ...]:
    by_id = {item.source_id: item for item in record.evidence}
    return tuple(by_id[item] for item in record.gold_evidence_ids)


def _match_evidence(
    predicted: tuple[PredictedVisualEvidence, ...],
    expected: tuple[VisualEvidence, ...],
    min_iou: float,
) -> tuple[dict[str, str], tuple[str, ...], list[float]]:
    if len(predicted) != len(expected):
        return {}, ("EVIDENCE_SET_MISMATCH",), []
    remaining = {item.source_id: item for item in expected}
    matched: dict[str, str] = {}
    errors: list[str] = []
    ious: list[float] = []
    for item in predicted:
        candidates = [source for source in remaining.values() if _same_semantics(item, source)]
        if len(candidates) != 1:
            errors.append("EVIDENCE_MISMATCH")
            continue
        source = candidates[0]
        del remaining[source.source_id]
        matched[item.evidence_id] = source.source_id
        overlap = _iou(item.bbox, source.bbox)
        ious.append(overlap)
        if overlap < min_iou:
            errors.append("VISUAL_REGION_MISMATCH")
    if remaining:
        errors.append("EVIDENCE_SET_MISMATCH")
    return matched, tuple(dict.fromkeys(errors)), ious


def _same_semantics(predicted: PredictedVisualEvidence, expected: VisualEvidence) -> bool:
    return (
        normalize_text(predicted.row) == normalize_text(expected.row)
        and normalize_text(predicted.column) == normalize_text(expected.column)
        and _same_value(predicted.value, expected.value)
    )


def _same_value(left: str, right: str) -> bool:
    try:
        return decimal_to_text(normalize_number(left)) == decimal_to_text(normalize_number(right))
    except ValueError:
        return normalize_text(left) == normalize_text(right)


def _operation_matches(expected: Operation, predicted: Operation, matched: dict[str, str]) -> bool:
    if expected.name != predicted.name or len(expected.arguments) != len(predicted.arguments):
        return False
    predicted_sources = tuple(matched.get(argument.value) for argument in predicted.arguments)
    expected_sources = tuple(argument.value for argument in expected.arguments)
    return predicted_sources == expected_sources


def _to_cell(item: PredictedVisualEvidence) -> EvidenceCell:
    from forgemm.data.schemas import EvidenceCell

    return EvidenceCell(item.evidence_id, item.row, item.column, item.value)


def _render_evidence(item: VisualEvidence, alias: str) -> str:
    row = json.dumps(item.row, ensure_ascii=False)
    column = json.dumps(item.column, ensure_ascii=False)
    value = json.dumps(item.value, ensure_ascii=False)
    bbox = ",".join(str(value) for value in item.bbox)
    return f"{alias}=cell(row={row}, column={column}, value={value}, bbox=[{bbox}])"


def _render_operation(operation: Operation, aliases: dict[str, str]) -> str:
    rendered = []
    for argument in operation.arguments:
        if not argument.is_reference or argument.value not in aliases:
            raise ValueError("invalid_controlled_gold_operation")
        rendered.append(f"ref={aliases[argument.value]}")
    return f"{operation.name}({', '.join(rendered)})"


def _parse_evidence(block: str) -> tuple[PredictedVisualEvidence, ...]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        raise ControlledParseError("empty_evidence")
    result: list[PredictedVisualEvidence] = []
    identifiers: set[str] = set()
    for line in lines:
        match = _EVIDENCE.fullmatch(line)
        if match is None:
            raise ControlledParseError("invalid_evidence_syntax")
        identifier = match.group("id")
        if identifier in identifiers:
            raise ControlledParseError("duplicate_evidence_id")
        identifiers.add(identifier)
        arguments = _named_arguments(match.group("args"))
        if tuple(name for name, _ in arguments) != ("row", "column", "value", "bbox"):
            raise ControlledParseError("invalid_evidence_arguments")
        result.append(
            PredictedVisualEvidence(
                evidence_id=identifier,
                row=_decode_string(arguments[0][1], "row"),
                column=_decode_string(arguments[1][1], "column"),
                value=_decode_value(arguments[2][1]),
                bbox=_decode_bbox(arguments[3][1]),
            )
        )
    return tuple(result)


def _parse_operation(block: str) -> Operation:
    if "\n" in block or "\r" in block:
        raise ControlledParseError("invalid_operation_syntax")
    match = _OPERATION.fullmatch(block.strip())
    if match is None:
        raise ControlledParseError("invalid_operation_syntax")
    name = "difference" if match.group("name") == "subtract" else match.group("name")
    if name not in _ALLOWED_OPERATIONS:
        raise ControlledParseError("operation_not_allowed")
    pairs = _named_arguments(match.group("args")) if match.group("args").strip() else []
    arguments: list[OperationArgument] = []
    for argument_name, raw_value in pairs:
        if argument_name != "ref" or _REFERENCE.fullmatch(raw_value) is None:
            raise ControlledParseError("operation_operand_must_reference_evidence")
        arguments.append(OperationArgument(argument_name, raw_value, True))
    return Operation(name=name, arguments=tuple(arguments))


def _named_arguments(text: str) -> list[tuple[str, str]]:
    parts = _split_arguments(text)
    result: list[tuple[str, str]] = []
    for part in parts:
        match = _ARGUMENT.fullmatch(part.strip())
        if match is None:
            raise ControlledParseError("invalid_argument_syntax")
        result.append((match.group("name"), match.group("value").strip()))
    return result


def _split_arguments(text: str) -> list[str]:
    if not text.strip():
        return []
    parts: list[str] = []
    start = 0
    quoted = False
    escaped = False
    square_depth = 0
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "\\" and quoted:
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif not quoted and character == "[":
            square_depth += 1
        elif not quoted and character == "]":
            square_depth -= 1
            if square_depth < 0:
                raise ControlledParseError("invalid_bbox")
        elif character == "," and not quoted and square_depth == 0:
            parts.append(text[start:index])
            start = index + 1
        elif character in "()" and not quoted:
            raise ControlledParseError("nested_argument")
    if quoted or escaped or square_depth != 0:
        raise ControlledParseError("invalid_quoted_argument")
    parts.append(text[start:])
    return parts


def _decode_string(raw: str, field: str) -> str:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControlledParseError(f"invalid_{field}") from exc
    if not isinstance(value, str) or not value:
        raise ControlledParseError(f"invalid_{field}")
    return value


def _decode_value(raw: str) -> str:
    if raw.startswith('"'):
        return _decode_string(raw, "value")
    if not raw or any(character in raw for character in "<>()[]"):
        raise ControlledParseError("invalid_value")
    return raw.strip()


def _decode_bbox(raw: str) -> tuple[int, int, int, int]:
    if not raw.startswith("[") or not raw.endswith("]"):
        raise ControlledParseError("invalid_bbox")
    components = raw[1:-1].split(",")
    if len(components) != 4:
        raise ControlledParseError("invalid_bbox")
    try:
        values = tuple(int(component.strip()) for component in components)
    except ValueError as exc:
        raise ControlledParseError("invalid_bbox") from exc
    if any(value < 0 for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ControlledParseError("invalid_bbox")
    return cast(tuple[int, int, int, int], values)


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        return 0.0
    intersection = (intersection_right - intersection_left) * (
        intersection_bottom - intersection_top
    )
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _operation_to_dict(operation: Operation) -> dict[str, Any]:
    return {
        "name": operation.name,
        "arguments": [
            {"name": argument.name, "value": argument.value, "is_reference": argument.is_reference}
            for argument in operation.arguments
        ],
    }


def _operation_from_dict(payload: Any) -> Operation:
    if not isinstance(payload, dict):
        raise ValueError("invalid_gold_operation")
    raw_arguments = payload.get("arguments")
    if not isinstance(raw_arguments, list) or not all(
        isinstance(item, dict) for item in raw_arguments
    ):
        raise ValueError("invalid_gold_operation")
    return Operation(
        name=_required_string(payload, "name"),
        arguments=tuple(
            OperationArgument(
                name=_required_string(item, "name"),
                value=_required_string(item, "value"),
                is_reference=_required_bool(item, "is_reference"),
            )
            for item in raw_arguments
        ),
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid_{key}")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"invalid_{key}")
    return value


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"invalid_{key}")
    return value


def _string_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"invalid_{key}")
    return tuple(value)


def _int_tuple(payload: dict[str, Any], key: str, length: int | None) -> tuple[int, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or (length is not None and len(value) != length)
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"invalid_{key}")
    return tuple(value)
