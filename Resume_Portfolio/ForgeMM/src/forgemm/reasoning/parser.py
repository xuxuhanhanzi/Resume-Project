"""Parser for ForgeMM's three-block structured output protocol."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from forgemm.data.schemas import EvidenceCell, Operation, OperationArgument

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


class ParseError(ValueError):
    """Stable, machine-readable structured output failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class StructuredPrediction:
    """A fully parsed model completion."""

    evidence: tuple[EvidenceCell, ...]
    operation: Operation
    answer: str


def parse_prediction(text: str) -> StructuredPrediction:
    """Parse exactly one evidence/operation/answer sequence."""

    match = _PROTOCOL.fullmatch(text)
    if match is None:
        raise ParseError("invalid_protocol")
    if any(match.group(name).count("<") for name in ("evidence", "operation", "answer")):
        raise ParseError("nested_or_extra_tag")
    evidence = _parse_evidence(match.group("evidence"))
    operation = _parse_operation(match.group("operation"))
    references = {cell.evidence_id for cell in evidence}
    if any(arg.is_reference and arg.value not in references for arg in operation.arguments):
        raise ParseError("undefined_reference")
    answer = match.group("answer").strip()
    if not answer:
        raise ParseError("empty_answer")
    return StructuredPrediction(evidence=evidence, operation=operation, answer=answer)


def _parse_evidence(block: str) -> tuple[EvidenceCell, ...]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        raise ParseError("empty_evidence")
    cells: list[EvidenceCell] = []
    identifiers: set[str] = set()
    for line in lines:
        match = _EVIDENCE.fullmatch(line)
        if match is None:
            raise ParseError("invalid_evidence_syntax")
        identifier = match.group("id")
        if identifier in identifiers:
            raise ParseError("duplicate_evidence_id")
        identifiers.add(identifier)
        arguments = _named_arguments(match.group("args"))
        if tuple(name for name, _ in arguments) != ("row", "column", "value"):
            raise ParseError("invalid_evidence_arguments")
        row = _decode_string(arguments[0][1], "row")
        column = _decode_string(arguments[1][1], "column")
        value = _decode_value(arguments[2][1])
        cells.append(EvidenceCell(identifier, row, column, value))
    return tuple(cells)


def _parse_operation(block: str) -> Operation:
    if "\n" in block or "\r" in block:
        raise ParseError("invalid_operation_syntax")
    match = _OPERATION.fullmatch(block.strip())
    if match is None:
        raise ParseError("invalid_operation_syntax")
    name = match.group("name")
    if name == "subtract":
        name = "difference"
    if name not in _ALLOWED_OPERATIONS:
        raise ParseError("operation_not_allowed")
    pairs = _named_arguments(match.group("args")) if match.group("args").strip() else []
    arguments: list[OperationArgument] = []
    for argument_name, raw_value in pairs:
        if argument_name != "ref" or _REFERENCE.fullmatch(raw_value) is None:
            raise ParseError("operation_operand_must_reference_evidence")
        arguments.append(OperationArgument(argument_name, raw_value, True))
    return Operation(name=name, arguments=tuple(arguments))


def _named_arguments(text: str) -> list[tuple[str, str]]:
    parts = _split_arguments(text)
    result: list[tuple[str, str]] = []
    for part in parts:
        match = _ARGUMENT.fullmatch(part.strip())
        if match is None:
            raise ParseError("invalid_argument_syntax")
        result.append((match.group("name"), match.group("value").strip()))
    return result


def _split_arguments(text: str) -> list[str]:
    if not text.strip():
        return []
    parts: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "\\" and quoted:
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == "," and not quoted:
            parts.append(text[start:index])
            start = index + 1
        elif character in "()" and not quoted:
            raise ParseError("nested_argument")
    if quoted or escaped:
        raise ParseError("invalid_quoted_argument")
    parts.append(text[start:])
    return parts


def _decode_string(raw: str, field: str) -> str:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid_{field}") from exc
    if not isinstance(value, str) or not value:
        raise ParseError(f"invalid_{field}")
    return value


def _decode_value(raw: str) -> str:
    if raw.startswith('"'):
        return _decode_string(raw, "value")
    if not raw or any(character in raw for character in "<>()"):
        raise ParseError("invalid_value")
    return raw.strip()
