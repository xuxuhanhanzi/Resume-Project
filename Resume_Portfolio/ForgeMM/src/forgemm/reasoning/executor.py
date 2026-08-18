"""Safe whitelist executor for parsed chart operations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import reduce
from operator import mul

from forgemm.data.schemas import EvidenceCell, Operation
from forgemm.reasoning.normalizer import answers_match, decimal_to_text, normalize_number


@dataclass(frozen=True)
class ExecutionResult:
    """Execution outcome with a stable failure reason."""

    success: bool
    value: str | None = None
    error: str | None = None


def execute(operation: Operation, evidence: tuple[EvidenceCell, ...]) -> ExecutionResult:
    """Execute one registered operation without dynamic code evaluation."""

    cells = {cell.evidence_id: cell for cell in evidence}
    try:
        selected = [cells[arg.value] for arg in operation.arguments]
    except KeyError:
        return ExecutionResult(False, error="undefined_reference")
    try:
        return ExecutionResult(True, value=_execute(operation.name, selected))
    except (ArithmeticError, ValueError) as exc:
        return ExecutionResult(False, error=str(exc))


def _execute(name: str, cells: list[EvidenceCell]) -> str:
    arity = len(cells)
    if name == "lookup":
        _require_arity(name, arity, 1)
        return cells[0].value
    if name == "equal":
        _require_arity(name, arity, 2)
        return "Yes" if answers_match(cells[0].value, cells[1].value) else "No"
    if name == "count":
        return str(arity)
    if name in {"argmax", "argmin"}:
        _require_minimum_arity(name, arity, 1)
        values = _numeric_values(cells)
        target = max(values) if name == "argmax" else min(values)
        winner = cells[values.index(target)]
        return winner.row or winner.column
    if name == "compare":
        _require_arity(name, arity, 2)
        left, right = _numeric_values(cells)
        return "greater" if left > right else "less" if left < right else "equal"

    values = _numeric_values(cells)
    if name == "sum":
        _require_minimum_arity(name, arity, 1)
        result = sum(values, Decimal(0))
    elif name == "difference":
        _require_arity(name, arity, 2)
        result = values[0] - values[1]
    elif name == "average":
        _require_minimum_arity(name, arity, 1)
        result = sum(values, Decimal(0)) / Decimal(arity)
    elif name == "ratio":
        _require_arity(name, arity, 2)
        if values[1] == 0:
            raise ValueError("division_by_zero")
        result = values[0] / values[1]
    elif name == "product":
        _require_minimum_arity(name, arity, 1)
        result = reduce(mul, values, Decimal(1))
    elif name == "median":
        _require_minimum_arity(name, arity, 1)
        ordered = sorted(values)
        midpoint = arity // 2
        result = (
            ordered[midpoint]
            if arity % 2
            else (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)
        )
    else:
        raise ValueError("operation_not_allowed")
    return decimal_to_text(result)


def _numeric_values(cells: list[EvidenceCell]) -> list[Decimal]:
    try:
        return [normalize_number(cell.value) for cell in cells]
    except ValueError as exc:
        raise ValueError("non_numeric_operand") from exc


def _require_arity(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"invalid_arity:{name}:{expected}:{actual}")


def _require_minimum_arity(name: str, actual: int, minimum: int) -> None:
    if actual < minimum:
        raise ValueError(f"invalid_arity:{name}:min{minimum}:{actual}")
