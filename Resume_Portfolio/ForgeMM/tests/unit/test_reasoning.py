from __future__ import annotations

from decimal import Decimal

import pytest

from forgemm.data.schemas import EvidenceCell, Operation, OperationArgument
from forgemm.reasoning.executor import execute
from forgemm.reasoning.normalizer import answers_match, normalize_number
from forgemm.reasoning.parser import ParseError, parse_prediction

VALID = """<evidence>
e1=cell(row="2015", column="Favorable", value=38)
e2=cell(row="2016", column="Favorable", value="$43.00")
</evidence>
<operation>
subtract(ref=e2, ref=e1)
</operation>
<answer>
5
</answer>"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,250", Decimal("1250")),
        ("$42.50", Decimal("42.50")),
        ("38%", Decimal("38")),
        ("(12)", Decimal("-12")),
        ("1.2M", Decimal("1200000")),
    ],
)
def test_normalize_number(raw: str, expected: Decimal) -> None:
    assert normalize_number(raw) == expected


def test_answer_matching_uses_five_percent_relative_tolerance() -> None:
    assert answers_match("104", "100")
    assert not answers_match("106", "100")
    assert answers_match(" UNITED   Airlines ", "united airlines")


def test_parse_prediction_and_subtract_alias() -> None:
    prediction = parse_prediction(VALID)

    assert prediction.operation.name == "difference"
    assert prediction.evidence[1].value == "$43.00"
    assert execute(prediction.operation, prediction.evidence).value == "5"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (VALID + " trailing", "invalid_protocol"),
        (VALID.replace("</answer>", ""), "invalid_protocol"),
        (VALID.replace("e2=cell", "e1=cell"), "duplicate_evidence_id"),
        (VALID.replace("ref=e2", "ref=e9"), "undefined_reference"),
        (VALID.replace("subtract", "python"), "operation_not_allowed"),
        (VALID.replace("value=38", "value=(38)"), "nested_argument"),
    ],
)
def test_parser_rejects_malformed_or_unsafe_input(text: str, code: str) -> None:
    with pytest.raises(ParseError) as error:
        parse_prediction(text)
    assert error.value.code == code


def _operation(name: str, *references: str) -> Operation:
    return Operation(name, tuple(OperationArgument("ref", ref, True) for ref in references))


CELLS = (
    EvidenceCell("e1", "A", "Value", "10"),
    EvidenceCell("e2", "B", "Value", "4"),
    EvidenceCell("e3", "C", "Value", "2"),
)


@pytest.mark.parametrize(
    ("name", "refs", "expected"),
    [
        ("lookup", ("e1",), "10"),
        ("equal", ("e1", "e1"), "Yes"),
        ("sum", ("e1", "e2", "e3"), "16"),
        ("difference", ("e1", "e2"), "6"),
        ("average", ("e1", "e2", "e3"), "5.333333333333333333333333333"),
        ("ratio", ("e1", "e3"), "5"),
        ("product", ("e1", "e2", "e3"), "80"),
        ("median", ("e1", "e2", "e3"), "4"),
        ("count", ("e1", "e2", "e3"), "3"),
        ("argmax", ("e1", "e2"), "A"),
        ("argmin", ("e1", "e2"), "B"),
        ("compare", ("e1", "e2"), "greater"),
    ],
)
def test_executor_whitelist(name: str, refs: tuple[str, ...], expected: str) -> None:
    result = execute(_operation(name, *refs), CELLS)

    assert result.success
    assert result.value == expected


def test_executor_returns_stable_failures() -> None:
    zero = CELLS + (EvidenceCell("e4", "D", "Value", "0"),)
    assert execute(_operation("ratio", "e1", "e4"), zero).error == "division_by_zero"
    assert execute(_operation("sum", "e9"), CELLS).error == "undefined_reference"
    assert execute(_operation("difference", "e1"), CELLS).error == "invalid_arity:difference:2:1"
