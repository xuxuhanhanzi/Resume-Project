"""Deterministic numeric and answer normalization."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_SPACE = re.compile(r"\s+")
_NUMERIC = re.compile(r"^(?P<sign>[+-]?)(?P<number>(?:\d+(?:\.\d*)?|\.\d+))(?P<scale>[kKmMbB])?$")
_SCALES = {"k": Decimal("1000"), "m": Decimal("1000000"), "b": Decimal("1000000000")}


def normalize_number(value: object) -> Decimal:
    """Parse common chart numeric forms while preserving percent points."""

    text = str(value).strip()
    if not text:
        raise ValueError("empty_numeric_value")
    negative_parentheses = text.startswith("(") and text.endswith(")")
    if negative_parentheses:
        text = text[1:-1].strip()
    text = text.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    match = _NUMERIC.fullmatch(text)
    if match is None:
        raise ValueError(f"non_numeric_value:{value}")
    try:
        number = Decimal(f"{match.group('sign')}{match.group('number')}")
    except InvalidOperation as exc:
        raise ValueError(f"non_numeric_value:{value}") from exc
    scale = match.group("scale")
    if scale:
        number *= _SCALES[scale.lower()]
    return -number if negative_parentheses else number


def normalize_text(value: object) -> str:
    """Normalize case and insignificant whitespace for exact text comparison."""

    return _SPACE.sub(" ", str(value).strip()).casefold()


def decimal_to_text(value: Decimal) -> str:
    """Render a Decimal without exponent notation or redundant trailing zeroes."""

    if not value.is_finite():
        raise ValueError("non_finite_result")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def answers_match(
    prediction: object, reference: object, tolerance: Decimal = Decimal("0.05")
) -> bool:
    """Apply ChartQA-style exact text or 5% relative numeric correctness."""

    try:
        predicted_number = normalize_number(prediction)
        reference_number = normalize_number(reference)
    except ValueError:
        return normalize_text(prediction) == normalize_text(reference)
    if reference_number == 0:
        return predicted_number == 0
    return abs(predicted_number - reference_number) / abs(reference_number) <= tolerance
