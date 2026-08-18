"""Deterministic task, behavior and system metrics for Stage 4."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import cast

from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """Rule-based evaluation for one generated response."""

    exact_match: bool
    starts_with: bool | None
    required_substrings: bool | None
    forbidden_substrings: bool | None
    exact_word_count: bool | None
    valid_json: bool | None
    all_constraints_passed: bool

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible metrics."""
        return cast(dict[str, JsonValue], asdict(self))


def verify_constraints(
    response: str,
    *,
    expected_response: str | None = None,
    starts_with: str | None = None,
    required_substrings: list[str] | None = None,
    forbidden_substrings: list[str] | None = None,
    exact_word_count: int | None = None,
    require_json: bool = False,
) -> ConstraintResult:
    """Evaluate only explicit, deterministic response constraints."""
    stripped = response.strip()
    exact = expected_response is not None and stripped == expected_response.strip()
    prefix_result = None if starts_with is None else stripped.startswith(starts_with)
    required_result = (
        None
        if required_substrings is None
        else all(item in response for item in required_substrings)
    )
    forbidden_result = (
        None
        if forbidden_substrings is None
        else all(item not in response for item in forbidden_substrings)
    )
    word_result = (
        None if exact_word_count is None else len(re.findall(r"\S+", stripped)) == exact_word_count
    )
    json_result: bool | None = None
    if require_json:
        try:
            json.loads(stripped)
            json_result = True
        except json.JSONDecodeError:
            json_result = False
    checks = [prefix_result, required_result, forbidden_result, word_result, json_result]
    active_checks = [value for value in checks if value is not None]
    if expected_response is not None:
        active_checks.append(exact)
    if not active_checks:
        raise ValueError("at least one deterministic constraint must be configured")
    return ConstraintResult(
        exact_match=exact,
        starts_with=prefix_result,
        required_substrings=required_result,
        forbidden_substrings=forbidden_result,
        exact_word_count=word_result,
        valid_json=json_result,
        all_constraints_passed=all(active_checks),
    )


def repeated_ngram_fraction(text: str, *, n: int = 3) -> float:
    """Fraction of whitespace-token n-gram occurrences beyond their first appearance."""
    if n <= 0:
        raise ValueError("n must be positive")
    tokens = text.split()
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(total)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def repeated_character_ngram_fraction(text: str, *, n: int = 8) -> float:
    """Detect repetition that word splitting misses in malformed or unspaced output."""
    if n <= 0:
        raise ValueError("n must be positive")
    compact = "".join(text.split())
    total = len(compact) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [compact[index : index + n] for index in range(total)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)
