"""Deterministic output metrics kept separate from training rewards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

from forgellm.evaluation.schema import EvaluationCase
from forgellm.post_training.evaluation import (
    repeated_character_ngram_fraction,
    verify_constraints,
)


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    """One response scored only against its frozen case contract."""

    exact_match: bool
    all_constraints_passed: bool
    correct_prefix_but_extra: bool
    valid_json: bool | None
    character_8gram_repetition: float
    token_3gram_repetition: float
    response_tokens: int
    truncated: bool

    def as_dict(self) -> dict[str, bool | float | int | None]:
        """Return serializable metrics."""
        return asdict(self)


def _token_repetition(token_ids: list[int], *, n: int = 3) -> float:
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    ngrams = [tuple(token_ids[index : index + n]) for index in range(total)]
    return (len(ngrams) - len(set(ngrams))) / len(ngrams)


def evaluate_behavior(
    case: EvaluationCase,
    response: str,
    response_token_ids: list[int],
    *,
    reached_token_budget: bool,
) -> BehaviorResult:
    """Evaluate exactness, explicit constraints, repetition and truncation."""
    constraints = case.constraints
    starts_with = constraints.get("starts_with")
    required = constraints.get("required_substrings")
    forbidden = constraints.get("forbidden_substrings")
    exact_words = constraints.get("exact_word_count")
    require_json = constraints.get("require_json", False)
    if starts_with is not None and not isinstance(starts_with, str):
        raise ValueError("starts_with must be a string")
    if required is not None and (
        not isinstance(required, list) or not all(isinstance(item, str) for item in required)
    ):
        raise ValueError("required_substrings must be a string list")
    if forbidden is not None and (
        not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden)
    ):
        raise ValueError("forbidden_substrings must be a string list")
    if exact_words is not None and (
        isinstance(exact_words, bool) or not isinstance(exact_words, int)
    ):
        raise ValueError("exact_word_count must be an integer")
    if not isinstance(require_json, bool):
        raise ValueError("require_json must be a boolean")
    result = verify_constraints(
        response,
        expected_response=case.expected_response,
        starts_with=starts_with,
        required_substrings=cast(list[str] | None, required),
        forbidden_substrings=cast(list[str] | None, forbidden),
        exact_word_count=exact_words,
        require_json=require_json,
    )
    stripped = response.strip()
    correct_prefix_but_extra = (
        stripped.startswith(case.expected_response) and not result.exact_match
    )
    return BehaviorResult(
        exact_match=result.exact_match,
        all_constraints_passed=result.all_constraints_passed,
        correct_prefix_but_extra=correct_prefix_but_extra,
        valid_json=result.valid_json,
        character_8gram_repetition=repeated_character_ngram_fraction(response),
        token_3gram_repetition=_token_repetition(response_token_ids),
        response_tokens=len(response_token_ids),
        truncated=reached_token_budget,
    )


def aggregate_behavior(results: list[BehaviorResult]) -> dict[str, float | int]:
    """Aggregate rates without hiding sample count."""
    if not results:
        raise ValueError("behavior aggregation requires at least one result")
    count = len(results)
    return {
        "cases": count,
        "exact_match_rate": sum(item.exact_match for item in results) / count,
        "constraint_pass_rate": sum(item.all_constraints_passed for item in results) / count,
        "correct_prefix_but_extra_rate": sum(item.correct_prefix_but_extra for item in results)
        / count,
        "valid_json_rate": sum(item.valid_json is True for item in results)
        / max(1, sum(item.valid_json is not None for item in results)),
        "mean_character_8gram_repetition": sum(item.character_8gram_repetition for item in results)
        / count,
        "mean_token_3gram_repetition": sum(item.token_3gram_repetition for item in results) / count,
        "mean_response_tokens": sum(item.response_tokens for item in results) / count,
        "truncation_rate": sum(item.truncated for item in results) / count,
    }
