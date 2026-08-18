from __future__ import annotations

import math

import pytest

from forgellm.evaluation.behavioral import aggregate_behavior, evaluate_behavior
from forgellm.evaluation.contamination import (
    audit_contamination,
    jaccard_similarity,
    normalize_for_contamination,
)
from forgellm.evaluation.language_modeling import (
    LanguageModelingTotals,
    assert_perplexity_comparable,
    merge_language_modeling_totals,
)
from forgellm.evaluation.schema import EvaluationCase
from forgellm.evaluation.statistics import (
    cohens_kappa,
    paired_bootstrap_interval,
    wilson_interval,
)
from forgellm.post_training.schema import Message


def _case() -> EvaluationCase:
    return EvaluationCase(
        "x",
        "correctness",
        (Message("user", "return JSON"),),
        '{"ok":true}',
        "test",
        "test",
        {"require_json": True},
    )


def test_behavior_keeps_exact_constraint_and_repetition_separate() -> None:
    exact = evaluate_behavior(_case(), '{"ok":true}', [1, 2, 3], reached_token_budget=False)
    extra = evaluate_behavior(_case(), '{"ok":true} extra', [1, 1, 1, 1], reached_token_budget=True)
    assert exact.exact_match and exact.all_constraints_passed
    assert extra.correct_prefix_but_extra and not extra.all_constraints_passed
    assert extra.truncated
    assert aggregate_behavior([exact, extra])["exact_match_rate"] == 0.5


def test_contamination_normalization_and_exact_match() -> None:
    assert normalize_for_contamination("Ａ  B\n") == "a b"
    matches = audit_contamination(
        {"eval": "Hello   WORLD"},
        {"train": "hello world"},
        ngram_size=3,
        near_threshold=0.8,
    )
    assert len(matches) == 1 and matches[0].exact


def test_jaccard_empty_semantics() -> None:
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"a"}, set()) == 0.0


def test_language_model_totals_are_additive_not_mean_perplexity() -> None:
    left = LanguageModelingTotals(4.0, 2, 4, "a" * 64)
    right = LanguageModelingTotals(9.0, 3, 6, "a" * 64)
    merged = merge_language_modeling_totals([left, right])
    assert merged.loss_per_token == 13 / 5
    assert merged.target_bytes == 10
    assert merged.bits_per_byte == pytest.approx(13 / (10 * math.log(2)))


def test_cross_tokenizer_perplexity_comparison_fails() -> None:
    left = LanguageModelingTotals(1.0, 1, 1, "a" * 64)
    right = LanguageModelingTotals(1.0, 1, 1, "b" * 64)
    with pytest.raises(RuntimeError, match="BPB"):
        assert_perplexity_comparable(left, right)


def test_wilson_interval_contains_estimate() -> None:
    interval = wilson_interval(5, 10)
    assert interval.lower < interval.estimate < interval.upper
    assert interval.samples == 10


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_bootstrap_interval([0, 0, 1], [1, 0, 1], resamples=200, seed=9)
    second = paired_bootstrap_interval([0, 0, 1], [1, 0, 1], resamples=200, seed=9)
    assert first == second
    assert first.estimate == pytest.approx(1 / 3)


def test_cohens_kappa_boundaries() -> None:
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
