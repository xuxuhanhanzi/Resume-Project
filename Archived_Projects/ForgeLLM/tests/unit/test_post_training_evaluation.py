"""Rule-based Stage 4 evaluation metrics."""

import pytest

from forgellm.post_training.evaluation import (
    repeated_character_ngram_fraction,
    repeated_ngram_fraction,
    verify_constraints,
)


def test_constraint_verifier_combines_only_explicit_rules() -> None:
    result = verify_constraints(
        "RESULT red blue",
        starts_with="RESULT",
        required_substrings=["red", "blue"],
        forbidden_substrings=["green"],
        exact_word_count=3,
    )

    assert result.all_constraints_passed
    assert result.starts_with is True
    assert result.required_substrings is True
    assert result.forbidden_substrings is True
    assert result.exact_word_count is True


def test_json_and_exact_match_are_deterministic() -> None:
    result = verify_constraints(
        '{"answer": 4}', expected_response='{"answer": 4}', require_json=True
    )

    assert result.exact_match
    assert result.valid_json
    assert result.all_constraints_passed


def test_verifier_requires_at_least_one_rule() -> None:
    with pytest.raises(ValueError):
        verify_constraints("anything")


def test_repetition_fraction_counts_occurrences_after_first() -> None:
    assert repeated_ngram_fraction("a b c a b c", n=3) == pytest.approx(1 / 4)
    assert repeated_ngram_fraction("too short", n=3) == 0.0


def test_character_ngram_catches_unspaced_repetition() -> None:
    text = "prefix" + ("p��sito" * 20)

    assert repeated_ngram_fraction(text, n=3) == 0.0
    assert repeated_character_ngram_fraction(text, n=8) > 0.8
