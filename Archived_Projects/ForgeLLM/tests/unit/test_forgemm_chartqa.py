from forgellm.multimodal.chartqa import (
    chartqa_relaxed_correct,
    normalize_answer,
    normalized_exact_correct,
)


def test_normalize_answer_collapses_whitespace_and_case() -> None:
    assert normalize_answer("  Nigeria\n") == "nigeria"
    assert normalize_answer("Not too much/ not at all") == "not too much/not at all"


def test_metrics_distinguish_normalized_exact_and_relaxed_numeric_matching() -> None:
    assert normalized_exact_correct("Nigeria", " nigeria ")
    assert chartqa_relaxed_correct("104.9", "100")
    assert not chartqa_relaxed_correct("106", "100")
    assert not normalized_exact_correct("2006", "2018")
    assert chartqa_relaxed_correct("2006", "2018")


def test_metrics_do_not_extract_numbers_from_explanations() -> None:
    assert not chartqa_relaxed_correct("The answer is 100.", "100")
