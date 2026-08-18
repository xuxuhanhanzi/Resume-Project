import pytest

from forgellm.multimodal.data_audit import (
    classify_answer,
    classify_task,
    compare_distributions,
    deterministic_stratified_indices,
    jensen_shannon_divergence,
    summarize_records,
)


@pytest.mark.parametrize(
    ("question", "answer", "expected"),
    [
        ("Is the largest value 30?", "Yes", "boolean_verification"),
        ("What is the median value?", "40", "median"),
        ("What is the average of all bars?", "12.5", "average"),
        ("What's the ratio of A and B?", "2", "ratio"),
        ("What is the sum of two bars?", "20", "sum"),
        ("What is the difference between the bars?", "4", "difference"),
        ("How many bars are below 20?", "3", "counting"),
        ("Which line has the highest value?", "Blue", "extremum_or_comparison"),
        ("In which year was the value 10?", "2019", "temporal_lookup"),
        ("What color represents sales?", "Red", "direct_lookup"),
    ],
)
def test_classify_task(question: str, answer: str, expected: str) -> None:
    assert classify_task(question, answer) == expected


@pytest.mark.parametrize(
    ("question", "answer", "expected"),
    [
        ("Is it larger?", "No", "boolean"),
        ("Which labels?", "[A, B]", "list"),
        ("In which year?", "2018", "year_or_date"),
        ("What percentage?", "42", "percentage_numeric"),
        ("What is the value?", "42.5", "numeric"),
        ("Which category?", "Economy", "text"),
    ],
)
def test_classify_answer(question: str, answer: str, expected: str) -> None:
    assert classify_answer(question, answer) == expected


def test_jensen_shannon_divergence_is_symmetric_and_bounded() -> None:
    left = {"a": 8, "b": 2}
    right = {"a": 5, "b": 5}

    forward = jensen_shannon_divergence(left, right)
    reverse = jensen_shannon_divergence(right, left)

    assert forward == pytest.approx(reverse)
    assert 0 < forward < 1
    assert jensen_shannon_divergence(left, left) == pytest.approx(0)


def test_compare_distributions_reports_largest_shift() -> None:
    result = compare_distributions({"a": 80, "b": 20}, {"a": 50, "b": 50})
    assert result["max_absolute_proportion_difference"] == pytest.approx(0.3)


def test_summarize_records_counts_duplicates_and_other() -> None:
    records = [
        {"imgname": "a.png", "query": "Unknown phrasing", "label": "alpha"},
        {"imgname": "a.png", "query": "Unknown   phrasing", "label": "alpha"},
    ]
    summary = summarize_records(records)

    assert summary["records"] == 2
    assert summary["unique_images"] == 1
    assert summary["duplicate_normalized_question_answer"] == 1
    assert summary["other_task_count"] == 2


def test_deterministic_stratified_indices_are_reproducible_and_proportional() -> None:
    records = [
        {"imgname": f"{index}.png", "query": "What is the value?", "label": str(index)}
        for index in range(8)
    ] + [
        {"imgname": f"b{index}.png", "query": "Is it larger?", "label": "Yes"} for index in range(2)
    ]

    first = deterministic_stratified_indices(records, sample_size=5, seed=17)
    repeated = deterministic_stratified_indices(records, sample_size=5, seed=17)
    selected = [records[index] for index in first]

    assert first == repeated
    assert len(first) == len(set(first)) == 5
    assert sum(record["label"] == "Yes" for record in selected) == 1


def test_deterministic_stratified_indices_reject_invalid_size() -> None:
    records = [{"imgname": "a.png", "query": "What value?", "label": "1"}]
    with pytest.raises(ValueError, match="sample_size"):
        deterministic_stratified_indices(records, sample_size=0, seed=1)


def test_deterministic_stratified_indices_honor_exclusions() -> None:
    records = [
        {"imgname": f"{index}.png", "query": "What value?", "label": str(index)}
        for index in range(5)
    ]
    selected = deterministic_stratified_indices(
        records,
        sample_size=3,
        seed=1,
        excluded_indices={0, 1},
    )
    assert set(selected) <= {2, 3, 4}
