"""Deterministic ChartQA data-audit helpers."""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import mean, median

from forgellm.multimodal.chartqa import normalize_answer

Record = Mapping[str, str]

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?%?$")
_YEAR_QUERY = re.compile(r"\b(?:which|what) year\b|\bwhen\b", re.IGNORECASE)
_PERCENT_QUERY = re.compile(r"\bpercent(?:age)?\b|\bshare\b|\brate\b|\bproportion\b", re.IGNORECASE)

_TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "boolean_verification",
        re.compile(
            r"^(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would)\b",
            re.IGNORECASE,
        ),
    ),
    ("median", re.compile(r"\bmedian\b", re.IGNORECASE)),
    ("average", re.compile(r"\baverage\b|\bmean\b", re.IGNORECASE)),
    ("ratio", re.compile(r"\bratio\b|\bdivid(?:e|ed|ing)\b", re.IGNORECASE)),
    ("product", re.compile(r"\bproduct\b|\bmultip(?:ly|lied|lication)\b", re.IGNORECASE)),
    (
        "sum",
        re.compile(r"\bsum\b|\btotal\b|\badd(?:ed|ing)?\b|\bplus\b", re.IGNORECASE),
    ),
    (
        "difference",
        re.compile(
            r"\bdifference\b|\bdeduct\b|\bminus\b|\bsubtract(?:ed|ion)?\b|\bgap\b",
            re.IGNORECASE,
        ),
    ),
    (
        "counting",
        re.compile(
            r"\bhow many\b|\bnumber of\b|\bfor how many\b|\bhow often\b|\bhow much time",
            re.IGNORECASE,
        ),
    ),
    (
        "extremum_or_comparison",
        re.compile(
            r"\b(?:highest|lowest|largest|smallest|maximum|minimum|maxiumum|maxium|peak|"
            r"tallest|shortest|most|least|greater|smaller|bigger|less|closest|nearest|"
            r"divergent|exceed|above|below|over|under)\b",
            re.IGNORECASE,
        ),
    ),
    ("temporal_lookup", _YEAR_QUERY),
    (
        "direct_lookup",
        re.compile(
            r"\b(?:what|which|who|where|how much)\b|\b(?:value|color|colour|represent|denote)\b",
            re.IGNORECASE,
        ),
    ),
)


def normalize_question(value: str) -> str:
    """Normalize question text for duplicate and leakage checks."""
    return re.sub(r"\s+", " ", value.strip().casefold())


def classify_task(question: str, answer: str = "") -> str:
    """Assign one mutually exclusive, heuristic project task category."""
    normalized_answer = normalize_answer(answer)
    if normalized_answer in {"yes", "no"}:
        return "boolean_verification"
    for label, pattern in _TASK_PATTERNS:
        if pattern.search(question):
            return label
    return "other"


def classify_answer(question: str, answer: str) -> str:
    """Classify short-answer surface form with limited question context."""
    normalized = normalize_answer(answer)
    if normalized in {"yes", "no"}:
        return "boolean"
    if normalized.startswith("[") and normalized.endswith("]"):
        return "list"
    compact = normalized.replace(" ", "")
    if _NUMBER.fullmatch(compact):
        numeric = compact.replace(",", "").rstrip("%")
        if _YEAR_QUERY.search(question):
            try:
                value = float(numeric)
            except ValueError:
                pass
            else:
                if value.is_integer() and 1800 <= value <= 2200:
                    return "year_or_date"
        if compact.endswith("%") or _PERCENT_QUERY.search(question):
            return "percentage_numeric"
        return "numeric"
    return "text"


def distribution(labels: Iterable[str]) -> dict[str, int]:
    """Return stable label counts."""
    return dict(sorted(Counter(labels).items()))


def proportions(counts: Mapping[str, int]) -> dict[str, float]:
    """Normalize counts to proportions."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def jensen_shannon_divergence(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    """Return base-2 Jensen-Shannon divergence in the inclusive range [0, 1]."""
    p = proportions(left)
    q = proportions(right)
    keys = set(p) | set(q)

    def kl(values: Mapping[str, float], mixture: Mapping[str, float]) -> float:
        return sum(
            value * math.log2(value / mixture[key]) for key, value in values.items() if value > 0
        )

    mixed = {key: (p.get(key, 0.0) + q.get(key, 0.0)) / 2 for key in keys}
    return (kl(p, mixed) + kl(q, mixed)) / 2


def compare_distributions(
    reference: Mapping[str, int], subset: Mapping[str, int]
) -> dict[str, object]:
    """Compare a subset distribution with its reference population."""
    ref = proportions(reference)
    sub = proportions(subset)
    keys = sorted(set(ref) | set(sub))
    deltas = {key: sub.get(key, 0.0) - ref.get(key, 0.0) for key in keys}
    return {
        "jensen_shannon_divergence": jensen_shannon_divergence(reference, subset),
        "max_absolute_proportion_difference": max(
            (abs(value) for value in deltas.values()), default=0.0
        ),
        "proportion_deltas": deltas,
    }


def summarize_records(records: Sequence[Record]) -> dict[str, object]:
    """Summarize task, answer, text length, duplicates and image references."""
    tasks = [classify_task(record["query"], record["label"]) for record in records]
    answers = [classify_answer(record["query"], record["label"]) for record in records]
    lengths = [len(normalize_question(record["query"]).split()) for record in records]
    exact_keys = [
        (record["imgname"], normalize_question(record["query"]), normalize_answer(record["label"]))
        for record in records
    ]
    qa_keys = [
        (normalize_question(record["query"]), normalize_answer(record["label"]))
        for record in records
    ]
    sorted_lengths = sorted(lengths)
    p90_index = max(math.ceil(len(sorted_lengths) * 0.9) - 1, 0) if lengths else 0
    return {
        "records": len(records),
        "unique_images": len({record["imgname"] for record in records}),
        "task_counts": distribution(tasks),
        "task_proportions": proportions(distribution(tasks)),
        "answer_counts": distribution(answers),
        "answer_proportions": proportions(distribution(answers)),
        "question_words": {
            "mean": mean(lengths) if lengths else 0.0,
            "median": median(lengths) if lengths else 0.0,
            "p90": sorted_lengths[p90_index] if lengths else 0,
            "max": max(lengths, default=0),
        },
        "duplicate_exact_records": len(exact_keys) - len(set(exact_keys)),
        "duplicate_normalized_question_answer": len(qa_keys) - len(set(qa_keys)),
        "other_task_count": tasks.count("other"),
        "other_task_fraction": tasks.count("other") / len(tasks) if tasks else 0.0,
    }


def deterministic_stratified_indices(
    records: Sequence[Record],
    *,
    sample_size: int,
    seed: int,
    excluded_indices: Iterable[int] = (),
) -> list[int]:
    """Sample records proportionally across joint task/answer strata."""
    excluded = set(excluded_indices)
    eligible_count = len(records) - len(excluded)
    if any(index < 0 or index >= len(records) for index in excluded):
        raise ValueError("excluded_indices contain an out-of-range record")
    if not 0 < sample_size <= eligible_count:
        raise ValueError("sample_size must be within the available record range")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if index in excluded:
            continue
        task = classify_task(record["query"], record["label"])
        answer = classify_answer(record["query"], record["label"])
        key = f"{task}|{answer}"
        groups.setdefault(key, []).append(index)

    ideals = {key: sample_size * len(indices) / eligible_count for key, indices in groups.items()}
    quotas = {key: math.floor(value) for key, value in ideals.items()}
    remaining = sample_size - sum(quotas.values())
    remainder_order = sorted(
        groups,
        key=lambda key: (-(ideals[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    selected: list[int] = []
    for key in sorted(groups):
        candidates = groups[key].copy()
        random.Random(f"{seed}:{key}").shuffle(candidates)
        selected.extend(candidates[: quotas[key]])
    random.Random(seed).shuffle(selected)
    if len(selected) != sample_size or len(set(selected)) != sample_size:
        raise RuntimeError("Stratified sampler did not produce the requested unique sample")
    return selected
