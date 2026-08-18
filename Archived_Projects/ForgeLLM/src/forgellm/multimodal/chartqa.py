"""ChartQA answer normalization and relaxed-accuracy utilities."""

from __future__ import annotations

import re


def normalize_answer(value: str) -> str:
    """Normalize a short ChartQA answer without attempting semantic extraction."""
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    return re.sub(r"\s*/\s*", "/", normalized)


def normalized_exact_correct(prediction: str, reference: str) -> bool:
    """Return the project's primary short-answer metric."""
    return normalize_answer(prediction) == normalize_answer(reference)


def chartqa_relaxed_correct(prediction: str, reference: str, *, tolerance: float = 0.05) -> bool:
    """Return exact-match for text or relative-error match for numeric answers.

    This mirrors ChartQA's commonly used relaxed numerical comparison. It is kept
    as a secondary comparability metric because a percentage tolerance can make
    close-looking years appear correct. It deliberately does not pull a number out
    of a longer explanation: prompt compliance is part of the baseline.
    """
    if normalized_exact_correct(prediction, reference):
        return True
    normalized_prediction = normalize_answer(prediction)
    normalized_reference = normalize_answer(reference)

    try:
        predicted_number = float(normalized_prediction.replace(",", "").rstrip("%"))
        reference_number = float(normalized_reference.replace(",", "").rstrip("%"))
    except ValueError:
        return False
    if reference_number == 0.0:
        return predicted_number == 0.0
    return abs(predicted_number - reference_number) / abs(reference_number) <= tolerance
