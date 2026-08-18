"""Exact and normalized n-gram contamination audits."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass


def normalize_for_contamination(text: str) -> str:
    """Apply a documented conservative normalization without semantic rewriting."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_sha256(text: str) -> str:
    """Hash normalized text."""
    return hashlib.sha256(normalize_for_contamination(text).encode("utf-8")).hexdigest()


def character_ngrams(text: str, *, n: int = 13) -> set[str]:
    """Return normalized character n-grams for near-duplicate screening."""
    if n <= 0:
        raise ValueError("n must be positive")
    normalized = normalize_for_contamination(text)
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return exact Jaccard similarity, including empty-set semantics."""
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True, slots=True)
class ContaminationMatch:
    """One exact or near-overlap candidate requiring explicit reporting."""

    evaluation_id: str
    training_id: str
    exact: bool
    similarity: float

    def as_dict(self) -> dict[str, str | bool | float]:
        """Return serializable fields."""
        return asdict(self)


def audit_contamination(
    evaluation_texts: dict[str, str],
    training_texts: dict[str, str],
    *,
    ngram_size: int = 13,
    near_threshold: float = 0.8,
) -> list[ContaminationMatch]:
    """Find exact normalized hashes and high character-ngram Jaccard candidates."""
    if not 0 <= near_threshold <= 1:
        raise ValueError("near_threshold must be in [0, 1]")
    training_hashes: dict[str, list[str]] = {}
    training_ngrams: dict[str, set[str]] = {}
    for training_id, text in training_texts.items():
        training_hashes.setdefault(normalized_sha256(text), []).append(training_id)
        training_ngrams[training_id] = character_ngrams(text, n=ngram_size)
    matches: list[ContaminationMatch] = []
    for evaluation_id, text in evaluation_texts.items():
        digest = normalized_sha256(text)
        exact_ids = set(training_hashes.get(digest, []))
        for training_id in sorted(exact_ids):
            matches.append(ContaminationMatch(evaluation_id, training_id, True, 1.0))
        eval_ngrams = character_ngrams(text, n=ngram_size)
        for training_id, train_ngrams in training_ngrams.items():
            if training_id in exact_ids:
                continue
            similarity = jaccard_similarity(eval_ngrams, train_ngrams)
            if similarity >= near_threshold:
                matches.append(ContaminationMatch(evaluation_id, training_id, False, similarity))
    return sorted(
        matches,
        key=lambda item: (item.evaluation_id, not item.exact, -item.similarity, item.training_id),
    )
