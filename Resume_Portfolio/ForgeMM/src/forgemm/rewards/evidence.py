"""Evidence cell precision/recall/F1 reward."""

from __future__ import annotations

from dataclasses import dataclass

from forgemm.data.schemas import EvidenceCell
from forgemm.reasoning.normalizer import decimal_to_text, normalize_number, normalize_text


@dataclass(frozen=True)
class EvidenceReward:
    """Mask-aware evidence score and its audit components."""

    value: float
    precision: float
    recall: float
    applicable: bool


def evidence_reward(
    predicted: tuple[EvidenceCell, ...],
    gold: tuple[EvidenceCell, ...],
    *,
    applicable: bool = True,
) -> EvidenceReward:
    """Score canonical row/column/value tuples with set F1."""

    if not applicable:
        return EvidenceReward(0.0, 0.0, 0.0, False)
    predicted_set = {_canonical(cell) for cell in predicted}
    gold_set = {_canonical(cell) for cell in gold}
    if not predicted_set and not gold_set:
        return EvidenceReward(1.0, 1.0, 1.0, True)
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EvidenceReward(f1, precision, recall, True)


def _canonical(cell: EvidenceCell) -> tuple[str, str, str]:
    try:
        value = decimal_to_text(normalize_number(cell.value))
    except ValueError:
        value = normalize_text(cell.value)
    return (
        normalize_text(cell.row),
        normalize_text(cell.column),
        value,
    )
