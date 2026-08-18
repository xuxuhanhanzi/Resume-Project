"""Aggregate faithful chart-reasoning metrics without model dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from forgemm.data.schemas import EvidenceCell
from forgemm.rewards.scoring import score_completion


@dataclass(frozen=True)
class CompletionMetrics:
    samples: int
    format_compliance: float
    task_accuracy: float
    evidence_f1: float
    operation_consistency: float
    faithful_correct_rate: float
    inconsistency_rate: float
    truncation_rate: float


def evaluate_completions(
    completions: list[str],
    reference_answers: list[str],
    gold_evidence: list[tuple[EvidenceCell, ...]],
    evidence_masks: list[bool],
    operation_masks: list[bool],
    truncated: list[bool] | None = None,
) -> CompletionMetrics:
    count = len(completions)
    if count == 0 or not (
        len(reference_answers)
        == len(gold_evidence)
        == len(evidence_masks)
        == len(operation_masks)
        == count
    ):
        raise ValueError("evaluation inputs must be non-empty and aligned")
    truncated = [False] * count if truncated is None else truncated
    if len(truncated) != count:
        raise ValueError("truncation flags must align with completions")
    bundles = [
        score_completion(
            completions[index],
            reference_answers[index],
            gold_evidence[index],
            evidence_mask=evidence_masks[index],
            operation_mask=operation_masks[index],
        )
        for index in range(count)
    ]
    format_ok = [bundle.parse_error is None for bundle in bundles]
    fcr = [
        bundle.task == 1.0
        and (not evidence_masks[index] or bundle.evidence == 1.0)
        and (not operation_masks[index] or bundle.operation == 1.0)
        for index, bundle in enumerate(bundles)
    ]
    inconsistent = [bundle.task == 1.0 and not fcr[index] for index, bundle in enumerate(bundles)]
    evidence_values = [
        bundle.evidence for index, bundle in enumerate(bundles) if evidence_masks[index]
    ]
    operation_values = [
        bundle.operation for index, bundle in enumerate(bundles) if operation_masks[index]
    ]
    return CompletionMetrics(
        samples=count,
        format_compliance=_mean(format_ok),
        task_accuracy=_mean([bundle.task for bundle in bundles]),
        evidence_f1=_mean(evidence_values),
        operation_consistency=_mean(operation_values),
        faithful_correct_rate=_mean(fcr),
        inconsistency_rate=_mean(inconsistent),
        truncation_rate=_mean(truncated),
    )


def _mean(values: list[float] | list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
