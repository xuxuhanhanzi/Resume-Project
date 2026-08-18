"""End-to-end completion scoring with parse-failure semantics."""

from __future__ import annotations

from dataclasses import dataclass

from forgemm.data.schemas import EvidenceCell
from forgemm.reasoning.parser import ParseError, parse_prediction
from forgemm.rewards.answer import answer_reward
from forgemm.rewards.evidence import evidence_reward
from forgemm.rewards.operation import operation_reward


@dataclass(frozen=True)
class RewardBundle:
    """Three bounded reward channels plus a stable parser diagnostic."""

    task: float
    evidence: float
    operation: float
    parse_error: str | None = None


def score_completion(
    completion: str,
    reference_answer: str,
    gold_evidence: tuple[EvidenceCell, ...],
    *,
    evidence_mask: bool = True,
    operation_mask: bool = True,
) -> RewardBundle:
    """Parse and score a completion; any parse failure zeros all channels."""

    try:
        prediction = parse_prediction(completion)
    except ParseError as exc:
        return RewardBundle(0.0, 0.0, 0.0, exc.code)
    evidence_score = evidence_reward(
        prediction.evidence, gold_evidence, applicable=evidence_mask
    ).value
    operation_score = operation_reward(prediction, applicable=operation_mask).value
    return RewardBundle(
        task=answer_reward(prediction.answer, reference_answer),
        evidence=evidence_score,
        operation=operation_score,
    )
