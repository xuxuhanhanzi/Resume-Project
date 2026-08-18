"""Executable-operation consistency reward."""

from __future__ import annotations

from dataclasses import dataclass

from forgemm.reasoning.executor import ExecutionResult, execute
from forgemm.reasoning.normalizer import answers_match
from forgemm.reasoning.parser import StructuredPrediction


@dataclass(frozen=True)
class OperationReward:
    """Mask-aware execution reward with attributable failure state."""

    value: float
    applicable: bool
    execution: ExecutionResult | None


def operation_reward(
    prediction: StructuredPrediction, *, applicable: bool = True
) -> OperationReward:
    """Reward an executable operation whose result supports the predicted answer."""

    if not applicable:
        return OperationReward(0.0, False, None)
    result = execute(prediction.operation, prediction.evidence)
    score = 1.0 if result.success and answers_match(result.value, prediction.answer) else 0.0
    return OperationReward(score, True, result)
