"""Ablation report construction from fixed-scaffold trial groups."""

from __future__ import annotations

from dataclasses import dataclass

from repopilot.evaluation.metrics import EvaluationRecord, EvaluationSummary, summarize


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One named scaffold variant and its aggregate metrics."""

    name: str
    summary: EvaluationSummary


def build_ablation_report(groups: dict[str, list[EvaluationRecord]]) -> tuple[AblationResult, ...]:
    """Summarize variants in stable name order; callers must keep model/budget fixed."""
    if not groups:
        raise ValueError("ablation groups must not be empty")
    task_sets = [{record.task_id for record in records} for records in groups.values()]
    if any(task_set != task_sets[0] for task_set in task_sets[1:]):
        raise ValueError("all ablation variants must contain the same task IDs")
    return tuple(AblationResult(name, summarize(groups[name])) for name in sorted(groups))
