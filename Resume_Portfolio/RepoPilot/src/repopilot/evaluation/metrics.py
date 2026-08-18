"""Small deterministic metrics for the Stage 6 benchmark harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from repopilot.core.contracts import JSONValue, RunStatus


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One task trial, independent of final natural-language phrasing."""

    task_id: str
    status: RunStatus
    hidden_tests_passed: bool
    iterations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    changed_files: int
    security_blocks: int = 0


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate outcome, cost, latency, and safety evidence."""

    tasks: int
    resolved: int
    resolve_rate: float
    hidden_pass_rate: float
    average_iterations: float
    average_tool_calls: float
    average_tokens: float
    average_wall_seconds: float
    security_blocks: int

    def to_dict(self) -> dict[str, JSONValue]:
        return asdict(self)


def summarize(records: list[EvaluationRecord]) -> EvaluationSummary:
    """Aggregate registered trials; an empty suite is a hard error."""
    if not records:
        raise ValueError("evaluation records must not be empty")
    count = len(records)
    resolved = sum(record.status is RunStatus.COMPLETED for record in records)
    hidden = sum(record.hidden_tests_passed for record in records)
    return EvaluationSummary(
        tasks=count,
        resolved=resolved,
        resolve_rate=resolved / count,
        hidden_pass_rate=hidden / count,
        average_iterations=sum(record.iterations for record in records) / count,
        average_tool_calls=sum(record.tool_calls for record in records) / count,
        average_tokens=sum(record.input_tokens + record.output_tokens for record in records)
        / count,
        average_wall_seconds=sum(record.wall_seconds for record in records) / count,
        security_blocks=sum(record.security_blocks for record in records),
    )


def retrieval_metrics(
    rankings: dict[str, list[str]], relevant: dict[str, set[str]], *, k: int
) -> dict[str, float]:
    """Compute File Recall@k and MRR for a frozen localization set."""
    if k <= 0 or not rankings or set(rankings) != set(relevant):
        raise ValueError("rankings/relevant must share non-empty task IDs and positive k")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for task_id, ranked in rankings.items():
        expected = relevant[task_id]
        if not expected:
            raise ValueError(f"task {task_id} has no relevant files")
        top = ranked[:k]
        recalls.append(len(set(top) & expected) / len(expected))
        reciprocal = 0.0
        for index, path in enumerate(ranked, start=1):
            if path in expected:
                reciprocal = 1.0 / index
                break
        reciprocal_ranks.append(reciprocal)
    return {
        f"file_recall@{k}": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }
