"""One-task benchmark orchestration with an evaluator secrecy boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repopilot.benchmarks.contracts import (
    BenchmarkTask,
    EvaluationOutcome,
    TaskOutput,
)
from repopilot.benchmarks.registry import BenchmarkBundle


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Public output and evaluator result for one task."""

    task: BenchmarkTask
    output: TaskOutput
    outcome: EvaluationOutcome


class BenchmarkRunner:
    """Prepare, execute, score, and always close one isolated environment."""

    async def run_one(
        self, bundle: BenchmarkBundle, task: BenchmarkTask, *, run_dir: Path
    ) -> BenchmarkRun:
        if task.benchmark_id != bundle.adapter.benchmark_id:
            raise ValueError("task benchmark_id does not match registered adapter")
        prepared = bundle.adapter.prepare(task, run_dir)
        if prepared.task != task:
            raise ValueError("adapter returned a prepared task with a different public task")
        handle = await bundle.environment.start(prepared)
        try:
            output = await bundle.executor.execute(task, prepared, handle)
            case = bundle.adapter.evaluator_case(task.task_id)
            if (
                case.benchmark_id != task.benchmark_id
                or case.dataset_revision != task.dataset_revision
                or case.task_id != task.task_id
            ):
                raise ValueError("evaluator case identity does not match public task")
            outcome = await bundle.evaluator.evaluate(output, case, handle)
            return BenchmarkRun(task, output, outcome)
        finally:
            await bundle.environment.close(handle)
