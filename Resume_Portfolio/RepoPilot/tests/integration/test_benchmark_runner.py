from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from repopilot.benchmarks.contracts import (
    BenchmarkDomain,
    BenchmarkTask,
    EnvironmentHandle,
    EvaluationOutcome,
    EvaluatorCase,
    PreparedTask,
    SecretRef,
    TaskOutput,
)
from repopilot.benchmarks.registry import BenchmarkBundle, BenchmarkRegistry
from repopilot.benchmarks.runner import BenchmarkRunner


class FixtureAdapter:
    def __init__(self, domain: BenchmarkDomain) -> None:
        self.domain = domain
        self._benchmark_id = f"fixture-{domain.value}"

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]:
        assert split == "smoke"
        task = BenchmarkTask(
            self.benchmark_id,
            "fixture-revision",
            "task-1",
            self.domain,
            "Return 42",
            "fixture-environment",
        )
        if task_ids is None or task.task_id in task_ids:
            yield task

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask:
        return PreparedTask(task, run_dir, public_payload={"question": task.instruction})

    def evaluator_case(self, task_id: str) -> EvaluatorCase:
        return EvaluatorCase(
            self.benchmark_id,
            "fixture-revision",
            task_id,
            "exact-match-v1",
            SecretRef("fixture-oracles", task_id),
        )


class FixtureEnvironment:
    def __init__(self) -> None:
        self.closed = False

    async def start(self, prepared: PreparedTask) -> EnvironmentHandle:
        prepared.work_dir.mkdir(parents=True, exist_ok=True)
        return EnvironmentHandle("fixture", prepared.work_dir)

    async def reset(self, handle: EnvironmentHandle) -> None:
        del handle

    async def close(self, handle: EnvironmentHandle) -> None:
        del handle
        self.closed = True


class FixtureExecutor:
    async def execute(
        self, task: BenchmarkTask, prepared: PreparedTask, handle: EnvironmentHandle
    ) -> TaskOutput:
        assert prepared.public_payload == {"question": "Return 42"}
        assert handle.handle_id == "fixture"
        assert task.task_id == "task-1"
        return TaskOutput(final_answer="42")


class FixtureEvaluator:
    async def evaluate(
        self, output: TaskOutput, case: EvaluatorCase, handle: EnvironmentHandle
    ) -> EvaluationOutcome:
        assert case.oracle_ref.store_id == "fixture-oracles"
        assert handle.handle_id == "fixture"
        correct = output.final_answer == "42"
        return EvaluationOutcome(correct, "accuracy", float(correct))


@pytest.mark.integration
@pytest.mark.parametrize("domain", list(BenchmarkDomain))
def test_three_domains_follow_the_same_run_contract(
    tmp_path: Path, domain: BenchmarkDomain
) -> None:
    adapter = FixtureAdapter(domain)
    environment = FixtureEnvironment()
    bundle = BenchmarkBundle(adapter, environment, FixtureExecutor(), FixtureEvaluator())
    registry = BenchmarkRegistry()
    registry.register(bundle)
    task = next(iter(adapter.load("smoke")))

    result = asyncio.run(
        BenchmarkRunner().run_one(registry.get(adapter.benchmark_id), task, run_dir=tmp_path)
    )

    assert result.outcome.task_success
    assert result.outcome.primary_metric_name == "accuracy"
    assert environment.closed


def test_registry_rejects_duplicate_benchmark(tmp_path: Path) -> None:
    del tmp_path
    adapter = FixtureAdapter(BenchmarkDomain.RESEARCH)
    bundle = BenchmarkBundle(adapter, FixtureEnvironment(), FixtureExecutor(), FixtureEvaluator())
    registry = BenchmarkRegistry()
    registry.register(bundle)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(bundle)
