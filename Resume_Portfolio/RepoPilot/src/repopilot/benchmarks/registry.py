"""Explicit benchmark component registry."""

from __future__ import annotations

from dataclasses import dataclass

from repopilot.benchmarks.contracts import (
    BenchmarkAdapter,
    TaskEnvironment,
    TaskEvaluator,
    TaskExecutor,
)


@dataclass(frozen=True, slots=True)
class BenchmarkBundle:
    """All components required to execute one benchmark family."""

    adapter: BenchmarkAdapter
    environment: TaskEnvironment
    executor: TaskExecutor
    evaluator: TaskEvaluator


class BenchmarkRegistry:
    """Fail on duplicate or unknown benchmark registrations."""

    def __init__(self) -> None:
        self._bundles: dict[str, BenchmarkBundle] = {}

    def register(self, bundle: BenchmarkBundle) -> None:
        benchmark_id = bundle.adapter.benchmark_id
        if not benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if benchmark_id in self._bundles:
            raise ValueError(f"benchmark already registered: {benchmark_id}")
        self._bundles[benchmark_id] = bundle

    def get(self, benchmark_id: str) -> BenchmarkBundle:
        try:
            return self._bundles[benchmark_id]
        except KeyError as error:
            raise KeyError(f"unknown benchmark: {benchmark_id}") from error

    def benchmark_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._bundles))
