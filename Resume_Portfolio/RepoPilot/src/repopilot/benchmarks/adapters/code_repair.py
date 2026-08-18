"""Compatibility adapter for the original local Python repair tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from repopilot.benchmarks.contracts import (
    AssetRef,
    BenchmarkDomain,
    BenchmarkTask,
    EvaluatorCase,
    NetworkPolicy,
    PreparedTask,
    SecretRef,
)
from repopilot.task import EvaluatorTaskSpec


class CodeRepairAdapter:
    """Expose legacy task specs through the domain-neutral benchmark contract."""

    def __init__(
        self,
        specs: Sequence[EvaluatorTaskSpec],
        *,
        benchmark_id: str = "repopilot_python_micro_v1",
        dataset_revision: str = "local-v1",
    ) -> None:
        if not specs:
            raise ValueError("code repair adapter requires at least one task")
        self._benchmark_id = benchmark_id
        self.dataset_revision = dataset_revision
        self._specs = {spec.public.task_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("code repair task IDs must be unique")

    @property
    def benchmark_id(self) -> str:
        return self._benchmark_id

    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]:
        if split not in {"local", "test"}:
            raise ValueError("legacy code repair adapter supports local/test splits")
        selected = tuple(task_ids) if task_ids is not None else tuple(sorted(self._specs))
        for task_id in selected:
            try:
                spec = self._specs[task_id].public
            except KeyError as error:
                raise KeyError(f"unknown code repair task: {task_id}") from error
            yield BenchmarkTask(
                benchmark_id=self.benchmark_id,
                dataset_revision=self.dataset_revision,
                task_id=spec.task_id,
                domain=BenchmarkDomain.SOFTWARE_ENGINEERING,
                instruction=spec.problem_statement,
                environment_id="legacy-python-workspace",
                assets=(AssetRef(spec.workspace.as_uri(), mount_path="."),),
                allowed_tools=(
                    "list_files",
                    "read_file",
                    "search_text",
                    "apply_patch",
                    "run_tests",
                ),
                budget=spec.budget,
                network_policy=NetworkPolicy(spec.network_policy),
                public_metadata={
                    "language": spec.language,
                    "allowed_paths": list(spec.allowed_paths),
                    "forbidden_paths": list(spec.forbidden_paths),
                    "visible_tests": list(spec.visible_tests),
                    "setup_command": list(spec.setup_command),
                    "test_command": list(spec.test_command),
                    "max_changed_files": spec.max_changed_files,
                    "trusted_fixture": spec.trusted_fixture,
                },
            )

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask:
        del run_dir
        spec = self._specs.get(task.task_id)
        if spec is None or task.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to this code repair adapter")
        return PreparedTask(task, spec.public.workspace, public_payload=spec.public)

    def evaluator_case(self, task_id: str) -> EvaluatorCase:
        try:
            spec = self._specs[task_id]
        except KeyError as error:
            raise KeyError(f"unknown code repair task: {task_id}") from error
        return EvaluatorCase(
            benchmark_id=self.benchmark_id,
            dataset_revision=self.dataset_revision,
            task_id=task_id,
            evaluator_id="legacy-hidden-test-grader-v1",
            oracle_ref=SecretRef("code-repair-evaluator", task_id),
            evaluator_config={
                "hidden_test_count": len(spec.hidden_tests),
                "has_gold_patch_hash": spec.gold_patch_sha256 is not None,
            },
        )

    def evaluator_spec(self, task_id: str) -> EvaluatorTaskSpec:
        """Return private data only to trusted evaluator construction code."""
        try:
            return self._specs[task_id]
        except KeyError as error:
            raise KeyError(f"unknown code repair task: {task_id}") from error
