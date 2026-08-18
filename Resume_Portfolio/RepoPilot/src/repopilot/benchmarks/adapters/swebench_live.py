"""Pinned SWE-bench-Live task adapter and official-result boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from repopilot.benchmarks.contracts import (
    AssetRef,
    BenchmarkDomain,
    BenchmarkTask,
    EnvironmentHandle,
    EvaluationOutcome,
    EvaluatorCase,
    NetworkPolicy,
    PreparedTask,
    SecretRef,
    TaskEvaluator,
    TaskOutput,
)
from repopilot.core.contracts import JSONValue


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, JSONValue]]:
    records: list[dict[str, JSONValue]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
            records.append(cast(dict[str, JSONValue], raw))
    return records


def _string_tuple(value: JSONValue, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"SWE-bench-Live {field_name} must be a list")
    return tuple(str(item) for item in value)


def _official_test_pass_rate(value: object, *, field_name: str) -> float:
    if not isinstance(value, Mapping):
        raise ValueError(f"official evaluator {field_name} status must be an object")
    successes = value.get("success")
    failures = value.get("failure")
    if not isinstance(successes, list) or not isinstance(failures, list):
        raise ValueError(f"official evaluator {field_name} status is incomplete")
    total = len(successes) + len(failures)
    return 1.0 if total == 0 else len(successes) / total


def normalize_official_swebench_result(
    report: Mapping[str, object], task_id: str
) -> dict[str, JSONValue]:
    """Convert one untouched upstream instance report into RepoPilot's score boundary."""

    instance = report.get(task_id)
    if not isinstance(instance, Mapping):
        raise ValueError(f"official evaluator report is missing task {task_id}")
    resolved = instance.get("resolved")
    patch_applied = instance.get("patch_successfully_applied")
    tests_status = instance.get("tests_status")
    if not isinstance(resolved, bool) or not isinstance(patch_applied, bool):
        raise ValueError("official evaluator report is missing boolean result fields")
    if not isinstance(tests_status, Mapping):
        raise ValueError("official evaluator report is missing tests_status")
    return {
        "resolved": resolved,
        "environment_built": True,
        "patch_applied": patch_applied,
        "fail_to_pass": _official_test_pass_rate(
            tests_status.get("FAIL_TO_PASS"), field_name="FAIL_TO_PASS"
        ),
        "pass_to_pass": _official_test_pass_rate(
            tests_status.get("PASS_TO_PASS"), field_name="PASS_TO_PASS"
        ),
    }


@dataclass(frozen=True, slots=True)
class _SWEbenchRecord:
    task_id: str
    public: Mapping[str, JSONValue]
    evaluator: Mapping[str, JSONValue]


class SWEbenchLiveAdapter:
    """Load a public/private extraction from one pinned official parquet revision."""

    def __init__(
        self,
        public_path: Path,
        evaluator_path: Path,
        *,
        dataset_revision: str,
        public_sha256: str,
        evaluator_sha256: str,
        workspace_by_task: Mapping[str, Path] | None = None,
    ) -> None:
        self.public_path = public_path.resolve(strict=True)
        self.evaluator_path = evaluator_path.resolve(strict=True)
        self.dataset_revision = dataset_revision
        self.workspace_by_task = dict(workspace_by_task or {})
        if _file_sha256(self.public_path) != public_sha256.casefold():
            raise ValueError("SWE-bench-Live public extraction hash mismatch")
        if _file_sha256(self.evaluator_path) != evaluator_sha256.casefold():
            raise ValueError("SWE-bench-Live evaluator extraction hash mismatch")
        self._records = self._read_records()

    @property
    def benchmark_id(self) -> str:
        return "swebench_live_lite_python"

    def _read_records(self) -> dict[str, _SWEbenchRecord]:
        public_records = {
            str(record["instance_id"]): record for record in _load_jsonl(self.public_path)
        }
        evaluator_records = {
            str(record["instance_id"]): record for record in _load_jsonl(self.evaluator_path)
        }
        if not public_records or set(public_records) != set(evaluator_records):
            raise ValueError("SWE-bench-Live public/evaluator task IDs must match and be non-empty")
        return {
            task_id: _SWEbenchRecord(task_id, public_records[task_id], evaluator_records[task_id])
            for task_id in sorted(public_records)
        }

    def load(self, split: str, task_ids: Sequence[str] | None = None) -> Iterable[BenchmarkTask]:
        if split != "lite":
            raise ValueError("this frozen SWE-bench-Live adapter supports the lite split")
        selected = tuple(task_ids) if task_ids is not None else tuple(self._records)
        for task_id in selected:
            try:
                record = self._records[task_id]
            except KeyError as error:
                raise KeyError(f"unknown SWE-bench-Live task: {task_id}") from error
            public = record.public
            problem_statement = str(public["problem_statement"]).strip()
            if not problem_statement:
                raise ValueError(f"SWE-bench-Live task has no problem statement: {task_id}")
            yield BenchmarkTask(
                benchmark_id=self.benchmark_id,
                dataset_revision=self.dataset_revision,
                task_id=task_id,
                domain=BenchmarkDomain.SOFTWARE_ENGINEERING,
                instruction=problem_statement,
                environment_id=f"swebench-live-instance:{task_id}",
                assets=(AssetRef(str(public["commit_url"]), media_type="application/git"),),
                allowed_tools=(
                    "list_files",
                    "read_file",
                    "search_text",
                    "apply_patch",
                    "run_tests",
                ),
                network_policy=NetworkPolicy.DENY,
                public_metadata={
                    "repo": str(public["repo"]),
                    "pull_number": str(public["pull_number"]),
                    "issue_numbers": public.get("issue_numbers", []),
                    "base_commit": str(public["base_commit"]),
                    "difficulty": public.get("difficulty", {}),
                },
            )

    def prepare(self, task: BenchmarkTask, run_dir: Path) -> PreparedTask:
        del run_dir
        if task.task_id not in self._records or task.benchmark_id != self.benchmark_id:
            raise ValueError("task does not belong to this SWE-bench-Live adapter")
        workspace = self.workspace_by_task.get(task.task_id)
        if workspace is None:
            raise RuntimeError(
                "SWE-bench-Live instance image/workspace is not materialized; "
                "refusing local fallback"
            )
        resolved_workspace = workspace.resolve(strict=True)
        return PreparedTask(
            task, resolved_workspace, public_payload=self._records[task.task_id].public
        )

    def evaluator_case(self, task_id: str) -> EvaluatorCase:
        try:
            record = self._records[task_id]
        except KeyError as error:
            raise KeyError(f"unknown SWE-bench-Live task: {task_id}") from error
        fail_to_pass = _string_tuple(record.evaluator["FAIL_TO_PASS"], field_name="FAIL_TO_PASS")
        pass_to_pass = _string_tuple(record.evaluator["PASS_TO_PASS"], field_name="PASS_TO_PASS")
        return EvaluatorCase(
            benchmark_id=self.benchmark_id,
            dataset_revision=self.dataset_revision,
            task_id=task_id,
            evaluator_id="swebench-live-official-wrapper-v1",
            oracle_ref=SecretRef("swebench-live-evaluator", task_id),
            evaluator_config={
                "fail_to_pass_count": len(fail_to_pass),
                "pass_to_pass_count": len(pass_to_pass),
                "log_parser": str(record.evaluator["log_parser"]),
            },
        )

    def evaluator_record(self, task_id: str) -> Mapping[str, JSONValue]:
        """Return patches and test lists only to the trusted official evaluator wrapper."""
        try:
            return self._records[task_id].evaluator
        except KeyError as error:
            raise KeyError(f"unknown SWE-bench-Live task: {task_id}") from error

    @property
    def task_count(self) -> int:
        return len(self._records)


class SWEbenchLiveEvaluator(TaskEvaluator):
    """Accept only a normalized result produced by the official evaluator."""

    async def evaluate(
        self, output: TaskOutput, case: EvaluatorCase, handle: EnvironmentHandle
    ) -> EvaluationOutcome:
        del handle
        raw = output.structured_payload.get("official_evaluator")
        if not isinstance(raw, dict):
            raise ValueError("SWE-bench-Live scoring requires an official_evaluator result")
        required = {
            "resolved",
            "environment_built",
            "patch_applied",
            "fail_to_pass",
            "pass_to_pass",
        }
        if not required.issubset(raw):
            raise ValueError("official_evaluator result is missing required fields")
        resolved = bool(raw["resolved"])
        environment_built = bool(raw["environment_built"])
        patch_applied = bool(raw["patch_applied"])
        fail_to_pass = float(raw["fail_to_pass"])
        pass_to_pass = float(raw["pass_to_pass"])
        if not 0.0 <= fail_to_pass <= 1.0 or not 0.0 <= pass_to_pass <= 1.0:
            raise ValueError("official evaluator pass rates must be between zero and one")
        if resolved:
            failure_type = None
        elif not environment_built:
            failure_type = "environment"
        elif not patch_applied:
            failure_type = "patch_apply"
        elif fail_to_pass < 1.0:
            failure_type = "target_tests"
        else:
            failure_type = "regression"
        return EvaluationOutcome(
            task_success=resolved,
            primary_metric_name="resolved",
            primary_metric_value=float(resolved),
            domain_metrics={
                "environment_build_success": float(environment_built),
                "patch_apply_success": float(patch_applied),
                "fail_to_pass_rate": fail_to_pass,
                "pass_to_pass_rate": pass_to_pass,
            },
            shared_metrics=output.run_metrics,
            failure_type=failure_type,
            evaluator_trace_ref=f"swebench-live://{case.dataset_revision}/{case.task_id}",
        )
