"""Versioned, public experiment manifests for reproducible benchmark runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from repopilot.benchmarks.contracts import NetworkPolicy
from repopilot.core.budgets import RunBudget
from repopilot.core.contracts import JSONValue


@dataclass(frozen=True, slots=True)
class ManifestAsset:
    """One frozen task input recorded by content hash."""

    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("manifest asset uri must be non-empty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.sha256
        ):
            raise ValueError("manifest asset sha256 must be 64 hexadecimal characters")


@dataclass(frozen=True, slots=True)
class ManifestTask:
    """A preregistered task and its immutable public assets."""

    task_id: str
    assets: tuple[ManifestAsset, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("manifest task_id must be non-empty")


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Everything needed to identify a run, excluding evaluator secrets."""

    benchmark_id: str
    split: str
    dataset_revision: str
    evaluator_revision: str
    tasks: tuple[ManifestTask, ...]
    model_id: str
    model_revision: str
    prompt_revision: str
    tool_config_revision: str
    environment_revision: str
    dependency_lock_sha256: str
    random_seed: int
    budget: RunBudget = field(default_factory=RunBudget)
    network_policy: NetworkPolicy = NetworkPolicy.DENY
    model_parameters: Mapping[str, JSONValue] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        text_fields = (
            self.benchmark_id,
            self.split,
            self.dataset_revision,
            self.evaluator_revision,
            self.model_id,
            self.model_revision,
            self.prompt_revision,
            self.tool_config_revision,
            self.environment_revision,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("manifest identity and revision fields must be non-empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported manifest schema_version: {self.schema_version}")
        if not self.tasks or len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("manifest tasks must be non-empty with unique task IDs")
        if len(self.dependency_lock_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.dependency_lock_sha256
        ):
            raise ValueError("dependency_lock_sha256 must be 64 hexadecimal characters")

    def to_dict(self) -> dict[str, JSONValue]:
        data = cast(dict[str, JSONValue], asdict(self))
        data["network_policy"] = self.network_policy.value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, JSONValue]) -> BenchmarkManifest:
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("manifest tasks must be a list")
        tasks: list[ManifestTask] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                raise ValueError("each manifest task must be an object")
            raw_assets = raw_task.get("assets", [])
            if not isinstance(raw_assets, list):
                raise ValueError("manifest task assets must be a list")
            assets = tuple(
                ManifestAsset(str(asset["uri"]), str(asset["sha256"]))
                for asset in raw_assets
                if isinstance(asset, dict)
            )
            if len(assets) != len(raw_assets):
                raise ValueError("each manifest asset must be an object")
            tasks.append(ManifestTask(str(raw_task["task_id"]), assets))
        raw_budget = data.get("budget", {})
        if not isinstance(raw_budget, dict):
            raise ValueError("manifest budget must be an object")
        raw_parameters = data.get("model_parameters", {})
        if not isinstance(raw_parameters, dict):
            raise ValueError("model_parameters must be an object")
        return cls(
            benchmark_id=str(data["benchmark_id"]),
            split=str(data["split"]),
            dataset_revision=str(data["dataset_revision"]),
            evaluator_revision=str(data["evaluator_revision"]),
            tasks=tuple(tasks),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            prompt_revision=str(data["prompt_revision"]),
            tool_config_revision=str(data["tool_config_revision"]),
            environment_revision=str(data["environment_revision"]),
            dependency_lock_sha256=str(data["dependency_lock_sha256"]),
            random_seed=int(data["random_seed"]),
            budget=RunBudget(
                max_iterations=int(raw_budget.get("max_iterations", 12)),
                max_tool_calls=int(raw_budget.get("max_tool_calls", 32)),
                max_total_tokens=int(raw_budget.get("max_total_tokens", 32_000)),
                max_wall_seconds=float(raw_budget.get("max_wall_seconds", 600.0)),
            ),
            network_policy=NetworkPolicy(str(data.get("network_policy", "deny"))),
            model_parameters=cast(dict[str, JSONValue], raw_parameters),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def load(cls, path: Path) -> BenchmarkManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("benchmark manifest must contain a JSON object")
        return cls.from_dict(cast(dict[str, JSONValue], raw))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
