"""Immutable public and evaluator-only task contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from repopilot.core.budgets import RunBudget


def _safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"task path must be relative and traversal-free: {value!r}")
    return str(path)


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(str(item) for item in value)


def _command_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    command = _string_tuple(value, field_name=field_name)
    if any("\x00" in item or "\n" in item or "\r" in item for item in command):
        raise ValueError(f"{field_name} contains an invalid control character")
    return command


@dataclass(frozen=True, slots=True)
class PublicTaskSpec:
    """Task data that is safe to expose to the agent."""

    task_id: str
    workspace: Path
    problem_statement: str
    language: str = "python"
    allowed_paths: tuple[str, ...] = (".",)
    forbidden_paths: tuple[str, ...] = ()
    visible_tests: tuple[str, ...] = ()
    setup_command: tuple[str, ...] = ()
    test_command: tuple[str, ...] = ()
    network_policy: str = "deny"
    max_changed_files: int = 8
    budget: RunBudget = field(default_factory=RunBudget)
    trusted_fixture: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.resolve())
        if not self.task_id.strip() or not self.problem_statement.strip():
            raise ValueError("task_id and problem_statement must be non-empty")
        if self.language.lower() != "python":
            raise ValueError("the Stage 1-6 demo supports Python tasks only")
        if self.network_policy not in {"deny", "allowlist"}:
            raise ValueError("network_policy must be deny or allowlist")
        if self.max_changed_files <= 0:
            raise ValueError("max_changed_files must be positive")
        for path in (*self.allowed_paths, *self.forbidden_paths):
            _safe_relative_path(path)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], *, base_dir: Path) -> PublicTaskSpec:
        raw_workspace = str(data.get("workspace", "."))
        workspace_path = Path(raw_workspace)
        if not workspace_path.is_absolute():
            workspace_path = base_dir / workspace_path
        budget = RunBudget(
            max_iterations=int(data.get("max_iterations", 12)),
            max_tool_calls=int(data.get("max_tool_calls", 32)),
            max_total_tokens=int(data.get("token_budget", 32_000)),
            max_wall_seconds=float(data.get("time_budget_seconds", 600.0)),
        )
        allowed = _string_tuple(data.get("allowed_paths", ["."]), field_name="allowed_paths")
        forbidden = _string_tuple(data.get("forbidden_paths", []), field_name="forbidden_paths")
        return cls(
            task_id=str(data["task_id"]),
            workspace=workspace_path.resolve(),
            problem_statement=str(data["problem_statement"]),
            language=str(data.get("language", "python")),
            allowed_paths=tuple(_safe_relative_path(item) for item in allowed),
            forbidden_paths=tuple(_safe_relative_path(item) for item in forbidden),
            visible_tests=_string_tuple(data.get("visible_tests", []), field_name="visible_tests"),
            setup_command=_command_tuple(data.get("setup_command", []), field_name="setup_command"),
            test_command=_command_tuple(data.get("test_command", []), field_name="test_command"),
            network_policy=str(data.get("network_policy", "deny")),
            max_changed_files=int(data.get("max_changed_files", 8)),
            budget=budget,
            trusted_fixture=bool(data.get("trusted_fixture", False)),
        )

    @classmethod
    def load(cls, path: Path) -> PublicTaskSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("task file must contain a YAML object")
        if "hidden_tests" in raw:
            raise ValueError("public task files must not contain hidden_tests")
        return cls.from_mapping(cast(dict[str, Any], raw), base_dir=path.parent)


@dataclass(frozen=True, slots=True)
class EvaluatorTaskSpec:
    """Evaluator-only data that must never enter model context or its sandbox."""

    public: PublicTaskSpec
    hidden_tests: tuple[str, ...] = ()
    gold_patch_sha256: str | None = None

    @classmethod
    def load(cls, path: Path) -> EvaluatorTaskSpec:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("evaluator task file must contain a YAML object")
        mapping = cast(dict[str, Any], raw)
        public_mapping = {key: value for key, value in mapping.items() if key != "evaluator"}
        evaluator = mapping.get("evaluator", {})
        if not isinstance(evaluator, dict):
            raise ValueError("evaluator must be an object")
        return cls(
            public=PublicTaskSpec.from_mapping(public_mapping, base_dir=path.parent),
            hidden_tests=_string_tuple(
                evaluator.get("hidden_tests", []), field_name="hidden_tests"
            ),
            gold_patch_sha256=(
                str(evaluator["gold_patch_sha256"])
                if evaluator.get("gold_patch_sha256") is not None
                else None
            ),
        )
