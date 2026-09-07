"""Workspace contracts shared by benchmark tasks and interactive sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Protocol

from repopilot.core.budgets import RunBudget


class WorkspaceTask(Protocol):
    """The safety-relevant subset of a task exposed to filesystem tools."""

    @property
    def task_id(self) -> str: ...

    @property
    def workspace(self) -> Path: ...

    @property
    def problem_statement(self) -> str: ...

    @property
    def allowed_paths(self) -> tuple[str, ...]: ...

    @property
    def forbidden_paths(self) -> tuple[str, ...]: ...

    @property
    def visible_tests(self) -> tuple[str, ...]: ...

    @property
    def test_command(self) -> tuple[str, ...]: ...

    @property
    def max_changed_files(self) -> int: ...

    @property
    def budget(self) -> RunBudget: ...


@dataclass(frozen=True, slots=True)
class InteractiveTask:
    """A session turn scoped to a real project rather than a benchmark fixture."""

    task_id: str
    workspace: Path
    problem_statement: str
    allowed_paths: tuple[str, ...] = (".",)
    forbidden_paths: tuple[str, ...] = ()
    visible_tests: tuple[str, ...] = ()
    test_command: tuple[str, ...] = ()
    max_changed_files: int = 1_000
    budget: RunBudget = field(default_factory=RunBudget)

    def __post_init__(self) -> None:
        root = self.workspace.resolve()
        if not root.is_dir():
            raise ValueError(f"interactive workspace does not exist: {root}")
        if not self.task_id.strip() or not self.problem_statement.strip():
            raise ValueError("task_id and problem_statement must be non-empty")
        if self.max_changed_files <= 0:
            raise ValueError("max_changed_files must be positive")
        for value in (*self.allowed_paths, *self.forbidden_paths):
            portable = value.replace("\\", "/")
            if PurePath(portable).is_absolute() or ".." in PurePath(portable).parts:
                raise ValueError(f"workspace path must be relative and traversal-free: {value!r}")
        object.__setattr__(self, "workspace", root)
