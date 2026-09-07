"""Bounded command verification with provider-neutral structured reports."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tomllib import TOMLDecodeError
from tomllib import loads as load_toml

from repopilot.runtime.runner import CommandRunner, ExecutionResult
from repopilot.tools.shell import validate_tokenized_command
from repopilot.workspace.project import ProjectWorkspace


class VerificationKind(StrEnum):
    """Stable categories exposed by the interactive verification selector."""

    TEST = "test"
    LINT = "lint"
    TYPECHECK = "typecheck"
    BUILD = "build"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class VerificationCommand:
    """One explicit argv verification operation in a project workspace."""

    label: str
    command: tuple[str, ...]
    cwd: Path | None = None
    timeout_seconds: float = 120.0
    kind: VerificationKind = VerificationKind.CUSTOM

    def __post_init__(self) -> None:
        validate_tokenized_command(list(self.command))
        if not self.label.strip() or not 1 <= self.timeout_seconds <= 120.0:
            raise ValueError("verification needs a label and a 1-120 second timeout")


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """One command result, including launch failures that have no exit code."""

    verification: VerificationCommand
    execution: ExecutionResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(
            self.execution
            and self.execution.exit_code == 0
            and not self.execution.timed_out
            and self.error is None
        )


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Immutable aggregate report suitable for transcript persistence and final answers."""

    results: tuple[VerificationResult, ...]

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(result.ok for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "results": [
                {
                    "label": item.verification.label,
                    "kind": item.verification.kind.value,
                    "command": list(item.verification.command),
                    "ok": item.ok,
                    "error": item.error,
                    "exit_code": item.execution.exit_code if item.execution else None,
                    "stdout": item.execution.stdout if item.execution else "",
                    "stderr": item.execution.stderr if item.execution else "",
                    "timed_out": item.execution.timed_out if item.execution else False,
                }
                for item in self.results
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> VerificationReport:
        """Restore a persisted report without re-running any command."""

        raw_results = raw.get("results", [])
        if not isinstance(raw_results, list):
            raise ValueError("verification report results must be a list")
        results: list[VerificationResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError("verification report result must be an object")
            label = item.get("label")
            raw_command = item.get("command")
            raw_kind = item.get("kind", VerificationKind.CUSTOM.value)
            if (
                not isinstance(label, str)
                or not isinstance(raw_command, list)
                or not all(isinstance(part, str) for part in raw_command)
                or not isinstance(raw_kind, str)
            ):
                raise ValueError("verification report has invalid command metadata")
            try:
                kind = VerificationKind(raw_kind)
            except ValueError as error:
                raise ValueError("verification report has invalid command kind") from error
            verification = VerificationCommand(label, tuple(raw_command), kind=kind)
            exit_code = item.get("exit_code")
            stdout = item.get("stdout", "")
            stderr = item.get("stderr", "")
            timed_out = item.get("timed_out", False)
            raw_error = item.get("error")
            if (
                not isinstance(stdout, str)
                or not isinstance(stderr, str)
                or not isinstance(timed_out, bool)
            ):
                raise ValueError("verification report has invalid execution output")
            execution = (
                ExecutionResult(verification.command, exit_code, stdout, stderr, timed_out)
                if isinstance(exit_code, int) and not isinstance(exit_code, bool)
                else None
            )
            results.append(
                VerificationResult(
                    verification, execution, raw_error if isinstance(raw_error, str) else None
                )
            )
        return cls(tuple(results))


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    """A project-discovered, user-selectable set of safe argv verifications."""

    commands: tuple[VerificationCommand, ...]

    def select(self, selector: str) -> tuple[VerificationCommand, ...]:
        """Select all, one category, or one label without accepting shell syntax."""

        normalized = selector.strip().casefold()
        if normalized in {"", "all"}:
            return self.commands
        selected = tuple(
            command
            for command in self.commands
            if command.kind.value == normalized or command.label.casefold() == normalized
        )
        if not selected:
            raise ValueError(f"unknown verification selector: {selector}")
        return selected


class VerificationEngine:
    """Run explicitly selected verification commands through the existing runner."""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    async def run(
        self, workspace: Path, commands: Sequence[VerificationCommand]
    ) -> VerificationReport:
        """Execute commands serially so their diagnostics remain intelligible."""
        root = workspace.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"verification workspace does not exist: {root}")
        results: list[VerificationResult] = []
        for verification in commands:
            cwd = self._resolve_cwd(root, verification.cwd)
            try:
                execution = await self.runner.run(
                    verification.command,
                    cwd=cwd,
                    timeout_seconds=verification.timeout_seconds,
                )
            except (OSError, PermissionError, RuntimeError, ValueError) as error:
                results.append(VerificationResult(verification, None, str(error)))
                continue
            results.append(VerificationResult(verification, execution))
        return VerificationReport(tuple(results))

    @staticmethod
    def _resolve_cwd(workspace: Path, requested: Path | None) -> Path:
        cwd = (requested or workspace).resolve()
        if cwd != workspace and workspace not in cwd.parents:
            raise ValueError("verification cwd escaped its workspace")
        if not cwd.is_dir():
            raise ValueError("verification cwd must be an existing directory")
        return cwd


def default_verification_plan(project: ProjectWorkspace) -> VerificationPlan:
    """Discover conservative commands; discovery alone never starts a process."""

    commands: list[VerificationCommand] = []
    if "python" in project.languages and "pyproject.toml" in project.build_markers:
        commands.append(
            VerificationCommand(
                "pytest", (sys.executable, "-m", "pytest"), kind=VerificationKind.TEST
            )
        )
        commands.extend(_python_quality_commands(project.root / "pyproject.toml"))
    package_json = project.root / "package.json"
    if package_json.is_file():
        commands.extend(_node_verification_commands(package_json))
    if "Cargo.toml" in project.build_markers:
        commands.append(
            VerificationCommand("cargo test", ("cargo", "test"), kind=VerificationKind.TEST)
        )
    if "go.mod" in project.build_markers:
        commands.append(
            VerificationCommand("go test", ("go", "test", "./..."), kind=VerificationKind.TEST)
        )
    if "pom.xml" in project.build_markers:
        commands.append(
            VerificationCommand("maven test", ("mvn", "test"), kind=VerificationKind.TEST)
        )
    return VerificationPlan(tuple(commands))


def default_verification_commands(project: ProjectWorkspace) -> tuple[VerificationCommand, ...]:
    """Backward-compatible access to all discovered verification commands."""

    return default_verification_plan(project).commands


def _python_quality_commands(pyproject: Path) -> tuple[VerificationCommand, ...]:
    try:
        raw = load_toml(pyproject.read_text(encoding="utf-8"))
    except (OSError, TOMLDecodeError):
        return ()
    tool = raw.get("tool", {}) if isinstance(raw, dict) else {}
    if not isinstance(tool, dict):
        return ()
    commands: list[VerificationCommand] = []
    if isinstance(tool.get("ruff"), dict):
        commands.append(
            VerificationCommand(
                "ruff", (sys.executable, "-m", "ruff", "check", "."), kind=VerificationKind.LINT
            )
        )
    if isinstance(tool.get("mypy"), dict):
        commands.append(
            VerificationCommand(
                "mypy", (sys.executable, "-m", "mypy", "."), kind=VerificationKind.TYPECHECK
            )
        )
    return tuple(commands)


def _node_verification_commands(package_json: Path) -> tuple[VerificationCommand, ...]:
    """Use only known npm script names; never parse a script body as a shell command."""
    try:
        raw = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    scripts = raw.get("scripts", {}) if isinstance(raw, dict) else {}
    if not isinstance(scripts, dict):
        return ()
    kinds = {
        "test": VerificationKind.TEST,
        "lint": VerificationKind.LINT,
        "typecheck": VerificationKind.TYPECHECK,
        "build": VerificationKind.BUILD,
    }
    return tuple(
        VerificationCommand(f"npm {name}", ("npm", "run", name), kind=kinds[name])
        for name in kinds
        if isinstance(scripts.get(name), str) and scripts[name].strip()
    )
