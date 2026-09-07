"""High-risk, workspace-contained Git worktree creation."""

from __future__ import annotations

import re
from pathlib import Path

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import Tool, ToolContext
from repopilot.tools.shell import as_command_list, execution_error, shell_schema

_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,79}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class CreateWorktreeTool:
    """Create a new Git worktree only in RepoPilot's project-local worktree area."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "create_worktree",
            "Create a new branch worktree under .repopilot/worktrees after high-risk approval.",
            shell_schema(
                {
                    "name": {"type": "string", "minLength": 1, "maxLength": 64},
                    "branch": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                ["name", "branch"],
            ),
            permission=Permission.HIGH_RISK,
            timeout_seconds=30.0,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        name = call.arguments.get("name")
        branch = call.arguments.get("branch")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="worktree name must use letters, digits, underscores, or hyphens",
                error_type=ErrorType.VALIDATION,
            )
        if not isinstance(branch, str) or not _BRANCH.fullmatch(branch) or branch.startswith("-"):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="branch must be a safe 1-80 character Git branch name",
                error_type=ErrorType.VALIDATION,
            )
        target = context.task.workspace / ".repopilot" / "worktrees" / name
        if target.exists():
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="worktree target already exists",
                error_type=ErrorType.CONFLICT,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        command = ("git", "worktree", "add", "-b", branch, str(target))
        try:
            result = await context.runner.run(
                command,
                cwd=context.task.workspace,
                timeout_seconds=self.spec.timeout_seconds,
                cancellation=context.cancellation,
            )
        except (OSError, PermissionError, RuntimeError, ValueError) as error:
            return execution_error(call, error)
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            call.call_id,
            call.name,
            ok,
            {
                "command": as_command_list(result.command),
                "path": target.relative_to(context.task.workspace).as_posix(),
                "branch": branch,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            error=None if ok else "git worktree creation failed",
            error_type=None if ok else ErrorType.EXECUTION,
            recoverable=not ok,
            side_effect=True,
        )


class ListWorktreesTool:
    """List only worktrees rooted inside the currently opened workspace."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "list_worktrees",
            "List the primary checkout and RepoPilot-managed worktrees without changing Git state.",
            shell_schema({}, []),
            permission=Permission.READ,
            timeout_seconds=15.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            result = await context.runner.run(
                ("git", "worktree", "list", "--porcelain"),
                cwd=context.task.workspace,
                timeout_seconds=self.spec.timeout_seconds,
                cancellation=context.cancellation,
            )
        except (OSError, PermissionError, RuntimeError, ValueError) as error:
            return execution_error(call, error)
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            call.call_id,
            call.name,
            ok,
            {
                "command": as_command_list(result.command),
                "worktrees": _workspace_worktrees(result.stdout, context.task.workspace),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            error=None if ok else "git worktree listing failed",
            error_type=None if ok else ErrorType.EXECUTION,
            recoverable=not ok,
        )


class InspectWorktreeTool:
    """Inspect exactly one RepoPilot-managed worktree before a user keeps it or removes it."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "inspect_worktree",
            "Inspect dirty state for one RepoPilot-managed worktree without changing Git state.",
            shell_schema({"name": {"type": "string", "minLength": 1, "maxLength": 64}}, ["name"]),
            permission=Permission.READ,
            timeout_seconds=15.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        name = call.arguments.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="worktree name must use letters, digits, underscores, or hyphens",
                error_type=ErrorType.VALIDATION,
            )
        target = context.task.workspace / ".repopilot" / "worktrees" / name
        if not target.is_dir():
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="RepoPilot-managed worktree does not exist",
                error_type=ErrorType.NOT_FOUND,
            )
        try:
            result = await context.runner.run(
                ("git", "status", "--porcelain=v1", "--branch"),
                cwd=target,
                timeout_seconds=self.spec.timeout_seconds,
                cancellation=context.cancellation,
            )
        except (OSError, PermissionError, RuntimeError, ValueError) as error:
            return execution_error(call, error)
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            call.call_id,
            call.name,
            ok,
            {
                "name": name,
                "path": target.relative_to(context.task.workspace).as_posix(),
                "dirty": bool(result.stdout.strip().splitlines()[1:]),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            },
            error=None if ok else "git worktree inspection failed",
            error_type=None if ok else ErrorType.EXECUTION,
            recoverable=not ok,
        )


def _workspace_worktrees(stdout: str, workspace: Path) -> list[dict[str, str]]:
    """Parse Git porcelain while withholding unrelated checkout locations."""
    root = workspace.resolve()
    records: list[dict[str, str]] = []
    for block in stdout.split("\n\n"):
        path: Path | None = None
        branch = "(detached)"
        for line in block.splitlines():
            if line.startswith("worktree "):
                try:
                    path = Path(line.removeprefix("worktree ")).resolve()
                except OSError:
                    path = None
            elif line.startswith("branch refs/heads/"):
                branch = line.removeprefix("branch refs/heads/")
        if path is None or (path != root and root not in path.parents):
            continue
        records.append(
            {
                "path": "." if path == root else path.relative_to(root).as_posix(),
                "branch": branch,
            }
        )
    return records


def default_worktree_tools() -> list[Tool]:
    """Return safe creation/inventory; destructive worktree removal remains absent."""
    return [ListWorktreesTool(), InspectWorktreeTool(), CreateWorktreeTool()]
