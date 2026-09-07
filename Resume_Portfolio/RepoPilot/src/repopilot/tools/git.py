"""Structured, read-only Git observations for coding workflows."""

from __future__ import annotations

import re

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.security.paths import PathSecurityError, resolve_workspace_path
from repopilot.tools.base import Tool, ToolContext
from repopilot.tools.shell import as_command_list, execution_error, shell_schema

_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./@~^:+-]{0,199}\Z")


class _GitTool:
    """Base for a bounded Git read command rooted in the current project."""

    command: tuple[str, ...]

    @property
    def spec(self) -> ToolSpec:
        raise NotImplementedError

    async def _run(
        self, call: ToolCall, context: ToolContext, *, command: tuple[str, ...] | None = None
    ) -> ToolResult:
        try:
            result = await context.runner.run(
                command or self.command,
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
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            },
            error=None if ok else "git command failed",
            error_type=(
                None if ok else (ErrorType.TIMEOUT if result.timed_out else ErrorType.EXECUTION)
            ),
            recoverable=not ok and not result.timed_out,
        )


class GitStatusTool(_GitTool):
    command = ("git", "status", "--short", "--branch")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_status",
            "Show the current Git branch and concise working-tree status.",
            shell_schema({}, []),
            permission=Permission.READ,
            timeout_seconds=15.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return await self._run(call, context)


class GitLogTool(_GitTool):
    command = ("git", "log", "--oneline", "--decorate", "-n", "20")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_log",
            "Show up to twenty recent Git commits without changing the repository.",
            shell_schema({}, []),
            permission=Permission.READ,
            timeout_seconds=15.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return await self._run(call, context)


class GitShowTool(_GitTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_show",
            "Show a bounded Git object or commit using a safe ref expression.",
            shell_schema({"ref": {"type": "string", "minLength": 1, "maxLength": 200}}, ["ref"]),
            permission=Permission.READ,
            timeout_seconds=15.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        raw_ref = call.arguments.get("ref")
        if not isinstance(raw_ref, str) or not _REF_PATTERN.fullmatch(raw_ref):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="ref must be a safe 1-200 character Git revision",
                error_type=ErrorType.VALIDATION,
            )
        return await self._run(
            call,
            context,
            command=("git", "show", "--stat", "--format=fuller", raw_ref),
        )


class GitWorkingDiffTool(_GitTool):
    """Read the working-tree patch without allowing Git external-diff hooks."""

    command = ("git", "diff", "--no-ext-diff", "--unified=3", "--")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_working_diff",
            "Show the current uncommitted Git text diff without invoking external diff tools.",
            shell_schema({}, []),
            permission=Permission.READ,
            timeout_seconds=20.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return await self._run(call, context)


class GitStageTool(_GitTool):
    """Stage an explicit bounded list of workspace files after edit approval."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_stage",
            "Stage an explicit list of existing workspace paths after approval.",
            shell_schema(
                {
                    "paths": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": {"type": "string", "minLength": 1, "maxLength": 260},
                    }
                },
                ["paths"],
            ),
            permission=Permission.WRITE,
            timeout_seconds=15.0,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        raw_paths = call.arguments.get("paths")
        if not isinstance(raw_paths, list) or not 1 <= len(raw_paths) <= 50:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="paths must contain 1-50 workspace-relative strings",
                error_type=ErrorType.VALIDATION,
            )
        relative_paths: list[str] = []
        try:
            for raw_path in raw_paths:
                if not isinstance(raw_path, str):
                    raise ValueError("paths must contain strings")
                path = resolve_workspace_path(context.task, raw_path)
                if path == context.task.workspace or not path.exists():
                    raise ValueError("git_stage paths must name existing files or directories")
                relative_paths.append(path.relative_to(context.task.workspace).as_posix())
        except (PathSecurityError, ValueError) as error:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=str(error),
                error_type=(
                    ErrorType.SECURITY
                    if isinstance(error, PathSecurityError)
                    else ErrorType.VALIDATION
                ),
            )
        return await self._run(
            call,
            context,
            command=("git", "add", "--", *tuple(relative_paths)),
        )


class GitCommitTool(_GitTool):
    """Commit the current index only after dedicated high-risk confirmation."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_commit",
            "Create a Git commit from the current index after explicit high-risk approval.",
            shell_schema(
                {"message": {"type": "string", "minLength": 1, "maxLength": 300}}, ["message"]
            ),
            permission=Permission.HIGH_RISK,
            timeout_seconds=30.0,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        message = call.arguments.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 300:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="commit message must contain 1-300 characters",
                error_type=ErrorType.VALIDATION,
            )
        if "\x00" in message or "\r" in message or "\n" in message:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="commit message must be a single line",
                error_type=ErrorType.VALIDATION,
            )
        return await self._run(call, context, command=("git", "commit", "-m", message.strip()))


def default_git_tools() -> list[Tool]:
    """Return read observations plus explicitly governed stage/commit actions."""
    return [
        GitStatusTool(),
        GitLogTool(),
        GitShowTool(),
        GitWorkingDiffTool(),
        GitStageTool(),
        GitCommitTool(),
    ]
