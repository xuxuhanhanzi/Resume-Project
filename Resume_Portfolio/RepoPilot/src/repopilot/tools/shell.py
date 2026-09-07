"""Governed tokenized shell access for interactive coding sessions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePath

from repopilot.core.contracts import (
    ErrorType,
    JSONValue,
    Permission,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repopilot.security.paths import PathSecurityError, resolve_workspace_path
from repopilot.tools.base import ToolContext

_MAX_ARGUMENTS = 64
_MAX_ARGUMENT_CHARS = 1_024
_MAX_TIMEOUT_SECONDS = 120.0


def shell_schema(properties: dict[str, JSONValue], required: list[str]) -> dict[str, JSONValue]:
    """Return a strict JSON schema shared by execution-oriented tools."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def validate_tokenized_command(raw: JSONValue) -> tuple[str, ...]:
    """Validate argv input before it reaches a command runner.

    Shell syntax is deliberately unsupported: callers supply an executable and its
    arguments as separate tokens, and runners always invoke them with ``shell=False``.
    """
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_ARGUMENTS:
        raise ValueError(f"command must contain 1-{_MAX_ARGUMENTS} string arguments")
    if not all(isinstance(item, str) for item in raw):
        raise ValueError("command must be a list of strings")
    command = tuple(raw)
    for argument in command:
        if not argument or len(argument) > _MAX_ARGUMENT_CHARS:
            raise ValueError(
                f"each command argument must contain 1-{_MAX_ARGUMENT_CHARS} characters"
            )
        if any(character in argument for character in ("\x00", "\r", "\n")):
            raise ValueError("command arguments cannot contain NUL or line breaks")
    return command


def resolve_command_cwd(context: ToolContext, raw: JSONValue) -> Path:
    """Resolve an optional relative working directory within the workspace."""
    if raw is None:
        return context.task.workspace
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("cwd must be a non-empty relative workspace path")
    portable = raw.replace("\\", "/")
    if PurePath(portable).is_absolute() or ".." in PurePath(portable).parts:
        raise PathSecurityError("cwd must stay within the workspace")
    path = resolve_workspace_path(context.task, raw)
    if not path.is_dir():
        raise ValueError("cwd must be an existing directory")
    return path


def execution_error(call: ToolCall, error: Exception) -> ToolResult:
    """Convert safe runner and validation failures to an agent observation."""
    error_type = ErrorType.SECURITY if isinstance(error, PathSecurityError) else ErrorType.EXECUTION
    if isinstance(error, ValueError):
        error_type = ErrorType.VALIDATION
    return ToolResult(
        call.call_id,
        call.name,
        False,
        error=str(error),
        error_type=error_type,
        recoverable=error_type is not ErrorType.SECURITY,
    )


class ShellTool:
    """Run one user-approved argv command in a workspace-owned directory."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "run_shell",
            (
                "Run one tokenized project command after explicit approval; "
                "shell syntax is unavailable."
            ),
            shell_schema(
                {
                    "command": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_ARGUMENTS,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_ARGUMENT_CHARS,
                        },
                    },
                    "cwd": {"type": "string"},
                    "timeout_seconds": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": _MAX_TIMEOUT_SECONDS,
                    },
                },
                ["command"],
            ),
            permission=Permission.EXECUTE,
            timeout_seconds=_MAX_TIMEOUT_SECONDS,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            command = validate_tokenized_command(call.arguments.get("command"))
            cwd = resolve_command_cwd(context, call.arguments.get("cwd"))
            timeout = float(call.arguments.get("timeout_seconds", self.spec.timeout_seconds))
            if not 1 <= timeout <= _MAX_TIMEOUT_SECONDS:
                raise ValueError(f"timeout_seconds must be between 1 and {_MAX_TIMEOUT_SECONDS:g}")
            result = await context.runner.run(
                command,
                cwd=cwd,
                timeout_seconds=timeout,
                cancellation=context.cancellation,
            )
        except (OSError, PermissionError, RuntimeError, ValueError, PathSecurityError) as error:
            return execution_error(call, error)
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            call.call_id,
            call.name,
            ok,
            {
                "command": list(result.command),
                "cwd": cwd.relative_to(context.task.workspace).as_posix() or ".",
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            },
            error=None if ok else "command failed",
            error_type=(
                None if ok else (ErrorType.TIMEOUT if result.timed_out else ErrorType.EXECUTION)
            ),
            recoverable=not ok and not result.timed_out,
            side_effect=True,
        )


def as_command_list(command: Sequence[str]) -> list[str]:
    """Keep command rendering consistent in process and Git observations."""
    return list(command)
