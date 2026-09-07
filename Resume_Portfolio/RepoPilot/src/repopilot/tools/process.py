"""Permission-governed background-process tools for interactive sessions."""

from __future__ import annotations

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext
from repopilot.tools.shell import (
    as_command_list,
    execution_error,
    resolve_command_cwd,
    shell_schema,
    validate_tokenized_command,
)


def _manager_error(call: ToolCall) -> ToolResult:
    return ToolResult(
        call.call_id,
        call.name,
        False,
        error="background process support is unavailable in this runtime",
        error_type=ErrorType.VALIDATION,
    )


class StartBackgroundTool:
    """Start one tokenized process under the same approval boundary as ShellTool."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "start_background",
            "Start one tokenized project process in the background after approval.",
            shell_schema(
                {
                    "command": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 64,
                        "items": {"type": "string", "minLength": 1, "maxLength": 1024},
                    },
                    "cwd": {"type": "string"},
                },
                ["command"],
            ),
            permission=Permission.EXECUTE,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.process_manager is None:
            return _manager_error(call)
        try:
            command = validate_tokenized_command(call.arguments.get("command"))
            cwd = resolve_command_cwd(context, call.arguments.get("cwd"))
            process = await context.process_manager.start(command, cwd=cwd)
        except (OSError, RuntimeError, ValueError) as error:
            return execution_error(call, error)
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {
                "process_id": process.process_id,
                "command": as_command_list(process.command),
                "cwd": process.cwd.relative_to(context.task.workspace).as_posix() or ".",
                "started_at": process.started_at.isoformat(),
            },
            side_effect=True,
        )


class BackgroundStatusTool:
    """Read bounded output and status for a process started in this runtime."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "background_status",
            "Read status and bounded output for one known background process.",
            shell_schema({"process_id": {"type": "string", "minLength": 1}}, ["process_id"]),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.process_manager is None:
            return _manager_error(call)
        try:
            process_id = str(call.arguments.get("process_id", ""))
            snapshot = await context.process_manager.snapshot(process_id)
        except ValueError as error:
            return execution_error(call, error)
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {
                "process_id": snapshot.process.process_id,
                "running": snapshot.running,
                "return_code": snapshot.return_code,
                "stdout": snapshot.stdout,
                "stderr": snapshot.stderr,
            },
        )


class TerminateBackgroundTool:
    """Stop a known background child only after dedicated high-risk approval."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "terminate_background",
            "Terminate one RepoPilot-managed background process after approval.",
            shell_schema({"process_id": {"type": "string", "minLength": 1}}, ["process_id"]),
            permission=Permission.HIGH_RISK,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.process_manager is None:
            return _manager_error(call)
        try:
            process_id = str(call.arguments.get("process_id", ""))
            snapshot = await context.process_manager.terminate(process_id)
        except (RuntimeError, ValueError) as error:
            return execution_error(call, error)
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {"process_id": snapshot.process.process_id, "return_code": snapshot.return_code},
            side_effect=True,
        )


def default_process_tools() -> list[
    StartBackgroundTool | BackgroundStatusTool | TerminateBackgroundTool
]:
    """Return background tools; only SessionRuntime supplies their process manager."""
    return [StartBackgroundTool(), BackgroundStatusTool(), TerminateBackgroundTool()]
