"""Structured access to session-owned, read-only background analysts."""

from __future__ import annotations

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext


class StartSubagentTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "start_subagent",
            "Start one bounded background analyst over supplied evidence; it cannot use tools.",
            {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "minLength": 1, "maxLength": 120},
                    "question": {"type": "string", "minLength": 1, "maxLength": 5000},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 60000},
                },
                "required": ["description", "question", "evidence"],
                "additionalProperties": False,
            },
            permission=Permission.HIGH_RISK,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.subagent_manager is None:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="background subagents are unavailable outside an interactive session",
                error_type=ErrorType.VALIDATION,
            )
        if call.call_id not in context.approved_call_ids:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="background subagent disclosure was not explicitly approved",
                error_type=ErrorType.PERMISSION,
            )
        values = {
            name: call.arguments.get(name) for name in ("description", "question", "evidence")
        }
        if not all(isinstance(value, str) for value in values.values()):
            return ToolResult(
                call.call_id, call.name, False, error="subagent fields must be strings"
            )
        try:
            task_id = context.subagent_manager.start(
                description=str(values["description"]),
                question=str(values["question"]),
                evidence=str(values["evidence"]),
            )
        except (RuntimeError, ValueError) as error:
            return ToolResult(call.call_id, call.name, False, error=str(error), recoverable=True)
        return ToolResult(call.call_id, call.name, True, {"task_id": task_id}, side_effect=True)


class SubagentStatusTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "subagent_status",
            "Read the result or current status of one RepoPilot-owned background analyst.",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string", "minLength": 1}},
                "required": ["task_id"],
                "additionalProperties": False,
            },
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.subagent_manager is None:
            return ToolResult(
                call.call_id, call.name, False, error="background subagents unavailable"
            )
        task_id = call.arguments.get("task_id")
        if not isinstance(task_id, str):
            return ToolResult(call.call_id, call.name, False, error="task_id must be a string")
        try:
            snapshot = await context.subagent_manager.snapshot(task_id)
        except ValueError as error:
            return ToolResult(
                call.call_id, call.name, False, error=str(error), error_type=ErrorType.NOT_FOUND
            )
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {
                "task_id": snapshot.task_id,
                "description": snapshot.description,
                "running": snapshot.running,
                "result": snapshot.result,
                "error": snapshot.error,
            },
        )


def default_background_subagent_tools() -> list[StartSubagentTool | SubagentStatusTool]:
    return [StartSubagentTool(), SubagentStatusTool()]
