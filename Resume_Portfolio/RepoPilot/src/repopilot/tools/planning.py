"""A narrow model-facing TodoWrite capability backed by the active session only."""

from __future__ import annotations

from repopilot.core.contracts import ErrorType, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext


class TodoWriteTool:
    """Persist a bounded todo list; this never writes project files or permissions."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "write_todos",
            "Replace the active session todo list with up to 20 explicit tasks.",
            {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "minLength": 1, "maxLength": 240},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                                "active_form": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                            },
                            "required": ["content", "status", "active_form"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["todos"],
                "additionalProperties": False,
            },
            read_only=False,
            idempotent=True,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        raw_todos = call.arguments.get("todos")
        if context.todo_writer is None:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="todo updates are unavailable outside an interactive session",
                error_type=ErrorType.VALIDATION,
            )
        if not isinstance(raw_todos, list) or not all(isinstance(item, dict) for item in raw_todos):
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="todos must be a list of objects",
                error_type=ErrorType.VALIDATION,
            )
        try:
            data = context.todo_writer(raw_todos)
        except ValueError as error:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=str(error),
                error_type=ErrorType.VALIDATION,
            )
        return ToolResult(call.call_id, call.name, True, data, side_effect=True)
