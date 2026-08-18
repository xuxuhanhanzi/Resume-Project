"""Parallelize only tools whose schemas declare no side effects."""

from __future__ import annotations

import asyncio

from repopilot.core.contracts import ErrorType, ToolCall, ToolResult
from repopilot.tools.base import ToolContext, ToolRegistry


async def execute_parallel_read_only(
    calls: tuple[ToolCall, ...], registry: ToolRegistry, context: ToolContext
) -> tuple[ToolResult, ...]:
    """Run independent reads concurrently while preserving result order."""
    coroutines = []
    for call in calls:
        tool = registry.get(call.name)
        if tool is None:

            async def unknown(current: ToolCall = call) -> ToolResult:
                return ToolResult(
                    current.call_id,
                    current.name,
                    False,
                    error="unknown tool",
                    error_type=ErrorType.VALIDATION,
                )

            coroutines.append(unknown())
        elif not tool.spec.read_only:

            async def rejected(current: ToolCall = call) -> ToolResult:
                return ToolResult(
                    current.call_id,
                    current.name,
                    False,
                    error="side-effecting tools cannot run in parallel",
                    error_type=ErrorType.PERMISSION,
                )

            coroutines.append(rejected())
        else:
            coroutines.append(tool.run(call, context))
    return tuple(await asyncio.gather(*coroutines))
