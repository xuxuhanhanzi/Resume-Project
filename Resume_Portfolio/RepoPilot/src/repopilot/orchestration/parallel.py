"""Parallelize bounded read-only tools while preserving result order and isolation."""

from __future__ import annotations

import asyncio

from repopilot.core.contracts import ErrorType, ToolCall, ToolResult
from repopilot.tools.base import Tool, ToolContext, ToolRegistry


async def execute_parallel_read_only(
    calls: tuple[ToolCall, ...],
    registry: ToolRegistry,
    context: ToolContext,
    *,
    max_concurrency: int = 4,
) -> tuple[ToolResult, ...]:
    """Run independent reads concurrently while preserving result order.

    Policy and approval decisions must happen in the caller before entering this
    helper.  A semaphore prevents a large model tool batch from overwhelming the
    filesystem or a remote read-only service.
    """
    if not 1 <= max_concurrency <= 16:
        raise ValueError("max_concurrency must be between 1 and 16")
    semaphore = asyncio.Semaphore(max_concurrency)
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
            coroutines.append(_run_read_only(call, tool, context, semaphore))
    return tuple(await asyncio.gather(*coroutines))


async def _run_read_only(
    call: ToolCall,
    tool: Tool,
    context: ToolContext,
    semaphore: asyncio.Semaphore,
) -> ToolResult:
    """Contain an unexpected read failure so siblings still return evidence."""
    async with semaphore:
        try:
            return await tool.run(call, context)
        except (OSError, PermissionError, RuntimeError, ValueError) as error:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=f"read-only tool failed: {error}",
                error_type=ErrorType.EXECUTION,
                recoverable=True,
            )
