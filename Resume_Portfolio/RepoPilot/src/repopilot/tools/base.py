"""Tool protocol, execution context, and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from repopilot.core.contracts import JSONValue, ToolCall, ToolResult, ToolSpec
from repopilot.runtime.runner import CommandRunner
from repopilot.task import PublicTaskSpec


@dataclass(slots=True)
class ToolContext:
    """Runtime-only dependencies that are never exposed to the model."""

    task: PublicTaskSpec
    runner: CommandRunner
    baseline: dict[str, str] = field(default_factory=dict)
    recent_results: list[ToolResult] = field(default_factory=list)


class Tool(Protocol):
    """A narrow structured capability."""

    @property
    def spec(self) -> ToolSpec:
        """Return the model-visible schema and runtime safety metadata."""
        ...

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute a validated call."""
        ...


class ToolRegistry:
    """Reject duplicate or unknown tools and provide stable schema ordering."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools[name].spec for name in sorted(self._tools))

    def describe(self) -> list[dict[str, JSONValue]]:
        return [spec.provider_schema() for spec in self.specs()]
