"""Tool protocol, execution context, and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from repopilot.core.contracts import JSONValue, ToolCall, ToolResult, ToolSpec
from repopilot.runtime.runner import CommandRunner
from repopilot.workspace.contracts import WorkspaceTask

if TYPE_CHECKING:
    from repopilot.agents.background import BackgroundAgentManager
    from repopilot.process.manager import ProcessManager
    from repopilot.runtime.cancellation import CancellationToken


@dataclass(slots=True)
class ToolContext:
    """Runtime-only dependencies that are never exposed to the model."""

    task: WorkspaceTask
    runner: CommandRunner
    baseline: dict[str, str] = field(default_factory=dict)
    recent_results: list[ToolResult] = field(default_factory=list)
    process_manager: ProcessManager | None = None
    cancellation: CancellationToken | None = None
    snapshot_exclude_paths: tuple[str, ...] = ()
    snapshot_max_files: int | None = None
    snapshot_max_total_bytes: int | None = None
    # Session-only mutable state used by TodoWrite.  It never exposes a path,
    # shell, provider credential, or arbitrary session store to model tools.
    todo_writer: TodoWriter | None = None
    subagent_manager: BackgroundAgentManager | None = None
    # Filled exclusively by AgentKernel after an interactive high-risk approval.
    # Tools must not infer approval from their own arguments or model text.
    approved_call_ids: set[str] = field(default_factory=set)


class TodoWriter(Protocol):
    def __call__(self, todos: list[dict[str, JSONValue]]) -> dict[str, JSONValue]: ...


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
