"""Bounded-execution controls for agent runs."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from repopilot.core.contracts import AgentState


@dataclass(frozen=True, slots=True)
class RunBudget:
    """Hard limits that prevent an unbounded agent loop."""

    max_iterations: int = 12
    max_tool_calls: int = 32
    max_total_tokens: int = 32_000
    max_wall_seconds: float = 600.0

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_tool_calls", "max_total_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")


@dataclass(slots=True)
class BudgetGuard:
    """Tracks wall time and explains which hard limit was reached."""

    budget: RunBudget
    started_at: float = 0.0

    def __post_init__(self) -> None:
        if self.started_at == 0.0:
            self.started_at = monotonic()

    def exceeded_reason(self, state: AgentState) -> str | None:
        if state.iteration >= self.budget.max_iterations:
            return "iteration budget exceeded"
        if state.tool_calls >= self.budget.max_tool_calls:
            return "tool-call budget exceeded"
        if state.input_tokens + state.output_tokens >= self.budget.max_total_tokens:
            return "token budget exceeded"
        if monotonic() - self.started_at >= self.budget.max_wall_seconds:
            return "wall-time budget exceeded"
        return None
