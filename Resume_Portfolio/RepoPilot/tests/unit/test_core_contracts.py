from __future__ import annotations

from repopilot.core.budgets import BudgetGuard, RunBudget
from repopilot.core.contracts import AgentState, Message, RunStatus, ToolCall


def test_agent_state_round_trip_preserves_pending_action() -> None:
    state = AgentState("run-1", "task-1", status=RunStatus.RUNNING)
    state.messages.append(Message("user", "hello"))
    state.messages.append(
        Message(
            "assistant",
            "Requested tools: read_file",
            tool_calls=(ToolCall("call-history", "read_file", {"path": "a.py"}),),
        )
    )
    state.pending_calls.append(ToolCall("call-1", "read_file", {"path": "a.py"}))
    state.plan = ["inspect", "verify"]

    restored = AgentState.from_dict(state.to_dict())

    assert restored == state


def test_budget_guard_explains_hard_limit() -> None:
    state = AgentState("run", "task", iteration=2)
    guard = BudgetGuard(RunBudget(max_iterations=2))

    assert guard.exceeded_reason(state) == "iteration budget exceeded"


def test_budget_requires_positive_limits() -> None:
    try:
        RunBudget(max_tool_calls=0)
    except ValueError as error:
        assert "max_tool_calls" in str(error)
    else:
        raise AssertionError("zero tool-call budget should fail")
