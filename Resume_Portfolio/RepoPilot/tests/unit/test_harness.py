"""Unit tests for the multi-agent harness (P11)."""

from __future__ import annotations

from repopilot.orchestration.harness import (
    AgentMessage,
    AgentRole,
    MultiAgentHarness,
    StateGraph,
)


def test_state_graph_default_topology() -> None:
    graph = StateGraph()
    assert set(graph.nodes) == {"planner", "executor", "verifier", "reviewer"}
    assert graph.next_nodes("planner") == ["executor"]
    assert graph.next_nodes("executor") == ["verifier"]
    assert "reviewer" in graph.next_nodes("verifier")
    assert "executor" in graph.next_nodes("verifier")
    assert "finish" in graph.next_nodes("reviewer")


def test_agent_role_is_string_enum() -> None:
    assert AgentRole.PLANNER.value == "planner"
    assert AgentRole.EXECUTOR.value == "executor"
    assert AgentRole.VERIFIER.value == "verifier"
    assert AgentRole.REVIEWER.value == "reviewer"


def test_agent_message_envelope() -> None:
    msg = AgentMessage(
        sender="planner",
        recipient="executor",
        task_id="task-001",
        correlation_id="corr-001",
        content="Execute step 1",
        permission="read",
        budget_remaining=100.0,
    )
    assert msg.sender == "planner"
    assert msg.recipient == "executor"
    assert msg.task_id == "task-001"
    assert msg.correlation_id == "corr-001"
    assert msg.permission == "read"
    assert msg.budget_remaining == 100.0
    assert msg.timestamp > 0


def test_harness_send_and_trace() -> None:
    harness = MultiAgentHarness(correlation_id="test-corr")
    harness.send("planner", "executor", "task-001", "do step 1")
    harness.send("executor", "verifier", "task-001", "result: ok")
    assert len(harness.messages) == 2
    assert harness.messages[0].sender == "planner"
    assert harness.messages[1].recipient == "verifier"
    summary = harness.trace_summary()
    assert summary["correlation_id"] == "test-corr"
    assert summary["total_messages"] == 2
    assert len(summary["node_transitions"]) == 2
    assert "planner→executor" in summary["node_transitions"]
