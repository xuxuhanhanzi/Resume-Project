from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.orchestration.routing import RouteTarget, RoutingMode, TaskRouter
from repopilot.providers.scripted import ScriptedProvider
from repopilot.task import PublicTaskSpec


def _task(statement: str, *, max_changed_files: int = 3) -> PublicTaskSpec:
    return PublicTaskSpec(
        "route-task",
        Path.cwd(),
        statement,
        max_changed_files=max_changed_files,
        trusted_fixture=True,
    )


def test_fixed_and_rule_routes_are_deterministic() -> None:
    fixed = asyncio.run(
        TaskRouter(RoutingMode.FIXED, fixed_target=RouteTarget.ESCALATED).route(_task("Fix typo"))
    )
    simple = asyncio.run(TaskRouter(RoutingMode.RULE).route(_task("Fix a small bug")))
    complex_route = asyncio.run(
        TaskRouter(RoutingMode.RULE).route(_task("Investigate a concurrency deadlock"))
    )

    assert fixed.target is RouteTarget.ESCALATED
    assert simple.target is RouteTarget.STANDARD
    assert complex_route.target is RouteTarget.ESCALATED


def test_model_route_is_read_only_and_validated() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                content=('{"target":"escalated","reason":"cross-module task","confidence":0.91}')
            )
        ]
    )

    decision = asyncio.run(
        TaskRouter(RoutingMode.MODEL).route(_task("Improve the subsystem"), provider=provider)
    )

    assert decision.target is RouteTarget.ESCALATED
    assert provider.requests[0].tools == ()


def test_invalid_model_route_cascades_to_rule() -> None:
    provider = ScriptedProvider([ModelResponse(content="not-json")])

    decision = asyncio.run(
        TaskRouter(RoutingMode.MODEL).route(_task("Fix a small bug"), provider=provider)
    )

    assert decision.target is RouteTarget.STANDARD
    assert "model_failed:JSONDecodeError" in decision.cascade
    assert decision.cascade[-1] == "fallback:standard"


def test_model_tool_attempt_cascades_to_rule() -> None:
    provider = ScriptedProvider(
        [ModelResponse(tool_calls=(ToolCall("route", "search_text", {"pattern": "x"}),))]
    )

    decision = asyncio.run(
        TaskRouter(RoutingMode.MODEL).route(_task("Investigate"), provider=provider)
    )

    assert decision.target is RouteTarget.STANDARD
    assert "model_failed:ValueError" in decision.cascade


def test_hybrid_uses_confident_rule_and_model_for_uncertain_task() -> None:
    unused = ScriptedProvider([])
    confident = asyncio.run(
        TaskRouter(RoutingMode.HYBRID).route(
            _task("Investigate a security race condition"), provider=unused
        )
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                content=(
                    '{"target":"escalated","reason":"ambiguous broad change","confidence":0.8}'
                )
            )
        ]
    )
    uncertain = asyncio.run(
        TaskRouter(RoutingMode.HYBRID).route(_task("Improve this module"), provider=provider)
    )

    assert confident.target is RouteTarget.ESCALATED
    assert not unused.requests
    assert uncertain.target is RouteTarget.ESCALATED
    assert len(provider.requests) == 1
