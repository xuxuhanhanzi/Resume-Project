from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import pytest

from repopilot.core.contracts import ModelRequest, ModelResponse, RunStatus, ToolCall
from repopilot.core.loop import AgentRuntime
from repopilot.orchestration.harness import (
    MultiAgentCheckpointStore,
    MultiAgentHarness,
    RouteBinding,
)
from repopilot.orchestration.routing import RouteTarget, RoutingMode, TaskRouter
from repopilot.providers.scripted import ScriptedProvider
from repopilot.retrieval.tool import RetrieveCodeTool
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import default_coding_tools


def _copy_demo(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "examples" / "bugfix_demo"
    workspace = tmp_path / "workspace"
    shutil.copytree(source, workspace)
    return workspace


def _task(workspace: Path) -> PublicTaskSpec:
    return PublicTaskSpec(
        "multi-subtract-demo",
        workspace,
        "Fix the Python subtract bug",
        allowed_paths=("calculator.py",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )


def _executor(tmp_path: Path, provider: ScriptedProvider) -> AgentRuntime:
    return AgentRuntime(
        provider=provider,
        tools=[*default_coding_tools(), RetrieveCodeTool()],
        runner=LocalTrustedRunner(trusted=True),
        artifacts_root=tmp_path / "executor_artifacts",
    )


def _successful_executor_provider() -> ScriptedProvider:
    return ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "search_text", {"pattern": "subtract"}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "c2",
                        "apply_patch",
                        {
                            "path": "calculator.py",
                            "old_text": "return left + right",
                            "new_text": "return left - right",
                        },
                    ),
                )
            ),
            ModelResponse(tool_calls=(ToolCall("c3", "run_tests", {}),)),
            ModelResponse(content='{"type":"finish","answer":"fixed and verified"}'),
        ]
    )


@pytest.mark.integration
def test_multi_agent_harness_executes_verified_task_end_to_end(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    planner = ScriptedProvider([ModelResponse(content="Inspect, patch, test, and review.")])
    executor_provider = _successful_executor_provider()
    reviewer = ScriptedProvider(
        [ModelResponse(content='{"verdict":"pass","feedback":"verified patch is focused"}')]
    )
    harness = MultiAgentHarness(
        correlation_id="multi-e2e",
        artifacts_root=tmp_path / "multi_artifacts",
    )

    state = asyncio.run(
        harness.run(
            _task(workspace),
            executor=_executor(tmp_path, executor_provider),
            planner_provider=planner,
            reviewer_provider=reviewer,
            run_id="complete",
        )
    )

    assert state.status is RunStatus.COMPLETED
    assert state.current_node == "finish"
    assert state.cycle == 1
    assert state.route_target == "standard"
    assert state.routing_mode == "fixed"
    assert "return left - right" in (workspace / "calculator.py").read_text(encoding="utf-8")
    assert [f"{item.sender}→{item.recipient}" for item in state.messages] == [
        "planner→executor",
        "executor→verifier",
        "verifier→reviewer",
        "reviewer→finish",
    ]
    assert planner.requests[0].tools == ()
    assert reviewer.requests[0].tools == ()
    assert "return left - right" in reviewer.requests[0].messages[1].content
    assert "Read-only planner plan" in executor_provider.requests[0].messages[1].content
    stored = MultiAgentCheckpointStore(
        tmp_path / "multi_artifacts" / "complete" / "checkpoint.json"
    ).load()
    assert stored is not None and stored.status is RunStatus.COMPLETED


class _BlockingReviewer:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _SlowPlanner:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        await asyncio.sleep(1)
        return ModelResponse(content="late plan")


@pytest.mark.integration
def test_multi_agent_resume_skips_completed_executor_after_process_cancel(tmp_path: Path) -> None:
    async def exercise() -> None:
        workspace = _copy_demo(tmp_path)
        task = _task(workspace)
        planner = ScriptedProvider([ModelResponse(content="Plan once.")])
        executor_provider = _successful_executor_provider()
        executor = _executor(tmp_path, executor_provider)
        blocking_reviewer = _BlockingReviewer()
        artifacts = tmp_path / "multi_artifacts"
        first = MultiAgentHarness(artifacts_root=artifacts)
        interrupted = asyncio.create_task(
            first.run(
                task,
                executor=executor,
                planner_provider=planner,
                reviewer_provider=blocking_reviewer,
                run_id="resume",
            )
        )
        await asyncio.wait_for(blocking_reviewer.started.wait(), timeout=5)
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted

        checkpoint = MultiAgentCheckpointStore(artifacts / "resume" / "checkpoint.json").load()
        assert checkpoint is not None
        assert checkpoint.status is RunStatus.RUNNING
        assert checkpoint.current_node == "reviewer"
        executor_request_count = len(executor_provider.requests)

        resumed = await MultiAgentHarness(artifacts_root=artifacts).run(
            task,
            executor=executor,
            planner_provider=ScriptedProvider([]),
            reviewer_provider=ScriptedProvider(
                [ModelResponse(content='{"verdict":"pass","feedback":"resume review passed"}')]
            ),
            run_id="resume",
        )
        assert resumed.status is RunStatus.COMPLETED
        assert len(executor_provider.requests) == executor_request_count

    asyncio.run(exercise())


@pytest.mark.integration
def test_multi_agent_honors_pre_execution_cancellation(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    cancel_event = asyncio.Event()
    cancel_event.set()
    planner = ScriptedProvider([])
    executor_provider = ScriptedProvider([])
    state = asyncio.run(
        MultiAgentHarness(artifacts_root=tmp_path / "multi_artifacts").run(
            _task(workspace),
            executor=_executor(tmp_path, executor_provider),
            planner_provider=planner,
            reviewer_provider=ScriptedProvider([]),
            run_id="cancelled",
            cancel_event=cancel_event,
        )
    )
    assert state.status is RunStatus.CANCELLED
    assert not planner.requests
    assert not executor_provider.requests


@pytest.mark.integration
def test_multi_agent_converts_node_timeout_to_durable_failure(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    state = asyncio.run(
        MultiAgentHarness(artifacts_root=tmp_path / "multi_artifacts").run(
            _task(workspace),
            executor=_executor(tmp_path, ScriptedProvider([])),
            planner_provider=_SlowPlanner(),
            reviewer_provider=ScriptedProvider([]),
            run_id="timeout",
            timeout_seconds=0.01,
        )
    )
    assert state.status is RunStatus.FAILED
    assert state.failure_reason == "node timeout after 0.0 seconds"
    stored = MultiAgentCheckpointStore(
        tmp_path / "multi_artifacts" / "timeout" / "checkpoint.json"
    ).load()
    assert stored is not None and stored.status is RunStatus.FAILED


@pytest.mark.integration
def test_multi_agent_route_selects_escalated_runtime_and_persists_reason(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    task = PublicTaskSpec(
        "multi-routed-demo",
        workspace,
        "Fix the security-sensitive subtract bug",
        allowed_paths=("calculator.py",),
        test_command=(sys.executable, "verify.py"),
        max_changed_files=2,
        trusted_fixture=True,
    )
    unused_executor_provider = ScriptedProvider([])
    escalated_executor_provider = _successful_executor_provider()
    escalated = RouteBinding(
        _executor(tmp_path, escalated_executor_provider),
        ScriptedProvider([ModelResponse(content="Escalated plan.")]),
        ScriptedProvider([ModelResponse(content='{"verdict":"pass","feedback":"routed pass"}')]),
    )

    state = asyncio.run(
        MultiAgentHarness(artifacts_root=tmp_path / "multi_artifacts").run(
            task,
            executor=_executor(tmp_path, unused_executor_provider),
            planner_provider=ScriptedProvider([]),
            reviewer_provider=ScriptedProvider([]),
            run_id="routed",
            router=TaskRouter(RoutingMode.RULE),
            route_bindings={RouteTarget.ESCALATED: escalated},
        )
    )

    assert state.status is RunStatus.COMPLETED
    assert state.route_target == "escalated"
    assert "complexity terms: security" in state.routing_reason
    assert escalated_executor_provider.requests
    assert not unused_executor_provider.requests
    stored = MultiAgentCheckpointStore(
        tmp_path / "multi_artifacts" / "routed" / "checkpoint.json"
    ).load()
    assert stored is not None and stored.route_target == "escalated"
