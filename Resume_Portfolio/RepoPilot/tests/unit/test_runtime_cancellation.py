from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

from repopilot.core.contracts import ModelResponse, ToolCall, ToolResult, ToolSpec
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.cancellation import CancellationToken, OperationCancelledError
from repopilot.runtime.runner import LocalTrustedRunner, _bounded_output
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.base import ToolContext
from repopilot.tools.coding import default_coding_tools


def test_local_runner_stops_a_cancelled_command(tmp_path: Path) -> None:
    async def exercise() -> None:
        token = CancellationToken()
        runner = LocalTrustedRunner(trusted=True)
        operation = asyncio.create_task(
            runner.run(
                (sys.executable, "-c", "import time; time.sleep(30)"),
                cwd=tmp_path,
                timeout_seconds=60.0,
                cancellation=token,
            )
        )
        await asyncio.sleep(0.1)
        token.cancel("cancelled by test")
        with pytest.raises(OperationCancelledError, match="cancelled by test"):
            await asyncio.wait_for(operation, timeout=5.0)

    asyncio.run(exercise())


def test_cancellation_token_wakes_a_waiter_when_called_from_another_thread() -> None:
    async def exercise() -> None:
        token = CancellationToken()
        waiting = asyncio.create_task(token.wait())
        await asyncio.sleep(0)
        worker = threading.Thread(target=token.cancel, args=("service stop",))
        worker.start()
        worker.join()
        await asyncio.wait_for(waiting, timeout=1.0)
        assert token.cancelled
        assert token.reason == "service stop"

    asyncio.run(exercise())


def test_bounded_output_does_not_split_utf8_characters() -> None:
    rendered = _bounded_output("你好世界" * 10, 38)

    assert "\ufffd" not in rendered
    assert rendered.endswith("...[output truncated by RepoPilot]")
    assert len(rendered.encode("utf-8")) <= 38


def test_session_cancellation_clears_pending_calls_and_is_durable(tmp_path: Path) -> None:
    class WaitForCancellationTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec("wait_for_cancellation", "Wait for cancellation.", {})

        async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
            assert context.cancellation is not None
            await context.cancellation.wait()
            context.cancellation.raise_if_cancelled()
            raise AssertionError("unreachable")

    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "sessions")
        runtime = SessionRuntime(
            provider=ScriptedProvider(
                [
                    ModelResponse(tool_calls=(ToolCall("wait-1", "wait_for_cancellation", {}),)),
                    ModelResponse(content="Recovered after cancellation."),
                ]
            ),
            tools=[WaitForCancellationTool()],
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        metadata = runtime.start(project)
        token = CancellationToken()
        started = asyncio.Event()
        events: list[RuntimeEvent] = []

        def emit(event: RuntimeEvent) -> None:
            events.append(event)
            if event.kind is RuntimeEventKind.TOOL_CALL_STARTED:
                started.set()

        turn = asyncio.create_task(
            runtime.run_turn(metadata, "Wait until I stop you.", emit=emit, cancellation=token)
        )
        await asyncio.wait_for(started.wait(), timeout=2.0)
        assert runtime.cancel(metadata, reason="user stopped the turn")
        result = await asyncio.wait_for(turn, timeout=2.0)
        assert result.state.status.value == "cancelled"
        assert result.answer == "Cancelled: user stopped the turn"
        assert not result.state.pending_calls
        assert RuntimeEventKind.TURN_CANCELLED in [event.kind for event in events]
        assert not runtime.cancel(metadata)
        persisted = store.checkpoint(result.metadata).load()
        assert persisted is not None
        assert persisted.status.value == "cancelled"
        assert not persisted.pending_calls
        continued = await runtime.run_turn(result.metadata, "Continue after cancellation.")
        assert continued.answer == "Recovered after cancellation."
        assert continued.state.status.value == "completed"

    asyncio.run(exercise())


def test_session_runtime_closes_owned_background_processes(tmp_path: Path) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        runtime = SessionRuntime(
            provider=ScriptedProvider([]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=SessionStore(tmp_path / "sessions"),
            model="scripted",
        )
        metadata = runtime.start(project)
        manager = runtime._process_manager(metadata)
        process = await manager.start((sys.executable, "-c", "import time; time.sleep(30)"))
        await runtime.aclose()
        snapshot = await manager.snapshot(process.process_id)
        assert not snapshot.running

    asyncio.run(exercise())
