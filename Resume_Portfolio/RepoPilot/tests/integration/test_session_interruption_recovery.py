from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse, ToolCall, ToolResult, ToolSpec
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.base import ToolContext


def test_cancelled_interactive_turn_recovers_after_a_runtime_restart(tmp_path: Path) -> None:
    class WaitTool:
        @property
        def spec(self) -> ToolSpec:
            return ToolSpec("wait", "Wait until the active turn is cancelled.", {})

        async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
            assert context.cancellation is not None
            await context.cancellation.wait()
            context.cancellation.raise_if_cancelled()
            raise AssertionError("cancellation must raise before a tool result is produced")

    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "sessions")
        first_runtime = SessionRuntime(
            provider=ScriptedProvider(
                [ModelResponse(tool_calls=(ToolCall("wait-1", "wait", {}),))]
            ),
            tools=[WaitTool()],
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        metadata = first_runtime.start(project)
        tool_started = asyncio.Event()

        def emit(event: RuntimeEvent) -> None:
            if event.kind is RuntimeEventKind.TOOL_CALL_STARTED:
                tool_started.set()

        active_turn = asyncio.create_task(
            first_runtime.run_turn(metadata, "Wait for cancellation.", emit=emit)
        )
        await asyncio.wait_for(tool_started.wait(), timeout=2.0)
        assert first_runtime.cancel(metadata, reason="interactive stop")
        cancelled = await asyncio.wait_for(active_turn, timeout=2.0)
        assert cancelled.state.status.value == "cancelled"
        await first_runtime.aclose()

        restarted = SessionRuntime(
            provider=ScriptedProvider([ModelResponse(content="Recovered after restart.")]),
            tools=[],
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        loaded = store.load(project_root=project, session_id=cancelled.metadata.session_id)
        continued = await restarted.run_turn(loaded, "Continue from the saved checkpoint.")
        assert continued.state.status.value == "completed"
        assert continued.answer == "Recovered after restart."
        persisted = store.checkpoint(continued.metadata).load()
        assert persisted is not None
        assert any(message.content == "Wait for cancellation." for message in persisted.messages)
        assert any(
            message.content == "Continue from the saved checkpoint."
            for message in persisted.messages
        )
        await restarted.aclose()

    asyncio.run(exercise())
