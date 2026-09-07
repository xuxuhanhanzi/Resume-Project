from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


def test_fifty_turn_session_compacts_and_resumes_without_losing_history(tmp_path: Path) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "sessions")
        provider = ScriptedProvider(
            [ModelResponse(content=f"answer {index}") for index in range(51)]
        )
        runtime = SessionRuntime(
            provider=provider,
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        metadata = runtime.start(project)
        for index in range(50):
            turn = await runtime.run_turn(metadata, f"Turn {index}: inspect one concern.")
            metadata = turn.metadata
            assert turn.answer == f"answer {index}"
        usage_before = runtime.context_usage(metadata)
        assert usage_before.source_messages == 100
        assert usage_before.retained_messages <= 16

        compacted = runtime.compact(metadata)
        assert "Turn 49" in compacted.summary
        resumed = SessionRuntime(
            provider=provider,
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        loaded = store.load(project_root=project, session_id=compacted.metadata.session_id)
        continued = await resumed.run_turn(loaded, "Continue from the prior evidence.")
        assert continued.answer == "answer 50"
        state = store.checkpoint(continued.metadata).load()
        assert state is not None
        assert any(message.content == "Turn 0: inspect one concern." for message in state.messages)
        assert any(
            message.content == "Continue from the prior evidence." for message in state.messages
        )
        assert resumed.context_usage(continued.metadata).summary_characters > 0

    asyncio.run(exercise())
