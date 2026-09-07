from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.background_tasks import BackgroundTaskRecord
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


def test_background_task_receipt_marks_a_lost_live_child_unavailable_after_restart(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        runtime = SessionRuntime(
            provider=ScriptedProvider([]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        metadata = runtime.start(project)
        receipt = BackgroundTaskRecord.from_started_result(
            process_id="saved-process",
            command=(sys.executable, "-c", "print('retry')"),
            cwd=".",
            started_at="2026-01-01T00:00:00+00:00",
        )
        metadata = store.save_background_tasks(metadata, (receipt,))
        restarted = SessionRuntime(
            provider=ScriptedProvider([]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        observed = await restarted.background_tasks(metadata)

        assert observed.tasks[0].status == "unavailable"
        assert observed.tasks[0].retryable
        await restarted.aclose()

    asyncio.run(exercise())


def test_background_task_retry_needs_an_explicit_runtime_call_and_uses_new_identity(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        runtime = SessionRuntime(
            provider=ScriptedProvider([]),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
        )
        metadata = runtime.start(project)
        receipt = BackgroundTaskRecord.from_started_result(
            process_id="old-process",
            command=(sys.executable, "-c", "print('retry')"),
            cwd=".",
            started_at="2026-01-01T00:00:00+00:00",
        ).unavailable_after_restart()
        metadata = store.save_background_tasks(metadata, (receipt,))

        updated, retried = await runtime.retry_background_task(metadata, "old-process")

        assert retried.process_id != "old-process"
        assert retried.status == "running"
        assert len(store.load_background_tasks(updated)) == 2
        await runtime.aclose()

    asyncio.run(exercise())


def test_model_started_background_task_gets_a_redacted_durable_receipt(tmp_path: Path) -> None:
    async def exercise() -> None:
        project = tmp_path / "project"
        project.mkdir()
        store = SessionStore(tmp_path / "state")
        runtime = SessionRuntime(
            provider=ScriptedProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "start",
                                "start_background",
                                {"command": [sys.executable, "-c", "print('ready')"]},
                            ),
                        )
                    ),
                    ModelResponse(content="started"),
                ]
            ),
            tools=default_coding_tools(),
            runner=LocalTrustedRunner(trusted=True),
            store=store,
            model="scripted",
            approval=StaticApprovalHandler(True),
        )
        result = await runtime.run_turn(runtime.start(project), "Start the helper.")
        receipts = store.load_background_tasks(result.metadata)

        assert result.answer == "started"
        assert len(receipts) == 1
        assert receipts[0].command[0] == sys.executable
        assert receipts[0].retryable
        await runtime.aclose()

    asyncio.run(exercise())
