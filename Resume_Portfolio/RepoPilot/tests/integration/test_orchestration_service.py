from __future__ import annotations

import asyncio

import pytest

from repopilot.core.contracts import RunStatus
from repopilot.orchestration.harness import MultiAgentRunState
from repopilot.orchestration.service import TaskService


@pytest.mark.integration
def test_service_enforces_concurrency_and_streams_terminal_events() -> None:
    async def exercise() -> None:
        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            nonlocal active, peak
            assert not cancel_event.is_set()
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return MultiAgentRunState("child", "task", RunStatus.COMPLETED)

        service = TaskService(max_concurrency=4)
        for index in range(8):
            await service.submit(f"run-{index}", runner)
        statuses = await asyncio.gather(*(service.wait(f"run-{index}") for index in range(8)))
        events = [event async for event in service.stream_events("run-0")]

        assert peak == 4
        assert all(status.status is RunStatus.COMPLETED for status in statuses)
        assert [event.kind for event in events] == ["submitted", "started", "finished"]
        assert [event.sequence for event in events] == [0, 1, 2]

    asyncio.run(exercise())


@pytest.mark.integration
def test_service_cancel_reaches_cooperative_runner() -> None:
    async def exercise() -> None:
        started = asyncio.Event()

        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            started.set()
            await cancel_event.wait()
            return MultiAgentRunState(
                "child", "task", RunStatus.CANCELLED, failure_reason="cancelled by caller"
            )

        service = TaskService()
        await service.submit("cancel", runner)
        await started.wait()
        await service.cancel("cancel")
        status = await service.wait("cancel")

        assert status.status is RunStatus.CANCELLED
        assert status.failure_reason == "cancelled by caller"

    asyncio.run(exercise())


@pytest.mark.integration
def test_service_timeout_and_exception_are_attributable_failures() -> None:
    async def exercise() -> None:
        async def slow(cancel_event: asyncio.Event) -> MultiAgentRunState:
            del cancel_event
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

        async def broken(cancel_event: asyncio.Event) -> MultiAgentRunState:
            del cancel_event
            raise RuntimeError("injected")

        service = TaskService(max_concurrency=2)
        await service.submit("timeout", slow, timeout_seconds=0.01)
        await service.submit("broken", broken)
        timeout, broken_status = await asyncio.gather(
            service.wait("timeout"), service.wait("broken")
        )

        assert timeout.status is RunStatus.FAILED
        assert timeout.failure_reason == "service timeout after 0.010 seconds"
        assert broken_status.status is RunStatus.FAILED
        assert broken_status.failure_reason == "runner_error:RuntimeError:injected"

    asyncio.run(exercise())


def test_service_rejects_duplicate_and_unknown_run_ids() -> None:
    async def exercise() -> None:
        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            del cancel_event
            return MultiAgentRunState("child", "task", RunStatus.COMPLETED)

        service = TaskService()
        await service.submit("same", runner)
        with pytest.raises(ValueError, match="duplicate_run_id"):
            await service.submit("same", runner)
        with pytest.raises(KeyError, match="unknown_run_id"):
            service.status("missing")
        await service.wait("same")

    asyncio.run(exercise())
