"""Transport-neutral structured API for bounded asynchronous agent runs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

from repopilot.core.contracts import RunStatus
from repopilot.orchestration.harness import MultiAgentRunState

_TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}


class ServiceRunner(Protocol):
    """One submitted run that cooperatively observes service cancellation."""

    async def __call__(self, cancel_event: asyncio.Event) -> MultiAgentRunState: ...


@dataclass(frozen=True, slots=True)
class ServiceEvent:
    """One ordered lifecycle event suitable for SSE/WebSocket adapters."""

    sequence: int
    run_id: str
    kind: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Immutable public snapshot for a submitted run."""

    run_id: str
    status: RunStatus
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    failure_reason: str
    event_count: int


@dataclass(slots=True)
class _RunRecord:
    run_id: str
    runner: ServiceRunner
    timeout_seconds: float | None
    status: RunStatus = RunStatus.CREATED
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    failure_reason: str = ""
    result: MultiAgentRunState | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    events: list[ServiceEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    task: asyncio.Task[None] | None = None


class TaskService:
    """In-process service API with hard concurrency and lifecycle governance."""

    def __init__(self, *, max_concurrency: int = 1) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._runs: dict[str, _RunRecord] = {}

    async def submit(
        self,
        run_id: str,
        runner: ServiceRunner,
        *,
        timeout_seconds: float | None = None,
    ) -> ServiceStatus:
        """Submit one unique run and return immediately after durable in-memory registration."""

        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if run_id in self._runs:
            raise ValueError(f"duplicate_run_id:{run_id}")
        record = _RunRecord(run_id, runner, timeout_seconds)
        self._runs[run_id] = record
        await self._emit(record, "submitted", {"max_concurrency": self.max_concurrency})
        record.task = asyncio.create_task(self._execute(record), name=f"repopilot:{run_id}")
        return self.status(run_id)

    def status(self, run_id: str) -> ServiceStatus:
        """Return the current structured status snapshot."""

        record = self._get(run_id)
        return ServiceStatus(
            run_id=record.run_id,
            status=record.status,
            submitted_at=record.submitted_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            failure_reason=record.failure_reason,
            event_count=len(record.events),
        )

    async def cancel(self, run_id: str) -> ServiceStatus:
        """Request cooperative cancellation without discarding checkpoints."""

        record = self._get(run_id)
        if record.status not in _TERMINAL and not record.cancel_event.is_set():
            record.cancel_event.set()
            await self._emit(record, "cancel_requested")
        return self.status(run_id)

    async def wait(self, run_id: str) -> ServiceStatus:
        """Wait for one run's terminal state."""

        record = self._get(run_id)
        if record.task is not None:
            await asyncio.shield(record.task)
        return self.status(run_id)

    async def stream_events(self, run_id: str, *, after: int = 0) -> AsyncIterator[ServiceEvent]:
        """Yield ordered events after a sequence number until the run is terminal."""

        record = self._get(run_id)
        index = max(after, 0)
        while True:
            while index < len(record.events):
                event = record.events[index]
                index += 1
                yield event
            if record.status in _TERMINAL:
                return
            async with record.condition:
                await record.condition.wait_for(partial(_events_ready, record, index))

    async def _execute(self, record: _RunRecord) -> None:
        try:
            async with self._semaphore:
                if record.cancel_event.is_set():
                    record.status = RunStatus.CANCELLED
                    record.failure_reason = "cancelled while queued"
                    return
                record.status = RunStatus.RUNNING
                record.started_at = time.time()
                await self._emit(record, "started")
                call = record.runner(record.cancel_event)
                result = (
                    await call
                    if record.timeout_seconds is None
                    else await asyncio.wait_for(call, timeout=record.timeout_seconds)
                )
                record.result = result
                record.status = result.status
                record.failure_reason = result.failure_reason
        except TimeoutError:
            record.cancel_event.set()
            record.status = RunStatus.FAILED
            record.failure_reason = f"service timeout after {record.timeout_seconds:.3f} seconds"
        except asyncio.CancelledError:
            record.cancel_event.set()
            record.status = RunStatus.CANCELLED
            record.failure_reason = "service task cancelled"
        except Exception as exc:  # noqa: BLE001 - service boundary must retain a terminal record
            record.status = RunStatus.FAILED
            record.failure_reason = f"runner_error:{type(exc).__name__}:{exc}"
        finally:
            record.finished_at = time.time()
            await self._emit(
                record,
                "finished",
                {"status": record.status.value, "failure_reason": record.failure_reason},
            )

    async def _emit(
        self, record: _RunRecord, kind: str, payload: dict[str, Any] | None = None
    ) -> None:
        event = ServiceEvent(
            sequence=len(record.events),
            run_id=record.run_id,
            kind=kind,
            timestamp=time.time(),
            payload=payload or {},
        )
        async with record.condition:
            record.events.append(event)
            record.condition.notify_all()

    def _get(self, run_id: str) -> _RunRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown_run_id:{run_id}") from exc


def _events_ready(record: _RunRecord, index: int) -> bool:
    return index < len(record.events) or record.status in _TERMINAL
