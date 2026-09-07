"""Cooperative cancellation primitives for one agent operation."""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress


class OperationCancelledError(RuntimeError):
    """Raised when a runtime operation observes its cancellation token."""


class CancellationToken:
    """A small, loop-safe cancellation signal shared by one agent turn.

    The token deliberately carries no task-specific policy.  Callers can request
    cancellation from a UI or service while runners and tools use the same signal
    to stop work and leave a durable checkpoint in a known state.
    """

    def __init__(self) -> None:
        # ``asyncio`` synchronization primitives are deliberately not
        # thread-safe.  A cancellation can originate from a terminal UI or a
        # service thread, so retain the truth in a threading.Event and bridge
        # it into the loop that is actually waiting for the signal.
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._event: asyncio.Event | None = None
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled.is_set()

    @property
    def reason(self) -> str:
        """A stable, user-safe cancellation reason."""
        with self._lock:
            return self._reason or "operation cancelled"

    def cancel(self, reason: str = "operation cancelled") -> None:
        """Request cancellation once; the first reason is retained."""
        with self._lock:
            if self._cancelled.is_set():
                return
            self._reason = reason.strip() or "operation cancelled"
            self._cancelled.set()
            loop = self._loop
            event = self._event
        if loop is not None and event is not None:
            # The loop may have closed after its final observer finished.
            # Cancellation itself remains durably recorded either way.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)

    def raise_if_cancelled(self) -> None:
        """Raise at a cooperative operation boundary."""
        if self.cancelled:
            raise OperationCancelledError(self.reason)

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._event is None:
                self._loop = loop
                self._event = asyncio.Event()
            elif self._loop is not loop:
                raise RuntimeError("a CancellationToken can be awaited from one event loop")
            event = self._event
            already_cancelled = self._cancelled.is_set()
        if already_cancelled:
            event.set()
        await event.wait()
