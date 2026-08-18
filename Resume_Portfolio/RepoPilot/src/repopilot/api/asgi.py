"""Dependency-free ASGI JSON/SSE adapter around :class:`TaskService`."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol, cast
from urllib.parse import parse_qs

from repopilot.core.contracts import JSONValue
from repopilot.orchestration.service import (
    ServiceEvent,
    ServiceRunner,
    ServiceStatus,
    TaskService,
)

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class _HttpError(ValueError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


class RunnerFactory(Protocol):
    """Create one real service runner from an authenticated request payload."""

    def __call__(self, run_id: str, payload: dict[str, JSONValue]) -> ServiceRunner: ...


@dataclass(frozen=True, slots=True)
class ApiConfig:
    """Security and resource limits for the HTTP boundary."""

    bearer_token: str
    max_body_bytes: int = 64 * 1024
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0

    def __post_init__(self) -> None:
        if len(self.bearer_token) < 16:
            raise ValueError("bearer_token must contain at least 16 characters")
        if self.max_body_bytes <= 0 or self.rate_limit_requests <= 0:
            raise ValueError("body and rate limits must be positive")
        if self.rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be positive")


class FixedWindowRateLimiter:
    """Small in-memory limiter keyed by client address without storing credentials."""

    def __init__(self, requests: int, window_seconds: float) -> None:
        self.requests = requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        bucket = self._requests[key]
        boundary = current - self.window_seconds
        while bucket and bucket[0] <= boundary:
            bucket.popleft()
        if len(bucket) >= self.requests:
            return False
        bucket.append(current)
        return True


class RepoPilotASGI:
    """ASGI application exposing submit/status/cancel and resumable SSE events."""

    def __init__(
        self,
        service: TaskService,
        runner_factory: RunnerFactory,
        config: ApiConfig,
    ) -> None:
        self.service = service
        self.runner_factory = runner_factory
        self.config = config
        self.rate_limiter = FixedWindowRateLimiter(
            config.rate_limit_requests, config.rate_limit_window_seconds
        )

    async def __call__(self, scope: ASGIMessage, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            return
        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", "/"))
        if method == "GET" and path == "/health":
            await self._json(send, 200, {"status": "up"})
            return
        headers = self._headers(scope)
        if not self._authenticated(headers):
            await self._json(
                send,
                401,
                {"error": "unauthorized"},
                extra_headers=[(b"www-authenticate", b"Bearer")],
            )
            return
        if not self.rate_limiter.allow(self._client_key(scope)):
            await self._json(
                send,
                429,
                {"error": "rate_limit_exceeded"},
                extra_headers=[
                    (
                        b"retry-after",
                        str(int(self.config.rate_limit_window_seconds)).encode(),
                    )
                ],
            )
            return
        try:
            if method == "POST" and path == "/v1/runs":
                await self._submit(receive, send)
                return
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["v1", "runs"]:
                run_id = parts[2]
                if not _RUN_ID.fullmatch(run_id):
                    await self._json(send, 400, {"error": "invalid_run_id"})
                    return
                if method == "GET" and len(parts) == 3:
                    await self._status(run_id, send)
                    return
                if method == "DELETE" and len(parts) == 3:
                    await self._cancel(run_id, send)
                    return
                if method == "GET" and len(parts) == 4 and parts[3] == "events":
                    await self._events(run_id, scope, headers, receive, send)
                    return
            await self._json(send, 404, {"error": "not_found"})
        except KeyError:
            await self._json(send, 404, {"error": "unknown_run_id"})
        except _HttpError as error:
            await self._json(send, error.status, {"error": error.code})
        except ValueError as error:
            await self._json(send, 400, {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - transport must not leak a traceback
            await self._json(send, 500, {"error": f"internal_error:{type(error).__name__}"})

    async def _submit(self, receive: Receive, send: Send) -> None:
        payload = await self._read_json(receive)
        run_id = str(payload.get("run_id", ""))
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("invalid_run_id")
        timeout_raw = payload.get("timeout_seconds")
        timeout_seconds = None if timeout_raw is None else float(timeout_raw)
        runner = self.runner_factory(run_id, payload)
        try:
            status = await self.service.submit(run_id, runner, timeout_seconds=timeout_seconds)
        except ValueError as error:
            if str(error).startswith("duplicate_run_id"):
                await self._json(send, 409, {"error": str(error)})
                return
            raise
        await self._json(
            send,
            202,
            self._status_payload(status),
            extra_headers=[(b"location", f"/v1/runs/{run_id}".encode())],
        )

    async def _status(self, run_id: str, send: Send) -> None:
        await self._json(send, 200, self._status_payload(self.service.status(run_id)))

    async def _cancel(self, run_id: str, send: Send) -> None:
        status = await self.service.cancel(run_id)
        await self._json(send, 202, self._status_payload(status))

    async def _events(
        self,
        run_id: str,
        scope: ASGIMessage,
        headers: dict[str, str],
        receive: Receive,
        send: Send,
    ) -> None:
        self.service.status(run_id)
        query = parse_qs(bytes(scope.get("query_string", b"")).decode("ascii", errors="ignore"))
        raw_cursor = headers.get("last-event-id", query.get("after", ["-1"])[0])
        try:
            last_event_id = int(raw_cursor)
        except ValueError as error:
            raise ValueError("invalid_event_cursor") from error
        if last_event_id < -1:
            raise ValueError("invalid_event_cursor")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"cache-control", b"no-cache"),
                    (b"x-accel-buffering", b"no"),
                ],
            }
        )
        iterator = self.service.stream_events(run_id, after=last_event_id + 1).__aiter__()
        event_task: asyncio.Task[Any] | None = asyncio.create_task(_next_event(iterator))
        receive_task: asyncio.Task[Any] | None = asyncio.create_task(_receive(receive))
        try:
            while event_task is not None:
                waiters: set[asyncio.Task[Any]] = {event_task}
                if receive_task is not None:
                    waiters.add(receive_task)
                done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                if receive_task is not None and receive_task in done:
                    message = receive_task.result()
                    if message.get("type") == "http.disconnect":
                        event_task.cancel()
                        return
                    receive_task = asyncio.create_task(_receive(receive))
                if event_task in done:
                    try:
                        event = cast(ServiceEvent, event_task.result())
                    except StopAsyncIteration:
                        event_task = None
                        break
                    await send(
                        {
                            "type": "http.response.body",
                            "body": self._sse_payload(event),
                            "more_body": True,
                        }
                    )
                    event_task = asyncio.create_task(_next_event(iterator))
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            for task in (event_task, receive_task):
                if task is not None and not task.done():
                    task.cancel()

    async def _read_json(self, receive: Receive) -> dict[str, JSONValue]:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                raise ValueError("client_disconnected")
            chunk = bytes(message.get("body", b""))
            size += len(chunk)
            if size > self.config.max_body_bytes:
                raise _HttpError(413, "request_body_too_large")
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        try:
            value = json.loads(b"".join(chunks).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid_json") from error
        if not isinstance(value, dict):
            raise ValueError("json_body_must_be_object")
        return cast(dict[str, JSONValue], value)

    def _authenticated(self, headers: dict[str, str]) -> bool:
        scheme, separator, credential = headers.get("authorization", "").partition(" ")
        return bool(
            separator
            and scheme.lower() == "bearer"
            and hmac.compare_digest(credential, self.config.bearer_token)
        )

    @staticmethod
    def _headers(scope: ASGIMessage) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in cast(list[tuple[bytes, bytes]], scope.get("headers", [])):
            result[key.decode("latin-1").lower()] = value.decode("latin-1")
        return result

    @staticmethod
    def _client_key(scope: ASGIMessage) -> str:
        client = scope.get("client")
        if isinstance(client, (tuple, list)) and client:
            return str(client[0])
        return "unknown"

    @staticmethod
    def _status_payload(status: ServiceStatus) -> dict[str, JSONValue]:
        payload = asdict(status)
        payload["status"] = status.status.value
        return cast(dict[str, JSONValue], payload)

    @staticmethod
    def _sse_payload(event: ServiceEvent) -> bytes:
        data = {
            "sequence": event.sequence,
            "run_id": event.run_id,
            "kind": event.kind,
            "timestamp": event.timestamp,
            "payload": event.payload,
        }
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return f"id: {event.sequence}\nevent: {event.kind}\ndata: {encoded}\n\n".encode()

    @staticmethod
    async def _json(
        send: Send,
        status: int,
        payload: dict[str, JSONValue],
        *,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        headers = [(b"content-type", b"application/json; charset=utf-8")]
        headers.extend(extra_headers or [])
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})


async def _next_event(iterator: Any) -> ServiceEvent:
    return cast(ServiceEvent, await anext(iterator))


async def _receive(receive: Receive) -> ASGIMessage:
    return await receive()
