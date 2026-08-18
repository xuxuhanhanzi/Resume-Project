from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from repopilot.api.asgi import ApiConfig, RepoPilotASGI, RunnerFactory
from repopilot.api.server import LocalTaskRunnerFactory
from repopilot.core.contracts import JSONValue, RunStatus
from repopilot.orchestration.harness import MultiAgentRunState
from repopilot.orchestration.service import ServiceRunner, TaskService

TOKEN = "test-token-at-least-16-characters"
ASGIMessage = dict[str, Any]


async def _request(
    app: RepoPilotASGI,
    method: str,
    path: str,
    *,
    body: object | bytes = b"",
    headers: dict[str, str] | None = None,
    disconnect_after_request: bool = False,
) -> tuple[int, dict[str, str], bytes]:
    query_separator = path.partition("?")
    request_body = body if isinstance(body, bytes) else json.dumps(body).encode()
    messages: asyncio.Queue[ASGIMessage] = asyncio.Queue()
    await messages.put({"type": "http.request", "body": request_body, "more_body": False})
    if disconnect_after_request:
        await messages.put({"type": "http.disconnect"})
    sent: list[ASGIMessage] = []

    async def receive() -> ASGIMessage:
        return await messages.get()

    async def send(message: ASGIMessage) -> None:
        sent.append(message)

    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope: ASGIMessage = {
        "type": "http",
        "method": method,
        "path": query_separator[0],
        "query_string": query_separator[2].encode("ascii"),
        "headers": raw_headers,
        "client": ("127.0.0.1", 50000),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=2)
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_headers = {
        key.decode("latin-1"): value.decode("latin-1") for key, value in start["headers"]
    }
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(start["status"]), response_headers, response_body


def _factory(
    runner: Callable[[asyncio.Event], Awaitable[MultiAgentRunState]],
) -> RunnerFactory:
    def create(run_id: str, payload: dict[str, JSONValue]) -> ServiceRunner:
        assert payload["run_id"] == run_id
        return cast(ServiceRunner, runner)

    return cast(RunnerFactory, create)


@pytest.mark.integration
def test_health_is_public_and_run_endpoints_require_bearer_auth() -> None:
    async def exercise() -> None:
        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            del cancel_event
            return MultiAgentRunState("run", "task", RunStatus.COMPLETED)

        app = RepoPilotASGI(TaskService(), _factory(runner), ApiConfig(TOKEN))
        health, _, health_body = await _request(app, "GET", "/health")
        unauthorized, unauthorized_headers, body = await _request(app, "GET", "/v1/runs/missing")

        assert health == 200 and json.loads(health_body) == {"status": "up"}
        assert unauthorized == 401
        assert unauthorized_headers["www-authenticate"] == "Bearer"
        assert json.loads(body) == {"error": "unauthorized"}

    asyncio.run(exercise())


@pytest.mark.integration
def test_submit_status_and_resumable_sse_lifecycle() -> None:
    async def exercise() -> None:
        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            assert not cancel_event.is_set()
            await asyncio.sleep(0)
            return MultiAgentRunState("http-run", "task", RunStatus.COMPLETED)

        service = TaskService()
        app = RepoPilotASGI(service, _factory(runner), ApiConfig(TOKEN))
        auth = {"authorization": f"Bearer {TOKEN}"}
        submitted, submit_headers, submit_body = await _request(
            app, "POST", "/v1/runs", body={"run_id": "http-run"}, headers=auth
        )
        await service.wait("http-run")
        status_code, _, status_body = await _request(app, "GET", "/v1/runs/http-run", headers=auth)
        events_code, events_headers, all_events = await _request(
            app, "GET", "/v1/runs/http-run/events", headers=auth
        )
        _, _, resumed_events = await _request(
            app,
            "GET",
            "/v1/runs/http-run/events",
            headers={**auth, "last-event-id": "0"},
        )

        assert submitted == 202
        assert submit_headers["location"] == "/v1/runs/http-run"
        assert json.loads(submit_body)["run_id"] == "http-run"
        assert status_code == 200 and json.loads(status_body)["status"] == "completed"
        assert events_code == 200
        assert events_headers["content-type"].startswith("text/event-stream")
        assert b"event: submitted" in all_events
        assert b"event: started" in all_events
        assert b"event: finished" in all_events
        assert b"event: submitted" not in resumed_events
        assert b"id: 1" in resumed_events and b"id: 2" in resumed_events

    asyncio.run(exercise())


@pytest.mark.integration
def test_cancel_and_sse_disconnect_are_independent() -> None:
    async def exercise() -> None:
        started = asyncio.Event()

        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            started.set()
            await cancel_event.wait()
            return MultiAgentRunState(
                "long-run", "task", RunStatus.CANCELLED, failure_reason="caller cancelled"
            )

        service = TaskService()
        app = RepoPilotASGI(service, _factory(runner), ApiConfig(TOKEN))
        auth = {"authorization": f"Bearer {TOKEN}"}
        await _request(app, "POST", "/v1/runs", body={"run_id": "long-run"}, headers=auth)
        await started.wait()
        events_status, _, _ = await _request(
            app,
            "GET",
            "/v1/runs/long-run/events",
            headers=auth,
            disconnect_after_request=True,
        )
        assert events_status == 200
        assert service.status("long-run").status is RunStatus.RUNNING
        cancel_status, _, _ = await _request(app, "DELETE", "/v1/runs/long-run", headers=auth)
        final = await service.wait("long-run")
        assert cancel_status == 202
        assert final.status is RunStatus.CANCELLED

    asyncio.run(exercise())


@pytest.mark.integration
def test_duplicate_rate_limit_and_request_size_fail_closed() -> None:
    async def exercise() -> None:
        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            del cancel_event
            return MultiAgentRunState("same", "task", RunStatus.COMPLETED)

        auth = {"authorization": f"Bearer {TOKEN}"}
        service = TaskService()
        app = RepoPilotASGI(service, _factory(runner), ApiConfig(TOKEN))
        first, _, _ = await _request(app, "POST", "/v1/runs", body={"run_id": "same"}, headers=auth)
        duplicate, _, duplicate_body = await _request(
            app, "POST", "/v1/runs", body={"run_id": "same"}, headers=auth
        )
        assert first == 202 and duplicate == 409
        assert json.loads(duplicate_body)["error"].startswith("duplicate_run_id")
        await service.wait("same")

        limited = RepoPilotASGI(
            TaskService(),
            _factory(runner),
            ApiConfig(TOKEN, rate_limit_requests=1),
        )
        missing, _, _ = await _request(limited, "GET", "/v1/runs/missing", headers=auth)
        rate_limited, rate_headers, _ = await _request(
            limited, "GET", "/v1/runs/missing", headers=auth
        )
        assert missing == 404 and rate_limited == 429
        assert "retry-after" in rate_headers

        small = RepoPilotASGI(TaskService(), _factory(runner), ApiConfig(TOKEN, max_body_bytes=8))
        too_large, _, too_large_body = await _request(
            small, "POST", "/v1/runs", body=b'{"run_id":"large"}', headers=auth
        )
        assert too_large == 413
        assert json.loads(too_large_body) == {"error": "request_body_too_large"}

    asyncio.run(exercise())


def test_local_runner_factory_rejects_task_path_escape(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    task_root.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("task_id: secret\n", encoding="utf-8")
    factory = LocalTaskRunnerFactory(
        task_root=task_root,
        artifacts_root=tmp_path / "artifacts",
        base_url="http://127.0.0.1:11434",
        model="local-test",
    )

    with pytest.raises(ValueError, match="task_path_outside_allowlist"):
        factory("escape", {"task_path": "../outside.yaml"})
