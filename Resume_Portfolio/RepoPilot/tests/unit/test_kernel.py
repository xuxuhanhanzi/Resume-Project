from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from repopilot.core.contracts import (
    AgentState,
    ModelRequest,
    ModelResponse,
    ModelResponseDiagnostics,
    ToolCall,
)
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.core.kernel import AgentKernel, AgentRuntimeConfig
from repopilot.providers.base import ModelProviderError, TextDeltaHandler
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.checkpoint import ExecutionJournal
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext, ToolRegistry
from repopilot.tools.coding import default_coding_tools


def test_kernel_emits_ordered_model_and_tool_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("value = 1\n", encoding="utf-8")
    task = PublicTaskSpec(
        "kernel-events",
        workspace,
        "Read the calculator module.",
        allowed_paths=("calculator.py",),
        test_command=(sys.executable, "-c", "raise SystemExit(0)"),
        trusted_fixture=True,
    )
    provider = ScriptedProvider(
        [ModelResponse(tool_calls=(ToolCall("read-1", "read_file", {"path": "calculator.py"}),))]
    )
    kernel = AgentKernel(provider=provider, registry=ToolRegistry(default_coding_tools()))
    state = AgentState("kernel-run", task.task_id)
    context = ToolContext(task, LocalTrustedRunner(trusted=True))
    journal = ExecutionJournal(tmp_path / "journal.json")
    events: list[RuntimeEvent] = []

    async def exercise() -> None:
        request = ModelRequest(messages=(), tools=kernel.registry.specs())
        response = await kernel.request_model(request, iteration=1, emit=events.append)
        results = await kernel.execute_calls(
            response.tool_calls,
            task=task,
            context=context,
            state=state,
            journal=journal,
            emit=events.append,
        )
        assert results[0].ok

    asyncio.run(exercise())

    assert [event.kind for event in events] == [
        RuntimeEventKind.TURN_STARTED,
        RuntimeEventKind.MODEL_CALL_STARTED,
        RuntimeEventKind.MODEL_CALL_COMPLETED,
        RuntimeEventKind.TOOL_CALL_PROPOSED,
        RuntimeEventKind.POLICY_DECISION,
        RuntimeEventKind.TOOL_CALL_STARTED,
        RuntimeEventKind.TOOL_CALL_COMPLETED,
        RuntimeEventKind.TURN_COMPLETED,
    ]


def test_kernel_emits_streamed_text_deltas_for_an_optional_streaming_provider() -> None:
    class StreamingProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise AssertionError("streaming path should be selected")

        async def complete_stream(
            self, request: ModelRequest, *, on_text_delta: TextDeltaHandler
        ) -> ModelResponse:
            del request
            first = on_text_delta("Hello, ")
            if first is not None:
                await first
            second = on_text_delta("world!")
            if second is not None:
                await second
            return ModelResponse(content="Hello, world!")

    kernel = AgentKernel(
        provider=StreamingProvider(),
        registry=ToolRegistry(),
        config=AgentRuntimeConfig(stream_model_output=True),
    )
    events: list[RuntimeEvent] = []

    response = asyncio.run(
        kernel.request_model(ModelRequest(messages=(), tools=()), iteration=1, emit=events.append)
    )

    assert response.content == "Hello, world!"
    deltas = [
        event.data["text"] for event in events if event.kind is RuntimeEventKind.MODEL_OUTPUT_DELTA
    ]
    assert deltas == ["Hello, ", "world!"]


def test_kernel_retries_a_recoverable_provider_error_with_an_observable_event() -> None:
    class _FlakyProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, _request: ModelRequest) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                raise ModelProviderError(
                    "temporary overload",
                    recoverable=True,
                    status_code=429,
                    retry_after_seconds=0,
                )
            return ModelResponse(content="Recovered.")

    provider = _FlakyProvider()
    kernel = AgentKernel(
        provider=provider,
        registry=ToolRegistry(),
        config=AgentRuntimeConfig(model_retries=1),
    )
    events: list[RuntimeEvent] = []

    response = asyncio.run(
        kernel.request_model(ModelRequest(messages=(), tools=()), iteration=1, emit=events.append)
    )

    assert response.content == "Recovered."
    assert provider.calls == 2
    retry_event = next(event for event in events if event.kind is RuntimeEventKind.MODEL_RETRYING)
    assert retry_event.data == {"attempt": 1, "delay_seconds": 0, "status_code": 429}


def test_kernel_turn_fails_with_a_recovery_message_for_an_empty_final_response() -> None:
    kernel = AgentKernel(
        provider=ScriptedProvider(
            [
                ModelResponse(
                    diagnostics=ModelResponseDiagnostics(
                        transport="sse",
                        finish_reason="length",
                        stream_chunks=4,
                        stream_done_received=True,
                    )
                )
            ]
        ),
        registry=ToolRegistry(),
    )
    events: list[RuntimeEvent] = []

    with pytest.raises(ModelProviderError, match="output limit"):
        asyncio.run(
            kernel.request_model(
                ModelRequest(messages=(), tools=()), iteration=1, emit=events.append
            )
        )

    assert events[-1].kind is RuntimeEventKind.TURN_FAILED
    assert "shorter request" in str(events[-1].data["reason"])
