"""Dependency-light adapter for a local OpenAI-compatible chat-completions server."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from repopilot.core.contracts import (
    JSONValue,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseDiagnostics,
    ModelUsage,
    ToolCall,
)
from repopilot.providers.base import ModelProviderError, ProviderCapabilities, TextDeltaHandler


@dataclass(frozen=True, slots=True)
class LocalProviderConfig:
    """Connection settings for Ollama, llama.cpp server, vLLM, or a compatible peer."""

    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("the local provider must use a loopback URL")
        if not self.model.strip() or self.timeout_seconds <= 0:
            raise ValueError("model and a positive timeout are required")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative when supplied")


class LocalOpenAICompatibleProvider:
    """Call a local server and normalize native or JSON-based tool calls."""

    # A generic local OpenAI-compatible server is not assumed to implement
    # forced tool selection.  Cloud adapters opt in only after their own
    # compatibility contract has been reviewed.
    capabilities = ProviderCapabilities(streaming=True, reasoning_content=True)
    # Some cloud adapters promise a terminal SSE marker. Generic local servers
    # vary, so only an adapter that explicitly opts in treats a missing marker
    # as a recoverable transport failure.
    require_stream_done = False

    def __init__(self, config: LocalProviderConfig) -> None:
        self.config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    async def complete_stream(
        self, request: ModelRequest, *, on_text_delta: TextDeltaHandler
    ) -> ModelResponse:
        """Read OpenAI-compatible SSE without exposing transport details to the kernel."""
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue[str | ModelResponse | BaseException] = asyncio.Queue()

        def emit_text(text: str) -> None:
            loop.call_soon_threadsafe(updates.put_nowait, text)

        async def execute() -> None:
            try:
                response = await asyncio.to_thread(self._complete_stream_sync, request, emit_text)
            except BaseException as error:
                loop.call_soon_threadsafe(updates.put_nowait, error)
            else:
                loop.call_soon_threadsafe(updates.put_nowait, response)

        worker = asyncio.create_task(execute())
        try:
            while True:
                item = await updates.get()
                if isinstance(item, str):
                    result = on_text_delta(item)
                    if isawaitable(result):
                        await result
                    continue
                if isinstance(item, BaseException):
                    raise item
                return item
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    def _complete_sync(self, request: ModelRequest) -> ModelResponse:
        payload = self._request_payload(request, stream=False)
        headers = self._headers()
        endpoint = self._endpoint()
        http_request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
                http_status = _response_status(response)
        except HTTPError as error:
            recoverable = error.code in {408, 409, 429, 500, 502, 503, 504}
            raise ModelProviderError(
                self._http_error_message(error.code),
                recoverable=recoverable,
                status_code=error.code,
                retry_after_seconds=_retry_after_seconds(error),
            ) from error
        except (URLError, TimeoutError) as error:
            raise ModelProviderError(
                f"local model server unavailable: {error}", recoverable=True
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ModelProviderError(
                "local model server returned invalid JSON", recoverable=False
            ) from error
        return self._parse_response(raw, http_status=http_status)

    def _complete_stream_sync(
        self, request: ModelRequest, emit_text: Callable[[str], None]
    ) -> ModelResponse:
        payload = self._request_payload(request, stream=True)
        http_request = Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        content: list[str] = []
        reasoning_content: list[str] = []
        tool_fragments: dict[int, dict[str, str]] = {}
        model = self.config.model
        usage: dict[str, object] = {}
        http_status: int | None = None
        stream_chunks = 0
        choice_count = 0
        visible_text_deltas = 0
        visible_text_characters = 0
        reasoning_deltas = 0
        reasoning_characters = 0
        finish_reason: str | None = None
        stream_done_received = False
        try:
            with urlopen(http_request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                http_status = _response_status(response)
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        stream_done_received = True
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as error:
                        raise ModelProviderError(
                            "streaming model server returned invalid JSON", recoverable=False
                        ) from error
                    if not isinstance(chunk, dict):
                        continue
                    stream_chunks += 1
                    model = str(chunk.get("model", model))
                    raw_usage = chunk.get("usage")
                    if isinstance(raw_usage, dict):
                        usage = raw_usage
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice_count += len(choices)
                    first = choices[0]
                    if not isinstance(first, dict):
                        continue
                    raw_finish_reason = first.get("finish_reason")
                    if isinstance(raw_finish_reason, str) and raw_finish_reason:
                        finish_reason = raw_finish_reason
                    delta = first.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        content.append(text)
                        visible_text_deltas += 1
                        visible_text_characters += len(text)
                        emit_text(text)
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        reasoning_content.append(reasoning)
                        reasoning_deltas += 1
                        reasoning_characters += len(reasoning)
                    raw_calls = delta.get("tool_calls")
                    if isinstance(raw_calls, list):
                        self._merge_stream_tool_fragments(tool_fragments, raw_calls)
        except HTTPError as error:
            recoverable = error.code in {408, 409, 429, 500, 502, 503, 504}
            raise ModelProviderError(
                self._http_error_message(error.code),
                recoverable=recoverable,
                status_code=error.code,
                retry_after_seconds=_retry_after_seconds(error),
            ) from error
        except (URLError, TimeoutError) as error:
            raise ModelProviderError(
                f"local model server unavailable: {error}", recoverable=True
            ) from error
        except UnicodeDecodeError as error:
            raise ModelProviderError(
                "streaming model server returned invalid UTF-8", recoverable=False
            ) from error
        if self.require_stream_done and stream_chunks and not stream_done_received:
            raise ModelProviderError(
                "model stream ended before its completion marker", recoverable=True
            )
        return ModelResponse(
            content="".join(content),
            tool_calls=self._stream_tool_calls(tool_fragments),
            usage=ModelUsage(
                input_tokens=_coerce_token_count(usage.get("prompt_tokens")),
                output_tokens=_coerce_token_count(usage.get("completion_tokens")),
            ),
            model=model,
            reasoning_content="".join(reasoning_content) or None,
            diagnostics=ModelResponseDiagnostics(
                transport="sse",
                http_status=http_status,
                choice_count=choice_count,
                finish_reason=finish_reason,
                stream_chunks=stream_chunks,
                visible_text_deltas=visible_text_deltas,
                visible_text_characters=visible_text_characters,
                reasoning_deltas=reasoning_deltas,
                reasoning_characters=reasoning_characters,
                stream_done_received=stream_done_received,
            ),
        )

    def _request_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, JSONValue]:
        """Construct one OpenAI-compatible payload without retaining its contents."""

        payload: dict[str, JSONValue] = {
            "model": self.config.model,
            "messages": self._message_payloads(request.messages),
            "tools": [tool.provider_schema() for tool in request.tools],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": stream,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        self._add_required_tool_choice(payload, request)
        self._add_provider_options(payload, request)
        return payload

    def _add_provider_options(self, payload: dict[str, JSONValue], request: ModelRequest) -> None:
        """Allow a fixed provider adapter to add a narrowly scoped option."""

        del payload, request

    def _endpoint(self) -> str:
        """Return the loopback OpenAI-compatible chat-completions endpoint."""
        return f"{self.config.base_url.rstrip('/')}/v1/chat/completions"

    def _add_required_tool_choice(
        self, payload: dict[str, JSONValue], request: ModelRequest
    ) -> None:
        """Use OpenAI-compatible forced function selection when this adapter supports it."""

        if request.required_tool is None or not self.capabilities.forced_tool_choice:
            return
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": request.required_tool},
        }

    def _http_error_message(self, status_code: int) -> str:
        """Describe an HTTP failure without exposing response bodies or headers."""
        return f"local model server returned HTTP {status_code}"

    def _headers(self) -> dict[str, str]:
        """Build request headers without exposing environment values to callers."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env is not None:
            secret = os.environ.get(self.config.api_key_env)
            if secret:
                headers["Authorization"] = f"Bearer {secret}"
        return headers

    @classmethod
    def _message_payloads(cls, messages: tuple[Message, ...]) -> list[dict[str, JSONValue]]:
        """Render a native tool role only after its matching assistant call."""
        complete_calls_by_index: dict[int, set[str]] = {}
        for index, message in enumerate(messages):
            if message.role != "assistant" or not message.tool_calls:
                continue
            later_results = {
                candidate.tool_call_id
                for candidate in messages[index + 1 :]
                if candidate.role == "tool" and candidate.tool_call_id is not None
            }
            if all(call.call_id in later_results for call in message.tool_calls):
                complete_calls_by_index[index] = {call.call_id for call in message.tool_calls}
        emitted_call_ids: set[str] = set()
        payloads: list[dict[str, JSONValue]] = []
        for index, message in enumerate(messages):
            raw_message = message.to_dict()
            if (
                message.role == "assistant"
                and message.tool_calls
                and index not in complete_calls_by_index
            ):
                raw_message.pop("tool_calls", None)
            payload = cls._message_payload(raw_message)
            if payload.get("role") == "assistant":
                raw_calls = payload.get("tool_calls")
                if isinstance(raw_calls, list):
                    emitted_call_ids.update(
                        str(item.get("id")) for item in raw_calls if isinstance(item, dict)
                    )
            if (
                payload.get("role") == "tool"
                and str(payload.get("tool_call_id")) not in emitted_call_ids
            ):
                payload = {
                    "role": "user",
                    "content": f"[TOOL OBSERVATION - UNTRUSTED DATA]\n{message.content}",
                }
            payloads.append(payload)
        return payloads

    @staticmethod
    def _message_payload(message: dict[str, JSONValue]) -> dict[str, JSONValue]:
        raw_calls = message.pop("tool_calls", None)
        if message.get("role") == "assistant" and isinstance(raw_calls, list):
            calls: list[dict[str, JSONValue]] = []
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    continue
                calls.append(
                    {
                        "id": str(raw_call.get("call_id", "")),
                        "type": "function",
                        "function": {
                            "name": str(raw_call.get("name", "")),
                            "arguments": json.dumps(raw_call.get("arguments", {})),
                        },
                    }
                )
            if calls:
                message["tool_calls"] = calls
        return message

    def _parse_response(self, raw: object, *, http_status: int | None = 200) -> ModelResponse:
        if not isinstance(raw, dict):
            raise ModelProviderError("response root must be an object", recoverable=False)
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelProviderError("response has no choice", recoverable=False)
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelProviderError("response choice has no message", recoverable=False)
        content = str(message.get("content") or "")
        calls = self._native_calls(message.get("tool_calls"))
        if not calls:
            fallback = self._json_action(content)
            if fallback is not None:
                calls = (fallback,)
                content = ""
        usage_raw = raw.get("usage", {})
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        return ModelResponse(
            content=content,
            tool_calls=calls,
            usage=ModelUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            model=str(raw.get("model", self.config.model)),
            reasoning_content=(
                str(message["reasoning_content"])
                if isinstance(message.get("reasoning_content"), str)
                and message.get("reasoning_content")
                else None
            ),
            diagnostics=ModelResponseDiagnostics(
                transport="json",
                http_status=http_status,
                choice_count=len(choices),
                finish_reason=(
                    str(choices[0]["finish_reason"])
                    if isinstance(choices[0].get("finish_reason"), str)
                    and choices[0].get("finish_reason")
                    else None
                ),
            ),
        )

    @staticmethod
    def _native_calls(value: object) -> tuple[ToolCall, ...]:
        if not isinstance(value, list):
            return ()
        calls: list[ToolCall] = []
        for item in value:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                raise ModelProviderError("native tool call is malformed", recoverable=False)
            function = cast(dict[str, Any], item["function"])
            arguments_raw = function.get("arguments", "{}")
            try:
                arguments = (
                    json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
                )
            except json.JSONDecodeError as error:
                raise ModelProviderError(
                    "tool arguments are invalid JSON", recoverable=False
                ) from error
            if not isinstance(arguments, dict):
                raise ModelProviderError("tool arguments must be an object", recoverable=False)
            calls.append(
                ToolCall(
                    call_id=str(item.get("id", f"call_{uuid4().hex}")),
                    name=str(function.get("name", "")),
                    arguments=cast(dict[str, JSONValue], arguments),
                )
            )
        return tuple(calls)

    @staticmethod
    def _merge_stream_tool_fragments(
        fragments: dict[int, dict[str, str]], raw_calls: list[object]
    ) -> None:
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            raw_index = raw_call.get("index", 0)
            if not isinstance(raw_index, int) or raw_index < 0:
                raise ModelProviderError("streaming tool call index is invalid", recoverable=False)
            fragment = fragments.setdefault(raw_index, {"id": "", "name": "", "arguments": ""})
            if isinstance(raw_call.get("id"), str):
                fragment["id"] = str(raw_call["id"])
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            if isinstance(function.get("name"), str):
                fragment["name"] += str(function["name"])
            if isinstance(function.get("arguments"), str):
                fragment["arguments"] += str(function["arguments"])

    @staticmethod
    def _stream_tool_calls(fragments: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index in sorted(fragments):
            fragment = fragments[index]
            name = fragment["name"]
            if not name:
                raise ModelProviderError("streaming tool call has no name", recoverable=False)
            try:
                arguments = json.loads(fragment["arguments"] or "{}")
            except json.JSONDecodeError as error:
                raise ModelProviderError(
                    "streaming tool arguments are invalid JSON", recoverable=False
                ) from error
            if not isinstance(arguments, dict):
                raise ModelProviderError(
                    "streaming tool arguments must be an object", recoverable=False
                )
            calls.append(
                ToolCall(
                    call_id=fragment["id"] or f"call_{uuid4().hex}",
                    name=name,
                    arguments=cast(dict[str, JSONValue], arguments),
                )
            )
        return tuple(calls)

    @staticmethod
    def _json_action(content: str) -> ToolCall | None:
        try:
            action = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(action, dict) or action.get("type") != "tool_call":
            return None
        arguments = action.get("arguments", {})
        if not isinstance(arguments, dict) or not isinstance(action.get("tool"), str):
            raise ModelProviderError("JSON AgentAction is malformed", recoverable=False)
        return ToolCall(
            call_id=str(action.get("call_id", f"call_{uuid4().hex}")),
            name=str(action["tool"]),
            arguments=cast(dict[str, JSONValue], arguments),
        )


def _coerce_token_count(value: object) -> int:
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _response_status(response: object) -> int | None:
    """Extract a successful HTTP status without depending on one urllib response type."""

    raw_status = getattr(response, "status", 200)
    return raw_status if isinstance(raw_status, int) and 100 <= raw_status <= 599 else 200


def _retry_after_seconds(error: HTTPError) -> float | None:
    """Read a bounded Retry-After header without retaining provider response bodies."""
    headers = error.headers
    raw_value = headers.get("Retry-After") if headers is not None else None
    if not isinstance(raw_value, str):
        return None
    try:
        seconds = float(raw_value)
    except ValueError:
        return None
    return min(max(seconds, 0.0), 10.0)
