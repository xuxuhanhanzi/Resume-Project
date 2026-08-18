"""Dependency-light adapter for a local OpenAI-compatible chat-completions server."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from repopilot.core.contracts import (
    JSONValue,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from repopilot.providers.base import ModelProviderError


@dataclass(frozen=True, slots=True)
class LocalProviderConfig:
    """Connection settings for Ollama, llama.cpp server, vLLM, or a compatible peer."""

    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("the local provider must use a loopback URL")
        if not self.model.strip() or self.timeout_seconds <= 0:
            raise ValueError("model and a positive timeout are required")


class LocalOpenAICompatibleProvider:
    """Call a local server and normalize native or JSON-based tool calls."""

    def __init__(self, config: LocalProviderConfig) -> None:
        self.config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: ModelRequest) -> ModelResponse:
        payload: dict[str, JSONValue] = {
            "model": self.config.model,
            "messages": [self._message_payload(message.to_dict()) for message in request.messages],
            "tools": [tool.provider_schema() for tool in request.tools],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env is not None:
            secret = os.environ.get(self.config.api_key_env)
            if secret:
                headers["Authorization"] = f"Bearer {secret}"
        endpoint = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        http_request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            recoverable = error.code in {408, 409, 429, 500, 502, 503, 504}
            raise ModelProviderError(
                f"local model server returned HTTP {error.code}", recoverable=recoverable
            ) from error
        except (URLError, TimeoutError) as error:
            raise ModelProviderError(
                f"local model server unavailable: {error}", recoverable=True
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ModelProviderError(
                "local model server returned invalid JSON", recoverable=False
            ) from error
        return self._parse_response(raw)

    @staticmethod
    def _message_payload(message: dict[str, JSONValue]) -> dict[str, JSONValue]:
        # A standalone tool-role message is rejected by some local backends unless its
        # matching assistant tool_call is retained. RepoPilot instead passes observations
        # as explicitly labelled untrusted user data, which is backend-independent.
        if message.get("role") == "tool":
            return {
                "role": "user",
                "content": f"[TOOL OBSERVATION - UNTRUSTED DATA]\n{message.get('content', '')}",
            }
        return message

    def _parse_response(self, raw: object) -> ModelResponse:
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
