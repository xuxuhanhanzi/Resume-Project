"""Provider protocol kept independent from any vendor SDK."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from repopilot.core.contracts import ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Stable adapter facts used to select safe runtime behaviour.

    Values describe RepoPilot's adapter contract rather than a vendor marketing
    claim.  This lets a compatibility endpoint opt out of a feature that its
    upstream API may advertise, keeping the runtime conservative.
    """

    native_tool_calls: bool = True
    forced_tool_choice: bool = False
    streaming: bool = False
    reasoning_content: bool = False
    vision: bool = False


def capabilities_for(provider: object) -> ProviderCapabilities:
    """Read declared capabilities without forcing existing test fakes to change."""

    declared = getattr(provider, "capabilities", None)
    return declared if isinstance(declared, ProviderCapabilities) else ProviderCapabilities()


class ModelProviderError(RuntimeError):
    """A classified provider failure that the runtime may retry."""

    def __init__(
        self,
        message: str,
        *,
        recoverable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.recoverable = recoverable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def empty_response_recovery_message(response: ModelResponse) -> str | None:
    """Return an actionable, content-free reason for an unusable final response.

    A response with structured tool calls is valid even when it has no prose.
    For a final response, accepting an empty body hides the provider symptom and
    makes a completed session indistinguishable from a useful answer.
    """

    if response.tool_calls or response.content.strip():
        return None
    diagnostics = response.diagnostics
    if diagnostics.finish_reason == "length":
        return (
            "model reached its output limit before producing visible final text; "
            "retry with a shorter request or a larger output budget"
        )
    if diagnostics.finish_reason == "content_filter":
        return (
            "model response was filtered before visible final text was produced; "
            "retry with a safer request"
        )
    if (
        diagnostics.transport == "sse"
        and diagnostics.stream_chunks
        and not diagnostics.stream_done_received
    ):
        return "model stream ended before its completion signal; retry the request"
    if diagnostics.reasoning_characters:
        return (
            "model produced reasoning but no visible final text; retry with a shorter request "
            "or disable reasoning for this provider"
        )
    return "model returned no visible final text; retry the request"


class ModelProvider(Protocol):
    """The only model dependency understood by the Agent Runtime."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Produce text or one or more structured tool calls."""
        ...


TextDeltaHandler = Callable[[str], Awaitable[None] | None]


@runtime_checkable
class StreamingModelProvider(ModelProvider, Protocol):
    """Optional provider extension for live text while preserving final tool calls."""

    async def complete_stream(
        self, request: ModelRequest, *, on_text_delta: TextDeltaHandler
    ) -> ModelResponse:
        """Emit content fragments and return the normalized completed response."""
        ...
