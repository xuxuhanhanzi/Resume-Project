"""Provider protocol kept independent from any vendor SDK."""

from __future__ import annotations

from typing import Protocol

from repopilot.core.contracts import ModelRequest, ModelResponse


class ModelProviderError(RuntimeError):
    """A classified provider failure that the runtime may retry."""

    def __init__(self, message: str, *, recoverable: bool) -> None:
        super().__init__(message)
        self.recoverable = recoverable


class ModelProvider(Protocol):
    """The only model dependency understood by the Agent Runtime."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Produce text or one or more structured tool calls."""
        ...
