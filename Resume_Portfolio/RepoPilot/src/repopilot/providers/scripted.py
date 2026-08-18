"""Deterministic provider for tests, recovery drills, and offline demonstrations."""

from __future__ import annotations

from collections import deque

from repopilot.core.contracts import ModelRequest, ModelResponse
from repopilot.providers.base import ModelProviderError


class ScriptedProvider:
    """Return a fixed response sequence and retain requests for assertions."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise ModelProviderError("scripted provider has no response left", recoverable=False)
        return self._responses.popleft()
