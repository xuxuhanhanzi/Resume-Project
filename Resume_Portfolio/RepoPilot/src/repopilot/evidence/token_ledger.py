"""Conservative pre-call token reservations for comparable model trials."""

from __future__ import annotations

import json
from dataclasses import replace

from repopilot.core.contracts import ModelRequest, ModelResponse
from repopilot.providers.base import ModelProvider, ModelProviderError


def conservative_request_token_reservation(
    request: ModelRequest, *, framing_overhead_tokens: int = 512
) -> int:
    """Return a deliberately high input-token reservation for a request.

    The local OpenAI-compatible server reports exact usage only after a call.
    For a pre-call guard, the UTF-8 size of the complete provider-neutral
    payload is a safe conservative proxy: every serialized token requires at
    least one byte, while ``framing_overhead_tokens`` absorbs chat-template and
    transport fields outside the public request model.
    """

    if framing_overhead_tokens < 0:
        raise ValueError("framing_overhead_tokens must not be negative")
    payload = {
        "messages": [message.to_dict() for message in request.messages],
        "tools": [tool.provider_schema() for tool in request.tools],
        "temperature": request.temperature,
        "max_tokens": request.max_output_tokens,
        "reasoning_mode": request.reasoning_mode,
        "required_tool": request.required_tool,
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return len(serialized) + framing_overhead_tokens


class StrictTokenLedgerProvider:
    """Share one pre-reserved total-token budget across all roles in a trial."""

    def __init__(
        self,
        delegate: ModelProvider,
        *,
        token_budget: int,
        framing_overhead_tokens: int = 512,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.delegate = delegate
        self.token_budget = token_budget
        self.framing_overhead_tokens = framing_overhead_tokens
        self.input_tokens = 0
        self.output_tokens = 0
        self.reserved_input_tokens = 0
        self.reserved_output_tokens = 0
        self.calls = 0
        self.protocol_violation = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        input_reservation = conservative_request_token_reservation(
            request, framing_overhead_tokens=self.framing_overhead_tokens
        )
        remaining = self.token_budget - self.reserved_tokens
        output_cap = min(request.max_output_tokens, remaining - input_reservation)
        if input_reservation >= remaining or output_cap <= 0:
            raise ModelProviderError(
                "shared R2 token budget cannot reserve another model call",
                recoverable=False,
            )

        bounded_request = replace(request, max_output_tokens=output_cap)
        response = await self.delegate.complete(bounded_request)
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.reserved_input_tokens += input_reservation
        self.reserved_output_tokens += output_cap
        if (
            response.usage.input_tokens > input_reservation
            or response.usage.output_tokens > output_cap
            or self.total_tokens > self.reserved_tokens
        ):
            self.protocol_violation = True
        return response

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def reserved_tokens(self) -> int:
        return self.reserved_input_tokens + self.reserved_output_tokens

    @property
    def over_budget(self) -> bool:
        """Keep prior receipt semantics: a violation is not comparable."""

        return self.protocol_violation or self.reserved_tokens > self.token_budget
