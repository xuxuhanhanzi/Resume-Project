"""Regression tests for the R2 shared pre-call budget gate."""

from __future__ import annotations

import asyncio

import pytest

from repopilot.core.contracts import Message, ModelRequest, ModelResponse, ModelUsage
from repopilot.evidence.token_ledger import (
    StrictTokenLedgerProvider,
    conservative_request_token_reservation,
)
from repopilot.providers.base import ModelProviderError


class _CapturingProvider:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response


def _request(*, max_output_tokens: int = 20) -> ModelRequest:
    return ModelRequest(
        messages=(Message("user", "Inspect the repository and propose a patch."),),
        tools=(),
        max_output_tokens=max_output_tokens,
    )


def test_ledger_caps_output_and_blocks_a_second_unreserved_call() -> None:
    request = _request()
    input_reservation = conservative_request_token_reservation(request)
    delegate = _CapturingProvider(ModelResponse(usage=ModelUsage(input_tokens=1, output_tokens=3)))
    provider = StrictTokenLedgerProvider(delegate, token_budget=input_reservation + 3)

    asyncio.run(provider.complete(request))

    assert delegate.requests[0].max_output_tokens == 3
    assert provider.reserved_tokens == provider.token_budget
    assert not provider.over_budget
    with pytest.raises(ModelProviderError, match="cannot reserve"):
        asyncio.run(provider.complete(request))
    assert len(delegate.requests) == 1


def test_ledger_rejects_a_call_when_its_input_cannot_be_reserved() -> None:
    request = _request()
    delegate = _CapturingProvider(ModelResponse())
    provider = StrictTokenLedgerProvider(
        delegate,
        token_budget=conservative_request_token_reservation(request),
    )

    with pytest.raises(ModelProviderError, match="cannot reserve"):
        asyncio.run(provider.complete(request))

    assert not delegate.requests


def test_ledger_marks_unexpected_provider_usage_as_non_comparable() -> None:
    request = _request(max_output_tokens=5)
    input_reservation = conservative_request_token_reservation(request)
    delegate = _CapturingProvider(
        ModelResponse(usage=ModelUsage(input_tokens=input_reservation + 1, output_tokens=1))
    )
    provider = StrictTokenLedgerProvider(delegate, token_budget=input_reservation + 5)

    asyncio.run(provider.complete(request))

    assert provider.over_budget
    assert provider.protocol_violation
