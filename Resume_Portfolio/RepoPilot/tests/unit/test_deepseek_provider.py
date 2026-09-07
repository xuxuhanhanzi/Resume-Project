from __future__ import annotations

import json
from email.message import Message as EmailMessage
from typing import Any
from urllib.error import HTTPError

import pytest

import repopilot.cli as cli
from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProviderError
from repopilot.providers.deepseek import (
    SUPPORTED_DEEPSEEK_MODELS,
    DeepSeekProvider,
    DeepSeekProviderConfig,
)


def test_deepseek_provider_uses_a_fixed_endpoint_and_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> None:
            del exc_type, exc, traceback

        @staticmethod
        def read() -> bytes:
            return b'{"model":"deepseek-v4-flash","choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        observed["request"] = request
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setenv("REPOPILOT_TEST_DEEPSEEK_KEY", "test-secret")
    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", fake_urlopen)
    provider = DeepSeekProvider(
        DeepSeekProviderConfig(
            model="deepseek-v4-flash",
            api_key_env="REPOPILOT_TEST_DEEPSEEK_KEY",
            timeout_seconds=7.5,
        )
    )

    response = provider._complete_sync(  # noqa: SLF001
        ModelRequest((Message("user", "hello"),), (), reasoning_mode="disabled")
    )

    request = observed["request"]
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert observed["timeout"] == 7.5
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert response.content == "ok"
    assert response.diagnostics.to_dict() == {
        "transport": "json",
        "http_status": 200,
        "choice_count": 1,
        "finish_reason": None,
        "stream_chunks": 0,
        "visible_text_deltas": 0,
        "visible_text_characters": 0,
        "reasoning_deltas": 0,
        "reasoning_characters": 0,
        "stream_done_received": False,
    }


def test_deepseek_provider_refuses_to_start_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPOPILOT_TEST_DEEPSEEK_KEY", raising=False)
    provider = DeepSeekProvider(DeepSeekProviderConfig(api_key_env="REPOPILOT_TEST_DEEPSEEK_KEY"))

    with pytest.raises(ModelProviderError, match="REPOPILOT_TEST_DEEPSEEK_KEY"):
        provider._headers()  # noqa: SLF001


def test_deepseek_provider_labels_credential_rejection_as_a_cloud_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(request: Any, *, timeout: float) -> None:
        del timeout
        raise HTTPError(request.full_url, 401, "Unauthorized", EmailMessage(), None)

    monkeypatch.setenv("REPOPILOT_TEST_DEEPSEEK_KEY", "test-secret")
    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", reject)
    provider = DeepSeekProvider(DeepSeekProviderConfig(api_key_env="REPOPILOT_TEST_DEEPSEEK_KEY"))

    with pytest.raises(ModelProviderError, match="DeepSeek API rejected the saved credentials"):
        provider._complete_sync(ModelRequest((Message("user", "hello"),), ()))  # noqa: SLF001


def test_deepseek_provider_retries_a_stream_missing_its_terminal_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    def fake_urlopen(_request: object, *, timeout: float) -> Response:
        assert timeout > 0
        return Response()

    monkeypatch.setenv("REPOPILOT_TEST_DEEPSEEK_KEY", "test-secret")
    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", fake_urlopen)
    provider = DeepSeekProvider(DeepSeekProviderConfig(api_key_env="REPOPILOT_TEST_DEEPSEEK_KEY"))

    with pytest.raises(ModelProviderError, match="completion marker"):
        provider._complete_stream_sync(  # noqa: SLF001
            ModelRequest((Message("user", "hello"),), ()), lambda _text: None
        )


def test_deepseek_configuration_and_cli_preserve_provider_boundaries() -> None:
    with pytest.raises(ValueError, match="DeepSeek model"):
        DeepSeekProviderConfig(model="deepseek-chat")
    with pytest.raises(ValueError, match="environment-variable"):
        DeepSeekProviderConfig(api_key_env="BAD-NAME")
    assert SUPPORTED_DEEPSEEK_MODELS == ("deepseek-v4-flash", "deepseek-v4-pro")

    args = cli.build_parser().parse_args(["--provider", "deepseek", "--trust", "-p", "inspect"])
    provider = cli._provider_from_args(args)  # noqa: SLF001
    assert isinstance(provider, DeepSeekProvider)
    assert provider.deepseek_config.model == "deepseek-v4-flash"

    custom_endpoint = cli.build_parser().parse_args(
        [
            "--provider",
            "deepseek",
            "--base-url",
            "https://example.invalid",
            "--trust",
            "-p",
            "inspect",
        ]
    )
    with pytest.raises(ValueError, match="fixed DeepSeek"):
        cli._provider_from_args(custom_endpoint)  # noqa: SLF001
