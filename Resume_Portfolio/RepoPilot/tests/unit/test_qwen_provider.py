from __future__ import annotations

import json
from email.message import Message as EmailMessage
from typing import Any
from urllib.error import HTTPError

import pytest

import repopilot.cli as cli
from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProviderError
from repopilot.providers.qwen import QwenProvider, QwenProviderConfig


def test_qwen_provider_uses_the_fixed_dashscope_compatibility_endpoint(
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
            return b'{"model":"qwen-plus","choices":[{"message":{"content":"ok"}}]}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        observed["request"] = request
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setenv("REPOPILOT_TEST_QWEN_KEY", "test-secret")
    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", fake_urlopen)
    provider = QwenProvider(
        QwenProviderConfig(model="qwen-plus", api_key_env="REPOPILOT_TEST_QWEN_KEY")
    )

    response = provider._complete_sync(ModelRequest((Message("user", "hello"),), ()))  # noqa: SLF001

    request = observed["request"]
    assert request.full_url == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data.decode("utf-8"))["model"] == "qwen-plus"
    assert response.content == "ok"


def test_qwen_provider_requires_an_environment_key_and_cli_selects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPOPILOT_TEST_QWEN_KEY", raising=False)
    provider = QwenProvider(QwenProviderConfig(api_key_env="REPOPILOT_TEST_QWEN_KEY"))
    with pytest.raises(ModelProviderError, match="REPOPILOT_TEST_QWEN_KEY"):
        provider._headers()  # noqa: SLF001

    args = cli.build_parser().parse_args(["--provider", "qwen", "--trust", "-p", "inspect"])
    selected = cli._provider_from_args(args)  # noqa: SLF001
    assert isinstance(selected, QwenProvider)
    assert selected.qwen_config.model == "qwen-plus"
    assert selected.qwen_config.api_key_env == "DASHSCOPE_API_KEY"


def test_qwen_provider_labels_credential_rejection_as_a_cloud_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(request: Any, *, timeout: float) -> None:
        del timeout
        raise HTTPError(request.full_url, 401, "Unauthorized", EmailMessage(), None)

    monkeypatch.setenv("REPOPILOT_TEST_QWEN_KEY", "test-secret")
    monkeypatch.setattr("repopilot.providers.local_openai.urlopen", reject)
    provider = QwenProvider(QwenProviderConfig(api_key_env="REPOPILOT_TEST_QWEN_KEY"))

    with pytest.raises(ModelProviderError, match="Qwen API rejected the saved credentials"):
        provider._complete_sync(ModelRequest((Message("user", "hello"),), ()))  # noqa: SLF001
