"""Explicit DeepSeek cloud provider with a fixed public API boundary."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from repopilot.core.contracts import JSONValue, ModelRequest
from repopilot.providers.base import ModelProviderError, ProviderCapabilities
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig

_DEEPSEEK_CHAT_ENDPOINT = "https://api.deepseek.com/chat/completions"
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
SUPPORTED_DEEPSEEK_MODELS = ("deepseek-v4-flash", "deepseek-v4-pro")


@dataclass(frozen=True, slots=True)
class DeepSeekProviderConfig:
    """Configuration for DeepSeek's documented OpenAI-compatible chat endpoint.

    The endpoint is intentionally not configurable. This avoids turning a
    convenience CLI flag into a generic external-request primitive and keeps
    outbound source-code disclosure explicit to this one provider.
    """

    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: float = 120.0
    api_key: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_DEEPSEEK_MODELS:
            allowed = ", ".join(SUPPORTED_DEEPSEEK_MODELS)
            raise ValueError(f"DeepSeek model must be one of: {allowed}")
        if not _ENVIRONMENT_NAME.fullmatch(self.api_key_env):
            raise ValueError("api_key_env must be a valid environment-variable name")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must not be empty when provided")


class DeepSeekProvider(LocalOpenAICompatibleProvider):
    """DeepSeek adapter that reuses OpenAI message and tool-call normalization."""

    capabilities = ProviderCapabilities(
        streaming=True,
        forced_tool_choice=True,
        reasoning_content=True,
    )
    require_stream_done = True

    def __init__(self, config: DeepSeekProviderConfig | None = None) -> None:
        self.deepseek_config = config or DeepSeekProviderConfig()
        # Keep the parent configuration loopback-only. The overridden endpoint
        # below is fixed, so accepting a cloud provider does not loosen the
        # security invariant of ``LocalOpenAICompatibleProvider``.
        super().__init__(
            LocalProviderConfig(
                "http://127.0.0.1",
                self.deepseek_config.model,
                api_key_env=self.deepseek_config.api_key_env,
                timeout_seconds=self.deepseek_config.timeout_seconds,
            )
        )

    def _endpoint(self) -> str:
        return _DEEPSEEK_CHAT_ENDPOINT

    def _http_error_message(self, status_code: int) -> str:
        if status_code in {401, 403}:
            return (
                f"DeepSeek API rejected the saved credentials (HTTP {status_code}); "
                "run 'repopilot auth login deepseek' to replace the key"
            )
        return f"DeepSeek API returned HTTP {status_code}"

    def _headers(self) -> dict[str, str]:
        secret = self.deepseek_config.api_key or os.environ.get(self.deepseek_config.api_key_env)
        if not secret:
            raise ModelProviderError(
                (
                    "DeepSeek API key is unavailable; run 'repopilot auth login deepseek' "
                    "or set environment variable "
                    f"{self.deepseek_config.api_key_env} before starting RepoPilot"
                ),
                recoverable=False,
            )
        return {"Content-Type": "application/json", "Authorization": f"Bearer {secret}"}

    def _add_provider_options(self, payload: dict[str, JSONValue], request: ModelRequest) -> None:
        """Disable hidden reasoning only for explicitly bounded diagnostic requests."""

        if request.reasoning_mode == "disabled":
            payload["thinking"] = {"type": "disabled"}
