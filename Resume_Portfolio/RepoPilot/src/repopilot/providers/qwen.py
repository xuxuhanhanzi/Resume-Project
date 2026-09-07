"""Explicit Qwen Cloud provider using DashScope's OpenAI-compatible endpoint."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from repopilot.providers.base import ModelProviderError, ProviderCapabilities
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig

_QWEN_CHAT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class QwenProviderConfig:
    """Configuration for the fixed China DashScope compatibility endpoint."""

    model: str = "qwen-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 120.0
    api_key: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not _MODEL_NAME.fullmatch(self.model):
            raise ValueError("Qwen model must be a safe 1-128 character model identifier")
        if not _ENVIRONMENT_NAME.fullmatch(self.api_key_env):
            raise ValueError("api_key_env must be a valid environment-variable name")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.api_key is not None and not self.api_key.strip():
            raise ValueError("api_key must not be empty when provided")


class QwenProvider(LocalOpenAICompatibleProvider):
    """DashScope adapter reusing the normalized OpenAI chat/tool-call protocol."""

    capabilities = ProviderCapabilities(
        streaming=True,
        forced_tool_choice=True,
        reasoning_content=True,
    )

    def __init__(self, config: QwenProviderConfig | None = None) -> None:
        self.qwen_config = config or QwenProviderConfig()
        # The parent stays loopback-only; only the fixed endpoint below can make
        # a cloud request, preserving a narrow explicit outbound boundary.
        super().__init__(
            LocalProviderConfig(
                "http://127.0.0.1",
                self.qwen_config.model,
                api_key_env=self.qwen_config.api_key_env,
                timeout_seconds=self.qwen_config.timeout_seconds,
            )
        )

    def _endpoint(self) -> str:
        return _QWEN_CHAT_ENDPOINT

    def _http_error_message(self, status_code: int) -> str:
        if status_code in {401, 403}:
            return (
                f"Qwen API rejected the saved credentials (HTTP {status_code}); "
                "run 'repopilot auth login qwen' to replace the key"
            )
        return f"Qwen API returned HTTP {status_code}"

    def _headers(self) -> dict[str, str]:
        secret = self.qwen_config.api_key or os.environ.get(self.qwen_config.api_key_env)
        if not secret:
            raise ModelProviderError(
                (
                    "Qwen API key is unavailable; run 'repopilot auth login qwen' "
                    "or set environment variable "
                    f"{self.qwen_config.api_key_env} before starting RepoPilot"
                ),
                recoverable=False,
            )
        return {"Content-Type": "application/json", "Authorization": f"Bearer {secret}"}
