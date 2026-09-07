"""Model-provider adapters."""

from repopilot.providers.base import ModelProvider, ModelProviderError
from repopilot.providers.deepseek import DeepSeekProvider, DeepSeekProviderConfig
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider
from repopilot.providers.qwen import QwenProvider, QwenProviderConfig
from repopilot.providers.scripted import ScriptedProvider

__all__ = [
    "LocalOpenAICompatibleProvider",
    "DeepSeekProvider",
    "DeepSeekProviderConfig",
    "QwenProvider",
    "QwenProviderConfig",
    "ModelProvider",
    "ModelProviderError",
    "ScriptedProvider",
]
