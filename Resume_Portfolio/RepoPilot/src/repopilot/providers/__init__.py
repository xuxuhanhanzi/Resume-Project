"""Model-provider adapters."""

from repopilot.providers.base import ModelProvider, ModelProviderError
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider
from repopilot.providers.scripted import ScriptedProvider

__all__ = [
    "LocalOpenAICompatibleProvider",
    "ModelProvider",
    "ModelProviderError",
    "ScriptedProvider",
]
