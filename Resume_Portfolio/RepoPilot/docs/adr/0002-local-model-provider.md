# ADR 0002: Local OpenAI-compatible model provider

## Decision

Use a loopback-only OpenAI-compatible HTTP boundary for the primary local Qwen model. Keep the
runtime vendor-neutral and include a deterministic ScriptedProvider.

## Consequences

Ollama, llama.cpp server, or vLLM can be compared without changing the Agent scaffold. Native
tool calls and JSON AgentAction are normalized internally. Exact model/quantization selection is
an experiment result rather than a source-code assumption.
