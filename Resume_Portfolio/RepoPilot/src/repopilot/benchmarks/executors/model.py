"""A fixed, tool-free model baseline for domain benchmark tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import monotonic

from repopilot.benchmarks.contracts import (
    ArtifactRef,
    BenchmarkTask,
    EnvironmentHandle,
    PreparedTask,
    SharedRunMetrics,
    TaskOutput,
)
from repopilot.core.contracts import Message, ModelRequest
from repopilot.providers.base import ModelProvider


@dataclass(frozen=True, slots=True)
class DirectModelExecutorConfig:
    """Settings frozen for a no-tools comparison baseline."""

    system_prompt: str = (
        "Answer the task directly. Give only the final answer requested by the user and do not "
        "claim to have used tools or sources that were not provided."
    )
    temperature: float = 0.0
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.system_prompt.strip() or self.max_output_tokens <= 0:
            raise ValueError("direct model prompt and output budget must be valid")


class DirectModelExecutor:
    """Call one fixed provider once and persist its raw response."""

    def __init__(
        self, provider: ModelProvider, config: DirectModelExecutorConfig | None = None
    ) -> None:
        self.provider = provider
        self.config = config or DirectModelExecutorConfig()

    async def execute(
        self, task: BenchmarkTask, prepared: PreparedTask, handle: EnvironmentHandle
    ) -> TaskOutput:
        if prepared.task != task or handle.work_dir != prepared.work_dir:
            raise ValueError("direct executor received mismatched task environment")
        started_at = monotonic()
        response = await self.provider.complete(
            ModelRequest(
                messages=(
                    Message("system", self.config.system_prompt),
                    Message("user", task.instruction),
                ),
                tools=(),
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )
        )
        if response.tool_calls:
            raise ValueError("direct model baseline must not return tool calls")
        artifact_path = prepared.work_dir / "model_response.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "model": response.model,
                    "content": response.content,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        return TaskOutput(
            final_answer=response.content.strip(),
            structured_payload={"model": response.model},
            artifacts=(ArtifactRef(str(artifact_path), digest, "application/json"),),
            run_metrics=SharedRunMetrics(
                iterations=1,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                wall_seconds=monotonic() - started_at,
            ),
        )
