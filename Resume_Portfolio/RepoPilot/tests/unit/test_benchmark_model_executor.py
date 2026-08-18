from __future__ import annotations

import asyncio
import json
from pathlib import Path

from repopilot.benchmarks.contracts import (
    BenchmarkDomain,
    BenchmarkTask,
    EnvironmentHandle,
    PreparedTask,
)
from repopilot.benchmarks.executors import DirectModelExecutor
from repopilot.core.contracts import ModelResponse, ModelUsage
from repopilot.providers.scripted import ScriptedProvider


def test_direct_model_executor_persists_output_and_usage(tmp_path: Path) -> None:
    task = BenchmarkTask(
        "fixture",
        "revision",
        "task-1",
        BenchmarkDomain.RESEARCH,
        "Return 42",
        "local",
    )
    prepared = PreparedTask(task, tmp_path)
    handle = EnvironmentHandle("local", tmp_path)
    provider = ScriptedProvider([ModelResponse("42", usage=ModelUsage(11, 2), model="fixed-model")])

    output = asyncio.run(DirectModelExecutor(provider).execute(task, prepared, handle))

    assert output.final_answer == "42"
    assert output.run_metrics.input_tokens == 11
    assert output.run_metrics.output_tokens == 2
    artifact = json.loads((tmp_path / "model_response.json").read_text(encoding="utf-8"))
    assert artifact["model"] == "fixed-model"
    assert provider.requests[0].tools == ()
