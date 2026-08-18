from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from repopilot.benchmarks.contracts import (
    AssetRef,
    BenchmarkDomain,
    BenchmarkTask,
    EnvironmentHandle,
    PreparedTask,
)
from repopilot.benchmarks.executors import (
    OracleDocumentExecutorConfig,
    OracleDocumentModelExecutor,
)
from repopilot.core.contracts import ModelResponse, ModelUsage
from repopilot.providers.scripted import ScriptedProvider


def test_oracle_document_executor_retrieves_frozen_evidence(tmp_path: Path) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("The measured result is forty two.\n", encoding="utf-8")
    digest = hashlib.sha256(document_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "url": "https://example.test/result",
                        "path": "document.txt",
                        "text_sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    task = BenchmarkTask(
        "fixture",
        "revision",
        "task-1",
        BenchmarkDomain.RESEARCH,
        "What is the measured result?",
        "local",
        assets=(AssetRef("https://example.test/result"),),
    )
    provider = ScriptedProvider([ModelResponse("42", usage=ModelUsage(30, 2), model="fixed-model")])
    executor = OracleDocumentModelExecutor(provider, manifest_path)

    output = asyncio.run(
        executor.execute(task, PreparedTask(task, tmp_path), EnvironmentHandle("local", tmp_path))
    )

    assert output.final_answer == "42"
    assert output.evidence[0].source_id == "https://example.test/result"
    assert output.run_metrics.tool_calls == 1
    assert "forty two" in (tmp_path / "retrieval_context.txt").read_text(encoding="utf-8")
    assert "Retrieved source excerpts" in provider.requests[0].messages[1].content


def test_oracle_document_executor_can_plan_source_queries(tmp_path: Path) -> None:
    document_path = tmp_path / "document.txt"
    document_path.write_text("Parent: Jane Ann Buchanan Lane.\n", encoding="utf-8")
    digest = hashlib.sha256(document_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "url": "https://example.test/person",
                        "path": "document.txt",
                        "text_sha256": digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    task = BenchmarkTask(
        "fixture",
        "revision",
        "task-1",
        BenchmarkDomain.RESEARCH,
        "What was the person's mother's first name?",
        "local",
        assets=(AssetRef("https://example.test/person"),),
    )
    provider = ScriptedProvider(
        [
            ModelResponse("mother parent name", usage=ModelUsage(20, 8)),
            ModelResponse("Jane", usage=ModelUsage(25, 2)),
        ]
    )
    executor = OracleDocumentModelExecutor(
        provider,
        manifest_path,
        OracleDocumentExecutorConfig(plan_queries=True),
    )

    output = asyncio.run(
        executor.execute(task, PreparedTask(task, tmp_path), EnvironmentHandle("local", tmp_path))
    )

    assert output.final_answer == "Jane"
    assert output.run_metrics.iterations == 2
    assert output.run_metrics.input_tokens == 45
    plan = json.loads((tmp_path / "query_plan.json").read_text(encoding="utf-8"))
    assert plan["parsed_queries"]["https://example.test/person"] == "mother parent name"
