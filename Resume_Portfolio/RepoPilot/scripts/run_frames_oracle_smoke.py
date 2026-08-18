"""Run a frozen oracle-document FRAMES smoke experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from repopilot.benchmarks.adapters import FramesAdapter, FramesEvaluator, FramesMode
from repopilot.benchmarks.environments import LocalPreparedEnvironment
from repopilot.benchmarks.executors import (
    OracleDocumentExecutorConfig,
    OracleDocumentModelExecutor,
)
from repopilot.benchmarks.registry import BenchmarkBundle
from repopilot.benchmarks.runner import BenchmarkRunner
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig

REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
DATASET_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    project = Path(__file__).resolve().parents[1]
    dataset_path = project / "evaluation" / "benchmarks" / "data" / "frames" / REVISION / "test.tsv"
    corpus_manifest = (
        project
        / "evaluation"
        / "benchmarks"
        / "corpora"
        / "frames"
        / REVISION
        / f"smoke{arguments.count}"
        / "manifest.json"
    )
    adapter = FramesAdapter(
        dataset_path,
        dataset_revision=REVISION,
        dataset_sha256=DATASET_SHA256,
        mode=FramesMode.ORACLE_DOCUMENT,
    )
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(arguments.base_url, arguments.model, timeout_seconds=300.0)
    )
    planner_provider = (
        LocalOpenAICompatibleProvider(
            LocalProviderConfig(arguments.base_url, arguments.planner_model, timeout_seconds=300.0)
        )
        if arguments.planner_model
        else None
    )
    executor = OracleDocumentModelExecutor(
        provider,
        corpus_manifest,
        OracleDocumentExecutorConfig(
            plan_queries=arguments.planned,
            reasoned_answer=arguments.reasoned,
            review_answer=arguments.reviewed,
            reasoning_output_tokens=arguments.reasoning_output_tokens,
            retriever_type=arguments.retriever_type,
            dense_model=arguments.dense_model,
            base_url=arguments.base_url,
            use_reranker=arguments.use_reranker,
            query_rewrite=arguments.query_rewrite,
        ),
        planner_provider=planner_provider,
    )
    bundle = BenchmarkBundle(
        adapter,
        LocalPreparedEnvironment(),
        executor,
        FramesEvaluator(adapter),
    )
    task_ids = [f"frames-test-{index:04d}" for index in range(arguments.count)]
    tasks = list(adapter.load("test", task_ids))
    run_root = project / "artifacts" / "benchmarks" / arguments.run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    runner = BenchmarkRunner()
    records: list[dict[str, object]] = []
    scores: list[float] = []
    for task in tasks:
        result = await runner.run_one(bundle, task, run_dir=run_root)
        scores.append(result.outcome.primary_metric_value)
        records.append(
            {
                "task_id": task.task_id,
                "prediction": result.output.final_answer,
                "task_success": result.outcome.task_success,
                "primary_metric": result.outcome.primary_metric_value,
                "domain_metrics": dict(result.outcome.domain_metrics),
                "shared_metrics": {
                    "tool_calls": result.outcome.shared_metrics.tool_calls,
                    "input_tokens": result.outcome.shared_metrics.input_tokens,
                    "output_tokens": result.outcome.shared_metrics.output_tokens,
                    "wall_seconds": result.outcome.shared_metrics.wall_seconds,
                },
            }
        )
    summary: dict[str, object] = {
        "run_id": arguments.run_id,
        "benchmark_id": adapter.benchmark_id,
        "dataset_revision": REVISION,
        "dataset_sha256": DATASET_SHA256,
        "corpus_manifest_sha256": hashlib.sha256(corpus_manifest.read_bytes()).hexdigest(),
        "model": arguments.model,
        "model_revision": arguments.model_revision,
        "planner_model": arguments.planner_model or arguments.model,
        "planner_model_revision": arguments.planner_model_revision or arguments.model_revision,
        "setting": "oracle_documents_"
        + ("planned_" if arguments.planned else "fixed_")
        + ("reasoned_" if arguments.reasoned else "short_")
        + ("reviewed_" if arguments.reviewed else "unreviewed_")
        + arguments.retriever_type
        + ("_reranked" if arguments.use_reranker else ""),
        "tasks": len(records),
        "accuracy": sum(scores) / len(scores),
        "records": records,
    }
    (run_root / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--count", type=int, default=3, choices=range(1, 201))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-model-revision")
    parser.add_argument("--planned", action="store_true")
    parser.add_argument("--reasoned", action="store_true")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--reasoning-output-tokens", type=int, default=512)
    parser.add_argument(
        "--retriever-type", default="bm25", choices=["bm25", "dense", "hybrid", "no_retrieval"]
    )
    parser.add_argument("--dense-model", default="nomic-embed-text")
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--query-rewrite", action="store_true")
    arguments = parser.parse_args()
    if bool(arguments.planner_model) != bool(arguments.planner_model_revision):
        parser.error("--planner-model and --planner-model-revision must be provided together")
    summary = asyncio.run(_run(arguments))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
