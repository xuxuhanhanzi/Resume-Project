"""Run a preregistered no-tools FRAMES cost probe against a local model."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from repopilot.benchmarks.adapters import FramesAdapter, FramesEvaluator, FramesMode
from repopilot.benchmarks.environments import LocalPreparedEnvironment
from repopilot.benchmarks.executors import DirectModelExecutor, DirectModelExecutorConfig
from repopilot.benchmarks.registry import BenchmarkBundle
from repopilot.benchmarks.runner import BenchmarkRunner
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig

REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
DATASET_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    project = Path(__file__).resolve().parents[1]
    dataset_path = project / "evaluation" / "benchmarks" / "data" / "frames" / REVISION / "test.tsv"
    adapter = FramesAdapter(
        dataset_path,
        dataset_revision=REVISION,
        dataset_sha256=DATASET_SHA256,
        mode=FramesMode.RETRIEVAL,
    )
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(arguments.base_url, arguments.model, timeout_seconds=300.0)
    )
    executor = DirectModelExecutor(
        provider,
        DirectModelExecutorConfig(
            system_prompt=(
                "Answer using only the shortest final answer. Do not explain, show reasoning, "
                "cite sources, or repeat the question."
            ),
            temperature=0.0,
            max_output_tokens=arguments.max_output_tokens,
        ),
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
        "model": arguments.model,
        "model_revision": arguments.model_revision,
        "setting": "direct_no_tools_no_documents",
        "max_output_tokens": arguments.max_output_tokens,
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
    parser.add_argument("--count", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    summary = asyncio.run(_run(arguments))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
