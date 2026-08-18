"""Run a frozen DABench task set with Docker-isolated generated Python."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from repopilot.benchmarks.adapters import DABenchAdapter, DABenchEvaluator
from repopilot.benchmarks.environments import LocalPreparedEnvironment
from repopilot.benchmarks.executors import DockerDataAnalysisConfig, DockerDataAnalysisExecutor
from repopilot.benchmarks.registry import BenchmarkBundle
from repopilot.benchmarks.runner import BenchmarkRunner
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig

REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"


def _mean_primary_metric(records: list[dict[str, object]]) -> float:
    """Mean of each record's numeric ``primary_metric`` (non-numeric -> 0.0)."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        metric = r.get("primary_metric", 0.0)
        if isinstance(metric, (int, float)):
            total += float(metric)
    return total / len(records)


QUESTIONS_SHA256 = "430306c5813bf4f2b0e38f1b6ccee9693c36d5920b3322bd7c8ae46f475ce443"
LABELS_SHA256 = "83b8fb8133c1794fcb6b42435e9cce124d22f68b71e2eedb37c973e40b5a18c2"
TABLE_SHA256 = {
    "test_ave.csv": "411cf03455d6026823fbd3ab65e2839075a22f9a5c088b85aef0d272d79cca00",
    "GODREJIND.csv": "55770473647050ba8f11d46108aac81f38df600e8261b105321fffd5c5a6ecec",
    "unemployement_industry.csv": (
        "c6fce742e7177c3ea07f5080dac8a54fcf8069c9bfab11012a4d9e5e1ac94dd9"
    ),
}
TASK_IDS = (
    "dabench-dev-0000",
    "dabench-dev-0005",
    "dabench-dev-0006",
    "dabench-dev-0007",
    "dabench-dev-0008",
    "dabench-dev-0009",
    "dabench-dev-0010",
    "dabench-dev-0011",
    "dabench-dev-0014",
    "dabench-dev-0023",
)
IMAGE = "repopilot-dabench:py311-v1"


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    project = Path(__file__).resolve().parents[1]
    data_dir = project / "evaluation" / "benchmarks" / "data" / "dabench" / REVISION
    table_sha256 = TABLE_SHA256
    default_task_ids = TASK_IDS
    if arguments.manifest:
        manifest_data = json.loads(
            Path(arguments.manifest).resolve(strict=True).read_text(encoding="utf-8")
        )
        table_sha256 = manifest_data["tables"]
        default_task_ids = tuple(manifest_data["task_ids"])
    adapter = DABenchAdapter(
        data_dir / "da-dev-questions.jsonl",
        data_dir / "da-dev-labels.jsonl",
        data_dir,
        dataset_revision=REVISION,
        questions_sha256=QUESTIONS_SHA256,
        labels_sha256=LABELS_SHA256,
        table_sha256=table_sha256,
    )
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(arguments.base_url, arguments.model, timeout_seconds=300.0)
    )
    bundle = BenchmarkBundle(
        adapter,
        LocalPreparedEnvironment(),
        DockerDataAnalysisExecutor(
            provider,
            DockerDataAnalysisConfig(
                max_attempts=arguments.max_attempts,
                dtype_guardrail=not arguments.no_dtype_guardrail,
                parse_robustness_guardrail=arguments.parse_robustness,
            ),
        ),
        DABenchEvaluator(adapter),
    )
    run_root = project / "artifacts" / "benchmarks" / arguments.run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    runner = BenchmarkRunner()
    records: list[dict[str, object]] = []
    scores: list[float] = []
    selected_task_ids = tuple(arguments.task_id or default_task_ids)
    loaded = list(adapter.load("dev", selected_task_ids))

    def _write_partial(finished_records: list[dict[str, object]]) -> None:
        """Incrementally persist results so a crash never loses progress."""
        partial = {
            "run_id": arguments.run_id,
            "benchmark_id": adapter.benchmark_id,
            "dataset_revision": REVISION,
            "model": arguments.model,
            "model_revision": arguments.model_revision,
            "network_policy": "deny",
            "max_attempts": arguments.max_attempts,
            "table_sha256": table_sha256,
            "tasks": len(finished_records),
            "completed": len(finished_records),
            "total": len(loaded),
            "accuracy": (_mean_primary_metric(finished_records) if finished_records else 0.0),
            "records": finished_records,
            "partial": len(finished_records) < len(loaded),
        }
        (run_root / "results.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for idx, task in enumerate(loaded, start=1):
        try:
            result = await runner.run_one(bundle, task, run_dir=run_root)
            scores.append(result.outcome.primary_metric_value)
            records.append(
                {
                    "task_id": task.task_id,
                    "level": task.public_metadata["level"],
                    "prediction": result.output.final_answer,
                    "task_success": result.outcome.task_success,
                    "primary_metric": result.outcome.primary_metric_value,
                    "domain_metrics": dict(result.outcome.domain_metrics),
                    "shared_metrics": {
                        "iterations": result.outcome.shared_metrics.iterations,
                        "tool_calls": result.outcome.shared_metrics.tool_calls,
                        "input_tokens": result.outcome.shared_metrics.input_tokens,
                        "output_tokens": result.outcome.shared_metrics.output_tokens,
                        "wall_seconds": result.outcome.shared_metrics.wall_seconds,
                        "recovery_count": result.outcome.shared_metrics.recovery_count,
                        "invalid_tool_calls": result.outcome.shared_metrics.invalid_tool_calls,
                        "security_blocks": result.outcome.shared_metrics.security_blocks,
                    },
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 — never let one task kill the run
            import traceback as _tb

            scores.append(0.0)
            records.append(
                {
                    "task_id": task.task_id,
                    "level": task.public_metadata.get("level", "unknown"),
                    "prediction": None,
                    "task_success": False,
                    "primary_metric": 0.0,
                    "domain_metrics": {},
                    "shared_metrics": {},
                    "error": str(exc),
                    "traceback": _tb.format_exc(),
                }
            )
            print(f"[{idx}/{len(loaded)}] {task.task_id} FAILED: {exc}", flush=True)
        print(
            f"[{idx}/{len(loaded)}] {task.task_id} done acc_so_far={sum(scores) / len(scores):.3f}",
            flush=True,
        )
        _write_partial(records)
    try:
        image_id = subprocess.run(  # noqa: S603
            ["docker", "image", "inspect", "--format", "{{.Id}}", IMAGE],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        image_id = "unknown"
    summary: dict[str, object] = {
        "run_id": arguments.run_id,
        "benchmark_id": adapter.benchmark_id,
        "dataset_revision": REVISION,
        "model": arguments.model,
        "model_revision": arguments.model_revision,
        "docker_image": IMAGE,
        "docker_image_id": image_id,
        "network_policy": "deny",
        "max_attempts": arguments.max_attempts,
        "table_sha256": table_sha256,
        "tasks": len(records),
        "total": len(loaded),
        "accuracy": sum(scores) / len(scores) if scores else 0.0,
        "records": records,
        "partial": len(records) < len(loaded),
    }
    (run_root / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--manifest", help="Path to custom manifest JSON with tables and task_ids")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--parse-robustness",
        action="store_true",
        help="Inject the P8C D3 parse-robustness guardrail (additive on top of dtype_guardrail).",
    )
    parser.add_argument(
        "--no-dtype-guardrail",
        action="store_true",
        help="Disable the P8C D1/D2 numeric-coercion guardrail (pre-D1 base config).",
    )
    arguments = parser.parse_args()
    summary = asyncio.run(_run(arguments))
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
