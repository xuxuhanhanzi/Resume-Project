"""Measure real local-model service capacity through RepoPilot's provider contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from repopilot.core.contracts import Message, ModelRequest  # noqa: E402
from repopilot.providers.local_openai import (  # noqa: E402
    LocalOpenAICompatibleProvider,
    LocalProviderConfig,
)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile + 0.999999))
    return ordered[index]


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=10.0) as response:  # noqa: S310 - loopback URL is CLI-fixed
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_object:{url}")
    return payload


async def _benchmark_level(
    provider: LocalOpenAICompatibleProvider,
    concurrency: int,
    tasks: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    batch_started = time.perf_counter()

    async def one(index: int) -> dict[str, Any]:
        submitted = time.perf_counter()
        async with semaphore:
            response = await provider.complete(
                ModelRequest(
                    messages=(
                        Message(
                            "system",
                            "You are a concise coding assistant. Return only the requested token.",
                        ),
                        Message(
                            "user",
                            f"Request {index}: Python len([0, 1, 2]) equals 3. Reply only OK.",
                        ),
                    ),
                    tools=(),
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                )
            )
        return {
            "request_index": index,
            "latency_seconds": time.perf_counter() - submitted,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "content": response.content,
            "model": response.model,
        }

    records = await asyncio.gather(*(one(index) for index in range(tasks)))
    wall_seconds = time.perf_counter() - batch_started
    latencies = [float(record["latency_seconds"]) for record in records]
    input_tokens = sum(int(record["input_tokens"]) for record in records)
    output_tokens = sum(int(record["output_tokens"]) for record in records)
    nonempty = sum(bool(str(record["content"]).strip()) for record in records)
    return {
        "concurrency": concurrency,
        "tasks": tasks,
        "completed": len(records),
        "nonempty_responses": nonempty,
        "wall_seconds": wall_seconds,
        "throughput_requests_per_second": tasks / wall_seconds,
        "output_tokens_per_second": output_tokens / wall_seconds,
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "latency_p99_seconds": _percentile(latencies, 0.99),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "records": records,
    }


async def run(arguments: argparse.Namespace) -> dict[str, Any]:
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(arguments.base_url, arguments.model, timeout_seconds=300.0)
    )
    warmup = await provider.complete(
        ModelRequest(
            messages=(Message("user", "Reply only READY."),),
            tools=(),
            temperature=0.0,
            max_output_tokens=arguments.max_output_tokens,
        )
    )
    results = []
    levels = (1, 4, 8)
    for concurrency in levels:
        results.append(
            await _benchmark_level(
                provider,
                concurrency,
                arguments.tasks,
                arguments.max_output_tokens,
            )
        )
    tags = _get_json(f"{arguments.base_url.rstrip('/')}/api/tags")
    selected = next(
        (
            item
            for item in tags.get("models", [])
            if isinstance(item, dict) and item.get("name") == arguments.model
        ),
        None,
    )
    report = {
        "schema_version": 1,
        "experiment_scope": "real_local_qwen_service_capacity",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "server": _get_json(f"{arguments.base_url.rstrip('/')}/api/version"),
        },
        "model": arguments.model,
        "model_revision_preregistered": arguments.model_revision,
        "model_server_record": selected,
        "warmup": {
            "content": warmup.content,
            "input_tokens": warmup.usage.input_tokens,
            "output_tokens": warmup.usage.output_tokens,
        },
        "fixed_request": {
            "temperature": 0.0,
            "max_output_tokens": arguments.max_output_tokens,
            "tasks_per_level": arguments.tasks,
            "levels": list(levels),
        },
        "results": results,
        "all_checks_passed": all(
            row["completed"] == arguments.tasks
            and row["nonempty_responses"] == arguments.tasks
            and row["output_tokens"] > 0
            for row in results
        ),
        "claim_boundary": (
            "measures loopback Ollama plus Qwen inference through RepoPilot's provider for a "
            "fixed short request; it is not end-to-end coding-agent task throughput"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    arguments = parser.parse_args()
    if arguments.tasks < 1 or arguments.max_output_tokens < 1:
        parser.error("tasks and max-output-tokens must be positive")
    report = asyncio.run(run(arguments))
    compact = {key: value for key, value in report.items() if key != "results"}
    compact["results"] = [
        {key: value for key, value in row.items() if key != "records"} for row in report["results"]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
