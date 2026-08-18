"""Deterministic 1/4/8 concurrency smoke for the transport-neutral task service."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from repopilot.core.contracts import RunStatus
from repopilot.orchestration.harness import MultiAgentRunState
from repopilot.orchestration.service import TaskService


async def benchmark(concurrency: int, tasks: int, delay: float) -> dict[str, float | int]:
    service = TaskService(max_concurrency=concurrency)
    started: dict[str, float] = {}
    durations: list[float] = []

    async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
        assert not cancel_event.is_set()
        await asyncio.sleep(delay)
        return MultiAgentRunState("simulated", "load", RunStatus.COMPLETED)

    wall_start = time.perf_counter()
    for index in range(tasks):
        run_id = f"c{concurrency}-{index}"
        started[run_id] = time.perf_counter()
        await service.submit(run_id, runner)
    statuses = await asyncio.gather(*(service.wait(run_id) for run_id in started))
    for run_id in started:
        status = service.status(run_id)
        assert status.finished_at is not None
        durations.append(status.finished_at - status.submitted_at)
    wall_seconds = time.perf_counter() - wall_start
    ordered = sorted(durations)
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)
    return {
        "concurrency": concurrency,
        "tasks": tasks,
        "delay_seconds": delay,
        "completed": sum(status.status is RunStatus.COMPLETED for status in statuses),
        "failed": sum(status.status is RunStatus.FAILED for status in statuses),
        "wall_seconds": wall_seconds,
        "throughput_tasks_per_second": tasks / wall_seconds,
        "latency_p50_seconds": statistics.median(ordered),
        "latency_p95_seconds": ordered[p95_index],
    }


async def async_main(args: argparse.Namespace) -> int:
    results = [await benchmark(concurrency, args.tasks, args.delay) for concurrency in (1, 4, 8)]
    report = {
        "schema_version": "1.0.0",
        "workload": "deterministic simulated runner; not model inference",
        "results": results,
    }
    if args.output.exists():
        raise FileExistsError(f"load_report_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tasks <= 0 or args.delay < 0:
        raise ValueError("tasks must be positive and delay must be non-negative")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
