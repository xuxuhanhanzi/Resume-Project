"""Benchmark TaskService with real AgentRuntime, file edits, and deterministic verification."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, cast

from repopilot.core.contracts import ModelResponse, RunStatus, ToolCall
from repopilot.core.loop import AgentRuntime
from repopilot.orchestration.harness import MultiAgentHarness, MultiAgentRunState
from repopilot.orchestration.service import ServiceRunner, TaskService
from repopilot.providers.scripted import ScriptedProvider
from repopilot.retrieval.tool import RetrieveCodeTool
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import default_coding_tools

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "bugfix_demo"


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _memory_snapshot() -> dict[str, int]:
    if os.name != "nt":
        return {}
    kernel32 = cast(Any, ctypes.WinDLL("kernel32", use_last_error=True))
    psapi = cast(Any, ctypes.WinDLL("psapi", use_last_error=True))
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    get_process_memory_info.restype = ctypes.c_int
    handle = get_current_process()
    if not get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {
        "working_set_bytes": int(counters.working_set_size),
        "peak_working_set_bytes": int(counters.peak_working_set_size),
        "private_bytes": int(counters.pagefile_usage),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, int(percentile * len(ordered) + 0.999999) - 1)
    return ordered[index]


def _runner(run_root: Path, run_id: str) -> ServiceRunner:
    async def execute(cancel_event: asyncio.Event) -> MultiAgentRunState:
        workspace = run_root / "workspaces" / run_id
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(DEMO, workspace)
        task = PublicTaskSpec(
            f"task-{run_id}",
            workspace,
            "Fix the Python subtract bug",
            allowed_paths=("calculator.py",),
            test_command=(sys.executable, "verify.py"),
            trusted_fixture=True,
        )
        executor_provider = ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(f"{run_id}-search", "search_text", {"pattern": "subtract"}),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            f"{run_id}-patch",
                            "apply_patch",
                            {
                                "path": "calculator.py",
                                "old_text": "return left + right",
                                "new_text": "return left - right",
                            },
                        ),
                    )
                ),
                ModelResponse(tool_calls=(ToolCall(f"{run_id}-test", "run_tests", {}),)),
                ModelResponse(content='{"type":"finish","answer":"fixed and verified"}'),
            ]
        )
        executor = AgentRuntime(
            provider=executor_provider,
            tools=[*default_coding_tools(), RetrieveCodeTool()],
            runner=LocalTrustedRunner(trusted=True),
            artifacts_root=run_root / "executor",
        )
        harness = MultiAgentHarness(artifacts_root=run_root / "coordinator")
        return await harness.run(
            task,
            executor=executor,
            planner_provider=ScriptedProvider([ModelResponse(content="Inspect, patch, test.")]),
            reviewer_provider=ScriptedProvider(
                [ModelResponse(content='{"verdict":"pass","feedback":"verified"}')]
            ),
            run_id=run_id,
            cancel_event=cancel_event,
        )

    return execute


async def benchmark_level(run_root: Path, concurrency: int, tasks: int) -> dict[str, float | int]:
    service = TaskService(max_concurrency=concurrency)
    run_ids = [f"c{concurrency}-{index:02d}" for index in range(tasks)]
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for run_id in run_ids:
        await service.submit(run_id, _runner(run_root, run_id))
    statuses = await asyncio.gather(*(service.wait(run_id) for run_id in run_ids))
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    latencies = [
        status.finished_at - status.submitted_at
        for status in statuses
        if status.finished_at is not None
    ]
    result: dict[str, float | int] = {
        "concurrency": concurrency,
        "tasks": tasks,
        "completed": sum(status.status is RunStatus.COMPLETED for status in statuses),
        "failed": sum(status.status is RunStatus.FAILED for status in statuses),
        "cancelled": sum(status.status is RunStatus.CANCELLED for status in statuses),
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "throughput_tasks_per_second": tasks / wall_seconds,
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": _percentile(latencies, 0.95),
        "latency_p99_seconds": _percentile(latencies, 0.99),
    }
    result.update(_memory_snapshot())
    return result


async def async_main(args: argparse.Namespace) -> int:
    if args.output.exists() or args.run_root.exists():
        raise FileExistsError("output or run_root already exists; choose a new run ID")
    args.run_root.mkdir(parents=True)
    print(json.dumps({"pid": os.getpid(), "phase": "started"}), flush=True)
    results: list[dict[str, float | int]] = []
    for concurrency in (1, 4, 8):
        print(json.dumps({"phase": "level", "concurrency": concurrency}), flush=True)
        results.append(await benchmark_level(args.run_root, concurrency, args.tasks))
    report = {
        "schema_version": "1.0.0",
        "workload": (
            "real AgentRuntime + MultiAgentHarness + subprocess verifier; "
            "deterministic ScriptedProvider, not model inference"
        ),
        "python": sys.version,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(item["failed"] == 0 for item in results) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tasks <= 0:
        raise ValueError("tasks must be positive")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
