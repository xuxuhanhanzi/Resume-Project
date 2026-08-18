"""Runnable local HTTP service binding ASGI requests to the real harness."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from repopilot.api.asgi import ApiConfig, RepoPilotASGI
from repopilot.cli import build_runtime
from repopilot.core.contracts import JSONValue
from repopilot.orchestration.harness import MultiAgentHarness, MultiAgentRunState
from repopilot.orchestration.service import ServiceRunner, TaskService
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig
from repopilot.runtime.runner import (
    CommandRunner,
    DockerSandboxConfig,
    DockerSandboxRunner,
    LocalTrustedRunner,
)
from repopilot.task import PublicTaskSpec


@dataclass(frozen=True, slots=True)
class LocalTaskRunnerFactory:
    """Resolve allowlisted task specs and bind them to the executable harness."""

    task_root: Path
    artifacts_root: Path
    base_url: str
    model: str
    sandbox_image: str | None = None

    def __call__(self, run_id: str, payload: dict[str, JSONValue]) -> ServiceRunner:
        raw_task_path = payload.get("task_path")
        if not isinstance(raw_task_path, str) or not raw_task_path.strip():
            raise ValueError("task_path_required")
        root = self.task_root.resolve(strict=True)
        task_path = (root / raw_task_path).resolve(strict=True)
        if not task_path.is_relative_to(root) or not task_path.is_file():
            raise ValueError("task_path_outside_allowlist")
        task = PublicTaskSpec.load(task_path)
        workspace = (self.artifacts_root / "workspaces" / run_id).resolve()
        if not workspace.exists():
            workspace.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                task.workspace,
                workspace,
                ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache"),
            )
        task = replace(task, workspace=workspace)
        provider = LocalOpenAICompatibleProvider(LocalProviderConfig(self.base_url, self.model))
        if task.trusted_fixture:
            command_runner: CommandRunner = LocalTrustedRunner(trusted=True)
        else:
            if not self.sandbox_image:
                raise ValueError("untrusted_task_requires_sandbox_image")
            command_runner = DockerSandboxRunner(DockerSandboxConfig(self.sandbox_image))
        runtime = build_runtime(
            provider=provider,
            runner=command_runner,
            artifacts=self.artifacts_root / "executor",
        )
        harness = MultiAgentHarness(artifacts_root=self.artifacts_root / "coordinator")

        async def runner(cancel_event: asyncio.Event) -> MultiAgentRunState:
            return await harness.run(
                task,
                executor=runtime,
                planner_provider=provider,
                reviewer_provider=provider,
                run_id=run_id,
                cancel_event=cancel_event,
            )

        return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RepoPilot authenticated JSON/SSE service")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/service"))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sandbox-image")
    parser.add_argument("--token-env", default="REPOPILOT_API_TOKEN")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--rate-limit", type=int, default=120)
    return parser


def entrypoint(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if len(token) < 16:
        print(
            f"repopilot-api: set {args.token_env} to a token of at least 16 characters",
            file=sys.stderr,
        )
        return 2
    if args.port <= 0 or args.max_concurrency <= 0 or args.rate_limit <= 0:
        print("repopilot-api: port, concurrency and rate limit must be positive", file=sys.stderr)
        return 2
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError:
        print(
            "repopilot-api: install the service extra with pip install -e .[service]",
            file=sys.stderr,
        )
        return 2
    factory = LocalTaskRunnerFactory(
        task_root=args.task_root,
        artifacts_root=args.artifacts.resolve(),
        base_url=args.base_url,
        model=args.model,
        sandbox_image=args.sandbox_image,
    )
    app = RepoPilotASGI(
        TaskService(max_concurrency=args.max_concurrency),
        factory,
        ApiConfig(token, rate_limit_requests=args.rate_limit),
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
