"""Generate SWE-bench-Live patches without loading evaluator-only records."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from repopilot.context.builder import ContextBuilder
from repopilot.core.budgets import RunBudget
from repopilot.core.loop import AgentRuntime
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig
from repopilot.runtime.runner import DockerSandboxConfig, DockerSandboxRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import default_coding_tools

REVISION = "a637bd46829f3132e12938c8a0ca93173a977b8e"
PUBLIC_SHA256 = "1b8b16fba4eb6a892606382b1ad9d4b5bd7e5a3d2fe0d652b910c74d59cc8bb4"
TASK_CONFIG = {
    "aws-cloudformation__cfn-lint-3767": {
        "image": "starryzhang/sweb.eval.x86_64.aws-cloudformation_1776_cfn-lint-3767:latest",
        "allowed_paths": (
            "src/cfnlint/data/schemas/other/iam",
            "src/cfnlint/rules/resources/iam",
            "test/unit/rules/resources/iam",
        ),
        "test_command": (
            "pytest",
            "-q",
            "test/unit/rules/resources/iam/test_identity_policy.py",
        ),
    },
    "python-babel__babel-1141": {
        "image": "starryzhang/sweb.eval.x86_64.python-babel_1776_babel-1141:latest",
        "allowed_paths": ("babel", "tests"),
        "test_command": ("pytest", "-q", "tests/test_dates.py"),
    },
    "projectmesa__mesa-2394": {
        "image": "starryzhang/sweb.eval.x86_64.projectmesa_1776_mesa-2394:latest",
        "allowed_paths": ("mesa", "tests"),
        "test_command": ("pytest", "-q"),
    },
}
_PRIVATE_FIELDS = {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_public_records(path: Path) -> dict[str, dict[str, object]]:
    if _sha256(path) != PUBLIC_SHA256:
        raise ValueError("SWE-bench-Live public extraction hash mismatch")
    records: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        if not isinstance(raw, dict) or _PRIVATE_FIELDS.intersection(raw):
            raise ValueError("public extraction contains evaluator-only data")
        task_id = str(raw.get("instance_id", ""))
        if task_id not in TASK_CONFIG:
            raise ValueError(f"unexpected SWE-bench-Live task: {task_id}")
        records[task_id] = raw
    if set(records) != set(TASK_CONFIG):
        raise ValueError("public extraction task set mismatch")
    return records


def _git(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(workspace), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    project = Path(__file__).resolve().parents[1]
    data_path = (
        project
        / "evaluation"
        / "benchmarks"
        / "data"
        / "swebench_live"
        / REVISION
        / "smoke_public.jsonl"
    )
    workspace_root = (
        project / "evaluation" / "benchmarks" / "workspaces" / "swebench_live" / REVISION
    )
    records = _load_public_records(data_path)
    selected = tuple(arguments.task_id or TASK_CONFIG)
    run_root = project / "artifacts" / "benchmarks" / arguments.run_id
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(arguments.base_url, arguments.model, timeout_seconds=300.0)
    )
    submission_model_name = arguments.submission_model_name or arguments.model.replace(":", "_")
    predictions: list[dict[str, str]] = []
    results: list[dict[str, object]] = []
    for task_id in selected:
        record = records[task_id]
        config = TASK_CONFIG[task_id]
        workspace = (workspace_root / task_id).resolve(strict=True)
        base_commit = str(record["base_commit"])
        allowed_paths = tuple(str(item) for item in config["allowed_paths"])
        if _git(workspace, "rev-parse", "HEAD") != base_commit:
            raise ValueError(f"workspace commit mismatch: {task_id}")
        if _git(workspace, "status", "--short", "--", *allowed_paths):
            raise ValueError(f"allowed workspace paths are not clean: {task_id}")
        image = str(config["image"])
        task = PublicTaskSpec(
            task_id=task_id,
            workspace=workspace,
            problem_statement=str(record["problem_statement"]),
            allowed_paths=allowed_paths,
            forbidden_paths=(".git",),
            test_command=tuple(str(item) for item in config["test_command"]),
            max_changed_files=4,
            budget=RunBudget(
                max_iterations=arguments.max_iterations,
                max_tool_calls=arguments.max_tool_calls,
                max_total_tokens=arguments.max_total_tokens,
                max_wall_seconds=arguments.max_wall_seconds,
            ),
        )
        runtime = AgentRuntime(
            provider=provider,
            tools=default_coding_tools(),
            runner=DockerSandboxRunner(
                DockerSandboxConfig(
                    image=image,
                    memory="2g",
                    cpus=2.0,
                    pids_limit=256,
                )
            ),
            artifacts_root=run_root / "traces",
            context_builder=ContextBuilder(max_output_tokens=arguments.max_output_tokens),
        )
        state = await runtime.run(task, run_id=task_id, resume=False)
        patch = _git(
            workspace,
            "diff",
            "--no-ext-diff",
            "--binary",
            "--",
            *allowed_paths,
        )
        predictions.append(
            {
                "instance_id": task_id,
                "model_name_or_path": submission_model_name,
                "model_patch": patch,
            }
        )
        results.append(
            {
                "task_id": task_id,
                "status": state.status.value,
                "failure_reason": state.failure_reason,
                "iterations": state.iteration,
                "tool_calls": state.tool_calls,
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "patch_chars": len(patch),
                "changed_files": _git(
                    workspace, "diff", "--name-only", "--", *allowed_paths
                ).splitlines(),
                "image": image,
            }
        )
    predictions_path = run_root / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
        newline="\n",
    )
    summary: dict[str, object] = {
        "run_id": arguments.run_id,
        "dataset_revision": REVISION,
        "public_sha256": PUBLIC_SHA256,
        "model": arguments.model,
        "submission_model_name": submission_model_name,
        "model_revision": arguments.model_revision,
        "tasks": len(results),
        "results": results,
        "predictions_path": str(predictions_path),
    }
    (run_root / "generation_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", action="append", choices=tuple(TASK_CONFIG))
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--submission-model-name")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=32)
    parser.add_argument("--max-total-tokens", type=int, default=32_000)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    arguments = parser.parse_args()
    summary = asyncio.run(_run(arguments))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
