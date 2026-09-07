"""Generate public-only R2 patches for a frozen SWE-bench Verified split.

This runner deliberately does not evaluate a patch.  It reads only the
allowlisted public JSONL, writes predictions plus bounded runtime receipts, and
leaves official resolution to the separately pinned SWE-bench evaluator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic

from repopilot.context.builder import ContextBuilder
from repopilot.core.budgets import RunBudget
from repopilot.core.contracts import AgentState, RunStatus
from repopilot.core.loop import AgentRuntime
from repopilot.evaluation.coderag import load_manifest
from repopilot.evaluation.swebench_verified import (
    load_public_records,
    public_dataset_sha256,
)
from repopilot.evaluation.swebench_workspace import reset_owned_workspace
from repopilot.evidence.protocol import ExperimentReceipt, canonical_sha256, sha256_file
from repopilot.evidence.token_ledger import StrictTokenLedgerProvider
from repopilot.orchestration.harness import MultiAgentHarness
from repopilot.orchestration.pev import ReviewerStrategy
from repopilot.providers.base import ModelProvider, ModelProviderError
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig
from repopilot.runtime.runner import DockerSandboxConfig, DockerSandboxRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import Tool
from repopilot.tools.coding import (
    ApplyPatchTool,
    FindSymbolTool,
    GitDiffTool,
    InspectFailureTool,
    ListFilesTool,
    ReadFileTool,
    RunTestsTool,
    SearchTextTool,
)

_VARIANTS = (
    "direct_react",
    "pev_no_reviewer",
    "pev_always_reviewer",
    "pev_risk_gated",
)
_SPLITS = ("development", "validation", "final_holdout")
_TEST_COMMAND = (
    "bash",
    "-lc",
    "source /opt/miniconda3/bin/activate && conda activate testbed "
    "&& cd /workspace && python -m pytest -q",
)
_TOKEN_RESERVATION_FRAMING_OVERHEAD = 512


class CapturingExecutor:
    """Retain runtime states so PEV receipts include executor cost and status."""

    def __init__(self, delegate: AgentRuntime) -> None:
        self.delegate = delegate
        self.states: list[AgentState] = []

    async def run(
        self, task: PublicTaskSpec, *, run_id: str | None = None, resume: bool = True
    ) -> AgentState:
        state = await self.delegate.run(task, run_id=run_id, resume=resume)
        self.states.append(state)
        return state


def _coding_tools() -> list[Tool]:
    """Use the same bounded code tool surface for every R2 control variant."""

    return [
        ListFilesTool(),
        ReadFileTool(),
        SearchTextTool(),
        FindSymbolTool(),
        ApplyPatchTool(),
        RunTestsTool(),
        GitDiffTool(),
        InspectFailureTool(),
    ]


def _git(workspace: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(workspace), *arguments],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _source_commit() -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _image_digest(image: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _write_source_snapshot(run_root: Path) -> tuple[Path, str]:
    """Bind generation evidence to the executed source, including dirty worktrees."""

    project_root = Path(__file__).resolve().parents[1]
    candidates = [project_root / "pyproject.toml", Path(__file__).resolve()]
    candidates.extend(sorted((project_root / "src" / "repopilot").rglob("*.py")))
    files = sorted(
        {path.resolve() for path in candidates},
        key=lambda path: path.relative_to(project_root).as_posix(),
    )
    entries = [
        {
            "path": str(path.relative_to(project_root).as_posix()),
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    content_sha256 = canonical_sha256(entries)
    snapshot = {
        "schema_version": 1,
        "kind": "r2_generation_source_snapshot",
        "content_sha256": content_sha256,
        "files": entries,
        "notes": [
            "This records the executed source tree even when the Git worktree is dirty.",
            "The Git commit in the generation receipt remains a parent revision reference.",
        ],
    }
    snapshot_path = run_root / "source_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return snapshot_path, content_sha256


def _select_records(
    records: list[dict[str, str]],
    *,
    manifest_ids: Sequence[str],
    task_ids: list[str],
    max_instances: int | None,
) -> list[dict[str, str]]:
    by_id = {record["instance_id"]: record for record in records}
    selected_ids = sorted(task_ids) if task_ids else list(manifest_ids)
    if max_instances is not None:
        selected_ids = selected_ids[:max_instances]
    if not selected_ids:
        raise ValueError("R2 run selected no task IDs")
    if any(instance_id not in manifest_ids for instance_id in selected_ids):
        raise ValueError("selected task is outside the frozen requested split")
    missing_public = [instance_id for instance_id in selected_ids if instance_id not in by_id]
    if missing_public:
        raise ValueError(f"selected task is absent from public JSONL: {missing_public[:3]}")
    return [by_id[instance_id] for instance_id in selected_ids]


def _make_task(record: dict[str, str], arguments: argparse.Namespace) -> PublicTaskSpec:
    workspace = (arguments.workspace_root / record["instance_id"]).resolve(strict=True)
    if _git(workspace, "rev-parse", "HEAD") != record["base_commit"]:
        raise ValueError("workspace HEAD does not match the public SWE-bench base commit")
    if _git(workspace, "status", "--short"):
        raise ValueError("workspace is not clean before the R2 trial")
    return PublicTaskSpec(
        task_id=record["instance_id"],
        workspace=workspace,
        problem_statement=record["problem_statement"],
        allowed_paths=(".",),
        forbidden_paths=(".git",),
        test_command=_TEST_COMMAND,
        max_changed_files=arguments.max_changed_files,
        budget=RunBudget(
            max_iterations=arguments.max_iterations,
            max_tool_calls=arguments.max_tool_calls,
            max_total_tokens=arguments.max_total_tokens,
            max_wall_seconds=arguments.max_wall_seconds,
        ),
    )


def _runtime(provider: ModelProvider, *, image: str, artifacts_root: Path) -> AgentRuntime:
    return AgentRuntime(
        provider=provider,
        tools=_coding_tools(),
        runner=DockerSandboxRunner(
            DockerSandboxConfig(image=image, memory="2g", cpus=2.0, pids_limit=256)
        ),
        artifacts_root=artifacts_root,
        context_builder=ContextBuilder(max_output_tokens=1024),
    )


async def _run_trial(
    record: dict[str, str], *, arguments: argparse.Namespace, run_root: Path
) -> dict[str, object]:
    task = _make_task(record, arguments)
    base_provider = LocalOpenAICompatibleProvider(
        LocalProviderConfig(
            arguments.base_url,
            arguments.model,
            timeout_seconds=300.0,
            seed=arguments.seed,
        )
    )
    provider = StrictTokenLedgerProvider(
        base_provider,
        token_budget=arguments.max_total_tokens,
        framing_overhead_tokens=_TOKEN_RESERVATION_FRAMING_OVERHEAD,
    )
    started = monotonic()
    variant = arguments.variant
    runtime = _runtime(
        provider,
        image=record["image"],
        artifacts_root=run_root / "executor_artifacts",
    )
    reviewer_calls = 0
    reviewer_required = False
    reviewer_reasons: list[str] = []
    executor_states: list[AgentState] = []
    try:
        if variant == "direct_react":
            state = await runtime.run(task, run_id=task.task_id, resume=False)
            executor_states = [state]
            status = state.status.value
            failure_reason = state.failure_reason
            claimed_completed = state.status is RunStatus.COMPLETED
            replans = 0
        else:
            strategy = ReviewerStrategy(
                {
                    "pev_no_reviewer": ReviewerStrategy.NONE,
                    "pev_always_reviewer": ReviewerStrategy.ALWAYS,
                    "pev_risk_gated": ReviewerStrategy.RISK_GATED,
                }[variant]
            )
            executor = CapturingExecutor(runtime)
            harness = MultiAgentHarness(artifacts_root=run_root / "pev_artifacts")
            multi = await harness.run(
                task,
                executor=executor,
                planner_provider=provider,
                reviewer_provider=provider,
                run_id=task.task_id,
                resume=False,
                max_cycles=arguments.max_replans + 1,
                timeout_seconds=arguments.max_wall_seconds,
                reviewer_strategy=strategy,
            )
            executor_states = executor.states
            status = multi.status.value
            failure_reason = multi.failure_reason
            claimed_completed = multi.status is RunStatus.COMPLETED
            reviewer_calls = multi.reviewer_calls
            reviewer_required = multi.reviewer_required
            reviewer_reasons = list(multi.reviewer_reasons)
            replans = max(0, multi.cycle - 1)
    except (ModelProviderError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        status = "runtime_error"
        failure_reason = f"{type(error).__name__}: {error}"
        claimed_completed = False
        replans = 0

    patch = _git(task.workspace, "diff", "--no-ext-diff", "--binary", "--", check=False)
    changed_files = _git(task.workspace, "diff", "--name-only", "--", check=False).splitlines()
    return {
        "instance_id": task.task_id,
        "repo": record["repo"],
        "base_commit": record["base_commit"],
        "image": record["image"],
        "image_digest": _image_digest(record["image"]),
        "variant": variant,
        "status": status,
        "failure_reason": failure_reason,
        "agent_claimed_completed": claimed_completed,
        "replans": replans,
        "reviewer_calls": reviewer_calls,
        "reviewer_required": reviewer_required,
        "reviewer_reasons": reviewer_reasons,
        "executor_cycles": len(executor_states),
        "executor_iterations": sum(state.iteration for state in executor_states),
        "executor_tool_calls": sum(state.tool_calls for state in executor_states),
        "executor_input_tokens": sum(state.input_tokens for state in executor_states),
        "executor_output_tokens": sum(state.output_tokens for state in executor_states),
        "total_model_calls": provider.calls,
        "total_input_tokens": provider.input_tokens,
        "total_output_tokens": provider.output_tokens,
        "total_tokens": provider.total_tokens,
        "token_budget": provider.token_budget,
        "token_budget_exceeded": provider.over_budget,
        "reserved_input_tokens": provider.reserved_input_tokens,
        "reserved_output_tokens": provider.reserved_output_tokens,
        "reserved_tokens": provider.reserved_tokens,
        "patch_bytes": len(patch.encode("utf-8")),
        "changed_files": changed_files,
        "wall_seconds": round(monotonic() - started, 3),
        "model_patch": patch,
    }


async def _run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.reset_workspace_after_run:
        if arguments.workspace_reuse_root is None:
            raise ValueError("--reset-workspace-after-run requires --workspace-reuse-root")
        reuse_root = arguments.workspace_reuse_root.resolve(strict=True)
        if not reuse_root.is_dir():
            raise ValueError("--workspace-reuse-root must be an existing directory")
    else:
        reuse_root = None
    records = load_public_records(arguments.public_jsonl)
    manifest = load_manifest(arguments.manifest)
    if arguments.split == "final_holdout" and not arguments.allow_final_holdout:
        raise ValueError("final holdout requires --allow-final-holdout after validation selection")
    notes = " ".join(manifest.notes)
    public_sha = public_dataset_sha256(records)
    if public_sha not in notes:
        raise ValueError("manifest does not bind the supplied public SWE-bench JSONL digest")
    selected = _select_records(
        records,
        manifest_ids=manifest.assignments[arguments.split],
        task_ids=arguments.task_id,
        max_instances=arguments.max_instances,
    )
    run_root = arguments.output_root / arguments.run_id
    if run_root.exists():
        raise ValueError(f"R2 output already exists: {run_root}")
    run_root.mkdir(parents=True)
    source_snapshot_path, source_snapshot_sha256 = _write_source_snapshot(run_root)
    results = [
        await _run_trial(record, arguments=arguments, run_root=run_root) for record in selected
    ]
    predictions_path = run_root / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": result["instance_id"],
                    "model_name_or_path": arguments.submission_model_name,
                    "model_patch": result["model_patch"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for result in results
        ),
        encoding="utf-8",
        newline="\n",
    )
    raw = {
        "schema_version": 1,
        "kind": "r2_swebench_verified_generation",
        "run_id": arguments.run_id,
        "variant": arguments.variant,
        "split": arguments.split,
        "scope": "full_split" if arguments.max_instances is None else "bounded_smoke",
        "manifest_sha256": manifest.sha256,
        "public_dataset_sha256": public_sha,
        "model": arguments.model,
        "model_revision": arguments.model_revision,
        "seed": arguments.seed,
        "source_snapshot_path": str(source_snapshot_path),
        "source_snapshot_sha256": source_snapshot_sha256,
        "results": results,
        "predictions_path": str(predictions_path),
        "workspace_reset": {
            "requested": arguments.reset_workspace_after_run,
            "reuse_root": str(reuse_root) if reuse_root is not None else None,
            "policy": "receipt-bound git reset --hard public base only",
        },
        "notes": [
            "This is generation evidence only. Official Docker resolution must be run separately.",
            "Agent runtime consumes only the allowlisted public JSONL and exported workspace.",
            (
                "Every model call reserves a conservative pre-call input bound plus its "
                "capped output allowance from the shared ledger."
            ),
            "A provider usage violation of that reservation is marked non-comparable.",
        ],
    }
    raw_path = run_root / "generation_results.json"
    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipt = ExperimentReceipt(
        schema_version=1,
        run_id=arguments.run_id,
        variant=arguments.variant,
        manifest_sha256=manifest.sha256,
        source_commit=_source_commit(),
        environment={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "provider": "local_openai_compatible",
            "model": arguments.model,
            "model_revision": arguments.model_revision,
            "source_snapshot_sha256": source_snapshot_sha256,
        },
        fixed_controls={
            "temperature": 0.0,
            "seed": arguments.seed,
            "max_iterations": arguments.max_iterations,
            "max_tool_calls": arguments.max_tool_calls,
            "max_total_tokens": arguments.max_total_tokens,
            "token_budget_mode": "pre_call_conservative_reservation_v1",
            "token_reservation_framing_overhead": _TOKEN_RESERVATION_FRAMING_OVERHEAD,
            "max_wall_seconds": arguments.max_wall_seconds,
            "max_changed_files": arguments.max_changed_files,
            "max_replans": arguments.max_replans,
            "tool_surface": [tool.spec.name for tool in _coding_tools()],
            "visible_test_command": list(_TEST_COMMAND),
            "selected_instances": [record["instance_id"] for record in selected],
        },
        raw_output_sha256=sha256_file(raw_path),
        notes=(
            "Official resolution is not included in this generation receipt.",
            "No evaluator-only patch, test ID, test patch or evaluation script is exposed "
            "to agent roles.",
            "The source snapshot binds the executed worktree when it differs from source_commit.",
        ),
    )
    receipt_path = run_root / "generation.receipt.json"
    receipt.write_once(receipt_path)
    reset_receipt_path = run_root / "workspace_reset_receipt.json"
    reset_events: list[dict[str, str]] = []
    reset_errors: list[str] = []
    if reuse_root is not None:
        for result in results:
            try:
                reset_events.append(
                    reset_owned_workspace(
                        workspace=arguments.workspace_root / str(result["instance_id"]),
                        reuse_root=reuse_root,
                        task_id=str(result["instance_id"]),
                        base_commit=str(result["base_commit"]),
                        manifest_sha256=manifest.sha256,
                    )
                )
            except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
                reset_errors.append(f"{result['instance_id']}: {type(error).__name__}: {error}")
    reset_receipt = {
        "schema_version": 1,
        "kind": "r2_swebench_workspace_reset_receipt",
        "run_id": arguments.run_id,
        "generation_results_sha256": sha256_file(raw_path),
        "generation_receipt_sha256": sha256_file(receipt_path),
        "requested": arguments.reset_workspace_after_run,
        "reuse_root": str(reuse_root) if reuse_root is not None else None,
        "events": reset_events,
        "errors": reset_errors,
        "notes": [
            "Each successful event used only git reset --hard <public-base-commit>.",
            "No git clean or filesystem deletion is part of this lifecycle operation.",
        ],
    }
    reset_receipt_path.write_text(
        json.dumps(reset_receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if reset_errors:
        raise RuntimeError(
            "workspace reset failed after generation artifacts were persisted; "
            f"see {reset_receipt_path}"
        )
    return {
        "run_root": str(run_root),
        "raw_output": str(raw_path),
        "receipt": str(receipt_path),
        "workspace_reset_receipt": str(reset_receipt_path),
        "predictions": str(predictions_path),
        "tasks": len(results),
        "scope": raw["scope"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=_SPLITS, required=True)
    parser.add_argument("--variant", choices=_VARIANTS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--reset-workspace-after-run", action="store_true")
    parser.add_argument("--workspace-reuse-root", type=Path)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--allow-final-holdout", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--submission-model-name", default="qwen2.5_7b_repopilot_r2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-total-tokens", type=int, default=12_000)
    parser.add_argument("--max-wall-seconds", type=float, default=600.0)
    parser.add_argument("--max-changed-files", type=int, default=8)
    parser.add_argument("--max-replans", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.max_instances is not None and arguments.max_instances <= 0:
        raise ValueError("--max-instances must be positive")
    if not 0 <= arguments.max_replans <= 2:
        raise ValueError("--max-replans must be in the pre-registered range 0..2")
    print(json.dumps(asyncio.run(_run(arguments)), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
