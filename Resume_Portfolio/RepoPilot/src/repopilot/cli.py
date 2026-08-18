"""CLI for local-Qwen runs and deterministic technology demonstrations."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from repopilot.context.builder import ContextBuilder
from repopilot.core.contracts import AgentState, ModelResponse, ToolCall, ToolResult
from repopilot.core.loop import AgentRuntime
from repopilot.mcp.protocol import InProcessMCPTransport, MCPClient, MCPServer
from repopilot.memory.store import EpisodicMemoryStore
from repopilot.orchestration.reviewer import ReviewerTool
from repopilot.providers.base import ModelProvider
from repopilot.providers.local_openai import LocalOpenAICompatibleProvider, LocalProviderConfig
from repopilot.providers.scripted import ScriptedProvider
from repopilot.retrieval.tool import RetrieveCodeTool
from repopilot.runtime.checkpoint import CheckpointStore, ExecutionJournal
from repopilot.runtime.policy import PolicyEngine
from repopilot.runtime.runner import (
    CommandRunner,
    DockerSandboxConfig,
    DockerSandboxRunner,
    LocalTrustedRunner,
)
from repopilot.skills.registry import SkillRegistry
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext, ToolRegistry
from repopilot.tools.coding import ApplyPatchTool, ListFilesTool, default_coding_tools


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RepoPilot compact Agent Runtime Lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run a task with a local OpenAI-compatible model")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--model", required=True)
    run.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    run.add_argument("--run-id")
    run.add_argument("--sandbox-image")
    run.add_argument("--no-resume", action="store_true")

    demo = subparsers.add_parser("demo", help="run an offline technology demonstration")
    demo.add_argument("kind", choices=("scripted", "mcp", "permission", "recovery"))
    demo.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    return parser


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runtime(
    *, provider: ModelProvider, runner: CommandRunner, artifacts: Path
) -> AgentRuntime:
    root = _project_root()
    return AgentRuntime(
        provider=provider,
        tools=[*default_coding_tools(), RetrieveCodeTool(), ReviewerTool(provider)],
        runner=runner,
        artifacts_root=artifacts,
        context_builder=ContextBuilder(),
        skill_registry=SkillRegistry(root / "skills"),
        episodic_memory=EpisodicMemoryStore(artifacts / "episodic_memory.jsonl"),
    )


async def _run_command(args: argparse.Namespace) -> int:
    task = PublicTaskSpec.load(args.task)
    run_id = args.run_id or f"local_{uuid4().hex[:12]}"
    workspace = (args.artifacts / "workspaces" / run_id).resolve()
    if workspace.exists():
        if args.no_resume:
            raise ValueError("workspace already exists for --no-resume; choose a new --run-id")
    else:
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            task.workspace,
            workspace,
            ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache"),
        )
    task = replace(task, workspace=workspace)
    provider = LocalOpenAICompatibleProvider(LocalProviderConfig(args.base_url, args.model))
    if task.trusted_fixture:
        runner: CommandRunner = LocalTrustedRunner(trusted=True)
    else:
        if not args.sandbox_image:
            raise ValueError("untrusted tasks require --sandbox-image")
        runner = DockerSandboxRunner(DockerSandboxConfig(args.sandbox_image))
    runtime = build_runtime(provider=provider, runner=runner, artifacts=args.artifacts)
    state = await runtime.run(task, run_id=run_id, resume=not args.no_resume)
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if state.status.value == "completed" else 1


async def _scripted_demo(artifacts: Path) -> int:
    source = _project_root() / "examples" / "bugfix_demo"
    demo_id = f"demo_{uuid4().hex[:8]}"
    workspace = artifacts / "workspaces" / demo_id
    workspace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, workspace)
    task = PublicTaskSpec(
        task_id="demo_subtract",
        workspace=workspace,
        problem_statement="Fix the Python subtract function; two minus one must equal one.",
        allowed_paths=(".",),
        forbidden_paths=("verify.py",),
        visible_tests=(),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )
    responses = [
        ModelResponse(tool_calls=(ToolCall("c1", "list_files", {"path": "."}),), model="scripted"),
        ModelResponse(
            tool_calls=(ToolCall("c2", "read_file", {"path": "calculator.py"}),),
            model="scripted",
        ),
        ModelResponse(
            tool_calls=(
                ToolCall(
                    "c3",
                    "apply_patch",
                    {
                        "path": "calculator.py",
                        "old_text": "return left + right",
                        "new_text": "return left - right",
                    },
                ),
            ),
            model="scripted",
        ),
        ModelResponse(tool_calls=(ToolCall("c4", "run_tests", {}),), model="scripted"),
        ModelResponse(tool_calls=(ToolCall("c5", "git_diff", {}),), model="scripted"),
        ModelResponse(
            content='{"type":"finish","answer":"Fixed subtraction and verified it."}',
            model="scripted",
        ),
    ]
    runtime = build_runtime(
        provider=ScriptedProvider(responses),
        runner=LocalTrustedRunner(trusted=True),
        artifacts=artifacts,
    )
    state = await runtime.run(task, run_id=demo_id)
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if state.status.value == "completed" else 1


async def _mcp_demo(artifacts: Path) -> int:
    task = PublicTaskSpec(
        task_id="mcp_demo",
        workspace=_project_root(),
        problem_statement="List the RepoPilot project root.",
        allowed_paths=(".",),
        trusted_fixture=True,
    )
    context = ToolContext(task, LocalTrustedRunner(trusted=True))
    server = MCPServer(ToolRegistry([ListFilesTool()]), context)
    client = MCPClient(InProcessMCPTransport(server))
    initialized = await client.initialize()
    tools = await client.list_tools()
    result = await client.call_tool(ToolCall("mcp1", "list_files", {"path": ".", "max_depth": 1}))
    output = {
        "initialized": initialized,
        "tools": [tool.name for tool in tools],
        "result": result.to_dict(),
        "artifacts": str(artifacts),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ok else 1


async def _permission_demo() -> int:
    task = PublicTaskSpec(
        task_id="permission_demo",
        workspace=_project_root(),
        problem_statement="Demonstrate high-risk path approval.",
        trusted_fixture=True,
    )
    call = ToolCall(
        "approval1",
        "apply_patch",
        {"path": "pyproject.toml", "old_text": "x", "new_text": "y"},
    )
    decision = PolicyEngine().decide(call, ApplyPatchTool().spec, task)
    print(json.dumps({"outcome": decision.outcome.value, "reason": decision.reason}, indent=2))
    return 0 if decision.outcome.value == "require_approval" else 1


async def _recovery_demo(artifacts: Path) -> int:
    run_dir = artifacts / f"recovery_{uuid4().hex[:8]}"
    state = AgentState("recovery", "recovery_demo")
    state.pending_calls = [ToolCall("stable-action", "read_file", {"path": "README.md"})]
    checkpoint = CheckpointStore(run_dir / "checkpoint.json")
    checkpoint.save(state)
    journal = ExecutionJournal(run_dir / "journal.json")
    journal.put(ToolResult("stable-action", "read_file", True, {"content": "saved"}))
    recovered = checkpoint.load()
    replayed = journal.get("stable-action")
    output = {
        "pending_action_recovered": bool(recovered and recovered.pending_calls),
        "completed_result_replayed": bool(replayed and replayed.cached),
        "run_dir": str(run_dir),
    }
    print(json.dumps(output, indent=2))
    return (
        0
        if all(output[key] for key in ("pending_action_recovered", "completed_result_replayed"))
        else 1
    )


async def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return await _run_command(args)
    if args.kind == "scripted":
        return await _scripted_demo(args.artifacts)
    if args.kind == "mcp":
        return await _mcp_demo(args.artifacts)
    if args.kind == "permission":
        return await _permission_demo()
    return await _recovery_demo(args.artifacts)


def entrypoint(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_dispatch(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"repopilot: {error}", file=sys.stderr)
        return 2
