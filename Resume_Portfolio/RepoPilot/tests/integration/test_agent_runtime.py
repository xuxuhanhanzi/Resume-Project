from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from repopilot.core.budgets import RunBudget
from repopilot.core.contracts import AgentState, ModelResponse, RunStatus, ToolCall, ToolResult
from repopilot.core.loop import AgentRuntime
from repopilot.providers.scripted import ScriptedProvider
from repopilot.retrieval.tool import RetrieveCodeTool
from repopilot.runtime.checkpoint import CheckpointStore, ExecutionJournal
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.skills.registry import SkillRegistry
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import capture_text_snapshot, default_coding_tools


def _copy_demo(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[2] / "examples" / "bugfix_demo"
    workspace = tmp_path / "workspace"
    shutil.copytree(source, workspace)
    return workspace


def _task(workspace: Path) -> PublicTaskSpec:
    return PublicTaskSpec(
        "subtract-demo",
        workspace,
        "Fix the Python subtract bug",
        allowed_paths=("calculator.py",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )


def _runtime(tmp_path: Path, provider: ScriptedProvider) -> AgentRuntime:
    project = Path(__file__).resolve().parents[2]
    return AgentRuntime(
        provider=provider,
        tools=[*default_coding_tools(), RetrieveCodeTool()],
        runner=LocalTrustedRunner(trusted=True),
        artifacts_root=tmp_path / "artifacts",
        skill_registry=SkillRegistry(project / "skills"),
    )


@pytest.mark.integration
def test_scripted_agent_completes_coding_loop(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("c1", "search_text", {"pattern": "subtract"}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "c2",
                        "apply_patch",
                        {
                            "path": "calculator.py",
                            "old_text": "return left + right",
                            "new_text": "return left - right",
                        },
                    ),
                )
            ),
            ModelResponse(tool_calls=(ToolCall("c3", "run_tests", {}),)),
            ModelResponse(content='{"type":"finish","answer":"fixed and verified"}'),
        ]
    )

    state = asyncio.run(_runtime(tmp_path, provider).run(_task(workspace), run_id="complete"))

    assert state.status is RunStatus.COMPLETED
    assert state.iteration == 4
    assert state.tool_calls == 3
    assert all(
        '"ok": true' in message.content for message in state.messages if message.role == "tool"
    )
    assert "return left - right" in (workspace / "calculator.py").read_text(encoding="utf-8")
    events = (tmp_path / "artifacts" / "complete" / "events.jsonl").read_text(encoding="utf-8")
    assert "verification_finished" in events
    assert "run_finished" in events


@pytest.mark.integration
def test_pending_side_effect_is_replayed_after_recovery(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    task = _task(workspace)
    run_dir = tmp_path / "artifacts" / "recover"
    run_dir.mkdir(parents=True)
    baseline = capture_text_snapshot(workspace)
    (run_dir / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False), encoding="utf-8"
    )
    original = (workspace / "calculator.py").read_text(encoding="utf-8")
    (workspace / "calculator.py").write_text(
        original.replace("return left + right", "return left - right"), encoding="utf-8"
    )
    pending = ToolCall(
        "stable-patch",
        "apply_patch",
        {
            "path": "calculator.py",
            "old_text": "return left + right",
            "new_text": "return left - right",
        },
    )
    state = AgentState("recover", task.task_id, status=RunStatus.RUNNING, pending_calls=[pending])
    CheckpointStore(run_dir / "checkpoint.json").save(state)
    ExecutionJournal(run_dir / "tool_journal.json").put(
        ToolResult("stable-patch", "apply_patch", True, {"path": "calculator.py"}, side_effect=True)
    )
    provider = ScriptedProvider([ModelResponse(content='{"type":"finish","answer":"resumed"}')])

    recovered = asyncio.run(_runtime(tmp_path, provider).run(task, run_id="recover", resume=True))

    assert recovered.status is RunStatus.COMPLETED
    assert recovered.tool_calls == 0
    assert len(provider.requests) == 1
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "tool_result_replayed" in events


@pytest.mark.integration
def test_budget_stops_provider_that_never_finishes(tmp_path: Path) -> None:
    workspace = _copy_demo(tmp_path)
    task = PublicTaskSpec(
        "bounded",
        workspace,
        "Fix",
        allowed_paths=("calculator.py",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
        budget=RunBudget(max_iterations=2),
    )
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("a", "read_file", {"path": "calculator.py"}),)),
            ModelResponse(tool_calls=(ToolCall("b", "read_file", {"path": "calculator.py"}),)),
        ]
    )

    state = asyncio.run(_runtime(tmp_path, provider).run(task, run_id="bounded"))

    assert state.status is RunStatus.FAILED
    assert state.failure_reason == "iteration budget exceeded"
