from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from repopilot.core.contracts import Permission, PolicyOutcome, ToolCall
from repopilot.process.manager import ProcessManager, ProcessSnapshot
from repopilot.runtime.policy import PermissionEngine, PermissionMode
from repopilot.runtime.runner import ExecutionResult
from repopilot.tools.base import ToolContext
from repopilot.tools.git import GitCommitTool, GitShowTool, GitStageTool, GitStatusTool
from repopilot.tools.process import (
    BackgroundStatusTool,
    StartBackgroundTool,
    TerminateBackgroundTool,
)
from repopilot.tools.shell import ShellTool
from repopilot.tools.worktree import CreateWorktreeTool, ListWorktreesTool
from repopilot.workspace.contracts import InteractiveTask


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []

    async def run(
        self, command: Sequence[str], *, cwd: Path, timeout_seconds: float, **_: object
    ) -> ExecutionResult:
        normalized = tuple(command)
        self.calls.append((normalized, cwd, timeout_seconds))
        return ExecutionResult(normalized, 0, "observed\n", "")


def test_interactive_shell_is_approved_but_benchmark_execution_is_unchanged(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    task = InteractiveTask("interactive", workspace, "Run a narrow test command")
    shell_spec = ShellTool().spec
    call = ToolCall("shell", "run_shell", {"command": ["pytest", "-q"]})
    assert (
        PermissionEngine().decide(call, shell_spec, task).outcome is PolicyOutcome.REQUIRE_APPROVAL
    )
    assert (
        PermissionEngine(PermissionMode.PLAN).decide(call, shell_spec, task).outcome
        is PolicyOutcome.DENY
    )
    assert shell_spec.permission is Permission.EXECUTE


def test_shell_tool_uses_tokenized_runner_and_rejects_newlines(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    task = InteractiveTask("interactive", workspace, "Execute a command")
    runner = RecordingRunner()
    context = ToolContext(task, runner)
    tool = ShellTool()
    result = asyncio.run(
        tool.run(ToolCall("shell", "run_shell", {"command": ["python", "-V"]}), context)
    )
    assert result.ok
    assert runner.calls == [(("python", "-V"), workspace.resolve(), 120.0)]

    rejected = asyncio.run(
        tool.run(ToolCall("bad", "run_shell", {"command": ["python", "bad\narg"]}), context)
    )
    assert not rejected.ok
    assert rejected.error_type is not None


def test_git_tools_are_read_only_and_validate_refs(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    task = InteractiveTask("interactive", workspace, "Inspect Git")
    runner = RecordingRunner()
    context = ToolContext(task, runner)
    status = asyncio.run(GitStatusTool().run(ToolCall("status", "git_status", {}), context))
    assert status.ok
    assert runner.calls[0][0] == ("git", "status", "--short", "--branch")
    assert GitStatusTool().spec.read_only

    invalid_ref = asyncio.run(
        GitShowTool().run(ToolCall("show", "git_show", {"ref": "--upload-pack=x"}), context)
    )
    assert not invalid_ref.ok


def test_git_write_tools_are_explicit_and_keep_secret_paths_protected(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (workspace / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    task = InteractiveTask("interactive", workspace, "Prepare a focused commit")
    runner = RecordingRunner()
    context = ToolContext(task, runner)
    staged = asyncio.run(
        GitStageTool().run(ToolCall("stage", "git_stage", {"paths": ["module.py"]}), context)
    )
    assert staged.ok
    assert runner.calls[0][0] == ("git", "add", "--", "module.py")
    assert (
        PermissionEngine()
        .decide(ToolCall("secret", "git_stage", {"paths": [".env"]}), GitStageTool().spec, task)
        .outcome
        is PolicyOutcome.DENY
    )
    assert (
        PermissionEngine()
        .decide(
            ToolCall("commit", "git_commit", {"message": "Focus parser fix"}),
            GitCommitTool().spec,
            task,
        )
        .outcome
        is PolicyOutcome.REQUIRE_APPROVAL
    )


def test_process_manager_captures_and_terminates_one_child(tmp_path: Path) -> None:
    async def exercise() -> None:
        manager = ProcessManager(tmp_path)
        completed = await manager.start((sys.executable, "-c", "print('ready')"))
        snapshot = await _eventually_finished(manager, completed.process_id)
        assert snapshot.return_code == 0
        assert "ready" in snapshot.stdout

        sleeping = await manager.start((sys.executable, "-c", "import time; time.sleep(20)"))
        stopped = await manager.terminate(sleeping.process_id)
        assert stopped.return_code is not None
        assert not stopped.running

    asyncio.run(exercise())


def test_background_tools_use_session_scoped_manager_and_plan_mode_blocks_termination(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        workspace = tmp_path / "project"
        workspace.mkdir()
        task = InteractiveTask("interactive", workspace, "Start a local development server")
        manager = ProcessManager(workspace)
        context = ToolContext(task, RecordingRunner(), process_manager=manager)
        started = await StartBackgroundTool().run(
            ToolCall(
                "start",
                "start_background",
                {"command": [sys.executable, "-c", "import time; time.sleep(20)"]},
            ),
            context,
        )
        assert started.ok
        assert isinstance(started.data, dict)
        process_id = str(started.data["process_id"])
        status = await BackgroundStatusTool().run(
            ToolCall("status", "background_status", {"process_id": process_id}), context
        )
        assert status.ok
        terminated = await TerminateBackgroundTool().run(
            ToolCall("stop", "terminate_background", {"process_id": process_id}), context
        )
        assert terminated.ok
        decision = PermissionEngine(PermissionMode.PLAN).decide(
            ToolCall("plan-stop", "terminate_background", {"process_id": process_id}),
            TerminateBackgroundTool().spec,
            task,
        )
        assert decision.outcome is PolicyOutcome.DENY

    asyncio.run(exercise())


def test_worktree_creation_is_high_risk_and_workspace_contained(tmp_path: Path) -> None:
    task = InteractiveTask("interactive", tmp_path, "Create an isolated implementation branch")
    runner = RecordingRunner()
    context = ToolContext(task, runner)
    tool = CreateWorktreeTool()
    result = asyncio.run(
        tool.run(
            ToolCall("worktree", "create_worktree", {"name": "parser-fix", "branch": "fix/parser"}),
            context,
        )
    )
    assert result.ok
    assert runner.calls[0][0][:5] == ("git", "worktree", "add", "-b", "fix/parser")
    assert (
        PermissionEngine()
        .decide(
            ToolCall("worktree", "create_worktree", {"name": "parser-fix", "branch": "fix/parser"}),
            tool.spec,
            task,
        )
        .outcome
        is PolicyOutcome.REQUIRE_APPROVAL
    )


def test_worktree_listing_is_read_only_and_hides_external_checkouts(tmp_path: Path) -> None:
    task = InteractiveTask("interactive", tmp_path, "Inspect isolated worktrees")

    class WorktreeRunner(RecordingRunner):
        async def run(
            self, command: Sequence[str], *, cwd: Path, timeout_seconds: float, **_: object
        ) -> ExecutionResult:
            normalized = tuple(command)
            self.calls.append((normalized, cwd, timeout_seconds))
            return ExecutionResult(
                normalized,
                0,
                f"worktree {tmp_path}\nbranch refs/heads/main\n\n"
                f"worktree {tmp_path / '.repopilot' / 'worktrees' / 'fix'}\n"
                "branch refs/heads/fix/parser\n\n"
                "worktree C:/unrelated/checkout\nbranch refs/heads/private\n",
                "",
            )

    runner = WorktreeRunner()
    tool = ListWorktreesTool()
    result = asyncio.run(
        tool.run(ToolCall("list", "list_worktrees", {}), ToolContext(task, runner))
    )
    assert result.ok
    assert runner.calls[0][0] == ("git", "worktree", "list", "--porcelain")
    assert result.data["worktrees"] == [
        {"path": ".", "branch": "main"},
        {"path": ".repopilot/worktrees/fix", "branch": "fix/parser"},
    ]
    assert tool.spec.read_only


async def _eventually_finished(manager: ProcessManager, process_id: str) -> ProcessSnapshot:
    for _ in range(20):
        snapshot = await manager.snapshot(process_id)
        if not snapshot.running:
            return snapshot
        await asyncio.sleep(0.05)
    raise AssertionError("background process did not finish in time")
