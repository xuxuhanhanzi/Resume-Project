from __future__ import annotations

from pathlib import Path

from repopilot.core.contracts import Permission, PolicyOutcome, ToolCall, ToolSpec
from repopilot.runtime.policy import PermissionEngine, PermissionMode
from repopilot.workspace.contracts import InteractiveTask
from repopilot.workspace.settings import ToolPermissionRules


def _task(tmp_path: Path) -> InteractiveTask:
    return InteractiveTask(task_id="t", workspace=tmp_path, problem_statement="test")


def test_explicit_user_rule_allows_interactive_execution(tmp_path: Path) -> None:
    engine = PermissionEngine(
        PermissionMode.MANUAL,
        ToolPermissionRules(allow=("run_shell",)),
    )
    decision = engine.decide(
        ToolCall("call", "run_shell", {}),
        ToolSpec("run_shell", "run", {}, permission=Permission.EXECUTE),
        _task(tmp_path),
    )

    assert decision.outcome is PolicyOutcome.ALLOW


def test_denial_rule_overrides_an_allow_rule(tmp_path: Path) -> None:
    engine = PermissionEngine(
        PermissionMode.ACCEPT_EDITS,
        ToolPermissionRules(allow=("apply_patch",), deny=("apply_patch",)),
    )
    decision = engine.decide(
        ToolCall("call", "apply_patch", {}),
        ToolSpec("apply_patch", "patch", {}, permission=Permission.WRITE),
        _task(tmp_path),
    )

    assert decision.outcome is PolicyOutcome.DENY
