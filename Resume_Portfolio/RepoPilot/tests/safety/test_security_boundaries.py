from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot.core.contracts import PolicyOutcome, ToolCall
from repopilot.runtime.policy import PermissionEngine, PermissionMode, PolicyEngine
from repopilot.runtime.runner import DockerSandboxConfig, DockerSandboxRunner, LocalTrustedRunner
from repopilot.security.paths import PathSecurityError, resolve_workspace_path
from repopilot.task import PublicTaskSpec
from repopilot.tools.coding import ApplyPatchTool
from repopilot.workspace.contracts import InteractiveTask


def _task(tmp_path: Path) -> PublicTaskSpec:
    (tmp_path / "safe.py").write_text("x = 1\n", encoding="utf-8")
    return PublicTaskSpec("t", tmp_path, "fix", trusted_fixture=True)


@pytest.mark.safety
def test_path_traversal_and_absolute_paths_are_rejected(tmp_path: Path) -> None:
    task = _task(tmp_path)

    with pytest.raises(PathSecurityError):
        resolve_workspace_path(task, "../outside.txt")
    with pytest.raises(PathSecurityError):
        resolve_workspace_path(task, str((tmp_path / "safe.py").resolve()))


@pytest.mark.safety
def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = workspace / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    task = PublicTaskSpec("symlink", workspace, "fix", trusted_fixture=True)

    with pytest.raises(PathSecurityError, match="outside the workspace"):
        resolve_workspace_path(task, "escape.txt")


@pytest.mark.safety
def test_untrusted_local_execution_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="requires Docker"):
        asyncio.run(
            LocalTrustedRunner(trusted=False).run(("python", "-V"), cwd=tmp_path, timeout_seconds=1)
        )


@pytest.mark.safety
def test_docker_command_contains_minimum_isolation(tmp_path: Path) -> None:
    runner = DockerSandboxRunner(DockerSandboxConfig("repopilot-sandbox:test"))
    command = runner.build_command(("python", "verify.py"), cwd=tmp_path, timeout_seconds=10)

    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit" in command
    assert "--user" in command
    assert any(entry.endswith("target=/workspace,readonly") for entry in command)


@pytest.mark.safety
def test_high_risk_file_requires_approval(tmp_path: Path) -> None:
    task = _task(tmp_path)
    call = ToolCall(
        "p", "apply_patch", {"path": "pyproject.toml", "old_text": "a", "new_text": "b"}
    )

    decision = PolicyEngine().decide(call, ApplyPatchTool().spec, task)

    assert decision.outcome is PolicyOutcome.REQUIRE_APPROVAL


@pytest.mark.safety
@pytest.mark.parametrize("path", ["REPOPILOT.md", "AGENTS.md", ".repopilot/mcp.json"])
def test_project_agent_control_plane_requires_approval_even_in_accept_edits(
    tmp_path: Path, path: str
) -> None:
    task = InteractiveTask("interactive", tmp_path, "Update project files")
    decision = PermissionEngine(PermissionMode.ACCEPT_EDITS).decide(
        ToolCall("control", "apply_patch", {"path": path, "old_text": "", "new_text": "x"}),
        ApplyPatchTool().spec,
        task,
    )

    assert decision.outcome is PolicyOutcome.REQUIRE_APPROVAL
