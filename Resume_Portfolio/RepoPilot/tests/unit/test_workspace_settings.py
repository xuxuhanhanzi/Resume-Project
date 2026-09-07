from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.workspace.settings import ToolPermissionRules, load_tool_permission_rules


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_layered_rules_only_honor_allow_entries_from_user_controlled_sources(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "session"
    project = tmp_path / "project"
    project.mkdir()
    _write(session_root / "settings.json", {"permissions": {"allow": ["run_tests"]}})
    _write(
        project / ".repopilot" / "settings.json",
        {"permissions": {"allow": ["run_shell"], "deny": ["git_commit"]}},
    )
    _write(
        project / ".repopilot" / "settings.local.json",
        {"permissions": {"allow": ["apply_patch"], "deny": ["mcp__*"]}},
    )

    rules = load_tool_permission_rules(session_root=session_root, project_root=project)

    assert rules.allows("run_tests")
    assert rules.allows("apply_patch")
    assert not rules.allows("run_shell")
    assert rules.denies("git_commit")
    assert rules.denies("mcp__server__tool")


def test_cli_rules_have_highest_precedence_and_denial_wins(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    rules = load_tool_permission_rules(
        session_root=tmp_path / "session",
        project_root=project,
        command_allow=("run_shell",),
        command_deny=("run_shell",),
    )

    assert rules.allows("run_shell")
    assert rules.denies("run_shell")


def test_rules_reject_unsafe_pattern_characters() -> None:
    with pytest.raises(ValueError, match="glob characters"):
        ToolPermissionRules(allow=("run_shell/anything",))
