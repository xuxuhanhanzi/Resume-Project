from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot import __version__, cli
from repopilot.core.contracts import ModelResponse, Permission, PolicyOutcome, ToolCall, ToolSpec
from repopilot.extensions.plugins import discover_plugins
from repopilot.mcp.config import MCPServerConfig, load_project_mcp_config, upsert_project_mcp_config
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import PermissionEngine
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.planning import TodoStatus
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools
from repopilot.workspace.contracts import InteractiveTask


def _runtime(tmp_path: Path, responses: list[ModelResponse]) -> SessionRuntime:
    return SessionRuntime(
        provider=ScriptedProvider(responses),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
    )


def test_explicit_named_tool_requirement_rejects_an_unproven_final_answer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(content="I inspected it."),
            ModelResponse(tool_calls=(ToolCall("read", "read_file", {"path": "module.py"}),)),
            ModelResponse(content="The recorded file value is 1."),
        ]
    )
    runtime = SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "必须调用 read_file 后再回答"))

    assert result.answer == "The recorded file value is 1."
    assert provider.requests[0].required_tool == "read_file"
    assert any(
        "TRUSTED RUNTIME CORRECTION" in message.content
        for message in provider.requests[1].messages
        if message.role == "user"
    )


def test_plan_and_todo_state_survive_session_operations(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime = _runtime(tmp_path, [])
    metadata = runtime.start(project)

    metadata = runtime.draft_plan(metadata, ("inspect module", "run tests"))
    metadata = runtime.approve_plan(metadata)
    metadata = runtime.set_todo_status(metadata, 1, TodoStatus.COMPLETED)

    plan = runtime.plan(metadata)
    assert plan is not None
    assert plan.approved
    assert [item.status for item in plan.todos] == [TodoStatus.COMPLETED, TodoStatus.PENDING]


def test_plugins_are_manifest_only_and_project_overrides_user(tmp_path: Path) -> None:
    project = tmp_path / "project"
    user_root = tmp_path / "user"
    for root, version, source in (
        (user_root / "plugins" / "review", "1.0.0", "user"),
        (project / ".repopilot" / "plugins" / "review", "2.0.0", "project"),
    ):
        root.mkdir(parents=True)
        (root / "plugin.json").write_text(
            '{"name":"review","version":"' + version + '","description":"' + source + '"}',
            encoding="utf-8",
        )

    plugins = discover_plugins(project, user_root)

    assert [(plugin.name, plugin.version, plugin.source) for plugin in plugins] == [
        ("review", "2.0.0", "project-plugin")
    ]
    assert plugins[0].skills_root == project / ".repopilot" / "plugins" / "review" / "skills"


def test_http_mcp_is_limited_to_loopback_and_round_trips_config(tmp_path: Path) -> None:
    config = MCPServerConfig("local", url="http://127.0.0.1:3000/mcp")
    upsert_project_mcp_config(tmp_path, config)

    assert load_project_mcp_config(tmp_path) == (config,)
    with pytest.raises(ValueError, match="loopback HTTP"):
        MCPServerConfig("remote", url="https://example.com/mcp")


def test_subagent_and_web_actions_are_never_silently_authorized(tmp_path: Path) -> None:
    task = InteractiveTask("task", tmp_path, "review")
    engine = PermissionEngine()
    high_risk = ToolSpec("ignored", "", {}, permission=Permission.READ)

    assert engine.decide(ToolCall("agent", "start_subagent", {}), high_risk, task).outcome is (
        PolicyOutcome.REQUIRE_APPROVAL
    )
    assert (
        engine.decide(
            ToolCall("web", "web_fetch", {"url": "https://example.com/docs"}), high_risk, task
        ).outcome
        is PolicyOutcome.REQUIRE_APPROVAL
    )


def test_cli_exposes_installed_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        cli.build_parser().parse_args(["--version"])

    assert f"RepoPilot {__version__}" in capsys.readouterr().out
