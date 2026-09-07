from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from repopilot.context.builder import ContextBuilder
from repopilot.core.contracts import AgentState, ModelResponse, ToolCall
from repopilot.core.events import RuntimeEvent, RuntimeEventKind
from repopilot.extensions.hooks import HookDispatcher
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools
from repopilot.verification.engine import VerificationCommand
from repopilot.workspace.contracts import InteractiveTask
from repopilot.workspace.instructions import load_project_instructions, project_instruction_template
from repopilot.workspace.project import ProjectWorkspace
from repopilot.workspace.trust import WorkspaceTrustStore


def test_project_workspace_loads_only_bounded_instruction_locations(tmp_path: Path) -> None:
    (tmp_path / "REPOPILOT.md").write_text("Use focused tests.", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Never alter generated files.", encoding="utf-8")
    rules = tmp_path / ".repopilot" / "rules"
    rules.mkdir(parents=True)
    (rules / "python.md").write_text("Run ruff for Python edits.", encoding="utf-8")
    workspace = ProjectWorkspace.discover(tmp_path)
    assert [path for path, _ in workspace.instructions.documents] == [
        "REPOPILOT.md",
        "AGENTS.md",
        ".repopilot/rules/python.md",
    ]
    task = InteractiveTask("session", tmp_path, "Explain the project")
    request = ContextBuilder().build(
        task=task,
        state=AgentState("session", "run"),
        tools=(),
        project_instructions=workspace.instructions.render(),
    )
    assert "untrusted context" in request.messages[1].content
    assert "Use focused tests." in request.messages[1].content


def test_instruction_loader_supports_local_claude_compatibility_and_bounded_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "REPOPILOT.md").write_text("@docs/workflow.md\nNative rule.", encoding="utf-8")
    (tmp_path / "REPOPILOT.local.md").write_text("Personal rule.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Compatible rule.", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "workflow.md").write_text("Imported rule.", encoding="utf-8")
    (tmp_path / "outside.md").write_text("Must not be loaded.", encoding="utf-8")
    (docs / "escape.md").write_text("@../outside.md", encoding="utf-8")

    instructions = load_project_instructions(tmp_path)

    assert [path for path, _ in instructions.documents] == [
        "REPOPILOT.md",
        "docs/workflow.md",
        "REPOPILOT.local.md",
        "CLAUDE.md",
    ]
    assert "Imported rule." in instructions.render()
    assert "Must not be loaded." not in instructions.render()


def test_instruction_loader_layers_user_and_repository_hierarchy(tmp_path: Path) -> None:
    user_state = tmp_path / "user-state"
    user_state.mkdir()
    (user_state / "REPOPILOT.md").write_text("User-owned defaults.", encoding="utf-8")
    repository = tmp_path / "repository"
    nested = repository / "apps" / "cli"
    nested.mkdir(parents=True)
    (repository / ".git").mkdir()
    (repository / "REPOPILOT.md").write_text("Repository rules.", encoding="utf-8")
    (nested / "CLAUDE.md").write_text("Nested compatibility rules.", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "AGENTS.md").write_text("Must not inherit from outside.", encoding="utf-8")

    instructions = load_project_instructions(
        nested,
        user_instructions_root=user_state,
    )

    assert [path for path, _content in instructions.documents] == [
        "~/.repopilot/REPOPILOT.md",
        "REPOPILOT.md",
        "apps/cli/CLAUDE.md",
    ]
    rendered = instructions.render()
    assert "User-owned defaults." in rendered
    assert "Repository rules." in rendered
    assert "Nested compatibility rules." in rendered
    assert "Must not inherit from outside." not in rendered


def test_project_instruction_template_contains_detected_markers(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'", encoding="utf-8")

    template = project_instruction_template(tmp_path)

    assert template.startswith("# RepoPilot project instructions")
    assert "pyproject.toml" in template


def test_workspace_trust_is_scoped_to_one_canonical_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = WorkspaceTrustStore(tmp_path / "state")
    assert not store.is_trusted(project)
    record = store.trust(project)
    assert record.project_root == project.resolve()
    assert store.is_trusted(project)
    assert not store.is_trusted(other)


def test_project_skill_is_deferred_then_persistently_activated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    skill_dir = project / ".repopilot" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: review a focused patch\nallowed_tools:\n  - git_diff\n"
        "---\nAlways identify regression risks.",
        encoding="utf-8",
    )
    provider = ScriptedProvider([ModelResponse(content="Reviewed.")])
    runtime = SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
    )
    metadata = runtime.start(project)
    assert runtime.available_skills(metadata) == (("review", "review a focused patch"),)
    metadata = runtime.activate_skills(metadata, ("review",))
    result = asyncio.run(runtime.run_turn(metadata, "Explain a small change."))
    assert result.answer == "Reviewed."
    assert "SKILL review; declared tools: git_diff" in provider.requests[0].messages[1].content
    assert runtime.store.load_active_skills(result.metadata) == ("review",)


def test_declarative_hook_is_audited_without_executing_project_code(tmp_path: Path) -> None:
    project = tmp_path / "project"
    hooks = project / ".repopilot"
    hooks.mkdir(parents=True)
    (hooks / "hooks.json").write_text(
        '{"hooks": [{"event": "post_tool_use", "tools": ["list_files"], '
        '"message": "Inspect the diff before declaring success."}]}',
        encoding="utf-8",
    )
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("files", "list_files", {"path": "."}),)),
            ModelResponse(content="Observed files."),
        ]
    )
    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    metadata = runtime.start(project)
    result = asyncio.run(runtime.run_turn(metadata, "Inspect the project."))
    transcript = store.session_dir(project, result.metadata.session_id) / "transcript.jsonl"
    assert "hook_triggered" in transcript.read_text(encoding="utf-8")


def test_verification_completed_hook_is_dispatched_and_audited(tmp_path: Path) -> None:
    project = tmp_path / "project"
    hooks = project / ".repopilot"
    hooks.mkdir(parents=True)
    (hooks / "hooks.json").write_text(
        '{"hooks": [{"event": "verification_completed", '
        '"message": "Review the verification evidence."}]}',
        encoding="utf-8",
    )
    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
        approval=StaticApprovalHandler(True),
    )
    metadata = runtime.start(project)
    verified = asyncio.run(
        runtime.verify(
            metadata,
            (VerificationCommand("python no-op", (sys.executable, "-c", "pass")),),
        )
    )

    transcript = (
        store.session_dir(project, verified.metadata.session_id) / "transcript.jsonl"
    ).read_text(encoding="utf-8")
    assert '"event": "verification_completed"' in transcript
    assert "Review the verification evidence." in transcript


def test_hook_matcher_and_session_lifecycle_are_audited_without_execution(tmp_path: Path) -> None:
    project = tmp_path / "project"
    hooks = project / ".repopilot"
    hooks.mkdir(parents=True)
    (hooks / "hooks.json").write_text(
        '{"hooks": ['
        '{"event": "session_start", "message": "Read local rules."}, '
        '{"event": "session_stop", "message": "Record the handoff."}, '
        '{"event": "pre_tool_use", "matcher": "read_*", '
        '"message": "Keep reads focused."}'
        "]}",
        encoding="utf-8",
    )
    dispatcher = HookDispatcher.load(project)
    matching = dispatcher.dispatch(
        RuntimeEvent(RuntimeEventKind.TOOL_CALL_PROPOSED, {"tool": "read_file"})
    )
    non_matching = dispatcher.dispatch(
        RuntimeEvent(RuntimeEventKind.TOOL_CALL_PROPOSED, {"tool": "list_files"})
    )
    assert [notice.message for notice in matching] == ["Keep reads focused."]
    assert non_matching == ()

    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    metadata = runtime.start(project)
    assert [notice.event for notice in runtime.hook_notices(metadata)] == [
        "session_start",
        "session_stop",
        "pre_tool_use",
    ]
    asyncio.run(runtime.aclose())
    transcript = (store.session_dir(project, metadata.session_id) / "transcript.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"event": "session_start"' in transcript
    assert '"event": "session_stop"' in transcript
