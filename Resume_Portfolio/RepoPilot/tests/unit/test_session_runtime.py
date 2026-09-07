from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import pytest

from repopilot.core.contracts import (
    AgentState,
    Message,
    ModelResponse,
    Permission,
    PolicyOutcome,
    ToolCall,
    ToolSpec,
)
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import PermissionEngine, PermissionMode, StaticApprovalHandler
from repopilot.runtime.runner import ExecutionResult, LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools
from repopilot.workspace.contracts import InteractiveTask
from repopilot.workspace.project import ProjectWorkspace


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    async def run(self, command: Sequence[str], **_: object) -> ExecutionResult:
        normalized = tuple(command)
        self.commands.append(normalized)
        return ExecutionResult(normalized, 0, "1 passed\n", "")


def test_project_discovery_and_session_resume(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    discovered = ProjectWorkspace.discover(project)
    assert discovered.languages == ("python",)

    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=ScriptedProvider(
            [
                ModelResponse(tool_calls=(ToolCall("read", "read_file", {"path": "module.py"}),)),
                ModelResponse(content='{"type":"finish","answer":"VALUE is one."}'),
            ]
        ),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    metadata = runtime.start(project)
    result = asyncio.run(runtime.run_turn(metadata, "Explain module.py"))
    assert result.answer == "VALUE is one."
    assert result.state.tool_calls == 1
    transcript = store.session_dir(project, metadata.session_id) / "transcript.jsonl"
    events = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert any(event["kind"] == "tool_call_completed" for event in events)

    resumed = SessionRuntime(
        provider=ScriptedProvider([ModelResponse(content="Still available.")]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    loaded = store.load(project_root=project, session_id=metadata.session_id)
    continued = asyncio.run(resumed.run_turn(loaded, "Continue."))
    assert continued.answer == "Still available."
    assert len(continued.state.messages) >= 6


def test_session_baseline_ignores_reference_code(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    reference = project / "Reference code"
    reference.mkdir()
    (reference / "upstream.py").write_text("VALUE = 2\n", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )

    metadata = runtime.start(project)
    baseline = json.loads(store.baseline_path(metadata).read_text(encoding="utf-8"))

    assert baseline == {"module.py": "VALUE = 1\n"}


def test_interactive_session_supplies_the_project_test_command(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    runner = RecordingRunner()
    runtime = SessionRuntime(
        provider=ScriptedProvider(
            [
                ModelResponse(tool_calls=(ToolCall("test", "run_tests", {}),)),
                ModelResponse(content='{"type":"finish","answer":"Tests passed."}'),
            ]
        ),
        tools=default_coding_tools(),
        runner=runner,
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    metadata = runtime.start(project)
    result = asyncio.run(runtime.run_turn(metadata, "Check tests"))

    assert result.answer == "Tests passed."
    assert runner.commands == [(sys.executable, "-m", "pytest")]


def test_coding_turn_persists_evidence_backed_workflow_summary(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    runner = RecordingRunner()
    runtime = SessionRuntime(
        provider=ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "edit",
                            "apply_patch",
                            {
                                "path": "module.py",
                                "old_text": "VALUE = 1",
                                "new_text": "VALUE = 2",
                            },
                        ),
                    )
                ),
                ModelResponse(tool_calls=(ToolCall("tests", "run_tests", {}),)),
                ModelResponse(content="The change is complete."),
            ]
        ),
        tools=default_coding_tools(),
        runner=runner,
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "Update the value and test it."))

    assert result.answer == "The change is complete."
    assert result.state.workflow_stage == "answer"
    assert result.workflow.changed_files == ("module.py",)
    assert [(check.label, check.ok) for check in result.workflow.checks] == [("run_tests", True)]
    assert "module.py" in result.workflow.render()
    assert "passed: run_tests" in result.workflow.render()
    assert runner.commands == [(sys.executable, "-m", "pytest")]


def test_recoverable_patch_conflict_requires_a_fresh_standalone_read(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "conflict",
                        "apply_patch",
                        {
                            "path": "module.py",
                            "old_text": "VALUE = 0",
                            "new_text": "VALUE = 2",
                        },
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "unsafe-retry",
                        "apply_patch",
                        {
                            "path": "module.py",
                            "old_text": "VALUE = 1",
                            "new_text": "VALUE = 2",
                        },
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(ToolCall("read-current", "read_file", {"path": "module.py"}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "safe-retry",
                        "apply_patch",
                        {
                            "path": "module.py",
                            "old_text": "VALUE = 1",
                            "new_text": "VALUE = 2",
                        },
                    ),
                )
            ),
            ModelResponse(content="Patched after reading the current file."),
        ]
    )
    runtime = SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "Update the value."))

    assert result.answer == "Patched after reading the current file."
    assert (project / "module.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert all(
        call.call_id != "unsafe-retry"
        for request in provider.requests
        for message in request.messages
        for call in message.tool_calls
    )
    assert any(
        "recoverable apply_patch conflict" in message.content
        for request in provider.requests
        for message in request.messages
    )


def test_interactive_session_reports_an_actionable_empty_model_failure(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime = SessionRuntime(
        provider=ScriptedProvider([ModelResponse(content="")]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "Summarize the project."))

    assert result.answer == (
        "Model provider failed: model returned no visible final text; retry the request"
    )


def test_session_preserves_provider_reasoning_through_one_tool_continuation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("read", "read_file", {"path": "module.py"}),),
                reasoning_content="opaque-provider-continuation",
            ),
            ModelResponse(content="Finished."),
        ]
    )
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = SessionRuntime(
        provider=provider,
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="deepseek-v4-flash",
        provider_label="deepseek/deepseek-v4-flash",
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "Read module.py"))

    assert result.answer == "Finished."
    continuation = provider.requests[1]
    assistant_messages = [
        message
        for message in continuation.messages
        if message.role == "assistant" and message.tool_calls
    ]
    assert assistant_messages[0].reasoning_content == "opaque-provider-continuation"
    assert any(
        "deepseek/deepseek-v4-flash" in message.content
        for message in continuation.messages
        if message.role == "user"
    )


def test_interactive_permission_modes_protect_edits_and_secrets(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    task = InteractiveTask("interactive", workspace, "Inspect the project")
    edit = ToolSpec("apply_patch", "", {}, permission=Permission.WRITE, read_only=False)
    read = ToolSpec("read_file", "", {})
    manual = PermissionEngine()
    edit_decision = manual.decide(
        ToolCall("edit", "apply_patch", {"path": "module.py"}), edit, task
    )
    assert edit_decision.outcome is PolicyOutcome.REQUIRE_APPROVAL
    assert manual.decide(ToolCall("secret", "read_file", {"path": ".env"}), read, task).outcome is (
        PolicyOutcome.DENY
    )
    plan = PermissionEngine(PermissionMode.PLAN)
    plan_decision = plan.decide(ToolCall("edit", "apply_patch", {"path": "module.py"}), edit, task)
    assert plan_decision.outcome is PolicyOutcome.DENY


def test_session_fork_preserves_source_state_and_continues_independently(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "sessions")
    original_runtime = SessionRuntime(
        provider=ScriptedProvider([ModelResponse(content="Original answer.")]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    source = original_runtime.start(project)
    original_turn = asyncio.run(original_runtime.run_turn(source, "Describe the project."))
    fork = store.fork(original_turn.metadata)
    assert fork.session_id != source.session_id
    source_state = store.checkpoint(original_turn.metadata).load()
    fork_state = store.checkpoint(fork).load()
    assert source_state is not None
    assert fork_state is not None
    assert [message.content for message in fork_state.messages] == [
        message.content for message in source_state.messages
    ]

    child_runtime = SessionRuntime(
        provider=ScriptedProvider([ModelResponse(content="Forked answer.")]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    child_turn = asyncio.run(child_runtime.run_turn(fork, "Continue in the fork."))
    assert child_turn.answer == "Forked answer."
    unchanged_source = store.checkpoint(original_turn.metadata).load()
    assert unchanged_source is not None
    assert len(unchanged_source.messages) == len(source_state.messages)
    assert len(child_turn.state.messages) > len(unchanged_source.messages)


def test_session_store_rejects_path_like_ids_and_serializes_one_writer(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create(
        project_root=project,
        model="scripted",
        permission_mode=PermissionMode.MANUAL,
    )

    with pytest.raises(ValueError, match="session_id"):
        store.session_dir(project, "../another-project")
    with ExitStack() as stack:
        stack.enter_context(store.lease(metadata))
        with pytest.raises(RuntimeError, match="already open"):
            stack.enter_context(store.lease(metadata))
    with store.lease(metadata):
        pass


def test_session_store_lists_resolves_titles_and_keeps_explicit_project_memory(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "sessions")
    first = store.create(
        project_root=project,
        model="deepseek-v4-flash",
        permission_mode=PermissionMode.MANUAL,
        provider="deepseek",
    )
    renamed = store.set_title(first, "investigate parser")
    state = AgentState(renamed.session_id, "session")
    state.messages.append(Message("user", "Inspect the parser."))
    store.checkpoint(renamed).save(state)
    store.append_message(renamed, Message("assistant", "I will inspect it."))
    second = store.create(
        project_root=project,
        model="qwen-plus",
        permission_mode=PermissionMode.MANUAL,
        provider="qwen",
    )
    second = store.append_runtime_event(second, kind="session_touched", data={})

    sessions = store.list(project)
    assert sessions[0].session_id == second.session_id
    assert store.resolve(project, renamed.session_id[:8]).title == "investigate parser"
    assert store.resolve(project, "investigate parser").session_id == renamed.session_id
    assert store.recent_user_messages(renamed) == ("Inspect the parser.",)

    memory = store.remember_project_fact(project, "Tests use pytest from the virtual environment.")
    assert memory.facts == ["Tests use pytest from the virtual environment."]
    assert store.load_project_memory(project).facts == memory.facts


def test_session_transcript_redacts_common_credentials(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "sessions")
    metadata = store.create(
        project_root=project,
        model="scripted",
        permission_mode=PermissionMode.MANUAL,
    )
    raw_token = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    store.append_runtime_event(
        metadata,
        kind="tool_call_completed",
        data={"stdout": f"api_key=super-secret-value token={raw_token}"},
    )

    transcript = (store.session_dir(project, metadata.session_id) / "transcript.jsonl").read_text(
        encoding="utf-8"
    )
    assert "super-secret-value" not in transcript
    assert raw_token not in transcript
    assert "[REDACTED]" in transcript
