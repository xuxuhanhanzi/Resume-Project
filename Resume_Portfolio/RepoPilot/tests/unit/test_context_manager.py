from __future__ import annotations

import asyncio
import json
from pathlib import Path

from repopilot.context.builder import ContextBuilder
from repopilot.context.manager import ContextManager
from repopilot.core.contracts import AgentState, Message, ModelResponse, ToolCall
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools
from repopilot.workspace.contracts import InteractiveTask


def test_context_manager_prunes_tool_output_and_preserves_recent_messages() -> None:
    manager = ContextManager(
        max_history_messages=3,
        max_context_characters=1_000,
        max_tool_output_characters=80,
    )
    messages = [
        Message("user", "earlier goal"),
        Message("tool", "x" * 500, name="run_shell"),
        Message("assistant", "recent conclusion"),
        Message("user", "latest request"),
    ]
    projection = manager.project(messages, summary="carry this forward")
    assert [message.role for message in projection.messages] == ["tool", "assistant", "user"]
    assert [message.content for message in projection.messages[1:]] == [
        "recent conclusion",
        "latest request",
    ]
    assert len(projection.messages[0].content) <= 80
    assert projection.usage.pruned_tool_messages == 1
    assert projection.summary == "carry this forward"


def test_context_compaction_records_only_structured_tool_outcomes() -> None:
    manager = ContextManager()
    summary = manager.compact(
        [
            Message("user", "Fix the failing parser."),
            Message(
                "tool",
                json.dumps({"tool_name": "run_shell", "ok": False, "error": "tests failed"}),
                name="run_shell",
            ),
            Message("assistant", "The parser rejects empty input."),
        ]
    )
    assert "Fix the failing parser." in summary
    assert "run_shell: failed (tests failed)" in summary
    assert "The parser rejects empty input." in summary


def test_context_projection_keeps_only_complete_native_tool_exchanges() -> None:
    requested = Message(
        "assistant",
        "Requested tools: read_file",
        tool_calls=(ToolCall("call-1", "read_file", {"path": "module.py"}),),
    )
    result = Message(
        "tool",
        '{"ok": true}',
        name="read_file",
        tool_call_id="call-1",
    )
    manager = ContextManager(max_history_messages=3, max_context_characters=1_000)
    complete = manager.project([requested, result, Message("user", "continue")])
    assert [message.role for message in complete.messages] == ["assistant", "tool", "user"]
    assert complete.messages[0].tool_calls[0].call_id == "call-1"

    clipped = ContextManager(max_history_messages=2, max_context_characters=1_000).project(
        [requested, result, Message("user", "continue")]
    )
    assert [message.role for message in clipped.messages] == ["user"]
    assert clipped.usage.pruned_tool_messages == 1


def test_session_compact_persists_summary_without_discarding_transcript(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")
    runtime = SessionRuntime(
        provider=ScriptedProvider([ModelResponse(content="Completed inspection.")]),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
    )
    metadata = runtime.start(project)
    turn = asyncio.run(runtime.run_turn(metadata, "Inspect module.py"))
    compacted = runtime.compact(turn.metadata)
    assert "Inspect module.py" in compacted.summary
    assert compacted.usage.summary_characters == len(compacted.summary)
    context_path = store.session_dir(project, metadata.session_id) / "context.json"
    assert json.loads(context_path.read_text(encoding="utf-8"))["summary"] == compacted.summary
    state = store.checkpoint(compacted.metadata).load()
    assert state is not None
    assert len(state.messages) == len(turn.state.messages)


def test_context_builder_marks_a_persisted_handoff_as_an_explicit_compaction(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    task = InteractiveTask("session", project, "Continue a session")

    request = ContextBuilder().build(
        task=task,
        state=AgentState("session", "continue"),
        tools=(),
        session_summary="A focused handoff summary.",
    )

    assert "user explicitly compacted this session" in request.messages[1].content
