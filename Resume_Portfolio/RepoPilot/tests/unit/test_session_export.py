from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.core.contracts import AgentState, Message
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore


def _runtime(store: SessionStore) -> SessionRuntime:
    return SessionRuntime(
        provider=ScriptedProvider([]),
        tools=[],
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="deepseek-v4-flash",
        provider_label="deepseek/deepseek-v4-flash",
    )


def test_session_export_is_redacted_local_and_omits_tool_payloads(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SessionStore(tmp_path / "state")
    runtime = _runtime(store)
    metadata = runtime.start(project)
    raw_key = "sk-abcdefghijklmnopqrst"
    state = AgentState(metadata.session_id, "session")
    state.messages.extend(
        (
            Message("user", f"Please inspect API_KEY={raw_key}"),
            Message("assistant", "I will inspect the project."),
            Message("tool", "tool internal payload", name="read_file"),
        )
    )
    store.checkpoint(metadata).save(state)

    preview = runtime.export_preview(metadata, filename="review.md")

    assert preview.target.name == "review.md"
    assert preview.target.parent == store.session_dir(project, metadata.session_id) / "exports"
    assert raw_key not in preview.content
    assert "Tool payload omitted from export." in preview.content
    assert "tool internal payload" not in preview.content
    assert not preview.target.exists()

    exported = runtime.export_session(metadata, filename="review.md")

    assert exported.preview.target.read_text(encoding="utf-8") == preview.content
    events = store.runtime_events(exported.metadata)
    assert any(event["kind"] == "session_exported" for event in events)
    with pytest.raises(ValueError, match="already exists"):
        runtime.export_preview(exported.metadata, filename="review.md")


def test_session_export_refuses_paths_and_non_text_extensions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    runtime = _runtime(SessionStore(tmp_path / "state"))
    metadata = runtime.start(project)

    with pytest.raises(ValueError, match="plain filename"):
        runtime.export_preview(metadata, filename="../outside.md")
    with pytest.raises(ValueError, match=".md or .txt"):
        runtime.export_preview(metadata, filename="conversation.exe")
