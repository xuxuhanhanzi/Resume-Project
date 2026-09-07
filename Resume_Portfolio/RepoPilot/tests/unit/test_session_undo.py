from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


def test_session_undo_restores_the_latest_unchanged_agent_edit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
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
                ModelResponse(content="Updated the value."),
            ]
        ),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    metadata = runtime.start(project)
    result = asyncio.run(runtime.run_turn(metadata, "Update the value."))

    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    assert runtime.undo_last_edit(result.metadata) == (
        "Reverted RepoPilot's latest edit to module.py."
    )
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_session_undo_refuses_to_overwrite_a_later_manual_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
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
                ModelResponse(content="Updated the value."),
            ]
        ),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    metadata = runtime.start(project)
    result = asyncio.run(runtime.run_turn(metadata, "Update the value."))
    target.write_text("VALUE = 3\n", encoding="utf-8")

    assert "file changed after RepoPilot's edit" in runtime.undo_last_edit(result.metadata)
    assert target.read_text(encoding="utf-8") == "VALUE = 3\n"
