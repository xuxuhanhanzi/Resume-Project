from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


def _runtime(store: SessionStore, responses: list[ModelResponse]) -> SessionRuntime:
    return SessionRuntime(
        provider=ScriptedProvider(responses),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=store,
        model="scripted",
        approval=StaticApprovalHandler(True),
    )


def _patch(call_id: str, before: str, after: str) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                call_id,
                "apply_patch",
                {"path": "module.py", "old_text": before, "new_text": after},
            ),
        )
    )


def test_turn_revisions_persist_across_runtime_restart_and_rewind_safely(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")
    first_runtime = _runtime(
        store,
        [_patch("one", "VALUE = 1", "VALUE = 2"), ModelResponse(content="First change.")],
    )

    first = asyncio.run(first_runtime.run_turn(first_runtime.start(project), "Set value to two."))
    first_snapshot = first_runtime.turn_diff(first.metadata)
    assert first_snapshot.sequence == 1
    assert [(change.path, change.before, change.after) for change in first_snapshot.files] == [
        ("module.py", "VALUE = 1\n", "VALUE = 2\n")
    ]

    restarted_runtime = _runtime(
        store,
        [_patch("two", "VALUE = 2", "VALUE = 3"), ModelResponse(content="Second change.")],
    )
    second = asyncio.run(restarted_runtime.run_turn(first.metadata, "Set value to three."))
    assert target.read_text(encoding="utf-8") == "VALUE = 3\n"
    revisions = restarted_runtime.turn_snapshots(second.metadata)
    assert [snapshot.sequence for snapshot in revisions] == [1, 2]

    preview = restarted_runtime.rewind_preview(second.metadata, 1)
    assert preview.can_apply
    assert preview.changed_files == ("module.py",)
    rewound = restarted_runtime.rewind(second.metadata, 1)

    assert rewound.scope == "all"
    assert rewound.restored_files == ("module.py",)
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"
    restored_state = store.checkpoint(rewound.metadata).load()
    assert restored_state is not None
    assert [message.content for message in restored_state.messages] == [
        message.content for message in first_snapshot.state.messages
    ]
    assert [snapshot.kind for snapshot in restarted_runtime.turn_snapshots(rewound.metadata)] == [
        "turn",
        "turn",
        "rewind",
    ]


def test_rewind_refuses_to_overwrite_a_manual_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")
    runtime = _runtime(
        store,
        [
            _patch("one", "VALUE = 1", "VALUE = 2"),
            ModelResponse(content="First change."),
            _patch("two", "VALUE = 2", "VALUE = 3"),
            ModelResponse(content="Second change."),
        ],
    )

    first = asyncio.run(runtime.run_turn(runtime.start(project), "Set value to two."))
    second = asyncio.run(runtime.run_turn(first.metadata, "Set value to three."))
    target.write_text("VALUE = 99\n", encoding="utf-8")

    preview = runtime.rewind_preview(second.metadata, 1)
    assert preview.conflicts == ("module.py",)
    with pytest.raises(ValueError, match="files changed after their recorded state"):
        runtime.rewind(second.metadata, 1)
    assert target.read_text(encoding="utf-8") == "VALUE = 99\n"
