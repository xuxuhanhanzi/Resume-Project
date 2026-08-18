from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ToolCall
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext
from repopilot.tools.coding import ApplyPatchTool, capture_text_snapshot


def test_relative_workspace_is_normalized_before_patch(tmp_path: Path) -> None:
    workspace = tmp_path / "relative"
    workspace.mkdir()
    target = workspace / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    task = PublicTaskSpec("t", Path(str(workspace)), "fix", trusted_fixture=True)
    context = ToolContext(task, LocalTrustedRunner(trusted=True), capture_text_snapshot(workspace))

    result = asyncio.run(
        ApplyPatchTool().run(
            ToolCall(
                "p",
                "apply_patch",
                {"path": "module.py", "old_text": "value = 1", "new_text": "value = 2"},
            ),
            context,
        )
    )

    assert task.workspace.is_absolute()
    assert result.ok
    assert result.data == {"path": "module.py", "replacements": 1}
