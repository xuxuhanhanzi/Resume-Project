from __future__ import annotations

import asyncio
from pathlib import Path

from repopilot.core.contracts import ToolCall
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.tools.additional_directories import (
    AdditionalDirectory,
    ReadAddedFileTool,
    SearchAddedTextTool,
    additional_directory_tools,
)
from repopilot.tools.base import ToolContext
from repopilot.workspace.contracts import InteractiveTask


def test_added_directories_are_read_only_aliases_and_reject_traversal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "notes.txt").write_text("needle\n", encoding="utf-8")
    directories = (AdditionalDirectory("extra1", external),)
    context = ToolContext(
        InteractiveTask("session", project, "Inspect dependency"),
        LocalTrustedRunner(trusted=True),
    )
    tools = additional_directory_tools(directories)

    assert {tool.spec.name for tool in tools} == {
        "list_added_files",
        "read_added_file",
        "search_added_text",
    }
    assert all(tool.spec.read_only for tool in tools)
    read = asyncio.run(
        ReadAddedFileTool(directories).run(
            ToolCall("read", "read_added_file", {"directory": "extra1", "path": "notes.txt"}),
            context,
        )
    )
    assert read.ok
    assert read.data["path"] == "@extra1/notes.txt"
    escaped = asyncio.run(
        ReadAddedFileTool(directories).run(
            ToolCall("escape", "read_added_file", {"directory": "extra1", "path": "../project"}),
            context,
        )
    )
    assert not escaped.ok
    searched = asyncio.run(
        SearchAddedTextTool(directories).run(
            ToolCall("search", "search_added_text", {"directory": "extra1", "pattern": "needle"}),
            context,
        )
    )
    assert searched.ok
    assert searched.data["matches"][0]["path"] == "@extra1/notes.txt"
