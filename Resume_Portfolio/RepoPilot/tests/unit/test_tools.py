from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from repopilot.core.contracts import ToolCall
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext
from repopilot.tools.coding import (
    ApplyPatchTool,
    FindSymbolTool,
    GitDiffTool,
    ReadFileTool,
    RunTestsTool,
    SearchTextTool,
    capture_text_snapshot,
)


def _context(tmp_path: Path) -> ToolContext:
    (tmp_path / "module.py").write_text(
        "def subtract(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    (tmp_path / "verify.py").write_text(
        "from module import subtract\nassert subtract(2, 1) == 1\n", encoding="utf-8"
    )
    task = PublicTaskSpec(
        "tool-task",
        tmp_path,
        "Fix subtract",
        allowed_paths=(".",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )
    return ToolContext(task, LocalTrustedRunner(trusted=True), capture_text_snapshot(tmp_path))


def test_read_search_and_symbol_tools(tmp_path: Path) -> None:
    context = _context(tmp_path)

    read = asyncio.run(
        ReadFileTool().run(ToolCall("r", "read_file", {"path": "module.py"}), context)
    )
    search = asyncio.run(
        SearchTextTool().run(ToolCall("s", "search_text", {"pattern": "left \\+ right"}), context)
    )
    symbol = asyncio.run(
        FindSymbolTool().run(ToolCall("f", "find_symbol", {"name": "subtract"}), context)
    )

    assert read.ok and "return left + right" in str(read.data)
    assert search.ok and "module.py" in str(search.data)
    assert symbol.ok and "FunctionDef" in str(symbol.data)


def test_search_does_not_cross_forbidden_path(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.task = PublicTaskSpec(
        "restricted",
        tmp_path,
        "Fix subtract",
        allowed_paths=(".",),
        forbidden_paths=("verify.py",),
        test_command=(sys.executable, "verify.py"),
        trusted_fixture=True,
    )

    search = asyncio.run(
        SearchTextTool().run(ToolCall("s", "search_text", {"pattern": "assert"}), context)
    )

    assert search.ok
    assert "verify.py" not in str(search.data)


def test_patch_diff_and_test_closed_loop(tmp_path: Path) -> None:
    context = _context(tmp_path)
    patch = asyncio.run(
        ApplyPatchTool().run(
            ToolCall(
                "p",
                "apply_patch",
                {
                    "path": "module.py",
                    "old_text": "return left + right",
                    "new_text": "return left - right",
                },
            ),
            context,
        )
    )
    test = asyncio.run(RunTestsTool().run(ToolCall("t", "run_tests", {}), context))
    diff = asyncio.run(GitDiffTool().run(ToolCall("d", "git_diff", {}), context))

    assert patch.ok and patch.side_effect
    assert test.ok
    assert "return left - right" in str(diff.data)


def test_patch_conflict_is_recoverable(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = asyncio.run(
        ApplyPatchTool().run(
            ToolCall(
                "p",
                "apply_patch",
                {"path": "module.py", "old_text": "missing", "new_text": "replacement"},
            ),
            context,
        )
    )

    assert not result.ok
    assert result.recoverable
