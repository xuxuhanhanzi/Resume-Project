from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from repopilot.core.contracts import ToolCall
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.task import PublicTaskSpec
from repopilot.tools.base import ToolContext
from repopilot.tools.coding import (
    ApplyPatchTool,
    FindSymbolTool,
    GitDiffTool,
    ListFilesTool,
    ReadFileTool,
    RunTestsTool,
    SearchTextTool,
    SnapshotLimitError,
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


def test_list_files_respects_depth_while_walking(tmp_path: Path) -> None:
    (tmp_path / "root.py").write_text("ROOT = 1\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "child.py").write_text("CHILD = 1\n", encoding="utf-8")
    deep = nested / "deep"
    deep.mkdir()
    (deep / "grandchild.py").write_text("GRANDCHILD = 1\n", encoding="utf-8")
    task = PublicTaskSpec("list", tmp_path, "Inspect files", trusted_fixture=True)
    context = ToolContext(task, LocalTrustedRunner(trusted=True))

    shallow = asyncio.run(
        ListFilesTool().run(ToolCall("list", "list_files", {"path": ".", "max_depth": 1}), context)
    )
    nested_result = asyncio.run(
        ListFilesTool().run(ToolCall("list", "list_files", {"path": ".", "max_depth": 2}), context)
    )

    assert shallow.data == {"files": ["root.py"]}
    assert nested_result.data == {"files": ["nested/child.py", "root.py"]}


def test_default_workspace_listing_skips_session_exclusions(tmp_path: Path) -> None:
    (tmp_path / "active.py").write_text("ACTIVE = 1\n", encoding="utf-8")
    excluded = tmp_path / "artifacts"
    excluded.mkdir()
    (excluded / "trace.json").write_text("{}\n", encoding="utf-8")
    task = PublicTaskSpec("list", tmp_path, "Inspect files", trusted_fixture=True)
    context = ToolContext(
        task,
        LocalTrustedRunner(trusted=True),
        snapshot_exclude_paths=("artifacts",),
    )

    result = asyncio.run(
        ListFilesTool().run(ToolCall("list", "list_files", {"path": ".", "max_depth": 2}), context)
    )

    assert result.data == {"files": ["active.py"]}


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


def test_patch_rejects_non_integer_replacement_count_without_crashing(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = asyncio.run(
        ApplyPatchTool().run(
            ToolCall(
                "p",
                "apply_patch",
                {
                    "path": "module.py",
                    "old_text": "return left + right",
                    "new_text": "return left - right",
                    "expected_replacements": {"value": 1},
                },
            ),
            context,
        )
    )

    assert not result.ok
    assert result.error_type is not None
    assert result.error_type.value == "validation"
    assert "must be an integer" in str(result.error)


def test_snapshot_excludes_reference_material_and_enforces_a_file_limit(tmp_path: Path) -> None:
    (tmp_path / "active.py").write_text("ACTIVE = 1\n", encoding="utf-8")
    reference = tmp_path / "Reference code"
    reference.mkdir()
    (reference / "upstream.py").write_text("REFERENCE = 1\n", encoding="utf-8")

    snapshot = capture_text_snapshot(tmp_path, exclude_paths=("Reference code",))

    assert snapshot == {"active.py": "ACTIVE = 1\n"}
    with pytest.raises(SnapshotLimitError, match="file safety limit"):
        capture_text_snapshot(tmp_path, max_files=1)


def test_implicit_search_skips_reference_material_but_explicit_search_can_read_it(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    reference = tmp_path / "Reference code"
    reference.mkdir()
    (reference / "upstream.py").write_text("REFERENCE_ONLY = 1\n", encoding="utf-8")
    context.snapshot_exclude_paths = ("Reference code",)

    implicit = asyncio.run(
        SearchTextTool().run(
            ToolCall("implicit", "search_text", {"pattern": "REFERENCE_ONLY"}), context
        )
    )
    explicit = asyncio.run(
        SearchTextTool().run(
            ToolCall(
                "explicit",
                "search_text",
                {"pattern": "REFERENCE_ONLY", "paths": ["Reference code"]},
            ),
            context,
        )
    )

    assert implicit.ok and implicit.data == {"matches": []}
    assert explicit.ok and "Reference code/upstream.py" in str(explicit.data)
