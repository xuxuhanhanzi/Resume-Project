"""Safe filesystem, retrieval, editing, testing, and diff tools."""

from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path

from repopilot.core.contracts import (
    ErrorType,
    JSONValue,
    Permission,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repopilot.security.paths import (
    PathSecurityError,
    resolve_workspace_path,
    task_path_is_visible,
)
from repopilot.tools.base import Tool, ToolContext

_IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


def _schema(properties: dict[str, JSONValue], required: list[str]) -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _error(call: ToolCall, error: Exception, *, recoverable: bool = False) -> ToolResult:
    error_type = (
        ErrorType.SECURITY if isinstance(error, PathSecurityError) else ErrorType.VALIDATION
    )
    return ToolResult(
        call.call_id,
        call.name,
        False,
        error=str(error),
        error_type=error_type,
        recoverable=recoverable,
    )


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part in _IGNORED_PARTS for part in path.parts)
    )


def capture_text_snapshot(
    root: Path,
    *,
    include_paths: tuple[str, ...] | None = None,
    max_file_bytes: int = 1_000_000,
) -> dict[str, str]:
    """Capture a bounded text baseline for deterministic diffing."""
    snapshot: dict[str, str] = {}
    candidates: set[Path] = set()
    for relative in include_paths or (".",):
        selected = (root / relative).resolve(strict=True)
        if (selected == root or root not in selected.parents) and selected != root:
            raise ValueError("snapshot path escaped the workspace")
        if selected.is_file():
            candidates.add(selected)
        else:
            candidates.update(_iter_files(selected))
    for path in sorted(candidates):
        if path.stat().st_size > max_file_bytes:
            continue
        try:
            snapshot[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return snapshot


class ListFilesTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "list_files",
            "List repository files below a safe relative path.",
            _schema(
                {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 0, "maximum": 8},
                },
                ["path"],
            ),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            raw_path = str(call.arguments.get("path", "."))
            max_depth = int(call.arguments.get("max_depth", 4))
            if not 0 <= max_depth <= 8:
                raise ValueError("max_depth must be between 0 and 8")
            root = resolve_workspace_path(context.task, raw_path)
            if not root.is_dir():
                raise ValueError("list_files path must be a directory")
            files = []
            for path in _iter_files(root):
                if not task_path_is_visible(context.task, path):
                    continue
                relative = path.relative_to(context.task.workspace)
                if len(path.relative_to(root).parts) <= max_depth:
                    files.append(relative.as_posix())
                if len(files) >= 500:
                    break
            return ToolResult(call.call_id, call.name, True, {"files": files})
        except (OSError, ValueError) as error:
            return _error(call, error)


class ReadFileTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_file",
            "Read a bounded UTF-8 line range from one repository file.",
            _schema(
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                ["path"],
            ),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            path = resolve_workspace_path(context.task, str(call.arguments.get("path", "")))
            if not path.is_file() or path.stat().st_size > 1_000_000:
                raise ValueError("read_file requires a text file no larger than 1 MB")
            start = int(call.arguments.get("start_line", 1))
            end = int(call.arguments.get("end_line", start + 199))
            if start < 1 or end < start or end - start > 500:
                raise ValueError("requested line range is invalid or exceeds 501 lines")
            lines = path.read_text(encoding="utf-8").splitlines()
            selected = [
                f"{index}: {lines[index - 1]}" for index in range(start, min(end, len(lines)) + 1)
            ]
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {
                    "path": path.relative_to(context.task.workspace).as_posix(),
                    "content": "\n".join(selected),
                },
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            return _error(call, error)


class SearchTextTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "search_text",
            "Regex-search bounded UTF-8 repository files.",
            _schema(
                {
                    "pattern": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                ["pattern"],
            ),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            pattern = str(call.arguments.get("pattern", ""))
            if not pattern or len(pattern) > 500:
                raise ValueError("pattern must contain 1-500 characters")
            regex = re.compile(pattern)
            raw_paths = call.arguments.get("paths", list(context.task.allowed_paths))
            if not isinstance(raw_paths, list) or not all(
                isinstance(item, str) for item in raw_paths
            ):
                raise ValueError("paths must be a list of strings")
            limit = int(call.arguments.get("max_results", 50))
            if not 1 <= limit <= 200:
                raise ValueError("max_results must be between 1 and 200")
            matches: list[dict[str, JSONValue]] = []
            seen: set[Path] = set()
            for raw_path in raw_paths:
                root = resolve_workspace_path(context.task, raw_path)
                candidates = [root] if root.is_file() else _iter_files(root)
                for path in candidates:
                    if not task_path_is_visible(context.task, path):
                        continue
                    if path in seen or path.stat().st_size > 1_000_000:
                        continue
                    seen.add(path)
                    try:
                        lines = path.read_text(encoding="utf-8").splitlines()
                    except UnicodeDecodeError:
                        continue
                    for line_number, line in enumerate(lines, start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "path": path.relative_to(context.task.workspace).as_posix(),
                                    "line": line_number,
                                    "text": line[:500],
                                }
                            )
                            if len(matches) >= limit:
                                return ToolResult(
                                    call.call_id, call.name, True, {"matches": matches}
                                )
            return ToolResult(call.call_id, call.name, True, {"matches": matches})
        except (OSError, re.error, ValueError) as error:
            return _error(call, error)


class FindSymbolTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "find_symbol",
            "Find Python function, class, and assignment definitions with the stdlib AST.",
            _schema({"name": {"type": "string"}}, ["name"]),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        name = str(call.arguments.get("name", ""))
        if not name.isidentifier():
            return _error(call, ValueError("name must be a Python identifier"))
        matches: list[dict[str, JSONValue]] = []
        for path in _iter_files(context.task.workspace):
            if path.suffix != ".py" or not task_path_is_visible(context.task, path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                node_name = getattr(node, "name", None)
                if node_name == name:
                    matches.append(
                        {
                            "path": path.relative_to(context.task.workspace).as_posix(),
                            "line": int(getattr(node, "lineno", 0)),
                            "kind": type(node).__name__,
                        }
                    )
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(
                        isinstance(target, ast.Name) and target.id == name for target in targets
                    ):
                        matches.append(
                            {
                                "path": path.relative_to(context.task.workspace).as_posix(),
                                "line": node.lineno,
                                "kind": type(node).__name__,
                            }
                        )
        return ToolResult(call.call_id, call.name, True, {"matches": matches[:100]})


class ApplyPatchTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "apply_patch",
            "Replace an exact text block in one allowed UTF-8 file.",
            _schema(
                {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "expected_replacements": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                ["path", "old_text", "new_text"],
            ),
            permission=Permission.WRITE,
            read_only=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        original_content: str | None = None
        path: Path | None = None
        try:
            path = resolve_workspace_path(context.task, str(call.arguments.get("path", "")))
            relative_path = path.relative_to(context.task.workspace).as_posix()
            old_text = str(call.arguments.get("old_text", ""))
            new_text = str(call.arguments.get("new_text", ""))
            expected = int(call.arguments.get("expected_replacements", 1))
            if not old_text:
                raise ValueError("old_text must not be empty")
            original_content = path.read_text(encoding="utf-8")
            count = original_content.count(old_text)
            if count != expected:
                return ToolResult(
                    call.call_id,
                    call.name,
                    False,
                    error=f"expected {expected} exact matches but found {count}",
                    error_type=ErrorType.CONFLICT,
                    recoverable=True,
                    side_effect=False,
                )
            path.write_text(
                original_content.replace(old_text, new_text), encoding="utf-8", newline=""
            )
            changed = _changed_files(context)
            if len(changed) > context.task.max_changed_files:
                path.write_text(original_content, encoding="utf-8", newline="")
                raise PathSecurityError("patch would exceed max_changed_files")
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {"path": relative_path, "replacements": count},
                side_effect=True,
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            if original_content is not None and path is not None:
                try:
                    path.write_text(original_content, encoding="utf-8", newline="")
                except OSError:
                    return ToolResult(
                        call.call_id,
                        call.name,
                        False,
                        error=f"patch failed and rollback also failed: {error}",
                        error_type=ErrorType.INTERNAL,
                        side_effect=True,
                    )
            return _error(call, error)


def _changed_files(context: ToolContext) -> list[str]:
    current = capture_text_snapshot(
        context.task.workspace, include_paths=context.task.allowed_paths
    )
    paths = set(context.baseline) | set(current)
    return sorted(path for path in paths if context.baseline.get(path) != current.get(path))


class RunTestsTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "run_tests",
            "Run the immutable task test command, optionally narrowed to allowed visible test IDs.",
            _schema(
                {"test_ids": {"type": "array", "items": {"type": "string"}}},
                [],
            ),
            permission=Permission.EXECUTE,
            timeout_seconds=120.0,
            read_only=False,
            idempotent=False,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        raw_test_ids = call.arguments.get("test_ids", [])
        if not isinstance(raw_test_ids, list) or not all(
            isinstance(item, str) for item in raw_test_ids
        ):
            return _error(call, ValueError("test_ids must be a list of strings"))
        unknown = set(raw_test_ids) - set(context.task.visible_tests)
        if unknown:
            return _error(call, ValueError(f"test IDs are not allowlisted: {sorted(unknown)}"))
        if not context.task.test_command:
            return _error(call, ValueError("task has no test_command"))
        command = (*context.task.test_command, *tuple(str(item) for item in raw_test_ids))
        try:
            result = await context.runner.run(
                command, cwd=context.task.workspace, timeout_seconds=self.spec.timeout_seconds
            )
        except (OSError, RuntimeError, PermissionError, ValueError) as error:
            return _error(call, error)
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            call.call_id,
            call.name,
            ok,
            {
                "command": list(result.command),
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
            },
            error=None if ok else "test command failed",
            error_type=None
            if ok
            else (ErrorType.TIMEOUT if result.timed_out else ErrorType.EXECUTION),
            recoverable=not ok and not result.timed_out,
            side_effect=True,
        )


class GitDiffTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "git_diff",
            "Return a deterministic unified diff against the run baseline.",
            _schema({}, []),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        current = capture_text_snapshot(
            context.task.workspace, include_paths=context.task.allowed_paths
        )
        chunks: list[str] = []
        for relative in sorted(set(context.baseline) | set(current)):
            before = context.baseline.get(relative, "").splitlines(keepends=True)
            after = current.get(relative, "").splitlines(keepends=True)
            if before == after:
                continue
            candidate = context.task.workspace / relative
            if candidate.exists() and not task_path_is_visible(context.task, candidate):
                continue
            chunks.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        diff = "".join(chunks)
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {"changed_files": _changed_files(context), "diff": diff[:100_000]},
        )


class InspectFailureTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "inspect_failure",
            "Inspect the most recent failed tool observations.",
            _schema({}, []),
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        failures = [result.to_dict() for result in context.recent_results if not result.ok]
        return ToolResult(call.call_id, call.name, True, {"failures": failures[-5:]})


def default_coding_tools() -> list[Tool]:
    """Return the stable Stage 2 coding-tool surface."""
    return [
        ListFilesTool(),
        ReadFileTool(),
        SearchTextTool(),
        FindSymbolTool(),
        ApplyPatchTool(),
        RunTestsTool(),
        GitDiffTool(),
        InspectFailureTool(),
    ]
