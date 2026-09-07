"""Read-only access to explicitly user-selected directories outside a workspace.

Claude Code-style additional directories are useful for inspecting a sibling
library or a checked-out dependency.  They are intentionally represented by
aliases here, rather than being merged into the primary workspace.  That keeps
all existing edit, Git, shell, LSP and test tools strictly rooted in the main
project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePath

from repopilot.core.contracts import (
    ErrorType,
    JSONValue,
    Permission,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repopilot.tools.base import Tool, ToolContext

_MAX_FILES = 2_000
_MAX_RESULTS = 200
_MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True, slots=True)
class AdditionalDirectory:
    """A checked, read-only external root exposed under a short alias."""

    alias: str
    root: Path

    def __post_init__(self) -> None:
        if not re.fullmatch(r"extra[1-8]", self.alias):
            raise ValueError("additional directory aliases must be extra1 through extra8")
        canonical = self.root.resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError("additional directory root must be an existing directory")
        object.__setattr__(self, "root", canonical)


class _AdditionalDirectoryTool:
    def __init__(self, directories: tuple[AdditionalDirectory, ...]) -> None:
        self._directories = {item.alias: item.root for item in directories}

    def _root(self, alias: object) -> tuple[str, Path]:
        if not isinstance(alias, str) or alias not in self._directories:
            available = ", ".join(f"@{name}" for name in sorted(self._directories)) or "(none)"
            raise ValueError(f"unknown added-directory alias; available: {available}")
        return alias, self._directories[alias]

    @staticmethod
    def _path(root: Path, raw_path: object) -> Path:
        if not isinstance(raw_path, str) or "\x00" in raw_path:
            raise ValueError("path must be a relative string")
        portable = raw_path.replace("\\", "/")
        if Path(portable).is_absolute() or ".." in PurePath(portable).parts:
            raise ValueError("path must be relative and traversal-free")
        candidate = (root / portable).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("path resolved outside the added directory") from error
        return candidate

    @staticmethod
    def _display(alias: str, root: Path, path: Path) -> str:
        relative = path.relative_to(root).as_posix()
        return f"@{alias}" if relative == "." else f"@{alias}/{relative}"


class ListAddedFilesTool(_AdditionalDirectoryTool):
    """List a bounded subset of an explicit external root without following links."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "list_added_files",
            "List up to 200 files from a user-approved read-only directory alias (@extra1, etc.).",
            {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer", "minimum": 0, "maximum": 4},
                },
                "required": ["directory"],
                "additionalProperties": False,
            },
            permission=Permission.READ,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        try:
            alias, root = self._root(call.arguments.get("directory"))
            path = self._path(root, call.arguments.get("path", "."))
            depth = call.arguments.get("max_depth", 2)
            if not isinstance(depth, int) or not 0 <= depth <= 4:
                raise ValueError("max_depth must be an integer from 0 to 4")
            if not path.is_dir():
                raise ValueError("path must be a directory")
            files = [
                self._display(alias, root, item)
                for item in _walk_files(path, root=root, max_depth=depth)
            ]
            return ToolResult(call.call_id, call.name, True, {"files": files})
        except (OSError, ValueError) as error:
            return _error(call, error)


class ReadAddedFileTool(_AdditionalDirectoryTool):
    """Read bounded UTF-8 text from an explicit external root."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_added_file",
            "Read up to 501 lines from a file in a user-approved read-only directory alias.",
            {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["directory", "path"],
                "additionalProperties": False,
            },
            permission=Permission.READ,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        try:
            alias, root = self._root(call.arguments.get("directory"))
            path = self._path(root, call.arguments.get("path"))
            if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
                raise ValueError("read_added_file requires a UTF-8 file no larger than 1 MB")
            start = call.arguments.get("start_line", 1)
            end = call.arguments.get("end_line", start + 199 if isinstance(start, int) else 200)
            if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
                raise ValueError("requested line range is invalid")
            if end - start > 500:
                raise ValueError("requested line range exceeds 501 lines")
            lines = path.read_text(encoding="utf-8").splitlines()
            content = "\n".join(
                f"{index}: {lines[index - 1]}" for index in range(start, min(end, len(lines)) + 1)
            )
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {"path": self._display(alias, root, path), "content": content},
            )
        except (OSError, UnicodeDecodeError, ValueError) as error:
            return _error(call, error)


class SearchAddedTextTool(_AdditionalDirectoryTool):
    """Search bounded text files in an explicit external root."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "search_added_text",
            "Regex-search a user-approved read-only directory alias; limited to 200 matches.",
            {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": _MAX_RESULTS},
                },
                "required": ["directory", "pattern"],
                "additionalProperties": False,
            },
            permission=Permission.READ,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        try:
            alias, root = self._root(call.arguments.get("directory"))
            base = self._path(root, call.arguments.get("path", "."))
            pattern = call.arguments.get("pattern")
            limit = call.arguments.get("max_results", 50)
            if not isinstance(pattern, str) or not 1 <= len(pattern) <= 500:
                raise ValueError("pattern must contain 1-500 characters")
            if not isinstance(limit, int) or not 1 <= limit <= _MAX_RESULTS:
                raise ValueError("max_results must be an integer from 1 to 200")
            regex = re.compile(pattern)
            candidates = [base] if base.is_file() else _walk_files(base, root=root, max_depth=8)
            matches: list[dict[str, JSONValue]] = []
            for path in candidates:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except UnicodeDecodeError:
                    continue
                for line_number, line in enumerate(lines, start=1):
                    if regex.search(line):
                        matches.append(
                            {
                                "path": self._display(alias, root, path),
                                "line": line_number,
                                "text": line[:500],
                            }
                        )
                        if len(matches) >= limit:
                            return ToolResult(call.call_id, call.name, True, {"matches": matches})
            return ToolResult(call.call_id, call.name, True, {"matches": matches})
        except (OSError, re.error, ValueError) as error:
            return _error(call, error)


def additional_directory_tools(
    directories: tuple[AdditionalDirectory, ...],
) -> list[Tool]:
    """Return no external capabilities unless the user passed ``--add-dir``."""

    if not directories:
        return []
    return [
        ListAddedFilesTool(directories),
        ReadAddedFileTool(directories),
        SearchAddedTextTool(directories),
    ]


def _walk_files(path: Path, *, root: Path, max_depth: int) -> list[Path]:
    """Collect a bounded, no-symlink file inventory under one already-checked root."""

    if path.is_file():
        return [path]
    result: list[Path] = []
    pending: list[tuple[Path, int]] = [(path, 0)]
    while pending and len(result) < _MAX_FILES:
        current, depth = pending.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            continue
        for entry in entries:
            if len(result) >= _MAX_FILES:
                break
            try:
                if entry.is_symlink():
                    continue
                resolved = entry.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                result.append(resolved)
            elif resolved.is_dir() and depth < max_depth:
                pending.append((resolved, depth + 1))
    return result


def _error(call: ToolCall, error: Exception) -> ToolResult:
    return ToolResult(
        call.call_id,
        call.name,
        False,
        error=str(error),
        error_type=ErrorType.VALIDATION,
    )
