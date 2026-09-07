"""Canonical workspace-path checks shared by every filesystem tool."""

from __future__ import annotations

from pathlib import Path, PurePath

from repopilot.workspace.contracts import WorkspaceTask


class PathSecurityError(ValueError):
    """A requested path crossed the task's workspace or allowlist boundary."""


def _relative_posix(root: Path, candidate: Path) -> str:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise PathSecurityError("path resolves outside the workspace") from error
    value = relative.as_posix()
    return value if value else "."


def _under(path: str, boundary: str) -> bool:
    if boundary in {"", "."}:
        return True
    return path == boundary or path.startswith(boundary.rstrip("/") + "/")


def resolve_workspace_path(
    task: WorkspaceTask, relative_path: str, *, require_exists: bool = True
) -> Path:
    """Resolve one model-provided path and enforce workspace/task boundaries."""
    if "\x00" in relative_path or Path(relative_path).is_absolute():
        raise PathSecurityError("path must be a relative path without null bytes")
    portable = relative_path.replace("\\", "/")
    if ".." in PurePath(portable).parts:
        raise PathSecurityError("path traversal is forbidden")
    root = task.workspace.resolve(strict=True)
    raw = root / portable
    if require_exists:
        try:
            candidate = raw.resolve(strict=True)
        except FileNotFoundError as error:
            raise PathSecurityError(f"path does not exist: {relative_path}") from error
    else:
        try:
            parent = raw.parent.resolve(strict=True)
        except FileNotFoundError as error:
            raise PathSecurityError("parent directory does not exist") from error
        candidate = parent / raw.name
    normalized = _relative_posix(root, candidate)
    allowed = any(
        _under(normalized, boundary.replace("\\", "/")) for boundary in task.allowed_paths
    )
    forbidden = any(
        _under(normalized, boundary.replace("\\", "/")) for boundary in task.forbidden_paths
    )
    if not allowed or forbidden:
        raise PathSecurityError(f"path is outside the task allowlist: {normalized}")
    return candidate


def task_path_is_visible(task: WorkspaceTask, candidate: Path) -> bool:
    """Return whether an existing candidate is inside the model-visible task boundary."""
    try:
        relative = candidate.resolve(strict=True).relative_to(task.workspace).as_posix()
        resolve_workspace_path(task, relative)
    except (OSError, ValueError):
        return False
    return True
