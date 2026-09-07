"""Layered, explicit tool-permission settings for interactive workspaces."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

_PATTERN = re.compile(r"[A-Za-z0-9_.*?-]{1,128}\Z")
_MAX_PATTERNS = 100


@dataclass(frozen=True, slots=True)
class ToolPermissionRules:
    """Glob patterns from trusted user sources and project deny sources.

    ``allow`` is only read from a user-global config, project-local config, or
    a flag supplied by the current user. A version-controlled project config
    can only add denials, so opening a repository cannot silently relax tool
    approvals.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_patterns(self.allow, field="allow")
        _validate_patterns(self.deny, field="deny")

    def allows(self, tool_name: str) -> bool:
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.allow)

    def denies(self, tool_name: str) -> bool:
        return any(fnmatch.fnmatchcase(tool_name, pattern) for pattern in self.deny)

    def summary(self) -> str:
        allowed = ", ".join(self.allow) if self.allow else "(none)"
        denied = ", ".join(self.deny) if self.deny else "(none)"
        return f"Allowed tools: {allowed}\nDenied tools: {denied}"


def load_tool_permission_rules(
    *,
    session_root: Path,
    project_root: Path,
    command_allow: tuple[str, ...] = (),
    command_deny: tuple[str, ...] = (),
) -> ToolPermissionRules:
    """Load rules with safe precedence from global, local, project, then CLI.

    Global and local settings represent an explicit user decision. Project
    settings may deny tools for a repository but their ``allow`` entries are
    intentionally ignored.
    """
    global_allow, global_deny = _read_rules(session_root.resolve() / "settings.json")
    project_allow, project_deny = _read_rules(
        project_root.resolve() / ".repopilot" / "settings.json"
    )
    local_allow, local_deny = _read_rules(
        project_root.resolve() / ".repopilot" / "settings.local.json"
    )
    del project_allow
    return ToolPermissionRules(
        allow=_ordered_unique((*global_allow, *local_allow, *command_allow)),
        deny=_ordered_unique((*global_deny, *project_deny, *local_deny, *command_deny)),
    )


def _read_rules(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not path.exists():
        return (), ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid RepoPilot settings at {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"RepoPilot settings at {path} must be an object")
    permissions = raw.get("permissions", {})
    if not isinstance(permissions, dict):
        raise ValueError(f"permissions at {path} must be an object")
    return (
        _string_tuple(permissions.get("allow", []), path=path, field="permissions.allow"),
        _string_tuple(permissions.get("deny", []), path=path, field="permissions.deny"),
    )


def _string_tuple(value: object, *, path: Path, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} at {path} must be a list of strings")
    result = tuple(value)
    try:
        _validate_patterns(result, field=field)
    except ValueError as error:
        raise ValueError(f"{error} at {path}") from error
    return result


def _validate_patterns(patterns: tuple[str, ...], *, field: str) -> None:
    if len(patterns) > _MAX_PATTERNS:
        raise ValueError(f"{field} may contain at most {_MAX_PATTERNS} patterns")
    if any(not _PATTERN.fullmatch(pattern) for pattern in patterns):
        raise ValueError(f"{field} patterns may contain only tool-name glob characters")


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
