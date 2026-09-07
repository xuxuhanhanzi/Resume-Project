"""Append-only, conflict-aware workspace revisions for interactive sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath

from repopilot.core.contracts import AgentState, JSONValue
from repopilot.session.planning import SessionPlan


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
    ):
        raise ValueError("turn snapshot paths must be workspace-relative POSIX paths")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class TurnFileChange:
    """The exact text state before and after one completed interactive turn."""

    path: str
    before: str | None
    after: str | None

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        if self.before is not None and not isinstance(self.before, str):
            raise ValueError("turn snapshot before content must be text or null")
        if self.after is not None and not isinstance(self.after, str):
            raise ValueError("turn snapshot after content must be text or null")
        if self.before == self.after:
            raise ValueError("turn snapshots must not store unchanged files")

    def to_dict(self) -> dict[str, JSONValue]:
        return {"path": self.path, "before": self.before, "after": self.after}

    @classmethod
    def from_dict(cls, raw: dict[str, JSONValue]) -> TurnFileChange:
        path = raw.get("path")
        before = raw.get("before")
        after = raw.get("after")
        if not isinstance(path, str) or before is not None and not isinstance(before, str):
            raise ValueError("turn snapshot file record is invalid")
        if after is not None and not isinstance(after, str):
            raise ValueError("turn snapshot file record is invalid")
        return cls(path, before, after)


@dataclass(frozen=True, slots=True)
class SessionTurnSnapshot:
    """One immutable post-turn checkpoint plus a bounded workspace text delta."""

    sequence: int
    state: AgentState
    files: tuple[TurnFileChange, ...]
    context_summary: str | None
    plan: SessionPlan | None
    kind: str = "turn"
    created_at: str = ""
    inventory_error: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("turn snapshot sequence must be positive")
        if self.kind not in {"turn", "rewind"}:
            raise ValueError("turn snapshot kind must be turn or rewind")
        if self.context_summary is not None and not self.context_summary.strip():
            raise ValueError("turn snapshot context summary must be text or null")
        if self.inventory_error is not None and not self.inventory_error.strip():
            raise ValueError("turn snapshot inventory error must be text or null")
        paths = [change.path for change in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("turn snapshot files must be unique")

    @property
    def rewindable(self) -> bool:
        return self.inventory_error is None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "created_at": self.created_at or _now(),
            "state": self.state.to_dict(),
            "files": [change.to_dict() for change in self.files],
            "context_summary": self.context_summary,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "inventory_error": self.inventory_error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, JSONValue]) -> SessionTurnSnapshot:
        sequence = raw.get("sequence")
        state = raw.get("state")
        files = raw.get("files", [])
        summary = raw.get("context_summary")
        plan = raw.get("plan")
        kind = raw.get("kind", "turn")
        created_at = raw.get("created_at", "")
        inventory_error = raw.get("inventory_error")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or not isinstance(state, dict)
            or not isinstance(files, list)
            or summary is not None
            and not isinstance(summary, str)
            or plan is not None
            and not isinstance(plan, dict)
            or not isinstance(kind, str)
            or not isinstance(created_at, str)
            or inventory_error is not None
            and not isinstance(inventory_error, str)
        ):
            raise ValueError("turn snapshot has an invalid shape")
        changes = tuple(TurnFileChange.from_dict(item) for item in files if isinstance(item, dict))
        if len(changes) != len(files):
            raise ValueError("turn snapshot files must be objects")
        return cls(
            sequence,
            AgentState.from_dict(state),
            changes,
            summary,
            SessionPlan.from_dict(plan) if plan is not None else None,
            kind,
            created_at,
            inventory_error,
        )


def workspace_delta(
    before: dict[str, str], after: dict[str, str], *, max_bytes: int
) -> tuple[tuple[TurnFileChange, ...], str | None]:
    """Create a reversible text delta, refusing an unexpectedly large history record."""

    changes: list[TurnFileChange] = []
    total_bytes = 0
    for path in sorted(set(before) | set(after)):
        previous = before.get(path)
        current = after.get(path)
        if previous == current:
            continue
        change = TurnFileChange(path, previous, current)
        total_bytes += sum(
            len(value.encode("utf-8")) for value in (previous, current) if value is not None
        )
        if total_bytes > max_bytes:
            return (), f"turn delta exceeds the {max_bytes:,}-byte history limit"
        changes.append(change)
    return tuple(changes), None
