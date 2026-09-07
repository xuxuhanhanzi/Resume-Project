"""Durable metadata for one interactive coding session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from repopilot.runtime.policy import PermissionMode


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Small, inspectable session identity kept separately from its transcript."""

    session_id: str
    project_root: Path
    model: str
    permission_mode: PermissionMode
    created_at: str
    updated_at: str
    title: str | None = None
    provider: str | None = None

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        project_root: Path,
        model: str,
        permission_mode: PermissionMode,
        provider: str | None = None,
    ) -> SessionMetadata:
        timestamp = _now()
        return cls(
            session_id,
            project_root.resolve(),
            model,
            permission_mode,
            timestamp,
            timestamp,
            provider=provider,
        )

    def touched(self) -> SessionMetadata:
        return SessionMetadata(
            self.session_id,
            self.project_root,
            self.model,
            self.permission_mode,
            self.created_at,
            _now(),
            self.title,
            self.provider,
        )

    def with_model(self, model: str, *, provider: str | None = None) -> SessionMetadata:
        """Record a user-selected model without changing session identity."""
        if not model.strip():
            raise ValueError("model must not be empty")
        return SessionMetadata(
            self.session_id,
            self.project_root,
            model,
            self.permission_mode,
            self.created_at,
            _now(),
            self.title,
            provider or self.provider,
        )

    def with_title(self, title: str) -> SessionMetadata:
        """Attach a small user-visible label without changing the session identity."""
        normalized = title.strip()
        if not 1 <= len(normalized) <= 120:
            raise ValueError("session title must contain 1-120 characters")
        return SessionMetadata(
            self.session_id,
            self.project_root,
            self.model,
            self.permission_mode,
            self.created_at,
            _now(),
            normalized,
            self.provider,
        )

    def to_dict(self) -> dict[str, str]:
        data = {
            "session_id": self.session_id,
            "project_root": str(self.project_root),
            "model": self.model,
            "permission_mode": self.permission_mode.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.title is not None:
            data["title"] = self.title
        if self.provider is not None:
            data["provider"] = self.provider
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SessionMetadata:
        return cls(
            str(data["session_id"]),
            Path(str(data["project_root"])).resolve(),
            str(data["model"]),
            PermissionMode(str(data["permission_mode"])),
            str(data["created_at"]),
            str(data["updated_at"]),
            str(data["title"]) if data.get("title") is not None else None,
            str(data["provider"]) if data.get("provider") is not None else None,
        )
