"""Local persistence for explicit workspace-trust decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WorkspaceTrust:
    """An explicit trust record tied to one canonical project path."""

    project_root: Path
    trusted_at: str


class WorkspaceTrustStore:
    """Store trust decisions outside projects so repositories remain unmodified."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def is_trusted(self, project_root: Path) -> bool:
        """Return true only for a valid record matching this exact canonical path."""
        path = self._path(project_root)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        return isinstance(raw, dict) and raw.get("project_root") == str(project_root.resolve())

    def trust(self, project_root: Path) -> WorkspaceTrust:
        """Persist a user-approved trust decision atomically."""
        resolved = project_root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace does not exist: {resolved}")
        record = WorkspaceTrust(resolved, datetime.now(UTC).isoformat())
        path = self._path(resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"project_root": str(record.project_root), "trusted_at": record.trusted_at},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
        return record

    def _path(self, project_root: Path) -> Path:
        canonical = str(project_root.resolve()).encode("utf-8")
        key = hashlib.sha256(canonical).hexdigest()[:24]
        return self.root / "projects" / key / "trust.json"
