"""Durable, secret-safe receipts for interactive background processes.

The actual child process intentionally remains owned by the open CLI session.
This ledger does not pretend a vanished parent can be reattached after a
restart.  Instead, it records what happened, marks a formerly running task as
unavailable on the next session, and permits an explicit, separately confirmed
retry only when the original argv contained no redacted secret.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from repopilot.process.manager import ProcessSnapshot
from repopilot.security.redaction import redact_text
from repopilot.tools.shell import validate_tokenized_command

_STATUSES = {"running", "exited", "stopped", "unavailable"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class BackgroundTaskRecord:
    """A redacted task receipt that survives an interactive session restart."""

    process_id: str
    command: tuple[str, ...]
    cwd: str
    started_at: str
    status: str
    return_code: int | None
    retryable: bool
    updated_at: str

    def __post_init__(self) -> None:
        if not self.process_id or len(self.process_id) > 128:
            raise ValueError("background task ID is invalid")
        validate_tokenized_command(list(self.command))
        if not self.cwd or self.cwd.startswith(("/", "\\")) or ".." in self.cwd.split("/"):
            raise ValueError("background task cwd is invalid")
        if self.status not in _STATUSES:
            raise ValueError("background task status is invalid")
        if self.return_code is not None and isinstance(self.return_code, bool):
            raise ValueError("background task return code is invalid")

    @classmethod
    def from_started_result(
        cls,
        *,
        process_id: str,
        command: tuple[str, ...],
        cwd: str,
        started_at: str,
    ) -> BackgroundTaskRecord:
        redacted = tuple(redact_text(item) for item in command)
        return cls(
            process_id,
            redacted,
            cwd,
            started_at,
            "running",
            None,
            redacted == command,
            _now(),
        )

    def observed(self, snapshot: ProcessSnapshot, *, stopped: bool = False) -> BackgroundTaskRecord:
        if snapshot.process.process_id != self.process_id:
            raise ValueError("background task receipt does not match process snapshot")
        return BackgroundTaskRecord(
            self.process_id,
            self.command,
            self.cwd,
            self.started_at,
            "running" if snapshot.running else ("stopped" if stopped else "exited"),
            snapshot.return_code,
            self.retryable,
            _now(),
        )

    def unavailable_after_restart(self) -> BackgroundTaskRecord:
        if self.status != "running":
            return self
        return BackgroundTaskRecord(
            self.process_id,
            self.command,
            self.cwd,
            self.started_at,
            "unavailable",
            self.return_code,
            self.retryable,
            _now(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "command": list(self.command),
            "cwd": self.cwd,
            "started_at": self.started_at,
            "status": self.status,
            "return_code": self.return_code,
            "retryable": self.retryable,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: object) -> BackgroundTaskRecord:
        if not isinstance(raw, dict):
            raise ValueError("background task receipt must be an object")
        command = raw.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("background task command must be a list of strings")
        process_id = raw.get("process_id")
        cwd = raw.get("cwd")
        started_at = raw.get("started_at")
        status = raw.get("status")
        return_code = raw.get("return_code")
        retryable = raw.get("retryable")
        updated_at = raw.get("updated_at")
        if (
            not isinstance(process_id, str)
            or not isinstance(cwd, str)
            or not isinstance(started_at, str)
            or not isinstance(status, str)
            or not isinstance(retryable, bool)
            or not isinstance(updated_at, str)
            or (
                return_code is not None
                and (not isinstance(return_code, int) or isinstance(return_code, bool))
            )
        ):
            raise ValueError("background task receipt has invalid fields")
        return cls(
            process_id,
            tuple(command),
            cwd,
            started_at,
            status,
            return_code,
            retryable,
            updated_at,
        )
