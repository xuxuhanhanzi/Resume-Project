"""Local, redacted receipts for explicitly requested MCP health probes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from repopilot.mcp.config import MCPServerConfig
from repopilot.security.redaction import redact_text

_MAX_HISTORY_RECORDS = 500
_MAX_AUDIT_BYTES = 1_000_000
_MAX_ERROR_CHARACTERS = 1_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _project_id(project_root: Path) -> str:
    return hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class MCPProbeRecord:
    """One no-secret, local receipt of a user-authorized health probe."""

    recorded_at: str
    server: str
    transport: str
    timeout_seconds: float
    ok: bool
    tools: tuple[str, ...]
    resources: int
    prompts: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    denied_tools: tuple[str, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.server
            or self.transport not in {"stdio", "loopback_http"}
            or not 1 <= self.timeout_seconds <= 120
            or self.resources < 0
            or len(self.tools) > 256
            or len(self.prompts) > 256
        ):
            raise ValueError("MCP probe record is invalid")
        if self.error is not None and (
            not self.error.strip() or len(self.error) > _MAX_ERROR_CHARACTERS
        ):
            raise ValueError("MCP probe error is invalid")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tools"] = list(self.tools)
        payload["prompts"] = list(self.prompts)
        payload["allowed_tools"] = list(self.allowed_tools)
        payload["denied_tools"] = list(self.denied_tools)
        payload["error"] = redact_text(self.error) if self.error else None
        return payload

    @classmethod
    def from_dict(cls, raw: object) -> MCPProbeRecord:
        if not isinstance(raw, dict):
            raise ValueError("MCP probe record must be an object")
        expected = {
            "recorded_at",
            "server",
            "transport",
            "timeout_seconds",
            "ok",
            "tools",
            "resources",
            "prompts",
            "allowed_tools",
            "denied_tools",
            "error",
        }
        if set(raw) != expected:
            raise ValueError("MCP probe record has unsupported or missing fields")
        list_fields = ("tools", "prompts", "allowed_tools", "denied_tools")
        if (
            not isinstance(raw["recorded_at"], str)
            or not isinstance(raw["server"], str)
            or not isinstance(raw["transport"], str)
            or not isinstance(raw["timeout_seconds"], int | float)
            or isinstance(raw["timeout_seconds"], bool)
            or not isinstance(raw["ok"], bool)
            or not isinstance(raw["resources"], int)
            or isinstance(raw["resources"], bool)
            or any(
                not isinstance(raw[field], list)
                or not all(isinstance(item, str) for item in raw[field])
                for field in list_fields
            )
            or (raw["error"] is not None and not isinstance(raw["error"], str))
        ):
            raise ValueError("MCP probe record has invalid field types")
        return cls(
            recorded_at=raw["recorded_at"],
            server=raw["server"],
            transport=raw["transport"],
            timeout_seconds=float(raw["timeout_seconds"]),
            ok=raw["ok"],
            tools=tuple(raw["tools"]),
            resources=raw["resources"],
            prompts=tuple(raw["prompts"]),
            allowed_tools=tuple(raw["allowed_tools"]),
            denied_tools=tuple(raw["denied_tools"]),
            error=redact_text(raw["error"]) if raw["error"] is not None else None,
        )


class MCPProbeAuditStore:
    """Append-only, per-project local audit storage; reading never contacts MCP."""

    def __init__(self, state_root: Path) -> None:
        self.root = state_root.resolve() / "mcp-probes"

    def record(
        self,
        *,
        project_root: Path,
        config: MCPServerConfig,
        timeout_seconds: float,
        tools: tuple[str, ...] = (),
        resources: int = 0,
        prompts: tuple[str, ...] = (),
        error: str | None = None,
    ) -> MCPProbeRecord:
        """Append one bounded result after an explicit probe attempt."""
        safe_error = redact_text(error)[:_MAX_ERROR_CHARACTERS] if error else None
        record = MCPProbeRecord(
            recorded_at=_now(),
            server=config.name,
            transport="loopback_http" if config.url is not None else "stdio",
            timeout_seconds=timeout_seconds,
            ok=safe_error is None,
            tools=tuple(sorted(set(tools))),
            resources=resources,
            prompts=tuple(sorted(set(prompts))),
            allowed_tools=config.allowed_tools,
            denied_tools=config.denied_tools,
            error=safe_error,
        )
        path = self._path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        try:
            existing = path.stat().st_size if path.exists() else 0
        except OSError:
            existing = _MAX_AUDIT_BYTES + 1
        if existing + len(encoded.encode("utf-8")) > _MAX_AUDIT_BYTES:
            raise ValueError(
                "MCP probe audit reached its 1 MB limit; archive it manually before probing"
            )
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
        return record

    def history(
        self, *, project_root: Path, server: str | None = None, limit: int = 20
    ) -> tuple[MCPProbeRecord, ...]:
        """Read bounded newest-first receipts without launching or contacting a server."""
        if not 1 <= limit <= _MAX_HISTORY_RECORDS:
            raise ValueError(f"MCP probe history limit must be 1-{_MAX_HISTORY_RECORDS}")
        path = self._path(project_root)
        if not path.exists():
            return ()
        try:
            if path.stat().st_size > _MAX_AUDIT_BYTES:
                raise ValueError("MCP probe audit exceeds its 1 MB safety limit")
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"could not read MCP probe audit: {error}") from error
        records: list[MCPProbeRecord] = []
        for line in lines[-_MAX_HISTORY_RECORDS:]:
            try:
                record = MCPProbeRecord.from_dict(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                continue
            if server is None or record.server == server:
                records.append(record)
        return tuple(reversed(records[-limit:]))

    def latest_success(self, *, project_root: Path, server: str) -> MCPProbeRecord | None:
        """Return one previous successful receipt for local capability-drift comparison."""
        return next(
            (
                record
                for record in self.history(project_root=project_root, server=server)
                if record.ok
            ),
            None,
        )

    def _path(self, project_root: Path) -> Path:
        return self.root / _project_id(project_root) / "history.jsonl"


def capability_drift(
    previous: MCPProbeRecord | None, current_tools: tuple[str, ...]
) -> dict[str, list[str]]:
    """Compare exposed (post-policy) tool names with the last successful local probe."""
    if previous is None:
        return {"added": [], "removed": []}
    previous_tools = set(previous.tools)
    current = set(current_tools)
    return {"added": sorted(current - previous_tools), "removed": sorted(previous_tools - current)}
