"""Append-only, redacted JSONL traces for inspectable agent runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from repopilot.core.contracts import JSONValue

_SENSITIVE_FRAGMENTS = ("api_key", "apikey", "authorization", "password", "secret", "token")


def _redact(value: JSONValue, key: str = "") -> JSONValue:
    normalized = key.lower().replace("-", "_")
    if any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class TraceRecorder:
    """Write one durable event per line and keep an in-memory test view."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[dict[str, JSONValue]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, fields: Mapping[str, JSONValue] | None = None) -> None:
        payload: dict[str, JSONValue] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
        }
        if fields is not None:
            payload.update({key: _redact(value, key) for key, value in fields.items()})
        self.events.append(payload)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
