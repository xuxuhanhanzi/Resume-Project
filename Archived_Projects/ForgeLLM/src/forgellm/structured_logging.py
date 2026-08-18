"""Small JSONL logger with conservative field redaction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
    }
)
_SENSITIVE_SUFFIXES = ("_api_key", "_password", "_secret", "_token")
_REDACTED = "[REDACTED]"


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def redact(value: object, *, key: str = "") -> JsonValue:
    """Convert a value to JSON-compatible data while redacting sensitive fields."""
    if key and _is_sensitive_key(key):
        return _REDACTED
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {
            str(item_key): redact(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [redact(item) for item in value]
    return repr(value)


def write_jsonl_event(
    path: Path,
    event: str,
    fields: Mapping[str, object] | None = None,
    *,
    now: datetime | None = None,
) -> None:
    """Append one redacted event to a UTF-8 JSONL file."""
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("log timestamps must be timezone-aware")

    payload: dict[str, JsonValue] = {
        "event": event,
        "timestamp": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }
    if fields:
        reserved = {"event", "timestamp"} & set(fields)
        if reserved:
            reserved_names = ", ".join(sorted(reserved))
            raise ValueError(f"log fields cannot replace reserved keys: {reserved_names}")
        payload.update({key: redact(value, key=key) for key, value in fields.items()})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
