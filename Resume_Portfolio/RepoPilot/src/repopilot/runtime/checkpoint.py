"""Atomic state checkpoints and idempotent tool-call journal."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from repopilot.core.contracts import AgentState, JSONValue, ToolResult


def _atomic_json_write(path: Path, data: dict[str, JSONValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


class CheckpointStore:
    """Persist exactly the state needed to resume the next model turn."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, state: AgentState) -> None:
        _atomic_json_write(self.path, state.to_dict())

    def load(self) -> AgentState | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("checkpoint root must be an object")
        return AgentState.from_dict(cast(dict[str, JSONValue], raw))


class ExecutionJournal:
    """Cache completed calls by action ID so a resumed run does not repeat effects."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._results = self._load()

    def _load(self) -> dict[str, ToolResult]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("journal root must be an object")
        results: dict[str, ToolResult] = {}
        for call_id, value in raw.items():
            if isinstance(value, dict):
                results[str(call_id)] = ToolResult.from_dict(cast(dict[str, JSONValue], value))
        return results

    def get(self, call_id: str) -> ToolResult | None:
        result = self._results.get(call_id)
        return replace(result, cached=True) if result is not None else None

    def put(self, result: ToolResult) -> None:
        self._results[result.call_id] = result
        _atomic_json_write(
            self.path, {call_id: item.to_dict() for call_id, item in self._results.items()}
        )
