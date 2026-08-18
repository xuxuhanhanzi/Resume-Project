from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .types import SceneContext


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_manifest(path: str | Path) -> list[tuple[dict[str, Any], SceneContext]]:
    result = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        if "scene" not in row:
            raise ValueError("each manifest row requires a 'scene' object")
        context = SceneContext.from_dict(row["scene"])
        if context.scene_id in seen:
            raise ValueError(f"duplicate scene_id in manifest: {context.scene_id}")
        seen.add(context.scene_id)
        result.append((row.get("model_input", {}), context))
    if not result:
        raise ValueError("manifest is empty")
    return result
