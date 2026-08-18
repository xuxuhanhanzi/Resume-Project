"""File-backed memory that demonstrates scope without requiring a vector database."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]")


@dataclass(slots=True)
class SessionMemory:
    """High-signal facts and a rolling summary for one run."""

    facts: list[str] = field(default_factory=list)
    summary: str = ""

    def remember(self, fact: str) -> None:
        normalized = fact.strip()
        if normalized and normalized not in self.facts:
            self.facts.append(normalized[:1000])
            self.facts[:] = self.facts[-20:]

    def render(self) -> str:
        sections = []
        if self.summary:
            sections.append(f"Session summary: {self.summary}")
        if self.facts:
            sections.append("Known facts:\n" + "\n".join(f"- {fact}" for fact in self.facts))
        return "\n".join(sections)


@dataclass(frozen=True, slots=True)
class Episode:
    """One compact cross-task experience."""

    task_id: str
    problem: str
    strategy: str
    outcome: str


class EpisodicMemoryStore:
    """Append and retrieve a tiny keyword-ranked experience log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, episode: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(asdict(episode), ensure_ascii=False, sort_keys=True) + "\n")

    def load(self) -> list[Episode]:
        if not self.path.exists():
            return []
        episodes: list[Episode] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            if isinstance(raw, dict):
                value = cast(dict[str, object], raw)
                episodes.append(
                    Episode(
                        task_id=str(value["task_id"]),
                        problem=str(value["problem"]),
                        strategy=str(value["strategy"]),
                        outcome=str(value["outcome"]),
                    )
                )
        return episodes

    def search(self, query: str, *, limit: int = 3) -> list[Episode]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for episode in self.load():
            text = f"{episode.problem} {episode.strategy} {episode.outcome}".lower()
            score = sum(term in text for term in terms)
            if score:
                scored.append((score, episode.task_id, episode))
        return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]
