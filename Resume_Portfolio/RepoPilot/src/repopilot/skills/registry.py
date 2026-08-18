"""Discover skill metadata first and load procedures only after selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True, slots=True)
class Skill:
    """A procedural-memory package backed by one SKILL.md."""

    name: str
    description: str
    path: Path

    def load_instructions(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        if text.startswith("---"):
            _, _, remainder = text.partition("---")
            _, _, body = remainder.partition("---")
            return body.strip()
        return text.strip()


class SkillRegistry:
    """Keep startup context small by separating discovery from activation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._skills = self._discover()

    def _discover(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self.root.exists():
            return skills
        for path in sorted(self.root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            metadata: dict[str, Any] = {}
            if text.startswith("---"):
                _, _, remainder = text.partition("---")
                header, separator, _ = remainder.partition("---")
                if separator:
                    raw = yaml.safe_load(header) or {}
                    if isinstance(raw, dict):
                        metadata = cast(dict[str, Any], raw)
            name = str(metadata.get("name", path.parent.name))
            description = str(metadata.get("description", ""))
            if not name or name in skills:
                raise ValueError(f"invalid or duplicate skill name: {name!r}")
            skills[name] = Skill(name, description, path)
        return skills

    def descriptors(self) -> tuple[tuple[str, str], ...]:
        return tuple((skill.name, skill.description) for skill in self._skills.values())

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def select(self, problem: str, *, limit: int = 2) -> tuple[Skill, ...]:
        lowered = problem.lower()
        scored: list[tuple[int, str, Skill]] = []
        for skill in self._skills.values():
            terms = set(f"{skill.name} {skill.description}".lower().replace("-", " ").split())
            score = sum(term in lowered for term in terms if len(term) >= 3)
            if score:
                scored.append((score, skill.name, skill))
        return tuple(
            item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
        )
