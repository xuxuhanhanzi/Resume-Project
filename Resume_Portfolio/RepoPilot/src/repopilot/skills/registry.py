"""Discover skill metadata first and load procedures only after selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

_MAX_SKILL_FILE_CHARACTERS = 64_000


@dataclass(frozen=True, slots=True)
class Skill:
    """A procedural-memory package backed by one SKILL.md."""

    name: str
    description: str
    path: Path
    allowed_tools: tuple[str, ...] = ()
    source: str = "project"

    def load_instructions(self) -> str:
        try:
            text = self.path.read_text(encoding="utf-8")[:_MAX_SKILL_FILE_CHARACTERS]
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError(f"could not load skill {self.name!r}: {error}") from error
        if text.startswith("---"):
            _, _, remainder = text.partition("---")
            _, _, body = remainder.partition("---")
            return body.strip()[:12_000]
        return text.strip()[:12_000]


class SkillRegistry:
    """Keep startup context small by separating discovery from activation."""

    def __init__(
        self,
        root: Path,
        *,
        additional_roots: tuple[Path, ...] = (),
        plugin_roots: tuple[Path, ...] = (),
    ) -> None:
        self.root = root
        self.roots = (*additional_roots, *plugin_roots, root)
        self._sources = (
            *("user" for _ in additional_roots),
            *("plugin" for _ in plugin_roots),
            "project",
        )
        self._skills = self._discover()

    def _discover(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        for root, source in zip(self.roots, self._sources, strict=True):
            self._discover_root(root, source=source, skills=skills)
        return skills

    @staticmethod
    def _discover_root(root: Path, *, source: str, skills: dict[str, Skill]) -> None:
        if not root.is_dir():
            return
        resolved_root = root.resolve()
        for path in sorted(root.glob("*/SKILL.md")):
            try:
                resolved_path = path.resolve(strict=True)
            except OSError:
                continue
            if resolved_path.parent.parent != resolved_root:
                continue
            try:
                if resolved_path.stat().st_size > _MAX_SKILL_FILE_CHARACTERS:
                    raise ValueError(f"skill file is too large: {resolved_path}")
                text = resolved_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                raise ValueError(f"could not read skill {resolved_path}: {error}") from error
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
            raw_allowed_tools = metadata.get("allowed_tools", [])
            if not isinstance(raw_allowed_tools, list) or not all(
                isinstance(item, str) and item for item in raw_allowed_tools
            ):
                raise ValueError(
                    f"skill {name!r} allowed_tools must be a list of non-empty strings"
                )
            if not name:
                raise ValueError(f"invalid skill name in {resolved_path}")
            # A project-local procedure deliberately overrides a same-named user
            # procedure, matching normal repository-over-user configuration layering.
            if name in skills and source != "project":
                raise ValueError(f"duplicate user skill name: {name!r}")
            skills[name] = Skill(
                name,
                description,
                resolved_path,
                tuple(raw_allowed_tools),
                source,
            )

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
