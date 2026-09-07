"""Manifest-only local plugins: procedures may extend context, never executable authority."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_VERSION = re.compile(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?\Z")


@dataclass(frozen=True, slots=True)
class PluginManifest:
    name: str
    version: str
    description: str
    root: Path
    source: str

    @property
    def skills_root(self) -> Path:
        return self.root / "skills"


def discover_plugins(project_root: Path, user_root: Path) -> tuple[PluginManifest, ...]:
    """Load bounded manifests from user/project folders; project names override user names."""

    discovered: dict[str, PluginManifest] = {}
    for source, root in (
        ("user-plugin", user_root.resolve() / "plugins"),
        ("project-plugin", project_root.resolve() / ".repopilot" / "plugins"),
    ):
        if not root.is_dir():
            continue
        children = sorted(path for path in root.iterdir() if path.is_dir())
        if len(children) > 32:
            raise ValueError(f"plugin root {root} contains more than 32 entries")
        for child in children:
            manifest_path = child / "plugin.json"
            if not manifest_path.is_file():
                continue
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid plugin manifest {manifest_path}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"plugin manifest {manifest_path} must be an object")
            name = raw.get("name")
            version = raw.get("version")
            description = raw.get("description", "")
            if (
                not isinstance(name, str)
                or not _NAME.fullmatch(name)
                or not isinstance(version, str)
                or not _VERSION.fullmatch(version)
                or not isinstance(description, str)
                or len(description.strip()) > 500
                or child.name != name
            ):
                raise ValueError(
                    f"plugin manifest {manifest_path} has invalid name, version, or description"
                )
            discovered[name] = PluginManifest(
                name, version, description.strip(), child.resolve(), source
            )
    return tuple(discovered[name] for name in sorted(discovered))


def plugin_skill_roots(project_root: Path, user_root: Path) -> tuple[Path, ...]:
    """Return only declared, local skill directories—plugins never contribute executable code."""

    return tuple(
        plugin.skills_root
        for plugin in discover_plugins(project_root, user_root)
        if plugin.skills_root.is_dir()
    )
