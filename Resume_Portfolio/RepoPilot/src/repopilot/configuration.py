"""User-owned model profiles and project shortcuts.

This module deliberately stores no API keys. Profiles live below the user's
home RepoPilot root, never inside a repository, while credentials are held by
the operating-system credential store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from repopilot.providers.deepseek import SUPPORTED_DEEPSEEK_MODELS

_PROFILE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONFIG_VERSION = 1
_CLOUD_PROVIDERS = ("deepseek", "qwen")


def cloud_providers() -> tuple[str, ...]:
    """Return the explicitly supported remote providers."""
    return _CLOUD_PROVIDERS


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One named provider and model pair, without a credential."""

    name: str
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.name):
            raise ValueError("profile names must be 1-64 lowercase letters, digits, '_' or '-'")
        if self.provider not in _CLOUD_PROVIDERS:
            allowed = ", ".join(_CLOUD_PROVIDERS)
            raise ValueError(f"profile provider must be one of: {allowed}")
        if not _MODEL_NAME.fullmatch(self.model):
            raise ValueError("model names must be a safe 1-128 character identifier")
        if self.provider == "deepseek" and self.model not in SUPPORTED_DEEPSEEK_MODELS:
            allowed = ", ".join(SUPPORTED_DEEPSEEK_MODELS)
            raise ValueError(f"DeepSeek model must be one of: {allowed}")


@dataclass(frozen=True, slots=True)
class ProjectShortcut:
    """A user-local, explicit path alias for launching one workspace."""

    name: str
    project_root: Path
    profile: str | None = None

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.name):
            raise ValueError("project names must be 1-64 lowercase letters, digits, '_' or '-'")
        if self.profile is not None and not _PROFILE_NAME.fullmatch(self.profile):
            raise ValueError("project profile must be a valid profile name")


@dataclass(frozen=True, slots=True)
class UserConfiguration:
    """The complete non-secret user configuration."""

    default_profile: str | None = None
    profiles: tuple[ModelProfile, ...] = ()
    projects: tuple[ProjectShortcut, ...] = ()

    def __post_init__(self) -> None:
        profile_names = tuple(profile.name for profile in self.profiles)
        if len(profile_names) != len(set(profile_names)):
            raise ValueError("profile names must be unique")
        project_names = tuple(project.name for project in self.projects)
        if len(project_names) != len(set(project_names)):
            raise ValueError("project names must be unique")
        if self.default_profile is not None and self.profile(self.default_profile) is None:
            raise ValueError("default_profile must name a configured profile")
        profile_set = set(profile_names)
        if any(
            shortcut.profile is not None and shortcut.profile not in profile_set
            for shortcut in self.projects
        ):
            raise ValueError("project profile must name a configured profile")

    def profile(self, name: str) -> ModelProfile | None:
        return next((profile for profile in self.profiles if profile.name == name), None)

    def project(self, name: str) -> ProjectShortcut | None:
        return next((shortcut for shortcut in self.projects if shortcut.name == name), None)


class UserConfigurationStore:
    """Read and write one strict, user-owned configuration document."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @property
    def path(self) -> Path:
        return self.root / "config.json"

    def load(self) -> UserConfiguration:
        if not self.path.exists():
            return UserConfiguration()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid RepoPilot user configuration at {self.path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("RepoPilot user configuration must be an object")
        expected = {"version", "default_profile", "profiles", "projects"}
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown user configuration keys: {', '.join(sorted(unknown))}")
        if raw.get("version") != _CONFIG_VERSION:
            raise ValueError(f"RepoPilot user configuration version must be {_CONFIG_VERSION}")
        default_profile = raw.get("default_profile")
        if default_profile is not None and not isinstance(default_profile, str):
            raise ValueError("default_profile must be a string or null")
        profiles = self._profiles(raw.get("profiles", {}))
        projects = self._projects(raw.get("projects", {}))
        return UserConfiguration(default_profile, profiles, projects)

    def save(self, configuration: UserConfiguration) -> None:
        payload = {
            "version": _CONFIG_VERSION,
            "default_profile": configuration.default_profile,
            "profiles": {
                profile.name: {"provider": profile.provider, "model": profile.model}
                for profile in sorted(configuration.profiles, key=lambda item: item.name)
            },
            "projects": {
                shortcut.name: {
                    "path": str(shortcut.project_root),
                    "profile": shortcut.profile,
                }
                for shortcut in sorted(configuration.projects, key=lambda item: item.name)
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def set_profile(
        self,
        configuration: UserConfiguration,
        profile: ModelProfile,
        *,
        make_default: bool = False,
    ) -> UserConfiguration:
        profiles = {item.name: item for item in configuration.profiles}
        profiles[profile.name] = profile
        default = (
            profile.name
            if make_default or configuration.default_profile is None
            else (configuration.default_profile)
        )
        updated = UserConfiguration(default, tuple(profiles.values()), configuration.projects)
        self.save(updated)
        return updated

    def set_default(self, configuration: UserConfiguration, profile_name: str) -> UserConfiguration:
        if configuration.profile(profile_name) is None:
            raise ValueError(f"unknown model profile: {profile_name}")
        updated = UserConfiguration(profile_name, configuration.profiles, configuration.projects)
        self.save(updated)
        return updated

    def set_project(
        self,
        configuration: UserConfiguration,
        shortcut: ProjectShortcut,
    ) -> UserConfiguration:
        canonical = shortcut.project_root.resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError(f"project path is not a directory: {canonical}")
        if shortcut.profile is not None and configuration.profile(shortcut.profile) is None:
            raise ValueError(f"unknown model profile: {shortcut.profile}")
        projects = {item.name: item for item in configuration.projects}
        projects[shortcut.name] = ProjectShortcut(shortcut.name, canonical, shortcut.profile)
        updated = UserConfiguration(
            configuration.default_profile, configuration.profiles, tuple(projects.values())
        )
        self.save(updated)
        return updated

    def resolve_project(self, configuration: UserConfiguration, name: str) -> ProjectShortcut:
        shortcut = configuration.project(name)
        if shortcut is None:
            raise ValueError(f"unknown project shortcut: {name}")
        canonical = shortcut.project_root.resolve(strict=True)
        if not canonical.is_dir():
            raise ValueError(f"project path is not a directory: {canonical}")
        return ProjectShortcut(shortcut.name, canonical, shortcut.profile)

    @staticmethod
    def _profiles(raw: object) -> tuple[ModelProfile, ...]:
        if not isinstance(raw, dict):
            raise ValueError("profiles must be an object")
        profiles: list[ModelProfile] = []
        for name, definition in raw.items():
            if not isinstance(name, str) or not isinstance(definition, dict):
                raise ValueError("profiles must map names to objects")
            if set(definition) != {"provider", "model"}:
                raise ValueError(f"profile {name!r} must contain only provider and model")
            provider = definition.get("provider")
            model = definition.get("model")
            if not isinstance(provider, str) or not isinstance(model, str):
                raise ValueError(f"profile {name!r} provider and model must be strings")
            profiles.append(ModelProfile(name, provider, model))
        return tuple(profiles)

    @staticmethod
    def _projects(raw: object) -> tuple[ProjectShortcut, ...]:
        if not isinstance(raw, dict):
            raise ValueError("projects must be an object")
        projects: list[ProjectShortcut] = []
        for name, definition in raw.items():
            if not isinstance(name, str) or not isinstance(definition, dict):
                raise ValueError("projects must map names to objects")
            if set(definition) != {"path", "profile"}:
                raise ValueError(f"project {name!r} must contain only path and profile")
            path = definition.get("path")
            profile = definition.get("profile")
            if not isinstance(path, str) or (profile is not None and not isinstance(profile, str)):
                raise ValueError(
                    f"project {name!r} path must be a string and profile a string or null"
                )
            projects.append(ProjectShortcut(name, Path(path), profile))
        return tuple(projects)
