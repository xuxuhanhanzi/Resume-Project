"""Validated configuration primitives for reproducible runs."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_RUN_FIELDS = frozenset({"name", "stage", "seed", "log_level"})
_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ConfigError(ValueError):
    """Raised when a ForgeLLM configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Minimal configuration required to initialize a run."""

    name: str
    stage: str
    seed: int
    log_level: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> RunConfig:
        """Validate and construct a run configuration."""
        unknown = set(values) - _RUN_FIELDS
        missing = _RUN_FIELDS - set(values)
        if unknown:
            raise ConfigError(f"Unknown run fields: {', '.join(sorted(unknown))}")
        if missing:
            raise ConfigError(f"Missing run fields: {', '.join(sorted(missing))}")

        name = values["name"]
        stage = values["stage"]
        seed = values["seed"]
        log_level = values["log_level"]

        if not isinstance(name, str) or not _SLUG_PATTERN.fullmatch(name):
            raise ConfigError("run.name must be a lowercase slug with at most 64 characters")
        if not isinstance(stage, str) or not _SLUG_PATTERN.fullmatch(stage):
            raise ConfigError("run.stage must be a lowercase slug with at most 64 characters")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ConfigError("run.seed must be an integer in [0, 2**32)")
        if not isinstance(log_level, str) or log_level not in _LOG_LEVELS:
            raise ConfigError(f"run.log_level must be one of: {', '.join(sorted(_LOG_LEVELS))}")

        return cls(name=name, stage=stage, seed=seed, log_level=log_level)

    def as_dict(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation."""
        return {
            "log_level": self.log_level,
            "name": self.name,
            "seed": self.seed,
            "stage": self.stage,
        }

    def fingerprint(self) -> str:
        """Return a stable SHA-256 hash of the resolved configuration."""
        payload = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_run_config(path: Path) -> RunConfig:
    """Load a TOML file containing exactly one ``run`` table."""
    try:
        with path.open("rb") as stream:
            document = cast(dict[str, object], tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Could not read config {path}: {error}") from error

    if set(document) != {"run"}:
        raise ConfigError("config must contain exactly one top-level [run] table")

    run_values = document["run"]
    if not isinstance(run_values, Mapping):
        raise ConfigError("[run] must be a TOML table")
    return RunConfig.from_mapping(cast(Mapping[str, object], run_values))
