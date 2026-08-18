"""Strict configuration for ForgeLLM byte-level BPE tokenizers."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from forgellm.structured_logging import JsonValue

BASE_VOCAB_SIZE = 260
_TOKENIZER_FIELDS = frozenset({"min_pair_frequency", "vocab_size"})


class TokenizerConfigError(ValueError):
    """Raised when a tokenizer configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Resolved byte-level BPE training settings."""

    vocab_size: int
    min_pair_frequency: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TokenizerConfig:
        """Validate and construct tokenizer settings."""
        unknown = set(values) - _TOKENIZER_FIELDS
        missing = _TOKENIZER_FIELDS - set(values)
        if unknown:
            raise TokenizerConfigError(f"Unknown tokenizer fields: {', '.join(sorted(unknown))}")
        if missing:
            raise TokenizerConfigError(f"Missing tokenizer fields: {', '.join(sorted(missing))}")

        vocab_size = values["vocab_size"]
        min_pair_frequency = values["min_pair_frequency"]
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TokenizerConfigError("tokenizer.vocab_size must be an integer")
        if not BASE_VOCAB_SIZE <= vocab_size <= 65536:
            raise TokenizerConfigError(
                f"tokenizer.vocab_size must be in [{BASE_VOCAB_SIZE}, 65536]"
            )
        if isinstance(min_pair_frequency, bool) or not isinstance(min_pair_frequency, int):
            raise TokenizerConfigError("tokenizer.min_pair_frequency must be an integer")
        if min_pair_frequency < 1:
            raise TokenizerConfigError("tokenizer.min_pair_frequency must be at least 1")
        return cls(vocab_size=vocab_size, min_pair_frequency=min_pair_frequency)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible mapping."""
        return {
            "min_pair_frequency": self.min_pair_frequency,
            "vocab_size": self.vocab_size,
        }

    def fingerprint(self) -> str:
        """Return the SHA-256 of the resolved configuration."""
        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_tokenizer_config(path: Path) -> TokenizerConfig:
    """Load a TOML file containing exactly one ``tokenizer`` table."""
    try:
        with path.open("rb") as stream:
            document = cast(dict[str, object], tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TokenizerConfigError(f"Could not read tokenizer config {path}: {error}") from error

    if set(document) != {"tokenizer"}:
        raise TokenizerConfigError("config must contain exactly one top-level [tokenizer] table")
    values = document["tokenizer"]
    if not isinstance(values, Mapping):
        raise TokenizerConfigError("[tokenizer] must be a TOML table")
    return TokenizerConfig.from_mapping(cast(Mapping[str, object], values))
