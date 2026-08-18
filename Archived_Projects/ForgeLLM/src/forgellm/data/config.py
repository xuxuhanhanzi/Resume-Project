"""Configuration for the deterministic data pipeline."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from forgellm.structured_logging import JsonValue

_DATA_FIELDS = frozenset(
    {
        "max_chars",
        "min_chars",
        "split_seed",
        "test_bps",
        "train_bps",
        "unicode_normalization",
        "validation_bps",
    }
)
_NORMALIZATION_FORMS = frozenset({"NFC", "NFD", "NFKC", "NFKD"})
NormalizationForm = Literal["NFC", "NFD", "NFKC", "NFKD"]


class DataConfigError(ValueError):
    """Raised when a data pipeline configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Resolved settings for one deterministic data pipeline run."""

    unicode_normalization: NormalizationForm
    min_chars: int
    max_chars: int
    split_seed: int
    train_bps: int
    validation_bps: int
    test_bps: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> DataConfig:
        """Validate and construct the data settings."""
        unknown = set(values) - _DATA_FIELDS
        missing = _DATA_FIELDS - set(values)
        if unknown:
            raise DataConfigError(f"Unknown data fields: {', '.join(sorted(unknown))}")
        if missing:
            raise DataConfigError(f"Missing data fields: {', '.join(sorted(missing))}")

        normalization = values["unicode_normalization"]
        if not isinstance(normalization, str) or normalization not in _NORMALIZATION_FORMS:
            raise DataConfigError("unicode_normalization must be NFC, NFD, NFKC, or NFKD")

        integer_fields = {
            name: values[name]
            for name in (
                "min_chars",
                "max_chars",
                "split_seed",
                "train_bps",
                "validation_bps",
                "test_bps",
            )
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise DataConfigError(f"{name} must be an integer")

        min_chars = cast(int, integer_fields["min_chars"])
        max_chars = cast(int, integer_fields["max_chars"])
        split_seed = cast(int, integer_fields["split_seed"])
        train_bps = cast(int, integer_fields["train_bps"])
        validation_bps = cast(int, integer_fields["validation_bps"])
        test_bps = cast(int, integer_fields["test_bps"])

        if min_chars < 0 or max_chars < min_chars:
            raise DataConfigError("require 0 <= min_chars <= max_chars")
        if not 0 <= split_seed < 2**32:
            raise DataConfigError("split_seed must be in [0, 2**32)")
        if any(value < 0 for value in (train_bps, validation_bps, test_bps)):
            raise DataConfigError("split basis points cannot be negative")
        if train_bps + validation_bps + test_bps != 10000:
            raise DataConfigError("train_bps + validation_bps + test_bps must equal 10000")

        return cls(
            unicode_normalization=cast(NormalizationForm, normalization),
            min_chars=min_chars,
            max_chars=max_chars,
            split_seed=split_seed,
            train_bps=train_bps,
            validation_bps=validation_bps,
            test_bps=test_bps,
        )

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible mapping."""
        return {
            "max_chars": self.max_chars,
            "min_chars": self.min_chars,
            "split_seed": self.split_seed,
            "test_bps": self.test_bps,
            "train_bps": self.train_bps,
            "unicode_normalization": self.unicode_normalization,
            "validation_bps": self.validation_bps,
        }

    def fingerprint(self) -> str:
        """Hash the resolved settings."""
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_data_config(path: Path) -> DataConfig:
    """Load a TOML file containing exactly one ``data`` table."""
    try:
        with path.open("rb") as stream:
            document = cast(dict[str, object], tomllib.load(stream))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DataConfigError(f"Could not read data config {path}: {error}") from error

    if set(document) != {"data"}:
        raise DataConfigError("config must contain exactly one top-level [data] table")
    values = document["data"]
    if not isinstance(values, Mapping):
        raise DataConfigError("[data] must be a TOML table")
    return DataConfig.from_mapping(cast(Mapping[str, object], values))
