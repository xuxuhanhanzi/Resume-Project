"""Tests for strict tokenizer configuration."""

from pathlib import Path

import pytest

from forgellm.tokenization.config import (
    TokenizerConfig,
    TokenizerConfigError,
    load_tokenizer_config,
)


def _valid_values() -> dict[str, object]:
    return {"min_pair_frequency": 2, "vocab_size": 320}


def test_tokenizer_config_has_stable_fingerprint() -> None:
    first = TokenizerConfig.from_mapping(_valid_values())
    second = TokenizerConfig.from_mapping(dict(reversed(list(_valid_values().items()))))

    assert first == second
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("vocab_size", 259, "must be in"),
        ("vocab_size", True, "must be an integer"),
        ("min_pair_frequency", 0, "at least 1"),
        ("min_pair_frequency", 1.5, "must be an integer"),
    ],
)
def test_invalid_tokenizer_config_fails(field: str, value: object, message: str) -> None:
    values = _valid_values()
    values[field] = value

    with pytest.raises(TokenizerConfigError, match=message):
        TokenizerConfig.from_mapping(values)


def test_unknown_and_missing_fields_fail() -> None:
    with pytest.raises(TokenizerConfigError, match="Unknown tokenizer fields"):
        TokenizerConfig.from_mapping({**_valid_values(), "seed": 7})
    with pytest.raises(TokenizerConfigError, match="Missing tokenizer fields"):
        TokenizerConfig.from_mapping({"vocab_size": 320})


def test_load_requires_exact_top_level_table(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[tokenizer]\nvocab_size=320\nmin_pair_frequency=2\n[extra]\nx=1\n")

    with pytest.raises(TokenizerConfigError, match="exactly one"):
        load_tokenizer_config(path)
