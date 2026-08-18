"""Tests for data pipeline configuration."""

import pytest

from forgellm.data.config import DataConfig, DataConfigError


def _valid_values() -> dict[str, object]:
    return {
        "max_chars": 100,
        "min_chars": 5,
        "split_seed": 7,
        "test_bps": 1000,
        "train_bps": 8000,
        "unicode_normalization": "NFC",
        "validation_bps": 1000,
    }


def test_data_config_has_stable_fingerprint() -> None:
    first = DataConfig.from_mapping(_valid_values())
    second = DataConfig.from_mapping(dict(reversed(list(_valid_values().items()))))

    assert first == second
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unicode_normalization", "UNKNOWN", "unicode_normalization"),
        ("min_chars", -1, "require 0 <= min_chars"),
        ("max_chars", 4, "require 0 <= min_chars"),
        ("split_seed", -1, "split_seed"),
        ("train_bps", 7000, "must equal 10000"),
        ("test_bps", True, "must be an integer"),
    ],
)
def test_invalid_data_config_fails(field: str, value: object, message: str) -> None:
    values = _valid_values()
    values[field] = value

    with pytest.raises(DataConfigError, match=message):
        DataConfig.from_mapping(values)
