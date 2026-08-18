"""Tests for validated run configuration."""

from pathlib import Path

import pytest

from forgellm.config import ConfigError, RunConfig, load_run_config


def test_config_fingerprint_is_order_independent() -> None:
    first = RunConfig.from_mapping(
        {"name": "smoke", "stage": "stage01", "seed": 7, "log_level": "INFO"}
    )
    second = RunConfig.from_mapping(
        {"seed": 7, "log_level": "INFO", "stage": "stage01", "name": "smoke"}
    )

    assert first == second
    assert first.fingerprint() == second.fingerprint()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Not A Slug"),
        ("stage", "../escape"),
        ("seed", -1),
        ("seed", True),
        ("log_level", "info"),
    ],
)
def test_invalid_config_values_fail(field: str, value: object) -> None:
    values: dict[str, object] = {
        "name": "smoke",
        "stage": "stage01",
        "seed": 7,
        "log_level": "INFO",
    }
    values[field] = value

    with pytest.raises(ConfigError):
        RunConfig.from_mapping(values)


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        '[run]\nname="smoke"\nstage="stage01"\nseed=7\nlog_level="INFO"\nsecret="x"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Unknown run fields: secret"):
        load_run_config(path)
