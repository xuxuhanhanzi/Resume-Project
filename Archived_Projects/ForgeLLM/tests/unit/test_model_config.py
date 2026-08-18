"""Validation tests for the Decoder-only model configuration."""

import pytest

from forgellm.model.config import ModelConfig, ModelConfigError


def test_default_model_config_has_consistent_head_dimensions() -> None:
    config = ModelConfig()

    assert config.head_dim == 32
    assert config.queries_per_kv == 2
    assert config.as_dict()["vocab_size"] == 320


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"d_model": 130}, "d_model must be divisible"),
        ({"n_heads": 6, "n_kv_heads": 4, "d_model": 120}, "n_heads must be divisible"),
        ({"d_model": 12, "n_heads": 4, "n_kv_heads": 2}, "head_dim must be even"),
        ({"dropout": 1.0}, "dropout"),
        ({"max_seq_len": 0}, "max_seq_len"),
    ],
)
def test_model_config_rejects_invalid_values(
    overrides: dict[str, int | float], message: str
) -> None:
    with pytest.raises(ModelConfigError, match=message):
        ModelConfig(**overrides)  # type: ignore[arg-type]
