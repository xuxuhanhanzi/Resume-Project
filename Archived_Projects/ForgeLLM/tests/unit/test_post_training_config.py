"""Strict Stage 4 run configuration tests."""

from pathlib import Path

import pytest

from forgellm.post_training.config import Stage4ConfigError, load_stage4_run_config


def test_real_stage4_config_loads_and_has_stable_fingerprint() -> None:
    path = Path("configs/post_training/stage4_qwen_lora_smoke.toml")

    first = load_stage4_run_config(path)
    second = load_stage4_run_config(path)

    assert first.model.revision == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert first.model.quantization == "none"
    assert first.training.warmup_ratio == 0.03
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = Path("configs/post_training/stage4_qwen_lora_smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "unknown.toml"
    path.write_text(
        source.replace('quantization = "none"', 'quantization = "none"\nmagic = 1'),
        encoding="utf-8",
    )

    with pytest.raises(Stage4ConfigError, match="fields differ"):
        load_stage4_run_config(path)


def test_config_rejects_boolean_integer(tmp_path: Path) -> None:
    source = Path("configs/post_training/stage4_qwen_lora_smoke.toml").read_text(encoding="utf-8")
    path = tmp_path / "boolean.toml"
    path.write_text(source.replace("max_steps = 2", "max_steps = true"), encoding="utf-8")

    with pytest.raises(Stage4ConfigError, match="max_steps must be an integer"):
        load_stage4_run_config(path)
