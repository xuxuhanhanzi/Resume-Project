from __future__ import annotations

from pathlib import Path

import pytest

from forgellm.alignment.config import Stage5ConfigError, load_stage5_run_config


def test_stage5_config_loads_and_is_fingerprinted() -> None:
    path = Path("configs/alignment/stage5_qwen_bounded.toml")
    first = load_stage5_run_config(path)
    second = load_stage5_run_config(path)
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64
    assert first.dpo.beta == 0.1
    assert first.grpo.group_size == 4


def test_stage5_config_rejects_unknown_fields(tmp_path: Path) -> None:
    source = Path("configs/alignment/stage5_qwen_bounded.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    path.write_text(source.replace("beta = 0.1", "beta = 0.1\nunknown = 1"), encoding="utf-8")
    with pytest.raises(Stage5ConfigError, match="unknown"):
        load_stage5_run_config(path)
