"""Integration test for config validation and run initialization."""

import json
from pathlib import Path

import pytest

from forgellm.cli import main


def test_init_run_from_toml(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "smoke.toml"
    config.write_text(
        '[run]\nname="smoke"\nstage="stage01"\nseed=7\nlog_level="INFO"\n',
        encoding="utf-8",
    )

    result = main(
        [
            "init-run",
            "--config",
            str(config),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--repo-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert Path(payload["directory"]).is_dir()
