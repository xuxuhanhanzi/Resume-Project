"""Tests for run initialization and provenance capture."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forgellm.config import RunConfig
from forgellm.runtime import initialize_run


def test_initialize_run_writes_traceable_metadata(tmp_path: Path) -> None:
    config = RunConfig(name="smoke", stage="stage01", seed=7, log_level="INFO")
    timestamp = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

    artifacts = initialize_run(config, tmp_path / "artifacts", tmp_path, now=timestamp)

    assert artifacts.run_id.startswith("20260711T120000Z_stage01_smoke_")
    assert sorted(path.name for path in artifacts.directory.iterdir()) == [
        "config.resolved.json",
        "environment.json",
        "events.jsonl",
        "git.json",
        "run.json",
    ]
    resolved = json.loads((artifacts.directory / "config.resolved.json").read_text("utf-8"))
    assert resolved["seed"] == 7
    assert resolved["sha256"] == config.fingerprint()

    with pytest.raises(FileExistsError):
        initialize_run(config, tmp_path / "artifacts", tmp_path, now=timestamp)
