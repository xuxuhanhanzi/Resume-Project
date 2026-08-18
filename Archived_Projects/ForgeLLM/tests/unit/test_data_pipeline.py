"""Unit tests for normalization, split assignment, and pipeline safety."""

from pathlib import Path

import pytest

from forgellm.data.config import DataConfig
from forgellm.data.pipeline import (
    DataPipelineError,
    assign_split,
    normalize_text,
    run_data_pipeline,
)


def _config() -> DataConfig:
    return DataConfig(
        unicode_normalization="NFC",
        min_chars=5,
        max_chars=1000,
        split_seed=7,
        train_bps=8000,
        validation_bps=1000,
        test_bps=1000,
    )


def test_normalize_text_handles_unicode_newlines_and_trailing_space() -> None:
    assert normalize_text("  Cafe\u0301   \r\nnext line  \r", "NFC") == "Café\nnext line"


def test_assign_split_is_deterministic() -> None:
    content_hash = "a" * 64
    assert assign_split(content_hash, _config()) == assign_split(content_hash, _config())
    assert assign_split(content_hash, _config()) in {"train", "validation", "test"}


def test_pipeline_refuses_to_overwrite_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id":"doc","text":"long enough text"}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        run_data_pipeline(
            _config(),
            input_path,
            output_dir,
            source_name="fixture",
            source_license="project-fixture",
        )


def test_pipeline_requires_source_provenance(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id":"doc","text":"long enough text"}\n', encoding="utf-8")

    with pytest.raises(DataPipelineError, match="source_name"):
        run_data_pipeline(
            _config(),
            input_path,
            tmp_path / "output",
            source_name="",
            source_license="project-fixture",
        )
