"""Integration tests for the complete deterministic data pipeline."""

import hashlib
import json
from pathlib import Path

from forgellm.data.config import DataConfig
from forgellm.data.pipeline import run_data_pipeline


def _config() -> DataConfig:
    return DataConfig(
        unicode_normalization="NFC",
        min_chars=12,
        max_chars=1000,
        split_seed=6,
        train_bps=8000,
        validation_bps=1000,
        test_bps=1000,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixture_pipeline_produces_manifest_and_explainable_rejections(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    input_path = repo_root / "tests" / "fixtures" / "data" / "sample_documents.jsonl"

    result = run_data_pipeline(
        _config(),
        input_path,
        tmp_path / "output",
        source_name="forgellm-test-fixture",
        source_license="project-test-fixture",
    )

    assert result.input_records == 9
    assert result.retained_records == 4
    assert result.rejected_records == 5
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["rejection_reasons"] == {
        "disallowed_control_character": 1,
        "duplicate_exact": 1,
        "empty_text": 1,
        "invalid_schema": 1,
        "too_short": 1,
    }
    assert report["split_records"] == {"test": 1, "train": 2, "validation": 1}

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "forgellm-data-manifest-v1"
    assert manifest["source"]["license"] == "project-test-fixture"
    assert set(manifest["outputs"]) == {
        "rejects.jsonl",
        "report.json",
        "test.jsonl",
        "train.jsonl",
        "validation.jsonl",
    }


def test_retained_outputs_do_not_depend_on_input_order(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    original = repo_root / "tests" / "fixtures" / "data" / "sample_documents.jsonl"
    reversed_input = tmp_path / "reversed.jsonl"
    lines = original.read_text(encoding="utf-8").splitlines()
    reversed_input.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    first = run_data_pipeline(
        _config(),
        original,
        tmp_path / "first",
        source_name="fixture",
        source_license="project-fixture",
    )
    second = run_data_pipeline(
        _config(),
        reversed_input,
        tmp_path / "second",
        source_name="fixture",
        source_license="project-fixture",
    )

    for file_name in ("train.jsonl", "validation.jsonl", "test.jsonl", "rejects.jsonl"):
        assert _sha256(first.output_dir / file_name) == _sha256(second.output_dir / file_name)
