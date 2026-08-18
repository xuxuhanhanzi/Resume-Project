"""CPU-only smoke test for tokenizer train and evaluate CLI commands."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_tokenizer_cli_train_then_evaluate(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    artifact_dir = tmp_path / "candidate"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgellm",
            "tokenizer-train",
            "--config",
            str(repo_root / "configs" / "tokenizer" / "bpe_v1.toml"),
            "--input",
            str(repo_root / "tests" / "fixtures" / "tokenizer" / "train.jsonl"),
            "--output-dir",
            str(artifact_dir),
            "--source-name",
            "forgellm-test-fixture",
            "--source-license",
            "project-test-fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert train.returncode == 0, train.stderr
    train_payload = json.loads(train.stdout)
    assert train_payload["vocab_size"] > 260
    model_path = Path(train_payload["model"])
    assert model_path.is_file()

    output_path = tmp_path / "evaluation.json"
    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgellm",
            "tokenizer-evaluate",
            "--model",
            str(model_path),
            "--input",
            str(repo_root / "tests" / "fixtures" / "tokenizer" / "validation.jsonl"),
            "--output",
            str(output_path),
            "--encode-repeats",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert evaluate.returncode == 0, evaluate.stderr
    assert output_path.is_file()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["results"]["forgellm_byte_bpe"]["all"]["round_trip_rate"] == 1.0


def test_tokenizer_reference_compare_cli(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    repo_root = Path(__file__).parents[2]
    artifact_dir = tmp_path / "candidate"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgellm",
            "tokenizer-train",
            "--config",
            str(repo_root / "configs" / "tokenizer" / "bpe_v1.toml"),
            "--input",
            str(repo_root / "tests" / "fixtures" / "tokenizer" / "train.jsonl"),
            "--output-dir",
            str(artifact_dir),
            "--source-name",
            "forgellm-test-fixture",
            "--source-license",
            "project-test-fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert train.returncode == 0, train.stderr

    comparison_dir = tmp_path / "comparison"
    compare = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgellm",
            "tokenizer-compare",
            "--config",
            str(repo_root / "configs" / "tokenizer" / "bpe_v1.toml"),
            "--model",
            str(artifact_dir / "tokenizer.json"),
            "--train-input",
            str(repo_root / "tests" / "fixtures" / "tokenizer" / "train.jsonl"),
            "--evaluation-input",
            str(repo_root / "tests" / "fixtures" / "tokenizer" / "validation.jsonl"),
            "--output-dir",
            str(comparison_dir),
            "--encode-repeats",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert compare.returncode == 0, compare.stderr
    report = json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    assert report["implementations"]["huggingface_bytelevel_bpe"]["library_version"] == ("0.22.2")
