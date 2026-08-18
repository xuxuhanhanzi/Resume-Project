"""Integration tests for tokenizer training, freezing, and evaluation."""

import json
from pathlib import Path
from typing import cast

import pytest

from forgellm.tokenization.config import TokenizerConfig
from forgellm.tokenization.workflow import (
    compare_with_huggingface_reference,
    evaluate_tokenizer_artifact,
    train_tokenizer_artifact,
)


def _fixtures() -> Path:
    return Path(__file__).parents[2] / "tests" / "fixtures" / "tokenizer"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def test_training_and_evaluation_freeze_auditable_artifacts(tmp_path: Path) -> None:
    training = train_tokenizer_artifact(
        TokenizerConfig(vocab_size=320, min_pair_frequency=2),
        _fixtures() / "train.jsonl",
        tmp_path / "candidate",
        source_name="forgellm-test-fixture",
        source_license="project-test-fixture",
    )
    report = evaluate_tokenizer_artifact(
        training.model_path,
        _fixtures() / "test.jsonl",
        tmp_path / "evaluation.json",
        encode_repeats=1,
    )

    manifest = json.loads(training.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "forgellm-tokenizer-training-manifest-v1"
    assert manifest["model"]["sha256"] == training.model_sha256
    assert report["schema_version"] == "forgellm-tokenizer-evaluation-v1"
    results = _mapping(report["results"])
    bpe_metrics = _mapping(_mapping(results["forgellm_byte_bpe"])["all"])
    raw_metrics = _mapping(_mapping(results["raw_utf8_bytes"])["all"])
    assert bpe_metrics["round_trip_rate"] == 1.0
    assert bpe_metrics["unknown_rate"] == 0.0
    bpe_bytes_per_token = bpe_metrics["bytes_per_token"]
    raw_bytes_per_token = raw_metrics["bytes_per_token"]
    assert isinstance(bpe_bytes_per_token, int | float)
    assert isinstance(raw_bytes_per_token, int | float)
    assert bpe_bytes_per_token > raw_bytes_per_token
    assert _mapping(report["special_token_accuracy"])["rate"] == 1.0


def test_model_hash_does_not_depend_on_training_document_order(tmp_path: Path) -> None:
    source = _fixtures() / "train.jsonl"
    reversed_source = tmp_path / "reversed.jsonl"
    reversed_source.write_text(
        "\n".join(reversed(source.read_text(encoding="utf-8").splitlines())) + "\n",
        encoding="utf-8",
    )
    config = TokenizerConfig(vocab_size=320, min_pair_frequency=2)

    first = train_tokenizer_artifact(
        config,
        source,
        tmp_path / "first",
        source_name="fixture",
        source_license="project-test-fixture",
    )
    second = train_tokenizer_artifact(
        config,
        reversed_source,
        tmp_path / "second",
        source_name="fixture",
        source_license="project-test-fixture",
    )

    assert first.model_sha256 == second.model_sha256


def test_optional_reference_comparison_uses_same_corpora_and_budget(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    config = TokenizerConfig(vocab_size=320, min_pair_frequency=2)
    training = train_tokenizer_artifact(
        config,
        _fixtures() / "train.jsonl",
        tmp_path / "candidate",
        source_name="fixture",
        source_license="project-test-fixture",
    )

    report = compare_with_huggingface_reference(
        config,
        training.model_path,
        _fixtures() / "train.jsonl",
        _fixtures() / "validation.jsonl",
        tmp_path / "comparison",
        encode_repeats=1,
    )

    assert report["schema_version"] == "forgellm-tokenizer-reference-comparison-v1"
    results = _mapping(report["results"])
    reference = _mapping(_mapping(results["huggingface_bytelevel_bpe"])["all"])
    assert reference["round_trip_rate"] == 1.0
    assert reference["unknown_rate"] == 0.0
    implementations = _mapping(report["implementations"])
    reference_implementation = _mapping(implementations["huggingface_bytelevel_bpe"])
    assert reference_implementation["library_version"] == "0.22.2"
    assert (tmp_path / "comparison" / "comparison.json").is_file()
