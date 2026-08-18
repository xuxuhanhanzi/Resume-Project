"""Reproducible tokenizer training and evaluation workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forgellm.structured_logging import JsonValue
from forgellm.tokenization.bpe import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    ByteBPETokenizer,
)
from forgellm.tokenization.config import TokenizerConfig
from forgellm.tokenization.corpus import read_tokenizer_jsonl, sha256_file
from forgellm.tokenization.evaluation import RawByteTokenizer, evaluate_with_subsets

_TRAINING_MANIFEST_SCHEMA = "forgellm-tokenizer-training-manifest-v1"
_EVALUATION_SCHEMA = "forgellm-tokenizer-evaluation-v1"


@dataclass(frozen=True, slots=True)
class TokenizerTrainingResult:
    """Artifacts returned by one tokenizer training run."""

    output_dir: Path
    model_path: Path
    manifest_path: Path
    documents: int
    vocab_size: int
    merge_count: int
    model_sha256: str


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def train_tokenizer_artifact(
    config: TokenizerConfig,
    input_path: Path,
    output_dir: Path,
    *,
    source_name: str,
    source_license: str,
) -> TokenizerTrainingResult:
    """Train a byte BPE model and freeze its provenance manifest."""
    if not source_name.strip() or not source_license.strip():
        raise ValueError("source_name and source_license must be non-empty")
    documents = read_tokenizer_jsonl(input_path)
    if not documents:
        raise ValueError("tokenizer training corpus must contain at least one document")

    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = ByteBPETokenizer.train(config, (document.text for document in documents))
    model_path = output_dir / "tokenizer.json"
    tokenizer.save(model_path)

    manifest: dict[str, JsonValue] = {
        "config": {"resolved": config.as_dict(), "sha256": config.fingerprint()},
        "input": {
            "bytes": input_path.stat().st_size,
            "documents": len(documents),
            "file_name": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "model": {
            "file_name": model_path.name,
            "merge_count": tokenizer.merge_count,
            "sha256": tokenizer.fingerprint(),
            "vocab_size": tokenizer.vocab_size,
        },
        "schema_version": _TRAINING_MANIFEST_SCHEMA,
        "source": {"license": source_license, "name": source_name},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return TokenizerTrainingResult(
        output_dir=output_dir,
        model_path=model_path,
        manifest_path=manifest_path,
        documents=len(documents),
        vocab_size=tokenizer.vocab_size,
        merge_count=tokenizer.merge_count,
        model_sha256=tokenizer.fingerprint(),
    )


def _special_token_checks(tokenizer: ByteBPETokenizer) -> dict[str, JsonValue]:
    checks = {
        "bos_id_is_1": BOS_ID == 1,
        "empty_with_bos_eos": tokenizer.encode("", add_bos=True, add_eos=True) == [BOS_ID, EOS_ID],
        "eos_id_is_2": EOS_ID == 2,
        "pad_id_is_0": PAD_ID == 0,
        "specials_skipped_on_decode": tokenizer.decode(
            [BOS_ID, *tokenizer.encode("ok"), EOS_ID, PAD_ID]
        )
        == "ok",
        "unk_id_is_3": UNK_ID == 3,
    }
    passed = sum(checks.values())
    assertions: dict[str, JsonValue] = dict(checks)
    return {
        "assertions": assertions,
        "correct": passed,
        "rate": passed / len(checks),
        "total": len(checks),
    }


def evaluate_tokenizer_artifact(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    encode_repeats: int = 5,
) -> dict[str, JsonValue]:
    """Compare the trained tokenizer with the raw-byte baseline."""
    if output_path.exists():
        raise FileExistsError(f"tokenizer evaluation already exists: {output_path}")
    documents = read_tokenizer_jsonl(input_path)
    if not documents:
        raise ValueError("tokenizer evaluation corpus must contain at least one document")
    tokenizer = ByteBPETokenizer.load(model_path)
    report: dict[str, JsonValue] = {
        "input": {
            "bytes": input_path.stat().st_size,
            "documents": len(documents),
            "file_name": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "metric_definitions": {
            "bytes_per_token": "UTF-8 bytes / ordinary encoded tokens",
            "fertility": "ordinary encoded tokens / non-empty segments from Python str.split()",
            "round_trip_rate": "documents with exact decode(encode(text)) equality / documents",
            "unknown_rate": "UNK tokens / ordinary encoded tokens",
        },
        "model": {
            "file_name": model_path.name,
            "merge_count": tokenizer.merge_count,
            "sha256": tokenizer.fingerprint(),
            "vocab_size": tokenizer.vocab_size,
        },
        "results": {
            "forgellm_byte_bpe": evaluate_with_subsets(
                tokenizer, documents, encode_repeats=encode_repeats
            ),
            "raw_utf8_bytes": evaluate_with_subsets(
                RawByteTokenizer(), documents, encode_repeats=encode_repeats
            ),
        },
        "schema_version": _EVALUATION_SCHEMA,
        "special_token_accuracy": _special_token_checks(tokenizer),
        "throughput_note": (
            "Micro-corpus wall-clock timing; compare only under the same interpreter and host."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    return report


def compare_with_huggingface_reference(
    config: TokenizerConfig,
    model_path: Path,
    train_input_path: Path,
    evaluation_input_path: Path,
    output_dir: Path,
    *,
    encode_repeats: int = 5,
) -> dict[str, JsonValue]:
    """Train the optional mature reference and compare all paths on one corpus."""
    from forgellm.tokenization.hf_reference import HuggingFaceByteBPE

    hand_written = ByteBPETokenizer.load(model_path)
    if hand_written.config != config:
        raise ValueError("comparison config does not match the hand-written tokenizer model")
    training_documents = read_tokenizer_jsonl(train_input_path)
    evaluation_documents = read_tokenizer_jsonl(evaluation_input_path)
    if not training_documents or not evaluation_documents:
        raise ValueError("reference comparison requires non-empty train and evaluation corpora")

    output_dir.mkdir(parents=True, exist_ok=False)
    reference = HuggingFaceByteBPE.train(config, [document.text for document in training_documents])
    reference_path = output_dir / "huggingface_tokenizer.json"
    reference.save(reference_path)
    reference_special_ids: dict[str, JsonValue] = {
        token: token_id for token, token_id in reference.special_token_ids().items()
    }
    report: dict[str, JsonValue] = {
        "contracts": {
            "forgellm": (
                "raw UTF-8 bytes, whole-document pair counts, deterministic integer-ID tie-break"
            ),
            "huggingface": (
                "ByteLevel(add_prefix_space=False,use_regex=False), explicit byte alphabet, BPE"
            ),
            "shared": {
                "min_pair_frequency": config.min_pair_frequency,
                "special_tokens": {"<bos>": 1, "<eos>": 2, "<pad>": 0, "<unk>": 3},
                "vocab_budget": config.vocab_size,
            },
        },
        "evaluation_input": {
            "documents": len(evaluation_documents),
            "file_name": evaluation_input_path.name,
            "sha256": sha256_file(evaluation_input_path),
        },
        "implementations": {
            "forgellm_byte_bpe": {
                "model_sha256": hand_written.fingerprint(),
                "vocab_size": hand_written.vocab_size,
            },
            "huggingface_bytelevel_bpe": {
                "artifact_file": reference_path.name,
                "artifact_sha256": sha256_file(reference_path),
                "library_version": reference.library_version,
                "special_token_ids": reference_special_ids,
                "vocab_size": reference.vocab_size,
            },
            "raw_utf8_bytes": {"vocab_size": 256},
        },
        "results": {
            "forgellm_byte_bpe": evaluate_with_subsets(
                hand_written, evaluation_documents, encode_repeats=encode_repeats
            ),
            "huggingface_bytelevel_bpe": evaluate_with_subsets(
                reference, evaluation_documents, encode_repeats=encode_repeats
            ),
            "raw_utf8_bytes": evaluate_with_subsets(
                RawByteTokenizer(), evaluation_documents, encode_repeats=encode_repeats
            ),
        },
        "schema_version": "forgellm-tokenizer-reference-comparison-v1",
        "training_input": {
            "documents": len(training_documents),
            "file_name": train_input_path.name,
            "sha256": sha256_file(train_input_path),
        },
    }
    _write_json(output_dir / "comparison.json", report)
    return report
