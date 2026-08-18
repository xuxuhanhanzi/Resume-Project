"""Unified, small-corpus experiments for learning tokenizer method choices."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from forgellm.structured_logging import JsonValue
from forgellm.tokenization.advanced_bpe import (
    EducationalBPE,
    train_classic_bpe,
    train_picky_bpe,
    train_super_bpe,
)
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig
from forgellm.tokenization.corpus import TokenizerDocument, read_tokenizer_jsonl, sha256_file
from forgellm.tokenization.entropy_patching import BigramEntropyPatcher
from forgellm.tokenization.evaluation import RawByteTokenizer, TokenizerLike, evaluate_with_subsets
from forgellm.tokenization.pretokenization import Pretokenization, split_text
from forgellm.tokenization.special_tokens import (
    DisallowedSpecialTokenError,
    encode_with_byte_offsets,
    encode_with_special_policy,
)
from forgellm.tokenization.unigram import EducationalUnigram, train_unigram


class MethodLabReport(TypedDict):
    """Typed top-level shape of the JSON method-lab report."""

    claim_boundary: str
    entropy_patching: dict[str, JsonValue]
    evaluation: dict[str, dict[str, JsonValue]]
    fixed_settings: dict[str, JsonValue]
    inputs: dict[str, JsonValue]
    pretokenization: dict[str, JsonValue]
    sampling: dict[str, JsonValue]
    schema_version: str
    special_tokens_and_offsets: dict[str, JsonValue]
    structure: dict[str, dict[str, JsonValue]]


def _cross_whitespace_tokens(vocabulary: Sequence[bytes]) -> int:
    count = 0
    for piece in vocabulary:
        has_whitespace = any(chr(byte).isspace() for byte in piece if byte < 128)
        has_non_whitespace = any(not chr(byte).isspace() for byte in piece if byte < 128) or any(
            byte >= 128 for byte in piece
        )
        count += has_whitespace and has_non_whitespace
    return count


def _token_usage(tokenizer: TokenizerLike, documents: Sequence[TokenizerDocument]) -> Counter[int]:
    usage: Counter[int] = Counter()
    for document in documents:
        usage.update(tokenizer.encode(document.text))
    return usage


def _structure_report(
    tokenizer: EducationalBPE | EducationalUnigram,
    train_documents: Sequence[TokenizerDocument],
) -> dict[str, JsonValue]:
    usage = _token_usage(tokenizer, train_documents)
    vocabulary = tokenizer.vocabulary
    report: dict[str, JsonValue] = {
        "cross_whitespace_tokens": _cross_whitespace_tokens(vocabulary),
        "learned_tokens": max(0, tokenizer.vocab_size - 256),
        "learned_tokens_used_once_or_never": sum(
            usage[token_id] <= 1 for token_id in range(256, len(vocabulary))
        ),
        "vocab_size": tokenizer.vocab_size,
    }
    if isinstance(tokenizer, EducationalBPE):
        report.update(
            {
                "merge_events": tokenizer.merge_count,
                "phase_two_merge_events": sum(event.phase == 2 for event in tokenizer.events),
                "pretokenization": tokenizer.pretokenization.value,
                "removal_events": tokenizer.removal_count,
            }
        )
    return report


def _sampling_report(
    classic: EducationalBPE,
    unigram: EducationalUnigram,
    sample_text: str,
) -> dict[str, JsonValue]:
    dropout_segmentations = {
        tuple(classic.encode(sample_text, dropout=0.35, seed=seed)) for seed in range(20)
    }
    unigram_segmentations = {
        tuple(unigram.sample_encode(sample_text, seed=seed, temperature=10.0)) for seed in range(20)
    }
    return {
        "bpe_dropout": {
            "dropout": 0.35,
            "round_trip_all": all(
                classic.decode(segmentation) == sample_text
                for segmentation in dropout_segmentations
            ),
            "unique_segmentations": len(dropout_segmentations),
        },
        "sample_text": sample_text,
        "unigram_sampling": {
            "round_trip_all": all(
                unigram.decode(segmentation) == sample_text
                for segmentation in unigram_segmentations
            ),
            "temperature": 10.0,
            "unique_segmentations": len(unigram_segmentations),
        },
    }


def _special_and_offset_report(
    tokenizer: ByteBPETokenizer, sample_text: str
) -> dict[str, JsonValue]:
    literal = f"{sample_text}<eos>"
    disallowed_was_rejected = False
    try:
        encode_with_special_policy(tokenizer, literal)
    except DisallowedSpecialTokenError:
        disallowed_was_rejected = True
    allowed_ids = encode_with_special_policy(tokenizer, literal, allowed_special={"<eos>"})
    token_ids, spans = encode_with_byte_offsets(tokenizer, sample_text)
    return {
        "allowed_eos_is_control_id": bool(allowed_ids and allowed_ids[-1] == 2),
        "byte_offsets_cover_input": bool(
            not spans or spans[-1].end_byte == len(sample_text.encode("utf-8"))
        ),
        "disallowed_literal_rejected": disallowed_was_rejected,
        "ordinary_token_count": len(token_ids),
        "span_count": len(spans),
    }


def _entropy_patch_report(texts: Sequence[str], sample_text: str) -> dict[str, JsonValue]:
    patcher = BigramEntropyPatcher.train(texts)
    patches = patcher.patch(sample_text, threshold_bits=4.0, max_patch_bytes=8)
    return {
        "exact_round_trip": patcher.decode(patches) == sample_text,
        "max_patch_bytes": 8,
        "patch_count": len(patches),
        "patches": [
            {
                "end": patch.end,
                "opening_surprisal_bits": patch.opening_surprisal_bits,
                "payload_hex": patch.payload.hex(),
                "start": patch.start,
            }
            for patch in patches
        ],
        "threshold_bits": 4.0,
    }


def run_method_lab(
    train_path: Path,
    evaluation_path: Path,
    output_path: Path,
    *,
    vocab_size: int = 300,
    min_pair_frequency: int = 1,
) -> MethodLabReport:
    """Run all G1-B learning experiments without vocabulary-size search or large data."""
    if output_path.exists():
        raise FileExistsError(f"method-lab report already exists: {output_path}")
    train_documents = read_tokenizer_jsonl(train_path)
    evaluation_documents = read_tokenizer_jsonl(evaluation_path)
    if not train_documents or not evaluation_documents:
        raise ValueError("method lab requires non-empty train and evaluation corpora")
    train_texts = [document.text for document in train_documents]

    classic_global = train_classic_bpe(
        train_texts,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
        pretokenization=Pretokenization.NONE,
    )
    classic_boundaries = train_classic_bpe(
        train_texts,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
        pretokenization=Pretokenization.UNICODE_CLASS,
    )
    super_bpe = train_super_bpe(
        train_texts,
        vocab_size=vocab_size,
        subword_vocab_size=max(256, vocab_size - 16),
        min_pair_frequency=min_pair_frequency,
    )
    picky_bpe = train_picky_bpe(
        train_texts,
        vocab_size=vocab_size,
        min_pair_frequency=min_pair_frequency,
        removal_threshold=1.0,
        refinement_steps=12,
    )
    unigram = train_unigram(
        train_texts,
        vocab_size=vocab_size,
        max_piece_bytes=8,
        em_iterations=4,
    )
    algorithms: dict[str, TokenizerLike] = {
        "classic_bpe_global": classic_global,
        "classic_bpe_unicode_boundaries": classic_boundaries,
        "picky_bpe_educational": picky_bpe,
        "raw_utf8_bytes": RawByteTokenizer(),
        "super_bpe_curriculum": super_bpe,
        "unigram_em": unigram,
    }
    sample_text = "repeat repeat repeat token"
    stable_tokenizer = ByteBPETokenizer.train(
        TokenizerConfig(vocab_size=vocab_size, min_pair_frequency=min_pair_frequency),
        train_texts,
    )
    pretokenization_example = "Token 123，中文🙂"
    report: MethodLabReport = {
        "claim_boundary": (
            "Small project-owned fixtures teach algorithm behavior only; results do not establish "
            "production quality or downstream language-model gains."
        ),
        "entropy_patching": _entropy_patch_report(train_texts, "byte byte 未见🙂"),
        "evaluation": {
            name: evaluate_with_subsets(tokenizer, evaluation_documents, encode_repeats=3)
            for name, tokenizer in algorithms.items()
        },
        "fixed_settings": {
            "min_pair_frequency": min_pair_frequency,
            "no_vocabulary_search": True,
            "vocab_size": vocab_size,
        },
        "inputs": {
            "evaluation": {
                "documents": len(evaluation_documents),
                "file_name": evaluation_path.name,
                "sha256": sha256_file(evaluation_path),
            },
            "train": {
                "documents": len(train_documents),
                "file_name": train_path.name,
                "sha256": sha256_file(train_path),
            },
        },
        "pretokenization": {
            "example": pretokenization_example,
            "none_hex": [
                piece.hex() for piece in split_text(pretokenization_example, Pretokenization.NONE)
            ],
            "unicode_class_hex": [
                piece.hex()
                for piece in split_text(pretokenization_example, Pretokenization.UNICODE_CLASS)
            ],
            "whitespace_hex": [
                piece.hex()
                for piece in split_text(pretokenization_example, Pretokenization.WHITESPACE)
            ],
        },
        "sampling": _sampling_report(classic_global, unigram, sample_text),
        "schema_version": "forgellm-tokenizer-method-lab-v1",
        "special_tokens_and_offsets": _special_and_offset_report(
            stable_tokenizer, "control 中文🙂"
        ),
        "structure": {
            "classic_bpe_global": _structure_report(classic_global, train_documents),
            "classic_bpe_unicode_boundaries": _structure_report(
                classic_boundaries, train_documents
            ),
            "picky_bpe_educational": _structure_report(picky_bpe, train_documents),
            "super_bpe_curriculum": _structure_report(super_bpe, train_documents),
            "unigram_em": _structure_report(unigram, train_documents),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
