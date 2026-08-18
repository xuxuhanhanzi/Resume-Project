"""Packing, byte accounting, and deterministic cursor tests."""

import json
from pathlib import Path

import torch

from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig
from forgellm.training.data import (
    DeterministicBatchStream,
    PackedTokenDataset,
    TrainingDocument,
    load_or_create_packed_jsonl,
)


def _tokenizer() -> ByteBPETokenizer:
    texts = ["alpha beta gamma " * 8, "你好，训练。" * 8]
    return ByteBPETokenizer.train(TokenizerConfig(280, 2), texts)


def _dataset() -> PackedTokenDataset:
    documents = [
        TrainingDocument("a", "alpha beta gamma " * 8),
        TrainingDocument("b", "你好，训练。" * 8),
    ]
    return PackedTokenDataset.from_documents(
        documents,
        _tokenizer(),
        sequence_length=9,
        source_path=Path("fixture.jsonl"),
    )


def test_packed_windows_overlap_only_at_the_context_boundary() -> None:
    dataset = _dataset()
    first, _ = dataset[0]
    second, _ = dataset[1]

    assert first[-1].item() == second[0].item()
    assert dataset.retained_target_tokens == len(dataset) * 8
    assert 0 <= dataset.dropped_tail_tokens < 8
    assert len(dataset.fingerprint) == 64


def test_batch_stream_resume_reproduces_the_next_batch_exactly() -> None:
    dataset = _dataset()
    first = DeterministicBatchStream(dataset, batch_size=3, seed=17)
    first.next_batch()
    checkpoint = first.state_dict()
    expected = first.next_batch()

    restored = DeterministicBatchStream(dataset, batch_size=3, seed=17)
    restored.load_state_dict(checkpoint)
    actual = restored.next_batch()

    assert torch.equal(actual.sample_indices, expected.sample_indices)
    assert torch.equal(actual.token_ids, expected.token_ids)
    assert actual.target_bytes == expected.target_bytes


def test_stream_wraps_epochs_without_dropping_a_partial_tail() -> None:
    dataset = _dataset()
    stream = DeterministicBatchStream(dataset, batch_size=len(dataset) + 2, seed=4)

    batch = stream.next_batch()

    assert batch.token_ids.size(0) == len(dataset) + 2
    assert stream.epoch == 1
    assert stream.position == 2


def test_fingerprinted_token_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    records = [
        {"id": "one", "text": "alpha beta gamma " * 8},
        {"id": "two", "text": "你好，训练。" * 8},
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    tokenizer = _tokenizer()
    first = load_or_create_packed_jsonl(path, tokenizer, sequence_length=9)
    cache_path = path.with_suffix(".seq9.tokens.pt")

    second = load_or_create_packed_jsonl(path, tokenizer, sequence_length=9)

    assert cache_path.is_file()
    assert first.fingerprint == second.fingerprint
    assert torch.equal(first[0][0], second[0][0])
