"""Tests for the educational byte Unigram model."""

import pytest

from forgellm.tokenization.unigram import train_unigram


@pytest.fixture
def unigram_model():  # type: ignore[no-untyped-def]
    return train_unigram(
        [
            "banana bandana banana",
            "token tokenizer tokenization",
            "中文分词 中文分词",
        ]
        * 4,
        vocab_size=280,
        max_piece_bytes=8,
        em_iterations=3,
    )


@pytest.mark.parametrize("text", ["", "banana", "未见文本🙂", "  exact\n"])
def test_unigram_viterbi_has_byte_fallback_and_round_trips(unigram_model, text: str) -> None:  # type: ignore[no-untyped-def]
    assert unigram_model.decode(unigram_model.encode(text)) == text


def test_unigram_sampling_produces_multiple_valid_segmentations(unigram_model) -> None:  # type: ignore[no-untyped-def]
    text = "banana bandana"
    samples = {
        tuple(unigram_model.sample_encode(text, seed=seed, temperature=10.0)) for seed in range(20)
    }

    assert len(samples) > 1
    assert all(unigram_model.decode(sample) == text for sample in samples)


def test_unigram_sampling_is_reproducible_for_a_seed(unigram_model) -> None:  # type: ignore[no-untyped-def]
    assert unigram_model.sample_encode("banana", seed=7) == unigram_model.sample_encode(
        "banana", seed=7
    )
