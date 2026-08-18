"""Tests for tokenizer metric definitions and baselines."""

from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig
from forgellm.tokenization.corpus import TokenizerDocument
from forgellm.tokenization.evaluation import RawByteTokenizer, evaluate_tokenizer


def test_raw_byte_metric_denominators_are_explicit() -> None:
    documents = [
        TokenizerDocument("one", "a b", "english"),
        TokenizerDocument("two", "中文", "chinese"),
    ]

    metrics = evaluate_tokenizer(RawByteTokenizer(), documents, encode_repeats=1)
    payload = metrics.as_dict()

    assert payload["utf8_bytes"] == 9
    assert payload["tokens"] == 9
    assert payload["bytes_per_token"] == 1.0
    assert payload["whitespace_segments"] == 3
    assert payload["fertility_tokens_per_whitespace_segment"] == 3.0
    assert payload["round_trip_rate"] == 1.0
    assert payload["unknown_rate"] == 0.0


def test_bpe_reduces_tokens_on_repeated_text() -> None:
    documents = [TokenizerDocument("one", "repeat repeat repeat", "english")]
    tokenizer = ByteBPETokenizer.train(
        TokenizerConfig(vocab_size=280, min_pair_frequency=1),
        [document.text for document in documents],
    )

    bpe = evaluate_tokenizer(tokenizer, documents, encode_repeats=1).as_dict()
    raw = evaluate_tokenizer(RawByteTokenizer(), documents, encode_repeats=1).as_dict()

    bpe_tokens = bpe["tokens"]
    raw_tokens = raw["tokens"]
    bpe_bytes_per_token = bpe["bytes_per_token"]
    raw_bytes_per_token = raw["bytes_per_token"]
    assert isinstance(bpe_tokens, int)
    assert isinstance(raw_tokens, int)
    assert isinstance(bpe_bytes_per_token, int | float)
    assert isinstance(raw_bytes_per_token, int | float)

    assert bpe_tokens < raw_tokens
    assert bpe_bytes_per_token > raw_bytes_per_token
