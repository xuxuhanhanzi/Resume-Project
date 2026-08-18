"""Tests for explicit control-token policy and byte offsets."""

import pytest

from forgellm.tokenization.bpe import EOS_ID, ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig
from forgellm.tokenization.special_tokens import (
    DisallowedSpecialTokenError,
    encode_with_byte_offsets,
    encode_with_special_policy,
)


@pytest.fixture
def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        TokenizerConfig(vocab_size=280, min_pair_frequency=1),
        ["hello hello 中文"],
    )


def test_special_literal_requires_explicit_permission(tokenizer: ByteBPETokenizer) -> None:
    with pytest.raises(DisallowedSpecialTokenError, match="<eos>"):
        encode_with_special_policy(tokenizer, "hello<eos>")

    encoded = encode_with_special_policy(tokenizer, "hello<eos>", allowed_special={"<eos>"})
    assert encoded[-1] == EOS_ID
    assert tokenizer.decode(encoded) == "hello"


def test_byte_offsets_cover_multibyte_text_exactly(tokenizer: ByteBPETokenizer) -> None:
    text = "A中文🙂"
    token_ids, spans = encode_with_byte_offsets(tokenizer, text)
    raw = text.encode()

    assert [span.token_id for span in spans] == token_ids
    assert b"".join(raw[span.start_byte : span.end_byte] for span in spans) == raw
    assert spans[-1].end_byte == len(raw)
