"""Tests for reversible pre-tokenization boundaries."""

import pytest

from forgellm.tokenization.pretokenization import Pretokenization, split_text


@pytest.mark.parametrize(
    "mode",
    [Pretokenization.NONE, Pretokenization.WHITESPACE, Pretokenization.UNICODE_CLASS],
)
@pytest.mark.parametrize("text", ["", "hello 123!", "中文 分词🙂", "a\tb\nc"])
def test_every_mode_preserves_exact_utf8_bytes(mode: Pretokenization, text: str) -> None:
    assert b"".join(split_text(text, mode)) == text.encode("utf-8")


def test_unicode_class_exposes_learning_boundaries() -> None:
    pieces = split_text("hello 123!中文", Pretokenization.UNICODE_CLASS)

    assert pieces == (b"hello", b" ", b"123", b"!", "中文".encode())
