"""Explicit special-token policies and byte-offset inspection helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from forgellm.tokenization.bpe import SPECIAL_TOKENS, ByteBPETokenizer


class DisallowedSpecialTokenError(ValueError):
    """Raised when ordinary input contains an unapproved control token literal."""


@dataclass(frozen=True, slots=True)
class TokenByteSpan:
    """The half-open byte range consumed by an ordinary token."""

    token_id: int
    start_byte: int
    end_byte: int


def encode_with_special_policy(
    tokenizer: ByteBPETokenizer,
    text: str,
    *,
    allowed_special: Iterable[str] = (),
    special_tokens: Mapping[str, int] = SPECIAL_TOKENS,
) -> list[int]:
    """Encode control-token literals only when the caller explicitly allows them."""
    allowed = frozenset(allowed_special)
    unknown_allowed = allowed - special_tokens.keys()
    if unknown_allowed:
        raise ValueError(f"unknown allowed special tokens: {sorted(unknown_allowed)}")
    output: list[int] = []
    position = 0
    while position < len(text):
        matches = [
            (text.find(token, position), token)
            for token in special_tokens
            if text.find(token, position) >= 0
        ]
        if not matches:
            output.extend(tokenizer.encode(text[position:]))
            break
        token_position, token = min(matches, key=lambda item: (item[0], -len(item[1]), item[1]))
        output.extend(tokenizer.encode(text[position:token_position]))
        if token not in allowed:
            raise DisallowedSpecialTokenError(
                f"ordinary input contains disallowed special token {token!r}"
            )
        output.append(special_tokens[token])
        position = token_position + len(token)
    return output


def encode_with_byte_offsets(
    tokenizer: ByteBPETokenizer, text: str
) -> tuple[list[int], list[TokenByteSpan]]:
    """Return ordinary token IDs and exact offsets in the original UTF-8 byte string."""
    token_ids = tokenizer.encode(text)
    spans: list[TokenByteSpan] = []
    cursor = 0
    for token_id in token_ids:
        payload = tokenizer.token_bytes(token_id)
        if payload is None:
            raise AssertionError("ordinary encoding unexpectedly contains a special token")
        spans.append(TokenByteSpan(token_id, cursor, cursor + len(payload)))
        cursor += len(payload)
    if cursor != len(text.encode("utf-8", errors="strict")):
        raise AssertionError("token offsets do not cover the original UTF-8 bytes")
    return token_ids, spans
