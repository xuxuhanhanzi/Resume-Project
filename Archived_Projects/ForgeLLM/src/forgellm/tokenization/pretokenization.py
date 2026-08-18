"""Lossless pre-tokenization strategies for tokenizer method experiments."""

from __future__ import annotations

from enum import StrEnum


class Pretokenization(StrEnum):
    """Boundaries that BPE is not allowed to cross during its first phase."""

    NONE = "none"
    WHITESPACE = "whitespace"
    UNICODE_CLASS = "unicode_class"


def _character_class(character: str, mode: Pretokenization) -> str:
    if mode is Pretokenization.WHITESPACE:
        return "space" if character.isspace() else "text"
    if character.isspace():
        return "space"
    if character.isalpha():
        return "letter"
    if character.isdecimal():
        return "number"
    return "other"


def split_text(text: str, mode: Pretokenization) -> tuple[bytes, ...]:
    """Split text at reversible boundaries and return exact UTF-8 byte pieces.

    This is deliberately smaller than the regular expressions used by production
    tokenizers.  It isolates the key idea: pre-tokenization changes which adjacent
    pairs BPE is allowed to count and merge without changing the original bytes.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if mode is Pretokenization.NONE:
        return (text.encode("utf-8", errors="strict"),) if text else ()
    if not text:
        return ()

    pieces: list[str] = []
    start = 0
    current_class = _character_class(text[0], mode)
    for index, character in enumerate(text[1:], start=1):
        next_class = _character_class(character, mode)
        if next_class != current_class:
            pieces.append(text[start:index])
            start = index
            current_class = next_class
    pieces.append(text[start:])
    encoded = tuple(piece.encode("utf-8", errors="strict") for piece in pieces)
    if b"".join(encoded) != text.encode("utf-8", errors="strict"):
        raise AssertionError("pre-tokenization must preserve every UTF-8 byte")
    return encoded
