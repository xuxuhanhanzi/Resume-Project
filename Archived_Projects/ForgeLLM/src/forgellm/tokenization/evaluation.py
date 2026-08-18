"""Tokenizer metrics with explicit denominators and a raw-byte baseline."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from forgellm.structured_logging import JsonValue
from forgellm.tokenization.corpus import TokenizerDocument


class TokenizerLike(Protocol):
    """Minimal interface needed by the evaluator."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def unknown_token_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Iterable[int]) -> str: ...


class RawByteTokenizer:
    """No-compression UTF-8 byte baseline."""

    @property
    def vocab_size(self) -> int:
        """Return the byte alphabet size."""
        return 256

    @property
    def unknown_token_id(self) -> None:
        """Raw bytes cover every possible byte."""
        return None

    def encode(self, text: str) -> list[int]:
        """Map UTF-8 bytes directly to IDs 0..255."""
        return list(text.encode("utf-8", errors="strict"))

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode a sequence of byte IDs as strict UTF-8."""
        values = list(token_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("raw-byte token IDs must be integers")
        if any(not 0 <= value <= 255 for value in values):
            raise ValueError("raw-byte token ID out of range")
        return bytes(values).decode("utf-8", errors="strict")


@dataclass(frozen=True, slots=True)
class TokenizerMetrics:
    """Aggregate metrics for one tokenizer and corpus slice."""

    documents: int
    characters: int
    utf8_bytes: int
    tokens: int
    whitespace_segments: int
    round_trip_documents: int
    unknown_tokens: int
    encode_repeats: int
    encode_seconds: float
    vocab_size: int

    def as_dict(self) -> dict[str, JsonValue]:
        """Return metrics and their derived ratios."""
        return {
            "bytes_per_second": (
                self.utf8_bytes * self.encode_repeats / self.encode_seconds
                if self.encode_seconds > 0
                else None
            ),
            "bytes_per_token": self.utf8_bytes / self.tokens if self.tokens else None,
            "characters": self.characters,
            "chars_per_token": self.characters / self.tokens if self.tokens else None,
            "documents": self.documents,
            "encode_repeats": self.encode_repeats,
            "encode_seconds": self.encode_seconds,
            "fertility_tokens_per_whitespace_segment": (
                self.tokens / self.whitespace_segments if self.whitespace_segments else None
            ),
            "round_trip_documents": self.round_trip_documents,
            "round_trip_rate": (
                self.round_trip_documents / self.documents if self.documents else None
            ),
            "tokens": self.tokens,
            "unknown_rate": self.unknown_tokens / self.tokens if self.tokens else 0.0,
            "unknown_tokens": self.unknown_tokens,
            "utf8_bytes": self.utf8_bytes,
            "vocab_size": self.vocab_size,
            "whitespace_segments": self.whitespace_segments,
        }


def evaluate_tokenizer(
    tokenizer: TokenizerLike,
    documents: Sequence[TokenizerDocument],
    *,
    encode_repeats: int = 5,
) -> TokenizerMetrics:
    """Evaluate round-trip, compression, fertility, OOV, and encode throughput."""
    if encode_repeats < 1:
        raise ValueError("encode_repeats must be at least 1")

    characters = 0
    utf8_bytes = 0
    tokens = 0
    whitespace_segments = 0
    round_trip_documents = 0
    unknown_tokens = 0
    unknown_id = tokenizer.unknown_token_id
    for document in documents:
        encoded = tokenizer.encode(document.text)
        characters += len(document.text)
        utf8_bytes += len(document.text.encode("utf-8"))
        tokens += len(encoded)
        whitespace_segments += len(document.text.split())
        unknown_tokens += sum(token_id == unknown_id for token_id in encoded)
        if tokenizer.decode(encoded) == document.text:
            round_trip_documents += 1

    start = time.perf_counter()
    for _ in range(encode_repeats):
        for document in documents:
            tokenizer.encode(document.text)
    encode_seconds = time.perf_counter() - start
    return TokenizerMetrics(
        documents=len(documents),
        characters=characters,
        utf8_bytes=utf8_bytes,
        tokens=tokens,
        whitespace_segments=whitespace_segments,
        round_trip_documents=round_trip_documents,
        unknown_tokens=unknown_tokens,
        encode_repeats=encode_repeats,
        encode_seconds=encode_seconds,
        vocab_size=tokenizer.vocab_size,
    )


def evaluate_with_subsets(
    tokenizer: TokenizerLike,
    documents: Sequence[TokenizerDocument],
    *,
    encode_repeats: int = 5,
) -> dict[str, JsonValue]:
    """Evaluate all documents and every declared subset."""
    subsets: defaultdict[str, list[TokenizerDocument]] = defaultdict(list)
    for document in documents:
        subsets[document.subset].append(document)
    return {
        "all": evaluate_tokenizer(tokenizer, documents, encode_repeats=encode_repeats).as_dict(),
        "subsets": {
            name: evaluate_tokenizer(
                tokenizer, subset_documents, encode_repeats=encode_repeats
            ).as_dict()
            for name, subset_documents in sorted(subsets.items())
        },
    }
