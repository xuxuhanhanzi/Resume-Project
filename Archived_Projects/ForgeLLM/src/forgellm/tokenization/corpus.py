"""Strict JSONL corpus reader for tokenizer training and evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_REQUIRED_FIELDS = frozenset({"id", "text"})
_OPTIONAL_FIELDS = frozenset({"content_sha256", "subset"})


class TokenizerCorpusError(ValueError):
    """Raised when a tokenizer corpus violates its documented schema."""


@dataclass(frozen=True, slots=True)
class TokenizerDocument:
    """One document and its evaluation subset."""

    document_id: str
    text: str
    subset: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tokenizer_jsonl(path: Path) -> list[TokenizerDocument]:
    """Read UTF-8 JSONL records with required ``id`` and ``text`` fields."""
    if not path.is_file():
        raise TokenizerCorpusError(f"tokenizer corpus does not exist: {path}")

    documents: list[TokenizerDocument] = []
    seen_ids: set[str] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as error:
        raise TokenizerCorpusError(f"Could not open tokenizer corpus {path}: {error}") from error

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise TokenizerCorpusError(f"blank line at {path}:{line_number}")
            try:
                parsed: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise TokenizerCorpusError(
                    f"invalid JSON at {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(parsed, dict):
                raise TokenizerCorpusError(f"record must be an object at {path}:{line_number}")
            record = cast(dict[object, object], parsed)
            fields = set(record)
            if not fields >= _REQUIRED_FIELDS or fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS:
                raise TokenizerCorpusError(
                    f"invalid fields at {path}:{line_number}; require id/text and allow "
                    "subset/content_sha256"
                )

            document_id = record["id"]
            text = record["text"]
            subset = record.get("subset", "all")
            content_sha256 = record.get("content_sha256")
            if not isinstance(document_id, str) or not document_id or len(document_id) > 256:
                raise TokenizerCorpusError(f"invalid id at {path}:{line_number}")
            if document_id in seen_ids:
                raise TokenizerCorpusError(f"duplicate id {document_id!r} at {path}:{line_number}")
            if not isinstance(text, str):
                raise TokenizerCorpusError(f"text must be a string at {path}:{line_number}")
            if not isinstance(subset, str) or not subset or len(subset) > 64:
                raise TokenizerCorpusError(f"invalid subset at {path}:{line_number}")
            if content_sha256 is not None:
                expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if not isinstance(content_sha256, str) or content_sha256 != expected_hash:
                    raise TokenizerCorpusError(f"content_sha256 mismatch at {path}:{line_number}")

            seen_ids.add(document_id)
            documents.append(TokenizerDocument(document_id, text, subset))
    return documents
