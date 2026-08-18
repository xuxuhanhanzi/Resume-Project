"""Internal schemas for data processing."""

from __future__ import annotations

from dataclasses import dataclass

from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    """A schema-valid, normalized document before filtering and deduplication."""

    document_id: str
    text: str
    content_sha256: str
    record_ref: str


@dataclass(frozen=True, slots=True)
class Rejection:
    """A rejected record without its raw text."""

    record_ref: str
    document_id: str | None
    reason: str

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a safe serialized representation."""
        return {
            "document_id": self.document_id,
            "reason": self.reason,
            "record_ref": self.record_ref,
        }


@dataclass(frozen=True, slots=True)
class OutputDocument:
    """A retained document written to one deterministic split."""

    document_id: str
    text: str
    content_sha256: str

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the stable output schema."""
        return {
            "content_sha256": self.content_sha256,
            "id": self.document_id,
            "text": self.text,
        }
