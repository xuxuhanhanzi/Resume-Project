"""Typed, source-bound episodic memory used by the R4 experiment.

Memory cards are indexes, not facts in their own right.  A caller must keep
the source session available and validate the card before it can reach a
reader prompt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    EVENT = "event"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class MemoryCard:
    """The fixed TPM schema defined in the evidence-driven experiment plan."""

    memory_id: str
    source_session_id: str
    source_turn_ids: tuple[str, ...]
    observed_at: str
    scope: str
    kind: MemoryKind
    statement: str
    entities: tuple[str, ...]
    valid_from: str | None
    valid_to: str | None
    confidence: float
    extraction_model: str
    content_hash: str

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("memory_id", self.memory_id, 160),
            ("source_session_id", self.source_session_id, 500),
            ("observed_at", self.observed_at, 64),
            ("scope", self.scope, 240),
            ("statement", self.statement, 4_000),
            ("extraction_model", self.extraction_model, 240),
        ):
            if not value.strip() or len(value) > limit:
                raise ValueError(f"{name} must be a bounded non-empty string")
        if not self.source_turn_ids or any(not item.strip() for item in self.source_turn_ids):
            raise ValueError("source_turn_ids must contain at least one source turn")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if len(self.content_hash) != 64 or any(
            char not in "0123456789abcdef" for char in self.content_hash.casefold()
        ):
            raise ValueError("content_hash must be a SHA-256 hex digest")
        _validate_date_or_datetime(self.observed_at, field="observed_at")
        if self.valid_from is not None:
            _validate_date_or_datetime(self.valid_from, field="valid_from")
        if self.valid_to is not None:
            _validate_date_or_datetime(self.valid_to, field="valid_to")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from must not be after valid_to")

    @classmethod
    def create(
        cls,
        *,
        source_session_id: str,
        source_turn_ids: tuple[str, ...],
        observed_at: str,
        scope: str,
        kind: MemoryKind,
        statement: str,
        entities: tuple[str, ...],
        valid_from: str | None,
        valid_to: str | None,
        confidence: float,
        extraction_model: str,
        source_content: str,
    ) -> MemoryCard:
        """Create an ID and a content binding from raw source content."""

        content_hash = sha256_text(source_content)
        identity = "\0".join(
            (source_session_id, "\0".join(source_turn_ids), statement, content_hash)
        )
        return cls(
            memory_id=sha256_text(identity)[:24],
            source_session_id=source_session_id,
            source_turn_ids=source_turn_ids,
            observed_at=observed_at,
            scope=scope,
            kind=kind,
            statement=statement,
            entities=tuple(sorted(set(entities))),
            valid_from=valid_from,
            valid_to=valid_to,
            confidence=confidence,
            extraction_model=extraction_model,
            content_hash=content_hash,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["source_turn_ids"] = list(self.source_turn_ids)
        data["entities"] = list(self.entities)
        return data

    @classmethod
    def from_dict(cls, raw: object) -> MemoryCard:
        required = {
            "memory_id",
            "source_session_id",
            "source_turn_ids",
            "observed_at",
            "scope",
            "kind",
            "statement",
            "entities",
            "valid_from",
            "valid_to",
            "confidence",
            "extraction_model",
            "content_hash",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("memory card must contain the fixed TPM schema only")
        turns, entities = raw["source_turn_ids"], raw["entities"]
        if (
            not isinstance(turns, list)
            or not all(isinstance(item, str) for item in turns)
            or not isinstance(entities, list)
            or not all(isinstance(item, str) for item in entities)
            or not isinstance(raw["confidence"], (int, float))
            or isinstance(raw["confidence"], bool)
            or raw["valid_from"] is not None
            and not isinstance(raw["valid_from"], str)
            or raw["valid_to"] is not None
            and not isinstance(raw["valid_to"], str)
        ):
            raise ValueError("memory card has invalid list, date, or confidence fields")
        try:
            kind = MemoryKind(str(raw["kind"]))
        except ValueError as error:
            raise ValueError("memory card kind is unsupported") from error
        return cls(
            memory_id=str(raw["memory_id"]),
            source_session_id=str(raw["source_session_id"]),
            source_turn_ids=tuple(turns),
            observed_at=str(raw["observed_at"]),
            scope=str(raw["scope"]),
            kind=kind,
            statement=str(raw["statement"]),
            entities=tuple(entities),
            valid_from=raw["valid_from"],
            valid_to=raw["valid_to"],
            confidence=float(raw["confidence"]),
            extraction_model=str(raw["extraction_model"]),
            content_hash=str(raw["content_hash"]),
        )


class ProvenanceValidator:
    """Reject cards whose source session/turn/content binding cannot be proved."""

    def validate(
        self,
        card: MemoryCard,
        *,
        source_sessions: dict[str, str],
        source_turn_ids: dict[str, set[str]],
    ) -> None:
        content = source_sessions.get(card.source_session_id)
        if content is None:
            raise ValueError("memory card source session is unavailable")
        if sha256_text(content) != card.content_hash:
            raise ValueError("memory card content hash does not match source session")
        known_turns = source_turn_ids.get(card.source_session_id, set())
        if not set(card.source_turn_ids) <= known_turns:
            raise ValueError("memory card cites a source turn that is unavailable")


class ContentAddressedExtractionCache:
    """A no-overwrite cache keyed solely by normalized raw session content."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get(self, source_content: str) -> tuple[MemoryCard, ...] | None:
        path = self._path(source_content)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read extraction cache: {error}") from error
        if not isinstance(raw, dict) or set(raw) != {"content_hash", "cards"}:
            raise ValueError("extraction cache has an invalid schema")
        expected = sha256_text(source_content)
        if raw["content_hash"] != expected or not isinstance(raw["cards"], list):
            raise ValueError("extraction cache does not bind to the requested content")
        cards = tuple(MemoryCard.from_dict(item) for item in raw["cards"])
        if any(card.content_hash != expected for card in cards):
            raise ValueError("extraction cache card hash differs from cache key")
        return cards

    def put_once(self, source_content: str, cards: tuple[MemoryCard, ...]) -> Path:
        digest = sha256_text(source_content)
        if any(card.content_hash != digest for card in cards):
            raise ValueError("all cards must bind to the cached source content")
        path = self._path(source_content)
        if path.exists():
            existing = self.get(source_content)
            if existing != cards:
                raise ValueError("refusing to replace an existing extraction cache entry")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"content_hash": digest, "cards": [card.to_dict() for card in cards]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return path

    def _path(self, source_content: str) -> Path:
        return self.root / f"{sha256_text(source_content)}.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_date_or_datetime(value: str, *, field: str) -> None:
    normalized = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
        return
    except ValueError:
        pass
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601 date or datetime") from error
