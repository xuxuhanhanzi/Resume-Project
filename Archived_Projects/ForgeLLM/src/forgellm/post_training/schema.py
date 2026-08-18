"""Strict conversational records and deterministic Stage 4 JSONL I/O."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from forgellm.structured_logging import JsonValue

Role = Literal["system", "user", "assistant"]


class InstructionDataError(ValueError):
    """Raised when a supervised conversation violates the Stage 4 contract."""


@dataclass(frozen=True, slots=True)
class Message:
    """One non-empty conversational message."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise InstructionDataError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise InstructionDataError("message content must be a non-empty string")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the canonical JSON representation."""
        return {"content": self.content, "role": self.role}


@dataclass(frozen=True, slots=True)
class InstructionRecord:
    """A traceable multi-turn SFT record with an assistant response."""

    record_id: str
    messages: tuple[Message, ...]
    source: str
    license: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise InstructionDataError("record_id must be non-empty")
        if not self.source.strip() or not self.license.strip():
            raise InstructionDataError("source and license must be non-empty")
        if not self.messages:
            raise InstructionDataError("messages must be non-empty")
        system_positions = [
            index for index, message in enumerate(self.messages) if message.role == "system"
        ]
        if system_positions not in ([], [0]):
            raise InstructionDataError(
                "a system message is optional but must appear once at index 0"
            )
        dialogue = self.messages[1:] if system_positions else self.messages
        if not dialogue or dialogue[0].role != "user":
            raise InstructionDataError("dialogue must start with a user message")
        expected: Role = "user"
        for message in dialogue:
            if message.role != expected:
                raise InstructionDataError("user and assistant messages must alternate")
            expected = "assistant" if expected == "user" else "user"
        if dialogue[-1].role != "assistant":
            raise InstructionDataError("a training conversation must end with assistant")
        if not _is_json_mapping(self.metadata):
            raise InstructionDataError("metadata must contain only JSON-compatible values")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return canonical JSON fields used by manifests and fingerprints."""
        return {
            "id": self.record_id,
            "license": self.license,
            "messages": [message.as_dict() for message in self.messages],
            "metadata": self.metadata,
            "source": self.source,
        }

    def content_fingerprint(self) -> str:
        """Hash message contents independently of the source-specific record ID."""
        payload = [message.as_dict() for message in self.messages]
        return _canonical_hash(cast(JsonValue, payload))

    def fingerprint(self) -> str:
        """Hash the complete canonical record."""
        return _canonical_hash(self.as_dict())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> InstructionRecord:
        """Validate one parsed JSON object."""
        expected = {"id", "license", "messages", "metadata", "source"}
        if set(raw) != expected:
            raise InstructionDataError("instruction record has unexpected fields")
        raw_messages = raw["messages"]
        if not isinstance(raw_messages, list):
            raise InstructionDataError("messages must be a list")
        messages: list[Message] = []
        for index, item in enumerate(raw_messages):
            if not isinstance(item, dict) or set(item) != {"content", "role"}:
                raise InstructionDataError(f"message {index} has unexpected fields")
            role = item["role"]
            content = item["content"]
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise InstructionDataError(f"message {index} has invalid role/content")
            messages.append(Message(cast(Role, role), content))
        metadata = raw["metadata"]
        if not isinstance(metadata, dict):
            raise InstructionDataError("metadata must be an object")
        scalar_fields = (raw["id"], raw["source"], raw["license"])
        if not all(isinstance(value, str) for value in scalar_fields):
            raise InstructionDataError("id, source, and license must be strings")
        return cls(
            record_id=cast(str, raw["id"]),
            messages=tuple(messages),
            source=cast(str, raw["source"]),
            license=cast(str, raw["license"]),
            metadata=cast(dict[str, JsonValue], metadata),
        )


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _is_json_mapping(value: object) -> bool:
    return isinstance(value, dict) and _is_json_value(value)


def _canonical_hash(value: JsonValue) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(records: Sequence[InstructionRecord]) -> str:
    """Hash an ordered collection of complete record fingerprints."""
    return _canonical_hash([record.fingerprint() for record in records])


def load_instruction_jsonl(path: Path) -> list[InstructionRecord]:
    """Load strict JSONL while rejecting duplicate IDs and duplicate contents."""
    records: list[InstructionRecord] = []
    ids: set[str] = set()
    contents: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise InstructionDataError(f"could not read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise InstructionDataError(f"blank line at {path}:{line_number}")
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise InstructionDataError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(raw, dict):
            raise InstructionDataError(f"record at {path}:{line_number} must be an object")
        record = InstructionRecord.from_mapping(cast(dict[str, object], raw))
        if record.record_id in ids:
            raise InstructionDataError(f"duplicate record ID: {record.record_id}")
        content_hash = record.content_fingerprint()
        if content_hash in contents:
            raise InstructionDataError(f"duplicate conversation content: {record.record_id}")
        ids.add(record.record_id)
        contents.add(content_hash)
        records.append(record)
    if not records:
        raise InstructionDataError(f"instruction dataset is empty: {path}")
    return records


def write_instruction_jsonl(path: Path, records: Iterable[InstructionRecord]) -> None:
    """Create a deterministic JSONL file without overwriting existing evidence."""
    materialized = list(records)
    if not materialized:
        raise InstructionDataError("cannot write an empty instruction dataset")
    ids = [record.record_id for record in materialized]
    contents = [record.content_fingerprint() for record in materialized]
    if len(set(ids)) != len(ids) or len(set(contents)) != len(contents):
        raise InstructionDataError("instruction dataset contains duplicate IDs or conversations")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"instruction dataset already exists: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in materialized:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
