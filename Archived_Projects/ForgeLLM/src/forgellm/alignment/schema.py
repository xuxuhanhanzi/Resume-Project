"""Strict, fingerprinted records for preference learning and online rollouts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

from forgellm.post_training.schema import Message
from forgellm.structured_logging import JsonValue


class AlignmentSchemaError(ValueError):
    """Raised when preference or rollout data violates a frozen contract."""


_SplitName = TypeVar("_SplitName", bound=str)


def _canonical_sha256(value: JsonValue) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _messages_to_json(messages: tuple[Message, ...]) -> list[JsonValue]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _messages_from_json(raw: object) -> tuple[Message, ...]:
    if not isinstance(raw, list) or not raw:
        raise AlignmentSchemaError("prompt must be a non-empty message list")
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise AlignmentSchemaError("each prompt message must contain role and content")
        role = item["role"]
        content = item["content"]
        if role not in ("system", "user") or not isinstance(content, str):
            raise AlignmentSchemaError("preference prompts allow only system/user text messages")
        messages.append(Message(role=role, content=content))
    if messages[-1].role != "user":
        raise AlignmentSchemaError("preference prompt must end with user")
    return tuple(messages)


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    """One auditable chosen/rejected comparison."""

    record_id: str
    prompt: tuple[Message, ...]
    chosen: str
    rejected: str
    task_family: str
    preference_source: str
    rejection_reason: str
    verifier_version: str
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        values = (
            self.record_id,
            self.chosen,
            self.rejected,
            self.task_family,
            self.preference_source,
            self.rejection_reason,
            self.verifier_version,
        )
        if any(not value.strip() for value in values):
            raise AlignmentSchemaError("preference string fields must be non-empty")
        if self.chosen == self.rejected:
            raise AlignmentSchemaError("chosen and rejected responses must differ")
        if not self.prompt or self.prompt[-1].role != "user":
            raise AlignmentSchemaError("preference prompt must end with a user message")
        if any(message.role not in ("system", "user") for message in self.prompt):
            raise AlignmentSchemaError("preference prompt cannot contain assistant messages")

    def prompt_fingerprint(self) -> str:
        """Hash only the task identity, before any rejected response is generated."""
        return _canonical_sha256(cast(JsonValue, _messages_to_json(self.prompt)))

    def fingerprint(self) -> str:
        """Hash the complete semantic pair while excluding its claimed fingerprint."""
        return _canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, JsonValue]:
        """Return a stable JSON representation."""
        value: dict[str, JsonValue] = {
            "id": self.record_id,
            "prompt": _messages_to_json(self.prompt),
            "chosen": self.chosen,
            "rejected": self.rejected,
            "task_family": self.task_family,
            "preference_source": self.preference_source,
            "rejection_reason": self.rejection_reason,
            "verifier_version": self.verifier_version,
            "metadata": self.metadata,
            "prompt_fingerprint": self.prompt_fingerprint(),
        }
        if include_fingerprint:
            value["pair_fingerprint"] = self.fingerprint()
        return value

    @classmethod
    def from_dict(cls, raw: object) -> PreferenceRecord:
        """Parse a pair and verify both stored fingerprints."""
        if not isinstance(raw, dict):
            raise AlignmentSchemaError("preference record must be an object")
        expected = {
            "id",
            "prompt",
            "chosen",
            "rejected",
            "task_family",
            "preference_source",
            "rejection_reason",
            "verifier_version",
            "metadata",
            "prompt_fingerprint",
            "pair_fingerprint",
        }
        if set(raw) != expected:
            raise AlignmentSchemaError(
                f"preference fields differ; missing={sorted(expected - set(raw))}, "
                f"unknown={sorted(set(raw) - expected)}"
            )
        metadata = raw["metadata"]
        string_keys = (
            "id",
            "chosen",
            "rejected",
            "task_family",
            "preference_source",
            "rejection_reason",
            "verifier_version",
            "prompt_fingerprint",
            "pair_fingerprint",
        )
        if not isinstance(metadata, dict) or any(
            not isinstance(raw[key], str) for key in string_keys
        ):
            raise AlignmentSchemaError("preference field types are invalid")
        record = cls(
            record_id=cast(str, raw["id"]),
            prompt=_messages_from_json(raw["prompt"]),
            chosen=cast(str, raw["chosen"]),
            rejected=cast(str, raw["rejected"]),
            task_family=cast(str, raw["task_family"]),
            preference_source=cast(str, raw["preference_source"]),
            rejection_reason=cast(str, raw["rejection_reason"]),
            verifier_version=cast(str, raw["verifier_version"]),
            metadata=cast(dict[str, JsonValue], metadata),
        )
        if raw["prompt_fingerprint"] != record.prompt_fingerprint():
            raise AlignmentSchemaError("prompt fingerprint mismatch")
        if raw["pair_fingerprint"] != record.fingerprint():
            raise AlignmentSchemaError("pair fingerprint mismatch")
        return record


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """One version-bound rollout eligible for an on-policy update."""

    prompt_id: str
    policy_revision: str
    adapter_sha256: str
    generation_config: dict[str, JsonValue]
    seed: int
    token_ids: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    reward_components: dict[str, float]
    verifier_version: str
    response: str
    timestamp_utc: str

    def __post_init__(self) -> None:
        if not self.prompt_id.strip() or not self.policy_revision.strip():
            raise AlignmentSchemaError("rollout identities must be non-empty")
        if len(self.adapter_sha256) != 64:
            raise AlignmentSchemaError("adapter_sha256 must be a complete SHA-256")
        if not self.token_ids or len(self.token_ids) != len(self.old_log_probs):
            raise AlignmentSchemaError("rollout token IDs and old log-probs must align")
        if any(token_id < 0 for token_id in self.token_ids):
            raise AlignmentSchemaError("rollout token IDs must be non-negative")
        if not self.reward_components or not self.verifier_version.strip():
            raise AlignmentSchemaError("rollout reward/verifier fields must be populated")
        if not self.timestamp_utc.endswith("Z"):
            raise AlignmentSchemaError("rollout timestamp must be UTC and end with Z")

    @property
    def response_length(self) -> int:
        """Return the number of sampled response tokens."""
        return len(self.token_ids)

    def fingerprint(self) -> str:
        """Hash all rollout fields."""
        return _canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, JsonValue]:
        """Return stable JSON values."""
        value: dict[str, JsonValue] = {
            "prompt_id": self.prompt_id,
            "policy_revision": self.policy_revision,
            "adapter_sha256": self.adapter_sha256,
            "generation_config": self.generation_config,
            "seed": self.seed,
            "token_ids": list(self.token_ids),
            "old_log_probs": list(self.old_log_probs),
            "reward_components": cast(dict[str, JsonValue], self.reward_components),
            "verifier_version": self.verifier_version,
            "response": self.response,
            "response_length": self.response_length,
            "timestamp_utc": self.timestamp_utc,
        }
        if include_fingerprint:
            value["rollout_fingerprint"] = self.fingerprint()
        return value


def write_preference_jsonl(path: Path, records: list[PreferenceRecord]) -> None:
    """Write canonical pairs after rejecting duplicate IDs, prompts and pairs."""
    ids = [record.record_id for record in records]
    prompts = [record.prompt_fingerprint() for record in records]
    pairs = [record.fingerprint() for record in records]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise AlignmentSchemaError("duplicate preference ID or prompt detected")
    if len(set(pairs)) != len(pairs):
        raise AlignmentSchemaError("duplicate preference pair detected")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def read_preference_jsonl(path: Path) -> list[PreferenceRecord]:
    """Read and revalidate canonical pairs."""
    records: list[PreferenceRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(PreferenceRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, AlignmentSchemaError) as error:
            raise AlignmentSchemaError(f"invalid {path}:{line_number}: {error}") from error
    if not records:
        raise AlignmentSchemaError(f"preference file is empty: {path}")
    return records


def assert_split_disjoint(
    splits: Mapping[_SplitName, Sequence[PreferenceRecord]],
) -> None:
    """Fail if prompt identities leak across named splits."""
    owners: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            fingerprint = record.prompt_fingerprint()
            if fingerprint in owners:
                raise AlignmentSchemaError(
                    f"prompt leakage between {owners[fingerprint]} and {split}: {record.record_id}"
                )
            owners[fingerprint] = split
