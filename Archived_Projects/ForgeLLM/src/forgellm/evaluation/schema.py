"""Strict identities and immutable records for final model evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, cast

from forgellm.post_training.schema import Message
from forgellm.structured_logging import JsonValue

TaskType = Literal["correctness", "preference", "robustness", "retention"]
StopReason = Literal["eos", "max_new_tokens", "error"]


class EvaluationSchemaError(ValueError):
    """Raised when an evaluation identity or record is ambiguous."""


def canonical_sha256(value: JsonValue) -> str:
    """Hash one JSON value with a stable UTF-8 representation."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """All immutable files needed to identify one evaluated policy."""

    model_key: str
    model_id: str
    revision: str
    tokenizer_sha256: str
    adapter_sha256: str | None

    def __post_init__(self) -> None:
        if not all((self.model_key.strip(), self.model_id.strip(), self.revision.strip())):
            raise EvaluationSchemaError("model identity strings must be non-empty")
        if not _valid_sha256(self.tokenizer_sha256):
            raise EvaluationSchemaError("tokenizer_sha256 must be a lowercase SHA-256")
        if self.adapter_sha256 is not None and not _valid_sha256(self.adapter_sha256):
            raise EvaluationSchemaError("adapter_sha256 must be null or a lowercase SHA-256")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible identity."""
        return cast(dict[str, JsonValue], asdict(self))

    def fingerprint(self) -> str:
        """Hash the complete policy identity."""
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Generation controls that must not be tuned on frozen test answers."""

    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    seed: int
    chat_template_sha256: str

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0 or self.seed < 0:
            raise EvaluationSchemaError(
                "generation token budget must be positive and seed non-negative"
            )
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise EvaluationSchemaError("invalid temperature/top_p")
        if not self.do_sample and self.temperature != 0.0:
            raise EvaluationSchemaError("greedy generation must use temperature=0")
        if not _valid_sha256(self.chat_template_sha256):
            raise EvaluationSchemaError("chat_template_sha256 must be a lowercase SHA-256")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible settings."""
        return cast(dict[str, JsonValue], asdict(self))


def _message_dicts(messages: tuple[Message, ...]) -> list[JsonValue]:
    return [message.as_dict() for message in messages]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One frozen prompt, answer and deterministic scoring contract."""

    case_id: str
    task_type: TaskType
    prompt: tuple[Message, ...]
    expected_response: str
    source: str
    split: str
    constraints: dict[str, JsonValue]
    parent_case_id: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.case_id.strip(),
                self.task_type.strip(),
                self.expected_response.strip(),
                self.source.strip(),
                self.split.strip(),
            )
        ):
            raise EvaluationSchemaError("evaluation case fields must be non-empty")
        if not self.prompt or self.prompt[-1].role != "user":
            raise EvaluationSchemaError("evaluation prompt must end with user")
        if self.split != "test":
            raise EvaluationSchemaError("Stage 6 cases must come from the frozen test split")
        if self.parent_case_id is not None and not self.parent_case_id.strip():
            raise EvaluationSchemaError("parent_case_id must be null or non-empty")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return canonical JSON fields."""
        return {
            "case_id": self.case_id,
            "constraints": self.constraints,
            "expected_response": self.expected_response,
            "parent_case_id": self.parent_case_id,
            "prompt": _message_dicts(self.prompt),
            "source": self.source,
            "split": self.split,
            "task_type": self.task_type,
        }

    def fingerprint(self) -> str:
        """Hash all frozen task semantics."""
        return canonical_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, raw: object) -> EvaluationCase:
        """Parse an exact frozen case without accepting silent extra fields."""
        if not isinstance(raw, dict):
            raise EvaluationSchemaError("evaluation case must be an object")
        expected = {
            "case_id",
            "constraints",
            "expected_response",
            "parent_case_id",
            "prompt",
            "source",
            "split",
            "task_type",
        }
        if set(raw) != expected:
            raise EvaluationSchemaError("evaluation case fields differ")
        prompt_raw = raw["prompt"]
        constraints = raw["constraints"]
        if not isinstance(prompt_raw, list) or not isinstance(constraints, dict):
            raise EvaluationSchemaError("evaluation prompt/constraints types are invalid")
        messages: list[Message] = []
        for item in prompt_raw:
            if not isinstance(item, dict) or set(item) != {"content", "role"}:
                raise EvaluationSchemaError("evaluation prompt message fields differ")
            role = item["role"]
            content = item["content"]
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise EvaluationSchemaError("evaluation prompt message is invalid")
            messages.append(Message(role=role, content=content))
        string_names = ("case_id", "expected_response", "source", "split", "task_type")
        if any(not isinstance(raw[name], str) for name in string_names):
            raise EvaluationSchemaError("evaluation case scalar types are invalid")
        parent = raw["parent_case_id"]
        if parent is not None and not isinstance(parent, str):
            raise EvaluationSchemaError("parent_case_id must be null or string")
        return cls(
            case_id=cast(str, raw["case_id"]),
            task_type=cast(TaskType, raw["task_type"]),
            prompt=tuple(messages),
            expected_response=cast(str, raw["expected_response"]),
            source=cast(str, raw["source"]),
            split=cast(str, raw["split"]),
            constraints=cast(dict[str, JsonValue], constraints),
            parent_case_id=parent,
        )


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """A run identity binding policies, tasks, generation and source code."""

    schema_version: str
    run_name: str
    models: tuple[ModelIdentity, ...]
    cases_sha256: str
    source_data_sha256: dict[str, JsonValue]
    generation: GenerationSettings
    code_revision: str

    def __post_init__(self) -> None:
        if self.schema_version != "forgellm-stage6-evaluation-v1":
            raise EvaluationSchemaError("unsupported evaluation manifest schema")
        if not self.run_name.strip() or not self.code_revision.strip():
            raise EvaluationSchemaError("run name and code revision must be non-empty")
        if not self.models or len({model.model_key for model in self.models}) != len(self.models):
            raise EvaluationSchemaError("manifest model keys must be non-empty and unique")
        if not _valid_sha256(self.cases_sha256):
            raise EvaluationSchemaError("cases_sha256 must be a lowercase SHA-256")
        if not self.source_data_sha256:
            raise EvaluationSchemaError("source_data_sha256 must be populated")

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, JsonValue]:
        """Return a stable manifest representation."""
        value: dict[str, JsonValue] = {
            "cases_sha256": self.cases_sha256,
            "code_revision": self.code_revision,
            "generation": self.generation.as_dict(),
            "models": [model.as_dict() for model in self.models],
            "run_name": self.run_name,
            "schema_version": self.schema_version,
            "source_data_sha256": self.source_data_sha256,
        }
        if include_fingerprint:
            value["run_fingerprint"] = self.fingerprint()
        return value

    def fingerprint(self) -> str:
        """Hash the manifest without recursively including its own fingerprint."""
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    @classmethod
    def from_dict(cls, raw: object) -> EvaluationManifest:
        """Parse an exact manifest and verify its stored fingerprint."""
        if not isinstance(raw, dict):
            raise EvaluationSchemaError("manifest must be an object")
        expected = {
            "cases_sha256",
            "code_revision",
            "generation",
            "models",
            "run_fingerprint",
            "run_name",
            "schema_version",
            "source_data_sha256",
        }
        if set(raw) != expected:
            raise EvaluationSchemaError("manifest fields differ from the frozen schema")
        models_raw = raw["models"]
        generation_raw = raw["generation"]
        source_hashes = raw["source_data_sha256"]
        if not isinstance(models_raw, list) or not isinstance(generation_raw, dict):
            raise EvaluationSchemaError("manifest model/generation types are invalid")
        if not isinstance(source_hashes, dict):
            raise EvaluationSchemaError("manifest source hashes must be an object")
        models: list[ModelIdentity] = []
        for item in models_raw:
            if not isinstance(item, dict) or set(item) != {
                "adapter_sha256",
                "model_id",
                "model_key",
                "revision",
                "tokenizer_sha256",
            }:
                raise EvaluationSchemaError("model identity fields differ")
            if any(
                not isinstance(item[name], str)
                for name in ("model_key", "model_id", "revision", "tokenizer_sha256")
            ):
                raise EvaluationSchemaError("model identity scalar types are invalid")
            adapter = item["adapter_sha256"]
            if adapter is not None and not isinstance(adapter, str):
                raise EvaluationSchemaError("adapter identity must be null or string")
            models.append(
                ModelIdentity(
                    model_key=cast(str, item["model_key"]),
                    model_id=cast(str, item["model_id"]),
                    revision=cast(str, item["revision"]),
                    tokenizer_sha256=cast(str, item["tokenizer_sha256"]),
                    adapter_sha256=adapter,
                )
            )
        if set(generation_raw) != {
            "chat_template_sha256",
            "do_sample",
            "max_new_tokens",
            "seed",
            "temperature",
            "top_p",
        }:
            raise EvaluationSchemaError("generation settings fields differ")
        scalar_names = ("schema_version", "run_name", "cases_sha256", "code_revision")
        if any(not isinstance(raw[name], str) for name in scalar_names):
            raise EvaluationSchemaError("manifest scalar types are invalid")
        integer_names = ("max_new_tokens", "seed")
        float_names = ("temperature", "top_p")
        if any(
            isinstance(generation_raw[name], bool) or not isinstance(generation_raw[name], int)
            for name in integer_names
        ):
            raise EvaluationSchemaError("generation integer controls are invalid")
        if any(
            isinstance(generation_raw[name], bool)
            or not isinstance(generation_raw[name], (int, float))
            for name in float_names
        ):
            raise EvaluationSchemaError("generation float controls are invalid")
        if not isinstance(generation_raw["do_sample"], bool) or not isinstance(
            generation_raw["chat_template_sha256"], str
        ):
            raise EvaluationSchemaError("generation scalar controls are invalid")
        manifest = cls(
            schema_version=cast(str, raw["schema_version"]),
            run_name=cast(str, raw["run_name"]),
            models=tuple(models),
            cases_sha256=cast(str, raw["cases_sha256"]),
            source_data_sha256=cast(dict[str, JsonValue], source_hashes),
            generation=GenerationSettings(
                max_new_tokens=cast(int, generation_raw["max_new_tokens"]),
                do_sample=generation_raw["do_sample"],
                temperature=float(cast(int | float, generation_raw["temperature"])),
                top_p=float(cast(int | float, generation_raw["top_p"])),
                seed=cast(int, generation_raw["seed"]),
                chat_template_sha256=generation_raw["chat_template_sha256"],
            ),
            code_revision=cast(str, raw["code_revision"]),
        )
        if raw["run_fingerprint"] != manifest.fingerprint():
            raise EvaluationSchemaError("run fingerprint mismatch")
        return manifest


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Raw model output from which all behavior metrics can be recomputed."""

    run_fingerprint: str
    model_key: str
    case_id: str
    response: str
    prompt_tokens: int
    response_tokens: int
    stop_reason: StopReason
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not _valid_sha256(self.run_fingerprint):
            raise EvaluationSchemaError("generation must bind to a complete run fingerprint")
        if not self.model_key.strip() or not self.case_id.strip():
            raise EvaluationSchemaError("generation identities must be non-empty")
        if self.prompt_tokens <= 0 or self.response_tokens < 0 or self.elapsed_seconds < 0:
            raise EvaluationSchemaError("generation counts/timing are invalid")

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible raw output fields."""
        return cast(dict[str, JsonValue], asdict(self))
