"""Strict Stage 6 TOML configuration."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from forgellm.structured_logging import JsonValue


class EvaluationConfigError(ValueError):
    """Raised when Stage 6 could silently change an evaluation conclusion."""


@dataclass(frozen=True, slots=True)
class Stage6ModelConfig:
    """All evaluated model and adapter paths."""

    model_id: str
    revision: str
    sft_adapter_path: str
    dpo_adapter_path: str
    grpo_adapter_path: str
    stage3_config_path: str
    stage3_checkpoint_path: str
    stage3_tokenizer_path: str

    def __post_init__(self) -> None:
        if any(not value.strip() for value in asdict(self).values()):
            raise EvaluationConfigError("model identity/path values must be non-empty")


@dataclass(frozen=True, slots=True)
class Stage6DataConfig:
    """Frozen test and contamination source paths."""

    cases_path: str
    correctness_train_path: str
    correctness_test_path: str
    preference_train_path: str
    preference_test_path: str
    smoltalk_train_path: str
    retention_train_path: str
    retention_test_path: str

    def __post_init__(self) -> None:
        if any(not value.strip() for value in asdict(self).values()):
            raise EvaluationConfigError("data paths must be non-empty")


@dataclass(frozen=True, slots=True)
class Stage6QualityConfig:
    """Frozen behavior and statistical evaluation controls."""

    run_name: str
    seed: int
    max_length: int
    max_new_tokens: int
    bootstrap_resamples: int
    near_duplicate_threshold: float
    near_duplicate_ngram: int
    retention_documents: int

    def __post_init__(self) -> None:
        integers = (
            self.seed,
            self.max_length,
            self.max_new_tokens,
            self.bootstrap_resamples,
            self.near_duplicate_ngram,
            self.retention_documents,
        )
        if not self.run_name.strip() or any(value <= 0 for value in integers):
            raise EvaluationConfigError("quality name/positive controls are invalid")
        if not 0 <= self.near_duplicate_threshold <= 1:
            raise EvaluationConfigError("near_duplicate_threshold must be in [0,1]")


@dataclass(frozen=True, slots=True)
class Stage6JudgeConfig:
    """Human audit package controls; no paid judge is configured."""

    blind_cases: int
    comparison_left: str
    comparison_right: str

    def __post_init__(self) -> None:
        if self.blind_cases <= 0 or self.comparison_left == self.comparison_right:
            raise EvaluationConfigError("blind comparison controls are invalid")
        if {self.comparison_left, self.comparison_right} - {"q0", "q1", "q2"}:
            raise EvaluationConfigError("blind comparison model key is unknown")


@dataclass(frozen=True, slots=True)
class Stage6SystemConfig:
    """Bounded system benchmark matrix."""

    repeats: int
    warmups: int
    batch_sizes: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    max_new_tokens: int
    effort_token_budgets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.repeats < 5 or self.warmups < 1 or self.max_new_tokens <= 1:
            raise EvaluationConfigError("system repeats/warmups/token budget are invalid")
        if not self.batch_sizes or not self.prompt_lengths:
            raise EvaluationConfigError("system benchmark matrix cannot be empty")
        if any(
            value <= 0
            for value in (*self.batch_sizes, *self.prompt_lengths, *self.effort_token_budgets)
        ):
            raise EvaluationConfigError("system matrix values must be positive")
        if self.effort_token_budgets != (32, 64):
            raise EvaluationConfigError("effort_token_budgets must freeze the 32/64 audit")


@dataclass(frozen=True, slots=True)
class Stage6Config:
    """Complete resolved Stage 6 configuration."""

    model: Stage6ModelConfig
    data: Stage6DataConfig
    quality: Stage6QualityConfig
    judge: Stage6JudgeConfig
    systems: Stage6SystemConfig

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible resolved values."""
        return cast(dict[str, JsonValue], asdict(self))

    def fingerprint(self) -> str:
        """Hash the resolved protocol."""
        payload = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _section(raw: dict[str, object], name: str, fields: set[str]) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict) or set(value) != fields:
        raise EvaluationConfigError(f"[{name}] fields differ from the frozen schema")
    return cast(dict[str, object], value)


def _int_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise EvaluationConfigError(f"{name} must be a non-empty integer list")
    return tuple(cast(list[int], value))


def load_stage6_config(path: Path) -> Stage6Config:
    """Load exactly five TOML sections and reject unknown controls."""
    try:
        parsed_raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise EvaluationConfigError(f"could not read Stage 6 config: {error}") from error
    parsed = cast(dict[str, object], parsed_raw)
    if set(parsed) != {"model", "data", "quality", "judge", "systems"}:
        raise EvaluationConfigError("Stage 6 config must contain exactly five sections")
    model = _section(
        parsed,
        "model",
        {
            "model_id",
            "revision",
            "sft_adapter_path",
            "dpo_adapter_path",
            "grpo_adapter_path",
            "stage3_config_path",
            "stage3_checkpoint_path",
            "stage3_tokenizer_path",
        },
    )
    data = _section(
        parsed,
        "data",
        {
            "cases_path",
            "correctness_train_path",
            "correctness_test_path",
            "preference_train_path",
            "preference_test_path",
            "smoltalk_train_path",
            "retention_train_path",
            "retention_test_path",
        },
    )
    quality = _section(
        parsed,
        "quality",
        {
            "run_name",
            "seed",
            "max_length",
            "max_new_tokens",
            "bootstrap_resamples",
            "near_duplicate_threshold",
            "near_duplicate_ngram",
            "retention_documents",
        },
    )
    judge = _section(
        parsed,
        "judge",
        {"blind_cases", "comparison_left", "comparison_right"},
    )
    systems = _section(
        parsed,
        "systems",
        {
            "repeats",
            "warmups",
            "batch_sizes",
            "prompt_lengths",
            "max_new_tokens",
            "effort_token_budgets",
        },
    )
    try:
        return Stage6Config(
            model=Stage6ModelConfig(**model),  # type: ignore[arg-type]
            data=Stage6DataConfig(**data),  # type: ignore[arg-type]
            quality=Stage6QualityConfig(**quality),  # type: ignore[arg-type]
            judge=Stage6JudgeConfig(**judge),  # type: ignore[arg-type]
            systems=Stage6SystemConfig(
                repeats=cast(int, systems["repeats"]),
                warmups=cast(int, systems["warmups"]),
                batch_sizes=_int_tuple(systems["batch_sizes"], name="batch_sizes"),
                prompt_lengths=_int_tuple(systems["prompt_lengths"], name="prompt_lengths"),
                max_new_tokens=cast(int, systems["max_new_tokens"]),
                effort_token_budgets=_int_tuple(
                    systems["effort_token_budgets"], name="effort_token_budgets"
                ),
            ),
        )
    except (TypeError, EvaluationConfigError) as error:
        raise EvaluationConfigError(f"invalid Stage 6 config: {error}") from error
