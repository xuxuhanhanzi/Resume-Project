"""Strict Stage 5 configuration shared by DPO and one-step GRPO runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from forgellm.structured_logging import JsonValue


class Stage5ConfigError(ValueError):
    """Raised when Stage 5 configuration drifts from the frozen protocol."""


@dataclass(frozen=True, slots=True)
class Stage5ModelConfig:
    """Frozen Base plus Stage 4 SFT Adapter identity."""

    model_id: str
    revision: str
    initial_adapter_path: str

    def __post_init__(self) -> None:
        if (
            not self.model_id.strip()
            or len(self.revision) < 7
            or not self.initial_adapter_path.strip()
        ):
            raise Stage5ConfigError("model ID, exact revision and initial Adapter are required")


@dataclass(frozen=True, slots=True)
class Stage5DataConfig:
    """Frozen preference splits and Stage 4 evaluation protocol."""

    train_path: str
    validation_path: str
    test_path: str
    stage4_evaluation_config: str

    def __post_init__(self) -> None:
        if any(not value.strip() for value in asdict(self).values()):
            raise Stage5ConfigError("all Stage 5 data paths must be non-empty")


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    """Single bounded DPO configuration; no hyperparameter search."""

    run_name: str
    seed: int
    max_length: int
    gradient_accumulation_steps: int
    max_steps: int
    max_pair_tokens: int
    max_duration_seconds: float
    learning_rate: float
    beta: float
    gradient_clip_norm: float
    evaluation_pairs: int
    generation_examples: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        integers = (
            self.seed,
            self.max_length,
            self.gradient_accumulation_steps,
            self.max_steps,
            self.max_pair_tokens,
            self.evaluation_pairs,
            self.generation_examples,
            self.max_new_tokens,
        )
        if not self.run_name.strip() or any(value <= 0 for value in integers):
            raise Stage5ConfigError("DPO name and integer controls must be positive")
        if self.max_length < 64 or self.max_duration_seconds <= 0:
            raise Stage5ConfigError("DPO length/duration controls are invalid")
        if self.learning_rate <= 0 or self.beta <= 0 or self.gradient_clip_norm <= 0:
            raise Stage5ConfigError("DPO optimization controls must be positive")


@dataclass(frozen=True, slots=True)
class GRPOSmokeConfig:
    """Exactly one real Qwen rollout/update smoke configuration."""

    run_name: str
    seed: int
    max_length: int
    prompts: int
    group_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    learning_rate: float
    clip_epsilon: float
    kl_beta: float
    gradient_clip_norm: float
    max_duration_seconds: float

    def __post_init__(self) -> None:
        integers = (self.seed, self.max_length, self.prompts, self.group_size, self.max_new_tokens)
        if not self.run_name.strip() or any(value <= 0 for value in integers):
            raise Stage5ConfigError("GRPO name and integer controls must be positive")
        if self.prompts > 4 or self.group_size != 4 or self.max_new_tokens > 32:
            raise Stage5ConfigError("GRPO smoke is capped at 4 prompts x 4 rollouts x 32 tokens")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise Stage5ConfigError("GRPO sampling controls are invalid")
        if self.learning_rate <= 0 or not 0 < self.clip_epsilon < 1 or self.kl_beta < 0:
            raise Stage5ConfigError("GRPO optimization controls are invalid")
        if self.gradient_clip_norm <= 0 or self.max_duration_seconds <= 0:
            raise Stage5ConfigError("GRPO gradient/time controls must be positive")


@dataclass(frozen=True, slots=True)
class Stage5RunConfig:
    """Resolved, fingerprinted Stage 5 protocol."""

    model: Stage5ModelConfig
    data: Stage5DataConfig
    dpo: DPOTrainingConfig
    grpo: GRPOSmokeConfig

    def as_dict(self) -> dict[str, JsonValue]:
        """Return stable JSON-compatible values."""
        return cast(dict[str, JsonValue], asdict(self))

    def fingerprint(self) -> str:
        """Hash the complete resolved protocol."""
        encoded = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _strict(raw: object, *, section: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != keys:
        actual = set(raw) if isinstance(raw, dict) else set()
        raise Stage5ConfigError(
            f"[{section}] fields differ; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return cast(dict[str, object], raw)


def _string(values: dict[str, object], key: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise Stage5ConfigError(f"{key} must be a string")
    return value


def _integer(values: dict[str, object], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage5ConfigError(f"{key} must be an integer")
    return value


def _number(values: dict[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage5ConfigError(f"{key} must be numeric")
    return float(value)


def load_stage5_run_config(path: Path) -> Stage5RunConfig:
    """Parse exactly four Stage 5 sections and reject silent defaults."""
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Stage5ConfigError(f"could not read Stage 5 config {path}: {error}") from error
    if set(parsed) != {"model", "data", "dpo", "grpo"}:
        raise Stage5ConfigError("Stage 5 config requires [model], [data], [dpo], [grpo]")
    model = _strict(
        parsed["model"],
        section="model",
        keys={"model_id", "revision", "initial_adapter_path"},
    )
    data = _strict(
        parsed["data"],
        section="data",
        keys={"train_path", "validation_path", "test_path", "stage4_evaluation_config"},
    )
    dpo = _strict(
        parsed["dpo"],
        section="dpo",
        keys={
            "run_name",
            "seed",
            "max_length",
            "gradient_accumulation_steps",
            "max_steps",
            "max_pair_tokens",
            "max_duration_seconds",
            "learning_rate",
            "beta",
            "gradient_clip_norm",
            "evaluation_pairs",
            "generation_examples",
            "max_new_tokens",
        },
    )
    grpo = _strict(
        parsed["grpo"],
        section="grpo",
        keys={
            "run_name",
            "seed",
            "max_length",
            "prompts",
            "group_size",
            "max_new_tokens",
            "temperature",
            "top_p",
            "learning_rate",
            "clip_epsilon",
            "kl_beta",
            "gradient_clip_norm",
            "max_duration_seconds",
        },
    )
    return Stage5RunConfig(
        model=Stage5ModelConfig(
            _string(model, "model_id"),
            _string(model, "revision"),
            _string(model, "initial_adapter_path"),
        ),
        data=Stage5DataConfig(
            _string(data, "train_path"),
            _string(data, "validation_path"),
            _string(data, "test_path"),
            _string(data, "stage4_evaluation_config"),
        ),
        dpo=DPOTrainingConfig(
            run_name=_string(dpo, "run_name"),
            seed=_integer(dpo, "seed"),
            max_length=_integer(dpo, "max_length"),
            gradient_accumulation_steps=_integer(dpo, "gradient_accumulation_steps"),
            max_steps=_integer(dpo, "max_steps"),
            max_pair_tokens=_integer(dpo, "max_pair_tokens"),
            max_duration_seconds=_number(dpo, "max_duration_seconds"),
            learning_rate=_number(dpo, "learning_rate"),
            beta=_number(dpo, "beta"),
            gradient_clip_norm=_number(dpo, "gradient_clip_norm"),
            evaluation_pairs=_integer(dpo, "evaluation_pairs"),
            generation_examples=_integer(dpo, "generation_examples"),
            max_new_tokens=_integer(dpo, "max_new_tokens"),
        ),
        grpo=GRPOSmokeConfig(
            run_name=_string(grpo, "run_name"),
            seed=_integer(grpo, "seed"),
            max_length=_integer(grpo, "max_length"),
            prompts=_integer(grpo, "prompts"),
            group_size=_integer(grpo, "group_size"),
            max_new_tokens=_integer(grpo, "max_new_tokens"),
            temperature=_number(grpo, "temperature"),
            top_p=_number(grpo, "top_p"),
            learning_rate=_number(grpo, "learning_rate"),
            clip_epsilon=_number(grpo, "clip_epsilon"),
            kl_beta=_number(grpo, "kl_beta"),
            gradient_clip_norm=_number(grpo, "gradient_clip_norm"),
            max_duration_seconds=_number(grpo, "max_duration_seconds"),
        ),
    )
