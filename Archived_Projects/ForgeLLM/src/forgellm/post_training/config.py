"""Strict TOML configuration for Stage 4 Hugging Face Adapter runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from forgellm.structured_logging import JsonValue

Quantization = Literal["none", "nf4"]


class Stage4ConfigError(ValueError):
    """Raised when a Stage 4 run config is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class HFModelConfig:
    """Frozen Hub identity and loading mode."""

    model_id: str
    revision: str
    quantization: Quantization = "none"

    def __post_init__(self) -> None:
        if not self.model_id.strip() or len(self.revision) < 7:
            raise Stage4ConfigError("model_id and exact revision are required")
        if self.quantization not in ("none", "nf4"):
            raise Stage4ConfigError("quantization must be none or nf4")


@dataclass(frozen=True, slots=True)
class Stage4DataConfig:
    """Frozen local split paths used by one run."""

    train_path: str
    validation_path: str
    task_test_path: str
    retention_path: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in asdict(self).values()):
            raise Stage4ConfigError("all Stage 4 data paths must be non-empty")


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """One deliberately unsearched Adapter configuration."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str = "all-linear"

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise Stage4ConfigError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise Stage4ConfigError("LoRA dropout must be in [0, 1)")
        if self.target_modules != "all-linear":
            raise Stage4ConfigError("Stage 4 freezes target_modules to all-linear")


@dataclass(frozen=True, slots=True)
class Stage4TrainingConfig:
    """Bounded local training and evaluation controls."""

    run_name: str
    seed: int = 20260728
    max_length: int = 512
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_steps: int = 20
    max_assistant_tokens: int = 20000
    max_duration_seconds: float = 900.0
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    eval_examples: int = 32
    generation_examples: int = 16
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        integer_fields = {
            "seed": self.seed,
            "max_length": self.max_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "max_assistant_tokens": self.max_assistant_tokens,
            "eval_examples": self.eval_examples,
            "generation_examples": self.generation_examples,
            "max_new_tokens": self.max_new_tokens,
        }
        if not self.run_name.strip():
            raise Stage4ConfigError("run_name must be non-empty")
        if any(isinstance(value, bool) or value <= 0 for value in integer_fields.values()):
            raise Stage4ConfigError("Stage 4 integer controls must be positive")
        if self.max_length < 64:
            raise Stage4ConfigError("max_length must be at least 64")
        if self.max_duration_seconds <= 0 or self.learning_rate <= 0:
            raise Stage4ConfigError("duration and learning rate must be positive")
        if not 0.0 < self.warmup_ratio < 1.0:
            raise Stage4ConfigError("warmup_ratio must be in (0, 1)")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise Stage4ConfigError("weight decay/gradient clip are invalid")


@dataclass(frozen=True, slots=True)
class Stage4RunConfig:
    """Complete resolved Stage 4 HF run."""

    model: HFModelConfig
    data: Stage4DataConfig
    lora: LoRAConfig
    training: Stage4TrainingConfig

    def as_dict(self) -> dict[str, JsonValue]:
        """Return stable JSON values."""
        return cast(dict[str, JsonValue], asdict(self))

    def fingerprint(self) -> str:
        """Hash the full resolved run."""
        encoded = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _strict_values(raw: object, *, section: str, expected: set[str]) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise Stage4ConfigError(f"[{section}] must be a TOML table")
    values = cast(dict[str, object], raw)
    if set(values) != expected:
        missing = expected - set(values)
        unknown = set(values) - expected
        raise Stage4ConfigError(
            f"[{section}] fields differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return values


def _string(values: dict[str, object], key: str) -> str:
    value = values[key]
    if not isinstance(value, str):
        raise Stage4ConfigError(f"{key} must be a string")
    return value


def _integer(values: dict[str, object], key: str) -> int:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage4ConfigError(f"{key} must be an integer")
    return value


def _number(values: dict[str, object], key: str) -> float:
    value = values[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Stage4ConfigError(f"{key} must be a number")
    return float(value)


def load_stage4_run_config(path: Path) -> Stage4RunConfig:
    """Load exactly four sections and validate every field."""
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Stage4ConfigError(f"could not read Stage 4 config {path}: {error}") from error
    if set(parsed) != {"data", "lora", "model", "training"}:
        raise Stage4ConfigError("config must contain [model], [data], [lora], [training]")
    model = _strict_values(
        parsed["model"], section="model", expected={"model_id", "revision", "quantization"}
    )
    data = _strict_values(
        parsed["data"],
        section="data",
        expected={"train_path", "validation_path", "task_test_path", "retention_path"},
    )
    lora = _strict_values(
        parsed["lora"],
        section="lora",
        expected={"rank", "alpha", "dropout", "target_modules"},
    )
    training = _strict_values(
        parsed["training"],
        section="training",
        expected={
            "run_name",
            "seed",
            "max_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "max_assistant_tokens",
            "max_duration_seconds",
            "learning_rate",
            "warmup_ratio",
            "weight_decay",
            "gradient_clip_norm",
            "eval_examples",
            "generation_examples",
            "max_new_tokens",
        },
    )
    quantization = _string(model, "quantization")
    if quantization not in ("none", "nf4"):
        raise Stage4ConfigError("quantization must be none or nf4")
    return Stage4RunConfig(
        model=HFModelConfig(
            model_id=_string(model, "model_id"),
            revision=_string(model, "revision"),
            quantization=cast(Quantization, quantization),
        ),
        data=Stage4DataConfig(
            train_path=_string(data, "train_path"),
            validation_path=_string(data, "validation_path"),
            task_test_path=_string(data, "task_test_path"),
            retention_path=_string(data, "retention_path"),
        ),
        lora=LoRAConfig(
            rank=_integer(lora, "rank"),
            alpha=_integer(lora, "alpha"),
            dropout=_number(lora, "dropout"),
            target_modules=_string(lora, "target_modules"),
        ),
        training=Stage4TrainingConfig(
            run_name=_string(training, "run_name"),
            seed=_integer(training, "seed"),
            max_length=_integer(training, "max_length"),
            micro_batch_size=_integer(training, "micro_batch_size"),
            gradient_accumulation_steps=_integer(training, "gradient_accumulation_steps"),
            max_steps=_integer(training, "max_steps"),
            max_assistant_tokens=_integer(training, "max_assistant_tokens"),
            max_duration_seconds=_number(training, "max_duration_seconds"),
            learning_rate=_number(training, "learning_rate"),
            warmup_ratio=_number(training, "warmup_ratio"),
            weight_decay=_number(training, "weight_decay"),
            gradient_clip_norm=_number(training, "gradient_clip_norm"),
            eval_examples=_integer(training, "eval_examples"),
            generation_examples=_integer(training, "generation_examples"),
            max_new_tokens=_integer(training, "max_new_tokens"),
        ),
    )
