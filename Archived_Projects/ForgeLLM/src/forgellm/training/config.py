"""Strict TOML configuration for bounded Stage 3 experiments."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, cast

from forgellm.model.config import ModelConfig, ModelConfigError
from forgellm.structured_logging import JsonValue

DeviceChoice = Literal["auto", "cpu", "cuda"]
Precision = Literal["fp32", "bf16", "fp16"]
OptimizerName = Literal["adamw", "muon"]


class TrainingConfigError(ValueError):
    """Raised when a training experiment configuration is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All controls that can change optimization or experiment termination."""

    run_name: str = "stage3-smoke"
    seed: int = 20260727
    device: DeviceChoice = "auto"
    precision: Precision = "fp32"
    optimizer: OptimizerName = "adamw"
    sequence_length: int = 64
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_steps: int = 20
    max_tokens: int | None = None
    max_duration_seconds: float | None = None
    learning_rate: float = 3e-4
    muon_learning_rate: float = 0.02
    min_lr_ratio: float = 0.1
    warmup_steps: int = 5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip_norm: float = 1.0
    validation_interval: int = 10
    validation_batches: int = 2
    checkpoint_interval: int = 10
    log_interval: int = 1
    mtp_future_tokens: int = 0
    mtp_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise TrainingConfigError("run_name must be non-empty")
        if self.device not in ("auto", "cpu", "cuda"):
            raise TrainingConfigError("device must be auto, cpu, or cuda")
        if self.precision not in ("fp32", "bf16", "fp16"):
            raise TrainingConfigError("precision must be fp32, bf16, or fp16")
        if self.optimizer not in ("adamw", "muon"):
            raise TrainingConfigError("optimizer must be adamw or muon")
        positive_integers = {
            "seed": self.seed,
            "sequence_length": self.sequence_length,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
            "log_interval": self.log_interval,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingConfigError(f"{name} must be a positive integer")
        if self.checkpoint_interval < 0 or self.warmup_steps < 0 or self.mtp_future_tokens < 0:
            raise TrainingConfigError(
                "checkpoint_interval, warmup_steps, and mtp_future_tokens must be non-negative"
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise TrainingConfigError("max_tokens must be positive when set")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise TrainingConfigError("max_duration_seconds must be positive when set")
        if self.learning_rate <= 0 or self.muon_learning_rate <= 0:
            raise TrainingConfigError("learning rates must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise TrainingConfigError("min_lr_ratio must be in [0, 1]")
        if self.weight_decay < 0.0:
            raise TrainingConfigError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise TrainingConfigError("AdamW beta values must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise TrainingConfigError("gradient_clip_norm must be positive")
        if self.mtp_future_tokens == 0 and self.mtp_loss_weight != 0.0:
            raise TrainingConfigError("mtp_loss_weight must be zero when MTP is disabled")
        if self.mtp_future_tokens > 0 and self.mtp_loss_weight <= 0.0:
            raise TrainingConfigError("mtp_loss_weight must be positive when MTP is enabled")
        if self.mtp_future_tokens >= self.sequence_length:
            raise TrainingConfigError("mtp_future_tokens must be smaller than sequence_length")

    @property
    def effective_tokens_per_step(self) -> int:
        """Return target tokens consumed by one optimizer update."""
        return self.micro_batch_size * self.gradient_accumulation_steps * (self.sequence_length - 1)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible mapping."""
        return cast(dict[str, JsonValue], asdict(self))


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Resolved architecture and training settings loaded from one file."""

    model: ModelConfig
    training: TrainingConfig

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the complete resolved experiment configuration."""
        return {
            "model": cast(dict[str, JsonValue], self.model.as_dict()),
            "training": self.training.as_dict(),
        }

    def fingerprint(self) -> str:
        """Hash the canonical resolved configuration."""
        encoded = json.dumps(
            self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _construct_dataclass_values(
    section: object,
    *,
    name: str,
    allowed_fields: set[str],
) -> dict[str, object]:
    if not isinstance(section, dict):
        raise TrainingConfigError(f"[{name}] must be a TOML table")
    values = cast(dict[str, object], section)
    unknown = set(values) - allowed_fields
    if unknown:
        raise TrainingConfigError(f"unknown [{name}] fields: {', '.join(sorted(unknown))}")
    return values


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a strict two-section experiment TOML file."""
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingConfigError(f"could not read training config {path}: {error}") from error
    if set(parsed) != {"model", "training"}:
        raise TrainingConfigError("config must contain exactly [model] and [training]")

    model_values = _construct_dataclass_values(
        parsed["model"],
        name="model",
        allowed_fields={field.name for field in fields(ModelConfig)},
    )
    training_values = _construct_dataclass_values(
        parsed["training"],
        name="training",
        allowed_fields={field.name for field in fields(TrainingConfig)},
    )
    try:
        model = ModelConfig(**model_values)  # type: ignore[arg-type]
        training = TrainingConfig(**training_values)  # type: ignore[arg-type]
    except (ModelConfigError, TypeError, TrainingConfigError) as error:
        raise TrainingConfigError(f"invalid experiment config: {error}") from error
    if training.sequence_length > model.max_seq_len:
        raise TrainingConfigError("sequence_length cannot exceed model.max_seq_len")
    if model.vocab_size < 260:
        raise TrainingConfigError("Stage 3 byte-BPE models require vocab_size >= 260")
    return ExperimentConfig(model=model, training=training)
