"""Numerically explicit loss, byte, throughput, and memory metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import cast

from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Metrics emitted for one optimizer update."""

    step: int
    loss: float
    main_loss: float
    mtp_loss: float | None
    learning_rates: tuple[float, ...]
    gradient_norm: float
    target_tokens: int
    tokens_seen: int
    step_seconds: float
    tokens_per_second: float
    peak_memory_mib: float | None

    def as_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Aggregate validation metrics with tokenizer-aware BPB."""

    loss: float
    perplexity: float
    bits_per_byte: float | None
    target_tokens: int
    target_bytes: int

    def as_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], asdict(self))


def perplexity_from_loss(loss: float) -> float:
    """Convert mean token NLL to perplexity without overflowing Python."""
    return math.exp(min(loss, 80.0))


def bits_per_byte(*, total_nll_nats: float, target_bytes: int) -> float | None:
    """Normalize total natural-log loss by original UTF-8 target bytes."""
    if target_bytes < 0:
        raise ValueError("target_bytes must be non-negative")
    if target_bytes == 0:
        return None
    return total_nll_nats / (math.log(2.0) * target_bytes)
