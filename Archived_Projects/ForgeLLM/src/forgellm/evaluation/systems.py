"""System benchmark contracts, timing summaries and KV-cache accounting."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Robust latency summary retaining repeat count."""

    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    repeats: int

    def as_dict(self) -> dict[str, float | int]:
        """Return serializable values."""
        return asdict(self)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_timings(values: list[float]) -> TimingSummary:
    """Summarize at least five finite non-negative measurements."""
    if len(values) < 5 or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing summary requires at least five finite non-negative values")
    return TimingSummary(
        median_seconds=_quantile(values, 0.5),
        p10_seconds=_quantile(values, 0.1),
        p90_seconds=_quantile(values, 0.9),
        repeats=len(values),
    )


def theoretical_kv_cache_bytes(
    *,
    layers: int,
    batch_size: int,
    sequence_length: int,
    key_value_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Compute K and V storage without allocator or framework overhead."""
    values = (layers, batch_size, sequence_length, key_value_heads, head_dim, bytes_per_element)
    if any(value <= 0 for value in values):
        raise ValueError("KV-cache dimensions must be positive")
    return (
        2 * layers * batch_size * sequence_length * key_value_heads * head_dim * bytes_per_element
    )


def approximate_decode_tokens_per_second(
    *,
    batch_size: int,
    generated_tokens: int,
    total_seconds: float,
    first_token_seconds: float,
) -> float:
    """Estimate steady decode rate after subtracting a separately measured first token."""
    if (
        batch_size <= 0
        or generated_tokens <= 1
        or first_token_seconds < 0
        or total_seconds <= first_token_seconds
    ):
        raise ValueError("invalid decode timing inputs")
    return batch_size * (generated_tokens - 1) / (total_seconds - first_token_seconds)
