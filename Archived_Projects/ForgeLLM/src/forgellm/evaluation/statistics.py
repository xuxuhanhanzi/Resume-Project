"""Small dependency-free confidence intervals and paired comparisons."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    """One estimate and lower/upper confidence bounds."""

    estimate: float
    lower: float
    upper: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        """Return serializable values."""
        return asdict(self)


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> Interval:
    """Compute a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or successes < 0 or successes > total or z <= 0:
        raise ValueError("invalid Wilson interval inputs")
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z_squared / (4 * total * total))
        / denominator
    )
    return Interval(proportion, max(0.0, center - half), min(1.0, center + half), total)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError("invalid percentile inputs")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def paired_bootstrap_interval(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = 2000,
    seed: int = 20260728,
    confidence: float = 0.95,
) -> Interval:
    """Bootstrap the paired mean difference candidate minus baseline."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap vectors must align and be non-empty")
    if resamples <= 0 or seed < 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap controls")
    differences = [right - left for left, right in zip(baseline, candidate, strict=True)]
    generator = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(resamples):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        sample_means.append(sum(sample) / len(sample))
    sample_means.sort()
    alpha = (1 - confidence) / 2
    estimate = sum(differences) / len(differences)
    return Interval(
        estimate,
        _percentile(sample_means, alpha),
        _percentile(sample_means, 1 - alpha),
        len(differences),
    )


def cohens_kappa(left: list[str], right: list[str]) -> float:
    """Compute nominal Cohen's kappa for two complete raters."""
    if len(left) != len(right) or not left:
        raise ValueError("two aligned non-empty rating vectors are required")
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    expected = 0.0
    for label in labels:
        expected += (left.count(label) / len(left)) * (right.count(label) / len(right))
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)
