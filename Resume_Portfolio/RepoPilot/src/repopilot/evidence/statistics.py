"""Deterministic confidence intervals used by frozen paired evaluations."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    successes: int
    trials: int
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    pairs: int
    estimate: float
    lower: float
    upper: float
    confidence: float
    repetitions: int
    seed: int


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> WilsonInterval:
    """Compute a two-sided Wilson score interval without optional libraries."""

    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("successes must be between zero and a positive trial count")
    if z <= 0:
        raise ValueError("z must be positive")
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (estimate + z * z / (2.0 * trials)) / denominator
    margin = (
        z * math.sqrt((estimate * (1.0 - estimate) + z * z / (4.0 * trials)) / trials) / denominator
    )
    return WilsonInterval(
        successes, trials, estimate, max(0.0, centre - margin), min(1.0, centre + margin)
    )


def paired_bootstrap_mean_difference(
    baseline: Sequence[float | int | bool],
    candidate: Sequence[float | int | bool],
    *,
    repetitions: int = 10_000,
    seed: int = 7,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Percentile CI of candidate minus baseline over paired task samples."""

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("baseline and candidate must have the same non-zero length")
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    differences = [
        float(after) - float(before) for before, after in zip(baseline, candidate, strict=True)
    ]
    generator = random.Random(seed)
    samples = [
        fmean(differences[generator.randrange(len(differences))] for _ in differences)
        for _ in range(repetitions)
    ]
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(alpha * repetitions))
    upper_index = min(repetitions - 1, math.ceil((1.0 - alpha) * repetitions) - 1)
    return BootstrapInterval(
        pairs=len(differences),
        estimate=fmean(differences),
        lower=samples[lower_index],
        upper=samples[upper_index],
        confidence=confidence,
        repetitions=repetitions,
        seed=seed,
    )
