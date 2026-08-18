"""Dependency-light paired uncertainty and significance utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class McNemarResult:
    baseline_only_correct: int
    candidate_only_correct: int
    exact_p_value: float


def paired_bootstrap_delta(
    baseline: NDArray[np.float64],
    candidate: NDArray[np.float64],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> dict[str, float | int]:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != candidate.shape or len(baseline) == 0:
        raise ValueError("paired samples must be non-empty aligned vectors")
    if samples < 100 or not 0 < confidence < 1:
        raise ValueError("samples must be >=100 and confidence must be in (0, 1)")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired samples must be finite")
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "pairs": len(delta),
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap, alpha / 2)),
        "ci_high": float(np.quantile(bootstrap, 1 - alpha / 2)),
        "confidence": confidence,
        "bootstrap_samples": samples,
        "seed": seed,
    }


def paired_mcnemar(baseline_correct: list[bool], candidate_correct: list[bool]) -> McNemarResult:
    if not baseline_correct or len(baseline_correct) != len(candidate_correct):
        raise ValueError("paired correctness vectors must be non-empty and aligned")
    baseline_only = sum(
        base and not candidate
        for base, candidate in zip(baseline_correct, candidate_correct, strict=True)
    )
    candidate_only = sum(
        candidate and not base
        for base, candidate in zip(baseline_correct, candidate_correct, strict=True)
    )
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) for index in range(min(baseline_only, candidate_only) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return McNemarResult(baseline_only, candidate_only, p_value)
