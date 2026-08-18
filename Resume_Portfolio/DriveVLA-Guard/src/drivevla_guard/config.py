from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GenerationConfig:
    primary_mode: str = "fast"
    fast_candidates: int = 4
    slow_candidates: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 768
    expected_action_tokens: int = 10
    seed: int = 17

    def __post_init__(self) -> None:
        if self.primary_mode not in {"fast", "slow"}:
            raise ValueError("generation.primary_mode must be 'fast' or 'slow'")
        if self.fast_candidates < 1 or self.slow_candidates < 1:
            raise ValueError("candidate counts must be positive")
        if self.temperature <= 0:
            raise ValueError("generation.temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("generation.top_p must be in (0, 1]")
        if self.top_k < 1:
            raise ValueError("generation.top_k must be positive")
        if self.max_new_tokens < 1 or self.expected_action_tokens < 1:
            raise ValueError("token counts must be positive")


@dataclass(frozen=True)
class RiskWeights:
    invalid: float = 100.0
    collision: float = 12.0
    ttc: float = 4.0
    drivable: float = 8.0
    comfort: float = 1.5
    progress: float = 2.0
    confidence: float = 0.15

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("risk weights must be non-negative")


@dataclass(frozen=True)
class RiskConfig:
    weights: RiskWeights = field(default_factory=RiskWeights)
    ego_radius_m: float = 1.4
    clearance_scale_m: float = 2.0
    ttc_horizon_s: float = 4.0
    expected_progress_m: float = 20.0
    max_accel_mps2: float = 4.0
    max_jerk_mps3: float = 6.0
    max_yaw_rate_rps: float = 0.8

    def __post_init__(self) -> None:
        positive = {
            "ego_radius_m": self.ego_radius_m,
            "clearance_scale_m": self.clearance_scale_m,
            "ttc_horizon_s": self.ttc_horizon_s,
            "expected_progress_m": self.expected_progress_m,
            "max_accel_mps2": self.max_accel_mps2,
            "max_jerk_mps3": self.max_jerk_mps3,
            "max_yaw_rate_rps": self.max_yaw_rate_rps,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"risk limits must be positive: {', '.join(invalid)}")


@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = True
    risk_threshold: float = 4.0
    margin_threshold: float = 0.20
    uncertainty_threshold: float = 0.55
    route_on_missing_context: bool = False

    def __post_init__(self) -> None:
        if self.risk_threshold < 0 or self.margin_threshold < 0:
            raise ValueError("router risk and margin thresholds must be non-negative")
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError("router.uncertainty_threshold must be in [0, 1]")


@dataclass(frozen=True)
class GuardConfig:
    rerank_enabled: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    router: RouterConfig = field(default_factory=RouterConfig)


def _merge_dataclass(cls: type, data: dict[str, Any] | None):
    data = data or {}
    if cls is RiskConfig:
        weights = RiskWeights(**data.get("weights", {}))
        return RiskConfig(weights=weights, **{k: v for k, v in data.items() if k != "weights"})
    return cls(**data)


def load_config(path: str | Path) -> GuardConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"rerank_enabled", "generation", "risk", "router"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    return GuardConfig(
        rerank_enabled=bool(raw.get("rerank_enabled", True)),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation")),
        risk=_merge_dataclass(RiskConfig, raw.get("risk")),
        router=_merge_dataclass(RouterConfig, raw.get("router")),
    )
