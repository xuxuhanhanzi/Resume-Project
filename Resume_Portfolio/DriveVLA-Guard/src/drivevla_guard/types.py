from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """Ego-frame trajectory with rows ``[x_m, y_m, heading_rad]``."""

    poses: np.ndarray
    dt_s: float = 0.5

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"poses must have shape [T, 3], got {poses.shape}")
        if poses.shape[0] == 0:
            raise ValueError("trajectory must contain at least one pose")
        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive")
        object.__setattr__(self, "poses", poses)

    @property
    def finite(self) -> bool:
        return bool(np.isfinite(self.poses).all())

    def to_dict(self) -> dict[str, Any]:
        return {"poses": self.poses.tolist(), "dt_s": self.dt_s}


@dataclass(frozen=True)
class DynamicObstacle:
    obstacle_id: str
    positions: np.ndarray
    radius_m: float = 1.2

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"obstacle positions must have shape [T, 2], got {positions.shape}")
        if self.radius_m <= 0:
            raise ValueError("obstacle radius must be positive")
        object.__setattr__(self, "positions", positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "positions": self.positions.tolist(),
            "radius_m": self.radius_m,
        }


@dataclass(frozen=True)
class SceneContext:
    scene_id: str
    obstacles: tuple[DynamicObstacle, ...] = ()
    lane_half_width_m: float | None = None
    source: str = "trajectory_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_environment(self) -> bool:
        return bool(self.obstacles) or self.lane_half_width_m is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "obstacles": [item.to_dict() for item in self.obstacles],
            "lane_half_width_m": self.lane_half_width_m,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneContext:
        return cls(
            scene_id=str(data["scene_id"]),
            obstacles=tuple(DynamicObstacle(**item) for item in data.get("obstacles", [])),
            lane_half_width_m=data.get("lane_half_width_m"),
            source=data.get("source", "trajectory_only"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    trajectory: Trajectory
    mode: str
    token_ids: tuple[int, ...] = ()
    log_probability: float | None = None
    uncertainty: float | None = None
    latency_ms: float = 0.0
    valid: bool = True
    error: str | None = None
    raw_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["trajectory"] = self.trajectory.to_dict()
        result["token_ids"] = list(self.token_ids)
        return result


@dataclass(frozen=True)
class RiskBreakdown:
    total: float
    invalid: float
    collision: float | None
    ttc: float | None
    drivable: float | None
    comfort: float
    progress: float
    confidence: float | None
    min_clearance_m: float | None
    min_ttc_s: float | None
    evidence_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    risk: RiskBreakdown

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate.to_dict(), "risk": self.risk.to_dict()}


@dataclass(frozen=True)
class PlanResult:
    scene_id: str
    selected: ScoredCandidate
    fast_candidates: tuple[ScoredCandidate, ...]
    slow_candidates: tuple[ScoredCandidate, ...]
    routed_to_slow: bool
    route_reasons: tuple[str, ...]
    total_latency_ms: float
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "selected": self.selected.to_dict(),
            "fast_candidates": [item.to_dict() for item in self.fast_candidates],
            "slow_candidates": [item.to_dict() for item in self.slow_candidates],
            "routed_to_slow": self.routed_to_slow,
            "route_reasons": list(self.route_reasons),
            "total_latency_ms": self.total_latency_ms,
            "config_hash": self.config_hash,
        }
