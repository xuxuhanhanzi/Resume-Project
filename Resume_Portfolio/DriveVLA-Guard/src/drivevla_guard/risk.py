from __future__ import annotations

import math

import numpy as np

from .config import RiskConfig
from .types import Candidate, RiskBreakdown, SceneContext


class RiskScorer:
    """Deterministic lower-is-better score using only declared evidence."""

    def __init__(self, config: RiskConfig):
        self.config = config

    def score(self, candidate: Candidate, context: SceneContext) -> RiskBreakdown:
        trajectory = candidate.trajectory
        poses = trajectory.poses
        invalid = 0.0 if candidate.valid and trajectory.finite else 1.0

        progress = self._progress_cost(poses)
        comfort = self._comfort_cost(poses, trajectory.dt_s)
        confidence = self._confidence_cost(candidate.log_probability)

        collision: float | None = None
        ttc: float | None = None
        min_clearance: float | None = None
        min_ttc: float | None = None
        if context.obstacles:
            collision, ttc, min_clearance, min_ttc = self._obstacle_costs(poses, trajectory.dt_s, context)

        drivable: float | None = None
        if context.lane_half_width_m is not None:
            drivable = float(np.mean(np.abs(poses[:, 1]) > context.lane_half_width_m))

        weights = self.config.weights
        total = weights.invalid * invalid + weights.comfort * comfort + weights.progress * progress
        if confidence is not None:
            total += weights.confidence * confidence
        if collision is not None:
            total += weights.collision * collision
        if ttc is not None:
            total += weights.ttc * ttc
        if drivable is not None:
            total += weights.drivable * drivable

        return RiskBreakdown(
            total=float(total),
            invalid=invalid,
            collision=collision,
            ttc=ttc,
            drivable=drivable,
            comfort=comfort,
            progress=progress,
            confidence=confidence,
            min_clearance_m=min_clearance,
            min_ttc_s=min_ttc,
            evidence_source=context.source,
        )

    def rank(self, candidates: list[Candidate], context: SceneContext):
        if not candidates:
            raise ValueError("at least one candidate is required")
        from .types import ScoredCandidate

        scored = [ScoredCandidate(candidate=item, risk=self.score(item, context)) for item in candidates]
        return sorted(scored, key=lambda item: (item.risk.total, item.candidate.candidate_id))

    def _progress_cost(self, poses: np.ndarray) -> float:
        progress = max(float(poses[-1, 0]), 0.0)
        ratio = min(progress / max(self.config.expected_progress_m, 1e-6), 1.0)
        return 1.0 - ratio

    def _comfort_cost(self, poses: np.ndarray, dt_s: float) -> float:
        xy = np.vstack([np.zeros((1, 2)), poses[:, :2]])
        velocity = np.diff(xy, axis=0) / dt_s
        acceleration = np.diff(velocity, axis=0) / dt_s
        jerk = np.diff(acceleration, axis=0) / dt_s
        headings = np.unwrap(np.concatenate([[0.0], poses[:, 2]]))
        yaw_rate = np.diff(headings) / dt_s

        accel_peak = float(np.linalg.norm(acceleration, axis=1).max()) if len(acceleration) else 0.0
        jerk_peak = float(np.linalg.norm(jerk, axis=1).max()) if len(jerk) else 0.0
        yaw_peak = float(np.abs(yaw_rate).max()) if len(yaw_rate) else 0.0
        return float(
            np.mean(
                [
                    min(accel_peak / self.config.max_accel_mps2, 2.0),
                    min(jerk_peak / self.config.max_jerk_mps3, 2.0),
                    min(yaw_peak / self.config.max_yaw_rate_rps, 2.0),
                ]
            )
        )

    @staticmethod
    def _confidence_cost(log_probability: float | None) -> float | None:
        if log_probability is None or not math.isfinite(log_probability):
            return None
        return float(min(max(-log_probability, 0.0), 20.0) / 20.0)

    def _obstacle_costs(
        self, poses: np.ndarray, dt_s: float, context: SceneContext
    ) -> tuple[float, float, float, float | None]:
        clearances: list[float] = []
        collision_steps: list[int] = []
        for obstacle in context.obstacles:
            length = min(len(poses), len(obstacle.positions))
            if length == 0:
                continue
            distance = np.linalg.norm(poses[:length, :2] - obstacle.positions[:length], axis=1)
            clearance = distance - (self.config.ego_radius_m + obstacle.radius_m)
            clearances.extend(clearance.tolist())
            collision_steps.extend(np.flatnonzero(clearance <= 0.0).tolist())

        if not clearances:
            return 0.0, 0.0, float("inf"), None
        min_clearance = float(min(clearances))
        collision = (
            1.0
            if collision_steps
            else float(np.exp(-max(min_clearance, 0.0) / self.config.clearance_scale_m))
        )
        min_ttc = (min(collision_steps) + 1) * dt_s if collision_steps else None
        ttc_cost = 0.0 if min_ttc is None else max(0.0, 1.0 - min_ttc / self.config.ttc_horizon_s)
        return float(collision), float(ttc_cost), min_clearance, min_ttc
