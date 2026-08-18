from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .types import Candidate, SceneContext, Trajectory


class CandidateBackend(Protocol):
    def generate(
        self,
        model_input: dict[str, Any],
        context: SceneContext,
        mode: str,
        count: int,
        seed: int,
    ) -> list[Candidate]: ...


class SyntheticBackend:
    """Deterministic backend for exercising the full guard pipeline without a GPU."""

    def __init__(self, num_poses: int = 10, dt_s: float = 0.5):
        self.num_poses = num_poses
        self.dt_s = dt_s

    def generate(
        self,
        model_input: dict[str, Any],
        context: SceneContext,
        mode: str,
        count: int,
        seed: int,
    ) -> list[Candidate]:
        rng = np.random.default_rng(seed + (10_000 if mode == "slow" else 0))
        x = np.linspace(2.0, 20.0, self.num_poses)
        obstacle_y = context.obstacles[0].positions[-1, 1] if context.obstacles else 0.0
        candidates: list[Candidate] = []
        for index in range(count):
            if mode == "slow" and context.obstacles:
                target_y = 3.0 if obstacle_y <= 0 else -3.0
            else:
                offsets = [0.0, 3.0, -3.0, 1.2, -1.2]
                target_y = offsets[index % len(offsets)]
            # Cubic smoothstep completes the lane offset in 2.5 s without the
            # discontinuous lateral acceleration of a piecewise-linear path.
            phase = np.clip(np.arange(self.num_poses, dtype=np.float64) / 5.0, 0.0, 1.0)
            y = target_y * (3.0 * phase**2 - 2.0 * phase**3)
            if count > 1:
                y += rng.normal(0.0, 0.025, size=self.num_poses)
            heading = np.arctan2(np.gradient(y), np.gradient(x))
            poses = np.column_stack([x, y, heading])
            uncertainty = 0.18 + 0.16 * index if mode == "fast" else 0.12
            candidates.append(
                Candidate(
                    candidate_id=f"{mode}-{index}",
                    trajectory=Trajectory(poses, dt_s=self.dt_s),
                    mode=mode,
                    token_ids=tuple(range(index * self.num_poses, (index + 1) * self.num_poses)),
                    log_probability=-0.35 - 0.15 * index,
                    uncertainty=min(uncertainty, 0.95),
                    latency_ms=4.0 if mode == "fast" else 14.0,
                    raw_text=f"synthetic {mode} candidate {index}",
                )
            )
        return candidates
