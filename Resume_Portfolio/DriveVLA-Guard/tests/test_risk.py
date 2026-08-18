import numpy as np

from drivevla_guard.config import RiskConfig
from drivevla_guard.risk import RiskScorer
from drivevla_guard.types import Candidate, DynamicObstacle, SceneContext, Trajectory


def candidate(candidate_id: str, y: float) -> Candidate:
    x = np.linspace(2.0, 20.0, 10)
    poses = np.column_stack([x, np.full_like(x, y), np.zeros_like(x)])
    return Candidate(candidate_id, Trajectory(poses), mode="fast")


def test_collision_candidate_is_ranked_below_clear_candidate() -> None:
    obstacle = DynamicObstacle("car", np.repeat([[10.0, 0.0]], 10, axis=0), radius_m=1.0)
    context = SceneContext("scene", (obstacle,), lane_half_width_m=4.0, source="test")
    ranked = RiskScorer(RiskConfig()).rank([candidate("collision", 0.0), candidate("clear", 3.0)], context)
    assert ranked[0].candidate.candidate_id == "clear"
    assert ranked[1].risk.collision == 1.0


def test_environment_metrics_are_none_without_context() -> None:
    score = RiskScorer(RiskConfig()).score(candidate("straight", 0.0), SceneContext("scene"))
    assert score.collision is None
    assert score.ttc is None
    assert score.drivable is None
    assert score.evidence_source == "trajectory_only"


def test_drivable_violation_is_reported() -> None:
    score = RiskScorer(RiskConfig()).score(
        candidate("outside", 3.0), SceneContext("scene", lane_half_width_m=2.0, source="test")
    )
    assert score.drivable == 1.0
