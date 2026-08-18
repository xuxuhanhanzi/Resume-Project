from __future__ import annotations

import numpy as np

from .types import DynamicObstacle, SceneContext


def build_synthetic_manifest(num_scenes: int = 12, num_poses: int = 10) -> list[dict]:
    if num_scenes < 1:
        raise ValueError("num_scenes must be positive")
    rows: list[dict] = []
    for index in range(num_scenes):
        kind = index % 4
        obstacles: tuple[DynamicObstacle, ...] = ()
        lane_half_width = 3.5
        label = "clear"
        if kind == 1:
            label = "stationary_blocker"
            positions = np.repeat(np.array([[10.0, 0.0]]), num_poses, axis=0)
            obstacles = (DynamicObstacle(f"vehicle-{index}", positions, radius_m=1.1),)
        elif kind == 2:
            label = "crossing_actor"
            positions = np.column_stack(
                [np.linspace(8.0, 12.0, num_poses), np.linspace(3.0, -1.0, num_poses)]
            )
            obstacles = (DynamicObstacle(f"pedestrian-{index}", positions, radius_m=0.7),)
        elif kind == 3:
            label = "narrow_lane"
            lane_half_width = 2.5

        context = SceneContext(
            scene_id=f"synthetic-{index:03d}",
            obstacles=obstacles,
            lane_half_width_m=lane_half_width,
            source="synthetic_current_observation",
            metadata={"scenario_type": label, "split": "smoke"},
        )
        rows.append(
            {
                "model_input": {"scenario_type": label, "instruction": "keep forward"},
                "scene": context.to_dict(),
            }
        )
    return rows
