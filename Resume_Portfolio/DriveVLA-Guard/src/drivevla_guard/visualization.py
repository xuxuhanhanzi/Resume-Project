from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .types import SceneContext


def render_result(row: dict[str, Any], context: SceneContext, output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("visualization requires matplotlib") from exc

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if context.lane_half_width_m is not None:
        axis.axhspan(-context.lane_half_width_m, context.lane_half_width_m, color="#edf5ff", alpha=0.8)
        axis.axhline(context.lane_half_width_m, color="#6b7280", linestyle="--")
        axis.axhline(-context.lane_half_width_m, color="#6b7280", linestyle="--")
    for obstacle in context.obstacles:
        axis.plot(obstacle.positions[:, 0], obstacle.positions[:, 1], color="#d9485f", linewidth=2)
        circle = plt.Circle(obstacle.positions[0], obstacle.radius_m, color="#d9485f", alpha=0.25)
        axis.add_patch(circle)

    selected_id = row["selected"]["candidate"]["candidate_id"]
    candidates = [*row.get("fast_candidates", []), *row.get("slow_candidates", [])]
    for item in candidates:
        candidate = item["candidate"]
        poses = np.asarray(candidate["trajectory"]["poses"])
        chosen = candidate["candidate_id"] == selected_id
        axis.plot(
            poses[:, 0],
            poses[:, 1],
            linewidth=3 if chosen else 1.2,
            alpha=1.0 if chosen else 0.45,
            label=f"{candidate['candidate_id']} risk={item['risk']['total']:.2f}",
        )
    axis.scatter([0], [0], marker="s", color="#222222", label="ego")
    axis.set_title(f"{context.scene_id} | selected={selected_id}")
    axis.set_xlabel("forward x (m)")
    axis.set_ylabel("left y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, loc="best")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160)
    plt.close(figure)
