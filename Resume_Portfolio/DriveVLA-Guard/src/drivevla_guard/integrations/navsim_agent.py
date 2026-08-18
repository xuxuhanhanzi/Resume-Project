from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from navsim.agents.autovla_agent import AutoVLAAgent as UpstreamAutoVLAAgent
from navsim.common.dataclasses import Scene
from navsim.common.dataclasses import Trajectory as NavsimTrajectory

from drivevla_guard.adapters.autovla import AutoVLACandidateBackend
from drivevla_guard.config import load_config
from drivevla_guard.pipeline import GuardPipeline
from drivevla_guard.types import DynamicObstacle, SceneContext


class DriveVLAGuardAgent(UpstreamAutoVLAAgent):
    """NAVSIM agent using only current-frame annotations for optional obstacle risk.

    Future ground-truth frames and metric cache are deliberately not exposed to the
    reranker. Current object velocity is propagated with a constant-velocity model.
    """

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling,
        checkpoint_path: str | None = None,
        sensor_data_path: str | None = None,
        codebook_cache_path: str | None = None,
        lora_conf: dict[str, Any] | None = None,
        config_path: str | None = None,
        guard_config_path: str = "configs/e4_guard.yaml",
        device: str = "cuda",
        skip_model_load: bool = False,
    ):
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=checkpoint_path,
            sensor_data_path=sensor_data_path,
            codebook_cache_path=codebook_cache_path,
            lora_conf=lora_conf or {"use_lora": False},
            config_path=config_path,
            device=device,
            skip_model_load=skip_model_load,
        )
        self.guard_config_path = str(Path(guard_config_path).resolve())
        self.guard_pipeline: GuardPipeline | None = None

    def initialize(self) -> None:
        super().initialize()
        guard_config = load_config(self.guard_config_path)
        backend = AutoVLACandidateBackend(self.autovla, guard_config.generation)
        self.guard_pipeline = GuardPipeline(backend, guard_config)

    def compute_trajectory(self, scene_data: dict[str, Any], scene: Scene):
        if self.guard_pipeline is None:
            raise RuntimeError("agent must be initialized before compute_trajectory")
        context = self._context_from_scene(scene)
        result = self.guard_pipeline.plan(scene_data, context)
        poses = result.selected.candidate.trajectory.poses
        poses = poses[: self._trajectory_sampling.num_poses]
        if len(poses) != self._trajectory_sampling.num_poses:
            raise ValueError("selected candidate length is incompatible with NAVSIM trajectory sampling")
        navsim_trajectory = NavsimTrajectory(poses.astype(np.float32), self._trajectory_sampling)
        reasoning = result.selected.candidate.raw_text or ""
        return navsim_trajectory, reasoning

    def _context_from_scene(self, scene: Scene) -> SceneContext:
        current_index = scene.scene_metadata.num_history_frames - 1
        frame = scene.frames[current_index]
        horizon = self._trajectory_sampling.num_poses
        dt_s = self._trajectory_sampling.interval_length
        times = np.arange(1, horizon + 1, dtype=np.float64)[:, None] * dt_s
        obstacles: list[DynamicObstacle] = []
        for index, (box, velocity) in enumerate(
            zip(frame.annotations.boxes, frame.annotations.velocity_3d, strict=True)
        ):
            position = np.asarray(box[:2], dtype=np.float64)
            velocity_xy = np.asarray(velocity[:2], dtype=np.float64)
            predicted = position[None, :] + times * velocity_xy[None, :]
            length, width = float(box[3]), float(box[4])
            radius = max(0.4, 0.5 * float(np.hypot(length, width)))
            token = frame.annotations.track_tokens[index] or f"object-{index}"
            obstacles.append(DynamicObstacle(str(token), predicted, radius_m=radius))
        return SceneContext(
            scene_id=scene.scene_metadata.scene_token,
            obstacles=tuple(obstacles),
            lane_half_width_m=None,
            source="navsim_current_annotations_constant_velocity",
            metadata={"map_name": scene.scene_metadata.map_name},
        )
