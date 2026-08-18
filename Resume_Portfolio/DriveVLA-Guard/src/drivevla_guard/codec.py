from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .types import Trajectory


class ActionCodebookCodec:
    """Pure NumPy decoder compatible with AutoVLA's ``agent_vocab.pkl`` layout."""

    def __init__(self, codebook: np.ndarray, action_start_id: int, dt_s: float = 0.5):
        array = np.asarray(codebook, dtype=np.float64)
        if array.ndim != 4 or array.shape[-2:] != (4, 2):
            raise ValueError(f"codebook must have shape [N, S, 4, 2], got {array.shape}")
        self.codebook = array
        self.action_start_id = int(action_start_id)
        self.dt_s = float(dt_s)

    @classmethod
    def from_pickle(cls, path: str | Path, action_start_id: int, dt_s: float = 0.5):
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        try:
            codebook = payload["token_all"]["veh"]
        except (KeyError, TypeError) as exc:
            raise ValueError("unsupported codebook: expected payload['token_all']['veh']") from exc
        return cls(codebook, action_start_id=action_start_id, dt_s=dt_s)

    @property
    def vocabulary_size(self) -> int:
        return int(self.codebook.shape[0])

    def token_ids_to_indices(self, token_ids: list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
        indices = np.asarray(token_ids, dtype=np.int64) - self.action_start_id
        if indices.ndim != 1:
            raise ValueError("token ids must be one-dimensional")
        if len(indices) == 0:
            raise ValueError("no action tokens were provided")
        if (indices < 0).any() or (indices >= self.vocabulary_size).any():
            raise ValueError("action token id outside codebook range")
        return indices

    def decode(self, token_ids: list[int] | tuple[int, ...] | np.ndarray) -> Trajectory:
        indices = self.token_ids_to_indices(token_ids)
        action_tokens = self.codebook[indices]
        position = np.array([0.0, 0.0], dtype=np.float64)
        heading = 0.0
        poses: list[list[float]] = []
        for token in action_tokens:
            cosine, sine = np.cos(heading), np.sin(heading)
            rotation = np.array([[cosine, sine], [-sine, cosine]])
            global_contours = token.reshape(-1, 2) @ rotation + position
            global_contours = global_contours.reshape(token.shape)
            final_contour = global_contours[-1]
            position = final_contour.mean(axis=0)
            delta = final_contour[0] - final_contour[3]
            heading = float(np.arctan2(delta[1], delta[0]))
            poses.append([float(position[0]), float(position[1]), heading])
        return Trajectory(np.asarray(poses), dt_s=self.dt_s)
