import pickle

import numpy as np
import pytest

from drivevla_guard.codec import ActionCodebookCodec


def make_codebook() -> np.ndarray:
    codebook = np.zeros((2, 2, 4, 2), dtype=np.float32)
    codebook[0, -1] = np.array([[1.0, 0.5], [1.0, -0.5], [0.0, -0.5], [0.0, 0.5]])
    codebook[1, -1] = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return codebook


def test_codec_rolls_out_relative_actions() -> None:
    codec = ActionCodebookCodec(make_codebook(), action_start_id=100)
    trajectory = codec.decode([100, 100])
    assert trajectory.poses.shape == (2, 3)
    assert np.allclose(trajectory.poses[:, 0], [0.5, 1.0])


def test_codec_rejects_out_of_range_tokens() -> None:
    codec = ActionCodebookCodec(make_codebook(), action_start_id=100)
    with pytest.raises(ValueError, match="outside codebook"):
        codec.decode([99])


def test_codec_loads_upstream_pickle_layout(tmp_path) -> None:
    path = tmp_path / "codebook.pkl"
    with path.open("wb") as handle:
        pickle.dump({"token_all": {"veh": make_codebook()}}, handle)
    codec = ActionCodebookCodec.from_pickle(path, action_start_id=100)
    assert codec.vocabulary_size == 2
