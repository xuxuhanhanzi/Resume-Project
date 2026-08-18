from pathlib import Path

import numpy as np

from forgemm.trainers.dual_state import DualConfig, DualState


def test_dual_update_direction_clip_and_mask() -> None:
    state = DualState(DualConfig(tau_evidence=0.9, tau_operation=0.8, dual_lr=1.0, lambda_max=0.5))
    report = state.update(
        np.array([0.0, 0.2]),
        np.array([1.0, 1.0]),
        np.array([True, True]),
        np.array([True, True]),
    )
    assert state.lambda_evidence == 0.5
    assert state.lambda_operation == 0.0
    assert report["update_step"] == 1


def test_dual_state_round_trip_continues_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "dual_state.json"
    state = DualState(DualConfig(), lambda_evidence=0.2, lambda_operation=0.3, update_step=7)
    state.save(path)
    restored = DualState.load(path)
    restored.update(
        np.array([0.5]),
        np.array([0.5]),
        np.array([True]),
        np.array([True]),
    )
    assert restored.update_step == 8
    assert restored.lambda_evidence > 0.2
    assert restored.lambda_operation > 0.3
