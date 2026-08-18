"""Atomic checkpoint envelope tests."""

from pathlib import Path

import pytest
import torch

from forgellm.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint_atomic


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.pt"
    payload: dict[str, object] = {"step": 3, "tensor": torch.arange(4)}

    save_checkpoint_atomic(path, payload)
    restored = load_checkpoint(path, map_location="cpu")

    assert restored["step"] == 3
    assert torch.equal(restored["tensor"], torch.arange(4))  # type: ignore[arg-type]
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_checkpoint_is_rejected_before_state_mutation(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(CheckpointError, match="could not load"):
        load_checkpoint(path, map_location="cpu")
