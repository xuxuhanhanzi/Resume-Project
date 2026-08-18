"""Atomic, schema-checked checkpoints including all reproducibility state."""

from __future__ import annotations

import os
import pickle
import random
import uuid
from pathlib import Path
from typing import cast

import torch

_CHECKPOINT_SCHEMA = "forgellm-stage3-checkpoint-v1"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete, corrupt, or incompatible."""


def capture_rng_state() -> dict[str, object]:
    """Capture every RNG currently used by the dependency-free training loop."""
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    """Restore Python, CPU torch, and optional CUDA generator states."""
    if set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise CheckpointError("checkpoint RNG state has unexpected fields")
    python_state = state["python"]
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    if not isinstance(python_state, tuple) or not isinstance(torch_cpu, torch.Tensor):
        raise CheckpointError("checkpoint RNG state has invalid values")
    random.setstate(python_state)
    torch.set_rng_state(torch_cpu.cpu())
    if torch_cuda is not None:
        if (
            not torch.cuda.is_available()
            or not isinstance(torch_cuda, list)
            or not all(isinstance(item, torch.Tensor) for item in torch_cuda)
        ):
            raise CheckpointError("checkpoint requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])


def save_checkpoint_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write one complete checkpoint and atomically replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": _CHECKPOINT_SCHEMA, "payload": payload}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(envelope, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as error:
        if temporary.is_file():
            temporary.unlink()
        raise CheckpointError(f"could not atomically save checkpoint {path}: {error}") from error


def load_checkpoint(path: Path, *, map_location: torch.device | str) -> dict[str, object]:
    """Load and validate the checkpoint envelope before state mutation."""
    try:
        raw: object = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise CheckpointError(f"could not load checkpoint {path}: {error}") from error
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint envelope must be a mapping")
    envelope = cast(dict[str, object], raw)
    if set(envelope) != {"schema_version", "payload"}:
        raise CheckpointError("checkpoint envelope has unexpected fields")
    if envelope["schema_version"] != _CHECKPOINT_SCHEMA:
        raise CheckpointError("unsupported checkpoint schema")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a mapping")
    return cast(dict[str, object], payload)
