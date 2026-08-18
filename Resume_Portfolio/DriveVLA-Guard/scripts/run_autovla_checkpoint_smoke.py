"""Run one evidence-producing AutoVLA checkpoint inference on 12 camera frames.

The script deliberately bypasses NAVSIM scoring.  It proves that the official
checkpoint can be loaded and can produce a finite trajectory from the same
three-camera/four-history-frame input contract used by AutoVLA.  A NAVSIM
PDMS claim still requires the complete benchmark data and metric cache.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autovla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        required=True,
        help="Repeat 12 times: front[0:4], front-left[0:4], front-right[0:4].",
    )
    parser.add_argument("--driving-command", default="go straight")
    parser.add_argument("--velocity-mps", type=float, default=5.0)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _absolute_existing(path: Path, *, kind: str) -> Path:
    resolved = path.expanduser().resolve()
    predicate = resolved.is_dir if kind == "directory" else resolved.is_file
    if not predicate():
        raise FileNotFoundError(f"expected {kind}: {resolved}")
    return resolved


def main() -> int:
    args = parse_args()
    if len(args.image) != 12:
        raise ValueError(f"--image must be supplied exactly 12 times, got {len(args.image)}")

    autovla_root = _absolute_existing(args.autovla_root, kind="directory")
    checkpoint = _absolute_existing(args.checkpoint, kind="file")
    qwen_model = _absolute_existing(args.qwen_model, kind="directory")
    config_path = _absolute_existing(args.config, kind="file")
    images = [_absolute_existing(path, kind="file") for path in args.image]

    # Keep third-party imports local so the repository's CPU quality gate does
    # not require the heavyweight AutoVLA environment.
    sys.path.insert(0, str(autovla_root))
    import numpy as np
    import torch
    import yaml
    from models.autovla import AutoVLA

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official AutoVLA checkpoint smoke")

    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["pretrained_model_path"] = str(qwen_model)
    config["model"]["codebook_cache_path"] = str(autovla_root / "codebook_cache" / "agent_vocab.pkl")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model = AutoVLA(config, device="cuda")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw_state = checkpoint_payload["state_dict"]
    state = {key.replace("autovla.", ""): value for key, value in raw_state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    model.eval()

    features = {
        "images": {
            "front_camera": [str(path) for path in images[0:4]],
            "front_left_camera": [str(path) for path in images[4:8]],
            "front_right_camera": [str(path) for path in images[8:12]],
        },
        "sensor_data_path": "",
        "vehicle_velocity": [args.velocity_mps, 0.0],
        "vehicle_acceleration": [0.0, 0.0],
        "driving_command": args.driving_command,
    }

    inference_started = time.perf_counter()
    with torch.inference_mode():
        trajectory, decoded_text = model.predict(features)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    trajectory_array = np.asarray(trajectory, dtype=np.float64)
    finite = bool(np.isfinite(trajectory_array).all())
    shape = list(trajectory_array.shape)
    success = bool(finite and len(shape) == 2 and shape[1:] == [3] and shape[0] > 0)
    result = {
        "schema_version": "drivevla-autovla-checkpoint-smoke-v1",
        "claim_boundary": (
            "official checkpoint inference on supplied camera frames; no NAVSIM PDMS, "
            "closed-loop, or real-road performance claim"
        ),
        "success": success,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "identity": {
            "autovla_root": str(autovla_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "qwen_model": str(qwen_model),
            "config": str(config_path),
            "images": [str(path) for path in images],
        },
        "load": {
            "state_tensors": len(state),
            "missing_keys_count": len(incompatible.missing_keys),
            "missing_keys_head": incompatible.missing_keys[:20],
            "unexpected_keys_count": len(incompatible.unexpected_keys),
            "unexpected_keys_head": incompatible.unexpected_keys[:20],
        },
        "output": {
            "trajectory_shape": shape,
            "trajectory_finite": finite,
            "trajectory": trajectory_array.tolist(),
            "decoded_text": decoded_text,
        },
        "runtime": {
            "inference_seconds": inference_seconds,
            "total_seconds": time.perf_counter() - started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
