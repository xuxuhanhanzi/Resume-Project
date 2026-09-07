"""Record the local WSL runtime identity before running a memory-sensitive job."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=6500)
    args = parser.parse_args()
    result = _runtime_identity(args.minimum_free_mib)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "pass" else 2


def _runtime_identity(minimum_free_mib: int) -> dict[str, Any]:
    packages = {}
    errors = []
    for package in ("torch", "ms-swift", "bitsandbytes", "transformers", "qwen-vl-utils"):
        try:
            packages[package] = version(package)
        except Exception:  # PackageNotFoundError and invalid metadata both fail closed.
            errors.append(f"missing_or_invalid:{package}")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "free_mib": free_bytes // (1024 * 1024),
                "total_mib": total_bytes // (1024 * 1024),
                "torch_cuda": torch.version.cuda,
            }
            if gpu["free_mib"] < minimum_free_mib:
                errors.append("insufficient_free_vram")
        else:
            gpu = {"name": None, "free_mib": 0, "total_mib": 0, "torch_cuda": None}
            errors.append("cuda_unavailable")
    except Exception as exc:
        gpu = {"name": None, "free_mib": 0, "total_mib": 0, "torch_cuda": None}
        errors.append(f"torch_probe_failed:{type(exc).__name__}")
    return {
        "schema_version": "forgemm-local-runtime-audit-v1",
        "platform": platform.platform(),
        "python": sys.version,
        "packages": packages,
        "gpu": gpu,
        "nvidia_smi": _nvidia_smi(),
        "minimum_free_mib": minimum_free_mib,
        "decision": "pass" if not errors else "stop",
        "errors": errors,
    }


def _nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


if __name__ == "__main__":
    raise SystemExit(main())
