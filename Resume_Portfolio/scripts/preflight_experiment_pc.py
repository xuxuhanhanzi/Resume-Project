#!/usr/bin/env python3
"""Verify the supported RTX 5090 experiment-PC baseline and write a receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = (
        args.output
        or root
        / "artifacts"
        / "experiment-suite"
        / os.environ.get("EXPERIMENT_RUN_ID", f"{args.profile}-latest")
        / "host-preflight.json"
    )
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str, remedy: str = "") -> None:
        checks.append(
            {"name": name, "passed": passed, "detail": detail, "remedy": remedy}
        )

    linux = sys.platform.startswith("linux")
    record(
        "linux", linux, sys.platform, "Run inside Ubuntu on WSL2, not from PowerShell."
    )
    if linux:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
        record(
            "wsl2",
            "microsoft" in version.lower(),
            version.strip(),
            "Use WSL2 Ubuntu with GPU integration enabled.",
        )

    for executable in ("bash", "git", "curl", "python3"):
        path = shutil.which(executable)
        record(
            executable,
            bool(path),
            path or "missing",
            f"Install {executable} in Ubuntu.",
        )

    mem_gib = 0.0
    if Path("/proc/meminfo").exists():
        first = Path("/proc/meminfo").read_text().splitlines()[0].split()[1]
        mem_gib = int(first) / 1024 / 1024
    record(
        "ram",
        mem_gib >= 58,
        f"{mem_gib:.1f} GiB visible",
        "Expose at least 60 GiB to WSL2 in .wslconfig.",
    )

    free_gib = shutil.disk_usage(root).free / 1024**3
    required = 300 if args.profile == "full" else 40
    record(
        "disk",
        free_gib >= required,
        f"{free_gib:.1f} GiB free; need {required} GiB",
        "Free space on the Linux filesystem.",
    )

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        code, rows = command_output(
            [
                nvidia,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
        first_row = rows.splitlines()[0] if rows else ""
        parts = [part.strip() for part in first_row.split(",")]
        try:
            vram_mib = int(parts[1])
        except (IndexError, ValueError):
            vram_mib = 0
        record(
            "nvidia-driver",
            code == 0,
            rows or "nvidia-smi failed",
            "Install/update the Windows NVIDIA driver and restart WSL.",
        )
        record(
            "gpu-vram",
            vram_mib >= 23500,
            f"{vram_mib} MiB",
            "A 24 GB GPU is required for the registered full profile.",
        )
    else:
        record(
            "nvidia-driver",
            False,
            "nvidia-smi missing",
            "Enable NVIDIA CUDA support in WSL2.",
        )

    docker = shutil.which("docker")
    record(
        "docker-cli",
        bool(docker),
        docker or "missing",
        "Install Docker Desktop and enable WSL integration.",
    )
    if docker:
        code, detail = command_output(
            [docker, "info", "--format", "{{.ServerVersion}}"]
        )
        record(
            "docker-daemon",
            code == 0,
            detail,
            "Start Docker Desktop and enable this WSL distribution.",
        )

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": args.profile,
        "platform": platform.platform(),
        "python": sys.version,
        "checks": checks,
        "ready": all(bool(item["passed"]) for item in checks),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for item in checks:
        print(
            f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}"
        )
        if not item["passed"] and item["remedy"]:
            print(f"       remedy: {item['remedy']}")
    print(f"receipt: {output}")
    return 0 if receipt["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
