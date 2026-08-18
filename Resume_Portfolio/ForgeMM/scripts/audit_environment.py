"""Capture a reproducible Stage 0 environment and dataset identity report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

MS_SWIFT_CANDIDATE = {
    "distribution": "ms-swift",
    "version": "4.2.2",
    "tag": "v4.2.2",
    "commit_short": "f279713",
    "release_url": "https://github.com/modelscope/ms-swift/releases/tag/v4.2.2",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _gpu_inventory() -> list[dict[str, str]]:
    command = (
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode:
        return []
    inventory: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            inventory.append({"name": parts[0], "memory_mib": parts[1], "driver_version": parts[2]})
    return inventory


def _verify_chartqa(root: Path) -> dict[str, Any]:
    manifest_path = root / "datasets" / "ChartQA" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    for relative, expected in manifest["key_file_sha256"].items():
        path = root / "datasets" / "ChartQA" / relative
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matched": actual == expected,
            }
        )
    return {
        "manifest": str(manifest_path.relative_to(root)),
        "checks": checks,
        "passed": all(item["matched"] for item in checks),
    }


def _verify_chartqapro(root: Path) -> dict[str, Any]:
    manifest_path = root / "datasets" / "ChartQAPro" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / manifest["target"]["relative_file"]
    actual = _sha256(path) if path.is_file() else None
    expected = str(manifest["file"]["sha256"])
    return {
        "manifest": str(manifest_path.relative_to(root)),
        "path": str(path.relative_to(root)),
        "exists": path.is_file(),
        "expected_bytes": int(manifest["file"]["bytes"]),
        "actual_bytes": path.stat().st_size if path.is_file() else None,
        "expected_sha256": expected,
        "actual_sha256": actual,
        "passed": path.is_file()
        and path.stat().st_size == int(manifest["file"]["bytes"])
        and actual == expected,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    packages = {
        name: _version(name)
        for name in (
            "numpy",
            "pillow",
            "pyarrow",
            "pyyaml",
            "pytest",
            "ruff",
            "mypy",
            "torch",
            "transformers",
            "ms-swift",
            "vllm",
            "bitsandbytes",
        )
    }
    chartqa = _verify_chartqa(project_root)
    chartqapro = _verify_chartqapro(project_root)
    dataset_payload_present = (
        any(item["exists"] for item in chartqa["checks"]) or chartqapro["exists"]
    )
    dataset_identity_status = (
        "verified"
        if chartqa["passed"] and chartqapro["passed"]
        else "failed"
        if dataset_payload_present
        else "unavailable"
    )
    return {
        "schema_version": "forgemm-environment-audit-v1",
        "project_root": str(project_root),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "gpu": _gpu_inventory(),
        "packages": packages,
        "ms_swift_candidate": MS_SWIFT_CANDIDATE,
        "datasets": {"chartqa": chartqa, "chartqapro": chartqapro},
        "gates": {
            "python_supported": (3, 10) <= sys.version_info[:2] < (3, 13),
            "cpu_dependencies_present": all(
                packages[name] is not None for name in ("numpy", "pillow", "pyarrow", "pyyaml")
            ),
            "dataset_identity_passed": chartqa["passed"] and chartqapro["passed"],
            "dataset_payload_present": dataset_payload_present,
            "dataset_identity_status": dataset_identity_status,
            "swift_smoke_ready": packages["ms-swift"] is not None and packages["torch"] is not None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-datasets",
        action="store_true",
        help="fail when the ignored dataset payload is unavailable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    report = build_report(root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    gates = report["gates"]
    required = gates["python_supported"] and gates["cpu_dependencies_present"]
    if args.require_datasets:
        required = required and gates["dataset_identity_passed"]
    elif gates["dataset_identity_status"] == "failed":
        required = False
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
