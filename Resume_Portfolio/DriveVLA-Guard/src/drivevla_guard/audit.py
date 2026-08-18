from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_autovla(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    required = [
        "LICENSE",
        "README.md",
        "models/autovla.py",
        "models/action_tokenizer.py",
        "codebook_cache/agent_vocab.pkl",
        "navsim/navsim/agents/autovla_agent.py",
    ]
    files = {name: (root / name).is_file() for name in required}
    commit = None
    source_kind = "git_checkout"
    source_archive = None
    with suppress(subprocess.CalledProcessError, FileNotFoundError):
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if commit is None:
        manifest_path = root / ".autovla-source.json"
        if manifest_path.is_file():
            with suppress(json.JSONDecodeError, KeyError, OSError, ValueError):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                archive = (root / str(manifest["archive"])).resolve()
                archive_sha256 = sha256_file(archive)
                archive_commit = str(manifest["commit"])
                expected_prefix = f"AutoVLA-{archive_commit}/"
                with zipfile.ZipFile(archive) as bundle:
                    roots_match = all(name.startswith(expected_prefix) for name in bundle.namelist())
                manifest_valid = (
                    manifest.get("schema_version") == "autovla-codeload-v1"
                    and manifest.get("upstream") == "https://github.com/ucla-mobility/AutoVLA"
                    and re.fullmatch(r"[0-9a-f]{40}", archive_commit) is not None
                    and archive_sha256 == manifest["archive_sha256"]
                    and roots_match
                )
                if manifest_valid:
                    commit = archive_commit
                    source_kind = "github_codeload_archive"
                    source_archive = {
                        "path": str(archive),
                        "sha256": archive_sha256,
                        "manifest": str(manifest_path),
                    }
    codebook = root / "codebook_cache/agent_vocab.pkl"
    return {
        "upstream": "https://github.com/ucla-mobility/AutoVLA",
        "root": str(root),
        "commit": commit,
        "source_kind": source_kind,
        "source_archive": source_archive,
        "required_files": files,
        "codebook_sha256": sha256_file(codebook) if codebook.is_file() else None,
        "license_class": "academic_noncommercial_nontransferable",
        "redistribution_allowed": False,
        "ready": bool(commit and all(files.values())),
    }


def write_audit(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _gpu_snapshot() -> dict[str, Any]:
    gpu: dict[str, Any] = {"available": False, "devices": [], "raw": None}
    if shutil.which("nvidia-smi") is None:
        return gpu
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return gpu
    devices = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        memory_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", fields[1])
        devices.append(
            {
                "name": fields[0],
                "memory_total_mib": float(memory_match.group(1)) if memory_match else None,
                "driver_version": fields[2],
            }
        )
    return {"available": bool(devices), "devices": devices, "raw": raw}


def _module_snapshot(names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        available = importlib.util.find_spec(name) is not None
        version = None
        if available:
            with suppress(importlib.metadata.PackageNotFoundError):
                version = importlib.metadata.version(name.replace("_", "-"))
        result[name] = {"available": available, "version": version}
    return result


def official_preflight(
    paths: dict[str, str | None],
    *,
    min_vram_gb: float = 24.0,
    expected_upstream_commit: str | None = None,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_kinds = {
        "autovla_root": "directory",
        "checkpoint": "file",
        "qwen_model": "directory",
        "model_config": "file",
        "json_data": "file_or_directory",
        "sensor_data": "directory",
        "metric_cache": "directory",
    }
    for name, value in paths.items():
        resolved = Path(value).resolve() if value else None
        exists = bool(resolved and resolved.exists())
        kind = expected_kinds.get(name, "file_or_directory")
        kind_ok = bool(
            exists
            and (
                kind == "file_or_directory"
                or (kind == "file" and resolved and resolved.is_file())
                or (kind == "directory" and resolved and resolved.is_dir())
            )
        )
        checks[name] = {
            "path": str(resolved) if resolved else None,
            "expected_kind": kind,
            "exists": exists,
            "kind_ok": kind_ok,
        }
    modules = _module_snapshot(["torch", "transformers", "qwen_vl_utils", "hydra", "nuplan", "navsim"])
    gpu = _gpu_snapshot()
    max_vram_mib = max(
        (device["memory_total_mib"] or 0.0 for device in gpu["devices"]),
        default=0.0,
    )
    gpu["minimum_required_mib"] = min_vram_gb * 1024.0
    gpu["capacity_ok"] = max_vram_mib >= gpu["minimum_required_mib"]

    blockers = [f"invalid_path:{name}" for name, item in checks.items() if not item["kind_ok"]]
    blockers.extend(
        f"missing_module:{name}" for name, snapshot in modules.items() if not snapshot["available"]
    )
    if not gpu["available"]:
        blockers.append("gpu_unavailable")
    elif not gpu["capacity_ok"]:
        blockers.append(f"insufficient_vram:{max_vram_mib:.0f}MiB<{gpu['minimum_required_mib']:.0f}MiB")

    upstream = None
    root_check = checks.get("autovla_root")
    if root_check and root_check["kind_ok"]:
        upstream = audit_autovla(root_check["path"])
        if not upstream["ready"]:
            blockers.append("invalid_autovla_checkout")
        if expected_upstream_commit and upstream["commit"] != expected_upstream_commit:
            blockers.append(f"upstream_commit_mismatch:{upstream['commit']}!={expected_upstream_commit}")
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "paths": checks,
        "modules": modules,
        "gpu": gpu,
        "upstream": upstream,
        "expected_upstream_commit": expected_upstream_commit,
        "checkpoint_repository_size_gb": 16.3,
        "checkpoint_license_metadata": "unspecified_on_model_card",
        "ready": not blockers,
        "blockers": blockers,
    }
