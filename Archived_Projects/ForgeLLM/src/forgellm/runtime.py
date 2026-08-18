"""Run initialization and provenance capture."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from forgellm import __version__
from forgellm.config import RunConfig
from forgellm.structured_logging import JsonValue, write_jsonl_event


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Locations produced when a run is initialized."""

    run_id: str
    directory: Path


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_command(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        timeout=10,
    )


def git_snapshot(repo_root: Path) -> dict[str, JsonValue]:
    """Capture a minimal Git snapshot without storing source or diff contents."""
    commit_result = _git_command(repo_root, "rev-parse", "HEAD")
    if commit_result.returncode != 0:
        return {"available": False, "commit": None, "dirty": None, "diff_sha256": None}

    status_result = _git_command(repo_root, "status", "--porcelain=v1")
    diff_result = _git_command(repo_root, "diff", "--binary", "HEAD")
    return {
        "available": True,
        "commit": commit_result.stdout.decode("ascii").strip(),
        "dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
        "diff_sha256": (
            hashlib.sha256(diff_result.stdout).hexdigest() if diff_result.returncode == 0 else None
        ),
    }


def environment_snapshot() -> dict[str, JsonValue]:
    """Capture a minimal non-sensitive runtime environment."""
    return {
        "forgellm": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def initialize_run(
    config: RunConfig,
    artifacts_root: Path,
    repo_root: Path,
    *,
    now: datetime | None = None,
) -> RunArtifacts:
    """Create an immutable run directory with resolved provenance metadata."""
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("run timestamps must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)

    config_hash = config.fingerprint()
    run_id = f"{timestamp:%Y%m%dT%H%M%SZ}_{config.stage}_{config.name}_{config_hash[:8]}"
    run_directory = artifacts_root / config.stage / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    created_at = timestamp.isoformat().replace("+00:00", "Z")
    _write_json(
        run_directory / "config.resolved.json",
        {**config.as_dict(), "sha256": config_hash},
    )
    _write_json(run_directory / "environment.json", environment_snapshot())
    _write_json(run_directory / "git.json", git_snapshot(repo_root))
    _write_json(
        run_directory / "run.json",
        {
            "created_at": created_at,
            "run_id": run_id,
            "stage": config.stage,
            "status": "initialized",
        },
    )
    write_jsonl_event(
        run_directory / "events.jsonl",
        "run_initialized",
        {"config_sha256": config_hash, "run_id": run_id, "stage": config.stage},
        now=timestamp,
    )
    return RunArtifacts(run_id=run_id, directory=run_directory)
