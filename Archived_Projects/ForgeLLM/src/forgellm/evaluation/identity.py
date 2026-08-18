"""File and directory identities used by Stage 6 manifests."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    """Hash one regular file without reading it wholly into memory."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative names and contents of every file in one artifact directory."""
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"artifact directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def code_revision(project_root: Path) -> str:
    """Return commit/state plus a hash of every Stage 6 executable input."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip() if completed.returncode == 0 else "git-unavailable"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    state = "dirty" if dirty.stdout.strip() or dirty.returncode != 0 else "clean"
    source_paths = sorted((project_root / "src/forgellm/evaluation").glob("*.py"))
    source_paths.extend(
        sorted(path for path in (project_root / "scripts").glob("*stage6*.py") if path.is_file())
    )
    source_paths.append(project_root / "scripts/prepare_stage6_evaluation.py")
    source_paths.append(project_root / "configs/evaluation/stage6_final.toml")
    digest = hashlib.sha256()
    for path in sorted(set(source_paths)):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return f"{commit}+{state}+stage6-source-sha256:{digest.hexdigest()}"
