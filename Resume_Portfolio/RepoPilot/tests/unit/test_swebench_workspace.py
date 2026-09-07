from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from repopilot.evaluation.swebench_workspace import reset_owned_workspace


def _git(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _workspace_with_receipt(tmp_path: Path) -> tuple[Path, Path, str, str, str]:
    reuse_root = tmp_path / "r2_scratch"
    workspace = reuse_root / "seed_7" / "owner__repo-1"
    workspace.mkdir(parents=True)
    _git(workspace, "init")
    _git(workspace, "config", "user.email", "tests@example.invalid")
    _git(workspace, "config", "user.name", "RepoPilot tests")
    module = workspace / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    _git(workspace, "add", "module.py")
    _git(workspace, "commit", "-m", "public base")
    base_commit = _git(workspace, "rev-parse", "HEAD")
    task_id = "owner__repo-1"
    manifest_sha256 = "a" * 64
    receipt = {
        "kind": "r2_swebench_workspace_receipt",
        "task_id": task_id,
        "base_commit": base_commit,
        "manifest_sha256": manifest_sha256,
        "workspace": str(workspace.resolve()),
    }
    (workspace.parent / f"{task_id}.workspace_receipt.json").write_text(
        json.dumps(receipt) + "\n", encoding="utf-8", newline="\n"
    )
    return reuse_root, workspace, task_id, base_commit, manifest_sha256


def test_reset_owned_workspace_restores_receipt_bound_public_base(tmp_path: Path) -> None:
    reuse_root, workspace, task_id, base_commit, manifest_sha256 = _workspace_with_receipt(tmp_path)
    (workspace / "module.py").write_text("VALUE = 2\n", encoding="utf-8", newline="\n")

    event = reset_owned_workspace(
        workspace=workspace,
        reuse_root=reuse_root,
        task_id=task_id,
        base_commit=base_commit,
        manifest_sha256=manifest_sha256,
    )

    assert (workspace / "module.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _git(workspace, "status", "--short") == ""
    assert event["head"] == base_commit


def test_reset_owned_workspace_refuses_an_external_workspace(tmp_path: Path) -> None:
    reuse_root, workspace, task_id, base_commit, manifest_sha256 = _workspace_with_receipt(tmp_path)
    other_root = tmp_path / "different_root"
    other_root.mkdir()

    with pytest.raises(ValueError, match="outside the explicitly authorized"):
        reset_owned_workspace(
            workspace=workspace,
            reuse_root=other_root,
            task_id=task_id,
            base_commit=base_commit,
            manifest_sha256=manifest_sha256,
        )
