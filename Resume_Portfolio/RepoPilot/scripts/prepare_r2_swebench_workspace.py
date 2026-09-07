"""Materialize one clean R2 workspace from a public SWE-bench instance image.

The image supplies the dependency-compatible ``/testbed`` tree.  This script
never reads evaluator-only dataset fields.  It then checks out the public
``base_commit`` and normalizes only Git's local Windows file-mode/line-ending
settings, producing a clean, dedicated agent workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from repopilot.evaluation.coderag import load_manifest
from repopilot.evaluation.swebench_verified import load_public_records


def _run(command: list[str], *, capture: bool = True) -> str:
    completed = subprocess.run(  # noqa: S603
        command,
        check=True,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _git(workspace: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(workspace), *arguments])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation", "final_holdout"))
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    arguments = parser.parse_args()

    records = {
        record["instance_id"]: record for record in load_public_records(arguments.public_jsonl)
    }
    record = records.get(arguments.task_id)
    if record is None:
        raise ValueError("task ID is not present in the public SWE-bench JSONL")
    manifest = load_manifest(arguments.manifest)
    if (
        arguments.split is not None
        and arguments.task_id not in manifest.assignments[arguments.split]
    ):
        raise ValueError(f"task ID is not assigned to manifest split {arguments.split!r}")
    workspace = arguments.workspace.resolve()
    receipt_path = workspace.parent / f"{arguments.task_id}.workspace_receipt.json"
    if workspace.exists() or receipt_path.exists():
        raise ValueError("workspace or its receipt already exists; refusing to overwrite evidence")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(arguments.task_id.encode("utf-8")).hexdigest()[:12]
    container_name = f"r2-workspace-export-{suffix}"
    existing = _run(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if container_name in existing.splitlines():
        raise ValueError(f"owned temporary container name already exists: {container_name}")

    created = False
    try:
        _run(["docker", "create", "--name", container_name, record["image"]])
        created = True
        _run(["docker", "cp", f"{container_name}:/testbed/.", str(workspace)])
    finally:
        if created:
            subprocess.run(  # noqa: S603
                ["docker", "rm", container_name],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

    if not (workspace / ".git").is_dir():
        raise ValueError("official image did not contain a Git testbed")
    # Docker-to-Windows export loses POSIX execute bits.  Configure Git before
    # checkout so that the portability-only mode difference cannot block the
    # explicit public base-commit materialization.
    _git(workspace, "config", "core.autocrlf", "false")
    _git(workspace, "config", "core.filemode", "false")
    _git(workspace, "checkout", "--detach", record["base_commit"])
    actual_commit = _git(workspace, "rev-parse", "HEAD")
    if actual_commit != record["base_commit"]:
        raise ValueError("exported testbed does not resolve to the public base commit")
    if _git(workspace, "status", "--short"):
        raise ValueError("exported R2 workspace is not clean after base-commit checkout")
    image_digest = _run(
        ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", record["image"]]
    )
    receipt = {
        "schema_version": 1,
        "kind": "r2_swebench_workspace_receipt",
        "task_id": arguments.task_id,
        "manifest_sha256": manifest.sha256,
        "split": arguments.split,
        "repo": record["repo"],
        "base_commit": actual_commit,
        "image": record["image"],
        "image_digest": image_digest,
        "workspace": str(workspace),
        "notes": [
            "Source was exported from the official public instance image's /testbed.",
            "Only public JSONL fields were read; patches, test metadata and evaluator scripts "
            "were excluded.",
            "Git core.autocrlf=false and core.filemode=false are workspace-local Windows "
            "portability settings.",
        ],
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
