"""Prepare one receipt-bound R2 SWE-bench split under a dedicated scratch root.

The script is intentionally resumable without overwriting evidence: an
already-created workspace is accepted only when its adjacent immutable receipt,
Git HEAD, and clean status match the frozen public record.  Missing workspaces
are delegated to the single-workspace preparer, which exports the official
public instance image's ``/testbed`` tree.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from repopilot.evaluation.coderag import load_manifest
from repopilot.evaluation.swebench_verified import (
    load_public_records,
    public_dataset_sha256,
)
from repopilot.evidence.protocol import sha256_file


def _git(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(workspace), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _validate_existing_workspace(
    *,
    workspace: Path,
    task_id: str,
    base_commit: str,
    manifest_sha256: str,
) -> Path:
    """Accept a prior workspace only when it still proves the same public base."""

    receipt_path = workspace.parent / f"{task_id}.workspace_receipt.json"
    if not workspace.is_dir() or not receipt_path.is_file():
        raise ValueError("existing workspace requires its adjacent workspace receipt")
    try:
        receipt: Any = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"workspace receipt is not valid JSON: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ValueError(f"workspace receipt is not an object: {receipt_path}")
    expected = {
        "kind": "r2_swebench_workspace_receipt",
        "task_id": task_id,
        "base_commit": base_commit,
        "manifest_sha256": manifest_sha256,
        "workspace": str(workspace.resolve(strict=True)),
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError(f"existing workspace receipt mismatch: {', '.join(mismatches)}")
    if _git(workspace, "rev-parse", "HEAD") != base_commit:
        raise ValueError("existing workspace does not have its recorded public base commit")
    if _git(workspace, "status", "--short"):
        raise ValueError("existing workspace is not clean")
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("development", "validation", "final_holdout"), required=True
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    arguments = parser.parse_args()

    records = {
        record["instance_id"]: record for record in load_public_records(arguments.public_jsonl)
    }
    manifest = load_manifest(arguments.manifest)
    public_sha = public_dataset_sha256(list(records.values()))
    if public_sha not in " ".join(manifest.notes):
        raise ValueError("manifest does not bind the supplied public SWE-bench JSONL digest")
    workspace_root = arguments.workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    batch_receipt_path = workspace_root / "workspace_batch.receipt.json"
    if batch_receipt_path.exists():
        raise ValueError(f"refusing to overwrite batch workspace receipt: {batch_receipt_path}")

    prepared: list[dict[str, str]] = []
    preparer = Path(__file__).with_name("prepare_r2_swebench_workspace.py")
    for task_id in manifest.assignments[arguments.split]:
        record = records.get(task_id)
        if record is None:
            raise ValueError(f"manifest task is absent from public JSONL: {task_id}")
        workspace = workspace_root / task_id
        receipt_path = workspace.parent / f"{task_id}.workspace_receipt.json"
        if workspace.exists() or receipt_path.exists():
            receipt_path = _validate_existing_workspace(
                workspace=workspace,
                task_id=task_id,
                base_commit=record["base_commit"],
                manifest_sha256=manifest.sha256,
            )
            disposition = "validated_existing"
        else:
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(preparer),
                    "--public-jsonl",
                    str(arguments.public_jsonl),
                    "--manifest",
                    str(arguments.manifest),
                    "--split",
                    arguments.split,
                    "--task-id",
                    task_id,
                    "--workspace",
                    str(workspace),
                ],
                check=True,
            )
            receipt_path = _validate_existing_workspace(
                workspace=workspace,
                task_id=task_id,
                base_commit=record["base_commit"],
                manifest_sha256=manifest.sha256,
            )
            disposition = "prepared"
        prepared.append(
            {
                "task_id": task_id,
                "workspace": str(workspace.resolve(strict=True)),
                "workspace_receipt": str(receipt_path),
                "workspace_receipt_sha256": sha256_file(receipt_path),
                "disposition": disposition,
            }
        )

    receipt = {
        "schema_version": 1,
        "kind": "r2_swebench_workspace_batch_receipt",
        "manifest_sha256": manifest.sha256,
        "public_dataset_sha256": public_sha,
        "split": arguments.split,
        "workspace_root": str(workspace_root),
        "prepared": prepared,
        "notes": [
            "Each workspace was exported from a public instance image or revalidated against "
            "its immutable receipt.",
            "This preparer does not delete or overwrite any workspace or receipt.",
        ],
    }
    batch_receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
