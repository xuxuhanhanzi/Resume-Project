"""Receipt-bound lifecycle operations for reusable R2 SWE-bench workspaces.

Only an explicitly supplied scratch root may be reused.  The reset operation
is deliberately narrower than a general cleanup: it validates the immutable
workspace receipt, runs ``git reset --hard`` to the public base commit, and
rejects any remaining working-tree content rather than deleting it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def reset_owned_workspace(
    *,
    workspace: Path,
    reuse_root: Path,
    task_id: str,
    base_commit: str,
    manifest_sha256: str,
) -> dict[str, str]:
    """Reset one receipt-bound workspace and prove it returned to its public base.

    This never invokes ``git clean`` and never removes files.  Consequently,
    an unexpected untracked file is evidence of a lifecycle violation, not a
    reason to broaden the operation's scope.
    """

    root = reuse_root.resolve(strict=True)
    resolved_workspace = workspace.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"workspace reuse root is not a directory: {root}")
    if resolved_workspace == root or root not in resolved_workspace.parents:
        raise ValueError("refusing workspace reset outside the explicitly authorized reuse root")

    receipt_path = resolved_workspace.parent / f"{task_id}.workspace_receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"missing workspace receipt required for reset: {receipt_path}")
    try:
        receipt: Any = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"workspace receipt is not valid JSON: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ValueError("workspace receipt must be a JSON object")
    expected = {
        "kind": "r2_swebench_workspace_receipt",
        "task_id": task_id,
        "base_commit": base_commit,
        "manifest_sha256": manifest_sha256,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError(f"workspace receipt mismatch for: {', '.join(mismatches)}")
    receipt_workspace = receipt.get("workspace")
    if not isinstance(receipt_workspace, str):
        raise ValueError("workspace receipt does not contain a workspace path")
    if Path(receipt_workspace).resolve(strict=True) != resolved_workspace:
        raise ValueError("workspace receipt path does not match reset target")

    _git(resolved_workspace, "reset", "--hard", base_commit)
    head = _git(resolved_workspace, "rev-parse", "HEAD")
    if head != base_commit:
        raise RuntimeError("workspace reset did not produce the recorded public base commit")
    if _git(resolved_workspace, "status", "--short"):
        raise RuntimeError("workspace is not clean after receipt-bound reset")
    return {
        "task_id": task_id,
        "workspace": str(resolved_workspace),
        "base_commit": base_commit,
        "head": head,
        "workspace_receipt": str(receipt_path),
        "operation": "git reset --hard <public-base-commit>",
    }


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
