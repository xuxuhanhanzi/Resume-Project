"""Verify docs/evidence/artifact_index.json against the live artifacts.

For every registered run it checks that the results file exists, that its
SHA-256 still matches the recorded digest, and that the primary metric can be
recomputed from the file. Fails (non-zero exit) on any mismatch, so it is safe
to drop into CI.

Usage:
    python scripts/verify_artifact_index.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "evidence" / "artifact_index.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(run: dict[str, Any]) -> tuple[int | None, int | None]:
    rp = ROOT / run["results_path"]
    if not rp.exists():
        return None, None
    top = json.loads(rp.read_text(encoding="utf-8"))
    if "resolved" in top:
        tasks = top.get("tasks")
        tot = len(tasks) if isinstance(tasks, list) else tasks
        return int(top["resolved"]), int(tot) if tot is not None else None
    recs = top.get("records", [])
    if not recs:
        return 0, 0
    ok = sum(
        1
        for r in recs
        if r.get("task_success") is True or float(r.get("primary_metric", 0.0)) >= 1.0
    )
    return ok, len(recs)


def main() -> int:
    if not INDEX.exists():
        print(f"MISSING index: {INDEX}")
        return 1
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    failures = 0
    print(f"model: {index['model']['name']} digest={index['model']['digest'][:12]}...")
    print(f"{'run':<44} {'score':<8} {'sha':<14} {'verify'}")
    print("-" * 78)
    for run in index["runs"]:
        rp = ROOT / run["results_path"]
        ok = True
        why = []
        if not rp.exists():
            ok = False
            why.append("results missing")
        else:
            live_sha = sha256(rp)
            if live_sha != run["results_sha256"]:
                ok = False
                why.append("sha mismatch")
            live_ok, live_tot = recompute(run)
            if live_ok != run["score"] or live_tot != run["total"]:
                ok = False
                why.append(f"metric {live_ok}/{live_tot} != {run['score']}/{run['total']}")
        if run.get("manifest"):
            mp = ROOT / run["manifest"]
            if mp.exists():
                if sha256(mp) != run["manifest_sha256"]:
                    ok = False
                    why.append("manifest sha mismatch")
            else:
                ok = False
                why.append("manifest missing")
        flag = "OK " if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"{run['id']:<44} {str(run['score']) + '/' + str(run['total']):<8} "
            f"{str(run['results_sha256'])[:12]:<14} {flag} {'; '.join(why)}"
        )
    print("-" * 78)
    if failures:
        print(f"VERIFY FAILED: {failures} run(s) mismatched")
        return 1
    print(f"VERIFY OK: {len(index['runs'])} runs consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
