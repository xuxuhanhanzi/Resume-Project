from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import SyntheticBackend
from .config import load_config
from .evaluation import compare_results, summarize_rows
from .io import read_jsonl
from .pipeline import GuardPipeline

VARIANTS = ("b0", "b1", "e1", "e2", "e3", "e4")
FORMAL_RUN = "formal_v5"
COMPARISONS = {
    "e2_vs_b0.json": ("b0", "e2"),
    "e3_vs_b0.json": ("b0", "e3"),
    "e4_vs_e2.json": ("e2", "e4"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_evidence_paths(root: Path) -> list[Path]:
    paths = [
        root / "data/synthetic/smoke_manifest.jsonl",
        root / "artifacts/stage00/official_preflight_local.json",
        root / "artifacts/stage00/upstream_lock.json",
        root / "docs/experiments/2026-08-14_synthetic_formal_v5.md",
        root / "docs/result_report.md",
        root / "src/drivevla_guard/backends.py",
        root / "src/drivevla_guard/pipeline.py",
        root / "src/drivevla_guard/risk.py",
        root / "src/drivevla_guard/types.py",
    ]
    for variant in VARIANTS:
        paths.extend(
            [
                root / f"configs/{variant if variant in {'b0', 'b1'} else variant}_"
                f"{_config_suffix(variant)}.yaml",
                root / f"artifacts/runs/{FORMAL_RUN}/{variant}/predictions.jsonl",
                root / f"artifacts/runs/{FORMAL_RUN}/{variant}/summary.json",
            ]
        )
    paths.extend(root / "artifacts/runs" / FORMAL_RUN / name for name in COMPARISONS)
    return paths


def _config_suffix(variant: str) -> str:
    return {
        "b0": "fast_greedy",
        "b1": "slow_greedy",
        "e1": "candidates_only",
        "e2": "rerank",
        "e3": "router",
        "e4": "guard",
    }[variant]


def _config_path(root: Path, variant: str) -> Path:
    return root / "configs" / f"{variant}_{_config_suffix(variant)}.yaml"


def evaluate_frozen_evidence(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    manifest_path = root / "data/synthetic/smoke_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    manifest_ids = [str(row["scene"]["scene_id"]) for row in manifest_rows]
    checks["manifest"] = {"scenes": len(manifest_ids), "unique": len(set(manifest_ids)) == len(manifest_ids)}
    if len(manifest_ids) != 12 or len(set(manifest_ids)) != len(manifest_ids):
        errors.append("formal manifest must contain exactly 12 unique scenes")

    for variant in VARIANTS:
        run_dir = root / "artifacts/runs" / FORMAL_RUN / variant
        predictions_path = run_dir / "predictions.jsonl"
        summary_path = run_dir / "summary.json"
        rows = read_jsonl(predictions_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_hash = GuardPipeline(SyntheticBackend(), load_config(_config_path(root, variant))).config_hash
        ids = [str(row.get("scene_id")) for row in rows]
        statuses_ok = all(row.get("status") == "ok" for row in rows)
        hashes_ok = all(row.get("config_hash") == config_hash for row in rows)
        summary_ok = summarize_rows(rows) == summary
        checks[variant] = {
            "scenes": len(rows),
            "manifest_order_ok": ids == manifest_ids,
            "statuses_ok": statuses_ok,
            "config_hash": config_hash,
            "config_hashes_ok": hashes_ok,
            "summary_recomputed_ok": summary_ok,
        }
        if ids != manifest_ids:
            errors.append(f"{variant}: predictions do not match the frozen manifest order")
        if not statuses_ok:
            errors.append(f"{variant}: one or more scenes failed")
        if not hashes_ok:
            errors.append(f"{variant}: config/source identity mismatch")
        if not summary_ok:
            errors.append(f"{variant}: summary does not match predictions")

    for filename, (baseline, candidate) in COMPARISONS.items():
        comparison_path = root / "artifacts/runs" / FORMAL_RUN / filename
        stored = json.loads(comparison_path.read_text(encoding="utf-8"))
        recomputed = compare_results(
            root / f"artifacts/runs/{FORMAL_RUN}/{baseline}/predictions.jsonl",
            root / f"artifacts/runs/{FORMAL_RUN}/{candidate}/predictions.jsonl",
        )
        comparison_ok = stored == recomputed
        checks[filename] = {"recomputed_ok": comparison_ok}
        if not comparison_ok:
            errors.append(f"{filename}: paired comparison does not match predictions")

    files: dict[str, str] = {}
    for path in _tracked_evidence_paths(root):
        relative = path.relative_to(root).as_posix()
        if not path.is_file():
            errors.append(f"missing evidence file: {relative}")
            continue
        files[relative] = _sha256(path)
    return {
        "schema_version": 1,
        "formal_run": f"synthetic_{FORMAL_RUN}",
        "ready": not errors,
        "errors": errors,
        "checks": checks,
        "files": dict(sorted(files.items())),
    }


def freeze_evidence(root: str | Path, output: str | Path) -> dict[str, Any]:
    report = evaluate_frozen_evidence(root)
    if not report["ready"]:
        raise ValueError("cannot freeze invalid evidence: " + "; ".join(report["errors"]))
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def verify_evidence(root: str | Path, snapshot: str | Path) -> dict[str, Any]:
    current = evaluate_frozen_evidence(root)
    expected = json.loads(Path(snapshot).read_text(encoding="utf-8"))
    hash_errors = []
    expected_files = expected.get("files", {})
    for relative, expected_hash in expected_files.items():
        actual_hash = current["files"].get(relative)
        if actual_hash != expected_hash:
            hash_errors.append(f"hash mismatch: {relative}")
    unexpected = sorted(set(current["files"]) - set(expected_files))
    if unexpected:
        hash_errors.extend(f"unfrozen evidence file: {path}" for path in unexpected)
    current["errors"].extend(hash_errors)
    current["ready"] = not current["errors"]
    current["snapshot"] = str(Path(snapshot).resolve())
    return current
