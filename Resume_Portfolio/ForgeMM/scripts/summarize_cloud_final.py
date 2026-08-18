"""Build a compact, hash-backed index of the completed cloud experiment evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DRIVER_EXITS = (
    "formal_matrix_driver.exit",
    "frozen_val_matrix_driver.exit",
    "frozen_val_summary.exit",
    "frozen_test_strict_driver.exit",
    "frozen_baseline_driver.exit",
    "frozen_task_driver.exit",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def summarize(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    exit_codes = {name: _read_exit(run_dir / name) for name in DRIVER_EXITS}
    if any(code != 0 for code in exit_codes.values()):
        raise RuntimeError(f"cloud driver failure: {exit_codes}")

    metric_paths = sorted(run_dir.glob("frozen_*.metrics.json"))
    if not metric_paths:
        raise RuntimeError("no frozen metrics found")
    metrics = {path.name: _load_json(path) for path in metric_paths}

    structured_paths = [
        run_dir / "formal_training_summary.json",
        run_dir / "frozen_val_summary.json",
        run_dir / "frozen_test_strict_e3_vs_e5.json",
    ]
    structured = {path.name: _load_json(path) for path in structured_paths if path.is_file()}
    evidence_paths = [*metric_paths, *structured_paths, *(run_dir / name for name in DRIVER_EXITS)]
    identities = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in evidence_paths
        if path.is_file()
    }

    return {
        "schema_version": "forgemm-cloud-final-summary-v1",
        "claim_boundary": (
            "official frozen ChartQA/ChartQAPro offline evaluation; no SOTA, deployment, "
            "human preference, or real-world reliability claim"
        ),
        "driver_exit_codes": exit_codes,
        "metrics": metrics,
        "structured_summaries": structured,
        "evidence_identities": identities,
    }


def _read_exit(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(path.read_text(encoding="utf-8").strip())


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
