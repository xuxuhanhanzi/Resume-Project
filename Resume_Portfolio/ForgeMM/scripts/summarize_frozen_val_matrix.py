"""Aggregate the preregistered frozen-validation matrix and select method checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_RUN = re.compile(
    r"frozen_val_(?P<method>e[345]_(?:task|fixed|dynamic)|a[12]_[a-z_]+)_seed(?P<seed>\d+)\.metrics\.json"
)
_METRICS = (
    "format_compliance",
    "task_accuracy",
    "evidence_f1",
    "evidence_exact",
    "operation_consistency",
    "fcr",
    "inconsistency_rate",
    "truncation_suspected_rate",
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
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(run_dir.glob("frozen_val_*.metrics.json")):
        match = _RUN.fullmatch(path.name)
        if match is None:
            continue
        method = match.group("method")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "method": method,
            "seed": int(match.group("seed")),
            "metrics_path": str(path),
            **{key: float(metrics[key]) for key in _METRICS},
        }
        rows.append(row)
        grouped[method].append(row)
    if not rows:
        raise ValueError("no frozen validation metrics found")

    aggregates = {}
    selections = {}
    for method, method_rows in sorted(grouped.items()):
        aggregates[method] = {
            "runs": len(method_rows),
            **{
                key: {
                    "mean": statistics.fmean(row[key] for row in method_rows),
                    "std": (
                        statistics.stdev(row[key] for row in method_rows)
                        if len(method_rows) > 1
                        else 0.0
                    ),
                }
                for key in _METRICS
            },
        }
        if method.startswith(("e3_", "e4_", "e5_")):
            best = max(
                method_rows,
                key=lambda row: (
                    row["fcr"],
                    row["task_accuracy"],
                    row["format_compliance"],
                    -row["seed"],
                ),
            )
            selections[method] = {
                "seed": best["seed"],
                "selection_rule": (
                    "max FCR, then task accuracy, then format compliance, then lower seed"
                ),
                "metrics_path": best["metrics_path"],
            }
    return {
        "schema_version": "forgemm-frozen-val-summary-v1",
        "claim_boundary": "selection uses only frozen ChartQA val strict labels",
        "rows": rows,
        "aggregates": aggregates,
        "selected_checkpoints": selections,
    }


if __name__ == "__main__":
    raise SystemExit(main())
