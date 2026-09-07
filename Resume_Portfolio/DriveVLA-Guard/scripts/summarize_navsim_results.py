#!/usr/bin/env python3
"""Freeze the latest official NAVSIM averages and the registered E4 claim boundary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from check_navsim_gate import METRICS, csv_files, read_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = {
        "upstream_b0": "autovla_upstream_b0",
        "guard_b0": "drivevla_guard_b0",
        "e2": "drivevla_guard_e2",
        "e3": "drivevla_guard_e3",
        "e4": "drivevla_guard_e4",
    }
    runs = {
        name: read_result(csv_files(args.experiment_root / directory)[-1])
        for name, directory in names.items()
    }
    baseline = runs["guard_b0"]["metrics"]
    e4 = runs["e4"]["metrics"]
    deltas = {name: e4[name] - baseline[name] for name in METRICS}
    baseline_safety_failures = (1.0 - baseline["no_at_fault_collisions"]) + (
        1.0 - baseline["drivable_area_compliance"]
    )
    e4_safety_failures = (1.0 - e4["no_at_fault_collisions"]) + (1.0 - e4["drivable_area_compliance"])
    checks = {
        "collision_or_drivable_failures_drop_10_percent": (
            e4_safety_failures <= baseline_safety_failures * 0.90
            if baseline_safety_failures > 0
            else e4_safety_failures <= 1e-9
        ),
        "pdms_regression_within_0p5pp": deltas["score"] >= -0.005,
        "ego_progress_regression_within_1pp": deltas["ego_progress"] >= -0.01,
    }
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "e4_minus_guard_b0": deltas,
        "registered_metric_checks": checks,
        "metric_claim_decision": "positive" if all(checks.values()) else "negative_or_mixed",
        "claim_boundary": (
            "Official NAVSIM metric summary. Latency, peak VRAM, invalid-rate and route-rate "
            "claims require their separately recorded runtime traces."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
