#!/usr/bin/env python3
"""Gate E3/E4 on reproducible B0, adapter parity, and non-regressing E2 safety."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
SAFETY = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
)


def csv_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/*.csv"), key=lambda item: item.stat().st_mtime_ns)
    if not files:
        raise FileNotFoundError(f"no NAVSIM result CSV under {root}")
    return files


def read_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    scenarios = {row["token"]: row for row in rows if row.get("token") != "average"}
    average = next((row for row in rows if row.get("token") == "average"), None)
    if not scenarios or average is None:
        raise ValueError(f"result has no scenario rows or average: {path}")
    if any(str(row.get("valid", "")).casefold() not in {"true", "1"} for row in scenarios.values()):
        raise ValueError(f"result contains invalid scenarios: {path}")
    return {
        "path": str(path.resolve()),
        "tokens": sorted(scenarios),
        "metrics": {name: float(average[name]) for name in METRICS},
    }


def max_delta(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(left[name] - right[name]) for name in METRICS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--e2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    upstream_paths = csv_files(args.upstream_root)
    if len(upstream_paths) < 2:
        raise ValueError("two upstream B0 repetitions are required")
    upstream_first = read_result(upstream_paths[-2])
    upstream_second = read_result(upstream_paths[-1])
    adapter = read_result(csv_files(args.adapter_root)[-1])
    e2 = read_result(csv_files(args.e2_root)[-1])
    same_tokens = upstream_first["tokens"] == upstream_second["tokens"] == adapter["tokens"] == e2["tokens"]
    repeat_delta = max_delta(upstream_first["metrics"], upstream_second["metrics"])
    parity_delta = max_delta(upstream_second["metrics"], adapter["metrics"])
    baseline = adapter["metrics"]
    candidate = e2["metrics"]
    safety_non_regression = all(candidate[name] >= baseline[name] - 1e-9 for name in SAFETY)
    safety_improvement = any(candidate[name] > baseline[name] + 1e-9 for name in SAFETY)
    pdms_ok = candidate["score"] >= baseline["score"] - 0.005
    progress_ok = candidate["ego_progress"] >= baseline["ego_progress"] - 0.01
    checks = {
        "same_frozen_tokens": same_tokens,
        "upstream_b0_repeat_delta_ok": repeat_delta <= args.parity_tolerance,
        "guard_b0_parity_ok": parity_delta <= args.parity_tolerance,
        "e2_safety_non_regression": safety_non_regression,
        "e2_has_safety_improvement": safety_improvement,
        "e2_pdms_regression_within_0p5pp": pdms_ok,
        "e2_progress_regression_within_1pp": progress_ok,
    }
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": "advance" if all(checks.values()) else "stop",
        "checks": checks,
        "max_deltas": {"upstream_repeat": repeat_delta, "guard_b0_parity": parity_delta},
        "runs": {
            "upstream_b0_first": upstream_first,
            "upstream_b0_second": upstream_second,
            "guard_b0": adapter,
            "e2": e2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["decision"] == "advance" else 3


if __name__ == "__main__":
    raise SystemExit(main())
