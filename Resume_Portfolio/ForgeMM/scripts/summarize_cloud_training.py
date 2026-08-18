"""Summarize formal GRPO stability, resource, reward, and dual-state evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

_RUN = re.compile(r"formal50_(?P<method>.+)_seed(?P<seed>\d+)$")


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
    runs: list[dict[str, Any]] = []
    for logging_path in sorted(run_dir.glob("formal50_*/logging.jsonl")):
        match = _RUN.fullmatch(logging_path.parent.name)
        if match is None:
            continue
        rows = [
            json.loads(line)
            for line in logging_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        steps = [row for row in rows if "loss" in row]
        reward_rows = [row for row in steps if "reward" in row]
        if not steps:
            raise ValueError(f"missing_step_logs:{logging_path}")
        losses = [float(row["loss"]) for row in steps]
        completion_lengths = [
            float(row["completions/mean_length"])
            for row in reward_rows
            if "completions/mean_length" in row
        ]
        clipped = [
            float(row["completions/clipped_ratio"])
            for row in reward_rows
            if "completions/clipped_ratio" in row
        ]
        run = {
            "run": logging_path.parent.name,
            "method": match.group("method"),
            "seed": int(match.group("seed")),
            "step_records": len(steps),
            "reward_records": len(reward_rows),
            "loss_finite": all(math.isfinite(value) for value in losses),
            "loss_mean": statistics.fmean(losses),
            "loss_min": min(losses),
            "loss_max": max(losses),
            "reward_std_nonzero_records": sum(
                float(row.get("reward_std", 0.0)) > 0 for row in reward_rows
            ),
            "completion_mean_length": (
                statistics.fmean(completion_lengths) if completion_lengths else None
            ),
            "completion_clipped_mean": statistics.fmean(clipped) if clipped else None,
            "completion_clipped_nonzero_records": sum(value > 0 for value in clipped),
            "peak_memory_gib": max(float(row.get("memory(GiB)", 0.0)) for row in steps),
        }
        dual_path = logging_path.parent / "chart_fgrpo_dual_state.json"
        if dual_path.is_file():
            run["dual_state"] = json.loads(dual_path.read_text(encoding="utf-8"))
        runs.append(run)
    if not runs:
        raise ValueError("no formal training logs found")
    return {
        "schema_version": "forgemm-formal-training-summary-v1",
        "runs": runs,
        "all_loss_finite": all(run["loss_finite"] for run in runs),
        "max_peak_memory_gib": max(float(run["peak_memory_gib"]) for run in runs),
        "total_reward_records": sum(int(run["reward_records"]) for run in runs),
        "total_nonzero_reward_std_records": sum(
            int(run["reward_std_nonzero_records"]) for run in runs
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
