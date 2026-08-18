from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io import append_jsonl, load_manifest, read_jsonl
from .pipeline import GuardPipeline


class EvaluationRunner:
    def __init__(self, pipeline: GuardPipeline):
        self.pipeline = pipeline

    def run(self, manifest_path: str | Path, output_path: str | Path, resume: bool = True) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        output = Path(output_path)
        if not resume and output.exists():
            raise FileExistsError(f"refusing to overwrite existing result file: {output}")
        existing = read_jsonl(output) if resume and output.exists() else []
        expected_ids = [context.scene_id for _, context in manifest]
        existing_ids = [row.get("scene_id") for row in existing]
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise ValueError("resume refused: existing results are not a strict manifest prefix")
        for row in existing:
            if row.get("config_hash") != self.pipeline.config_hash:
                raise ValueError("resume refused: config hash differs from existing results")

        for model_input, context in manifest[len(existing) :]:
            started = time.perf_counter()
            try:
                result = self.pipeline.plan(model_input, context)
                row = result.to_dict()
                row["status"] = "ok"
                row["scenario_type"] = context.metadata.get("scenario_type")
            except Exception as exc:  # preserve a per-scene failure instead of losing the run
                row = {
                    "scene_id": context.scene_id,
                    "config_hash": self.pipeline.config_hash,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            row["runner_wall_ms"] = (time.perf_counter() - started) * 1000.0
            append_jsonl(output, row)
            existing.append(row)
        return summarize_rows(existing)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        return {"metric_scope": "synthetic_proxy_not_navsim", "scenes": len(rows), "valid_scenes": 0}
    selected = [row["selected"]["risk"] for row in ok]
    collision_failures = sum((item.get("collision") or 0.0) >= 1.0 for item in selected)
    drivable_failures = sum((item.get("drivable") or 0.0) > 0.0 for item in selected)
    progress = [1.0 - float(item["progress"]) for item in selected]
    comfort = [1.0 - min(float(item["comfort"]), 1.0) for item in selected]
    safety = 1.0 - (collision_failures + drivable_failures) / (2.0 * len(ok))
    proxy_score = safety * float(np.mean(progress)) * float(np.mean(comfort))
    return {
        "metric_scope": "synthetic_proxy_not_navsim",
        "scenes": len(rows),
        "valid_scenes": len(ok),
        "failed_scenes": len(rows) - len(ok),
        "collision_failures": int(collision_failures),
        "drivable_failures": int(drivable_failures),
        "mean_progress": float(np.mean(progress)),
        "mean_comfort": float(np.mean(comfort)),
        "mean_risk": float(np.mean([item["total"] for item in selected])),
        "mean_latency_ms": float(np.mean([row["total_latency_ms"] for row in ok])),
        "p95_latency_ms": float(np.percentile([row["total_latency_ms"] for row in ok], 95)),
        "slow_route_rate": float(np.mean([row["routed_to_slow"] for row in ok])),
        "proxy_score": proxy_score,
    }


def compare_results(baseline_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    baseline = {row["scene_id"]: row for row in read_jsonl(baseline_path) if row.get("status") == "ok"}
    candidate = {row["scene_id"]: row for row in read_jsonl(candidate_path) if row.get("status") == "ok"}
    if baseline.keys() != candidate.keys():
        raise ValueError("paired comparison requires identical successful scene ids")
    deltas = []
    changed = 0
    improved = 0
    regressed = 0
    for scene_id in baseline:
        base = baseline[scene_id]
        new = candidate[scene_id]
        delta = float(new["selected"]["risk"]["total"] - base["selected"]["risk"]["total"])
        deltas.append(delta)
        if new["selected"]["candidate"]["candidate_id"] != base["selected"]["candidate"]["candidate_id"]:
            changed += 1
        improved += delta < -1e-9
        regressed += delta > 1e-9
    return {
        "metric_scope": "paired_internal_risk_not_navsim",
        "paired_scenes": len(deltas),
        "mean_risk_delta": float(np.mean(deltas)),
        "median_risk_delta": float(np.median(deltas)),
        "improved_scenes": int(improved),
        "regressed_scenes": int(regressed),
        "selection_changed_scenes": int(changed),
    }


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
