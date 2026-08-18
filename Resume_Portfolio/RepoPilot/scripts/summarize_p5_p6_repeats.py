"""Aggregate the frozen three-repeat P5/P6 experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

GROUPS = {
    "frames_b0_direct": (
        "20260807_frames_p5_b0_direct_qwen2_5_7b_quick10",
        "20260807_frames_p5_b0_direct_qwen2_5_7b_repeat2",
        "20260807_frames_p5_b0_direct_qwen2_5_7b_repeat3",
    ),
    "frames_b1_full_agent": (
        "20260807_frames_agent_qwen2_5_7b_smoke10_v1",
        "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat2",
        "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat3",
    ),
    "frames_a4_no_planner": (
        "20260807_frames_p6_a4_no_planner_qwen2_5_7b_quick10",
        "20260807_frames_p6_a4_no_planner_qwen2_5_7b_repeat2",
        "20260807_frames_p6_a4_no_planner_qwen2_5_7b_repeat3",
    ),
    "frames_a5_no_reviewer": (
        "20260807_frames_p6_a5_no_reviewer_qwen2_5_7b_quick10",
        "20260807_frames_p6_a5_no_reviewer_qwen2_5_7b_repeat2",
        "20260807_frames_p6_a5_no_reviewer_qwen2_5_7b_repeat3",
    ),
    "dabench_b1_feedback": (
        "20260807_dabench_p5_b1_agent_qwen2_5_7b_quick10",
        "20260807_dabench_p5_b1_agent_qwen2_5_7b_repeat2",
        "20260807_dabench_p5_b1_agent_qwen2_5_7b_repeat3",
    ),
    "dabench_a1_no_feedback": (
        "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_quick10",
        "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_repeat2",
        "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_repeat3",
    ),
}


def _rounded(value: float) -> float:
    return round(value, 6)


def _run_metrics(path: Path) -> dict[str, float | str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw["records"]
    shared_keys = ("input_tokens", "output_tokens", "tool_calls", "wall_seconds")
    metrics: dict[str, float | str] = {
        "run_id": str(raw["run_id"]),
        "accuracy": float(raw["accuracy"]),
    }
    for key in shared_keys:
        metrics[key] = sum(float(record["shared_metrics"].get(key, 0)) for record in records)
    metrics["recoveries"] = sum(
        float(record["shared_metrics"].get("recovery_count", 0)) for record in records
    )
    f1_values = [
        float(record["domain_metrics"]["answer_token_f1"])
        for record in records
        if "answer_token_f1" in record["domain_metrics"]
    ]
    if f1_values:
        metrics["answer_token_f1"] = mean(f1_values)
    return metrics


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    artifact_root = project / "artifacts" / "benchmarks"
    groups: dict[str, object] = {}
    for label, run_ids in GROUPS.items():
        runs = [_run_metrics(artifact_root / run_id / "results.json") for run_id in run_ids]
        numeric_keys = sorted(
            set.intersection(
                *(
                    set(key for key, value in run.items() if isinstance(value, float))
                    for run in runs
                )
            )
        )
        aggregate = {
            key: {
                "mean": _rounded(mean(float(run[key]) for run in runs)),
                "population_std": _rounded(pstdev(float(run[key]) for run in runs)),
            }
            for key in numeric_keys
        }
        groups[label] = {"runs": runs, "aggregate": aggregate}
    result = {
        "repetitions": 3,
        "tasks_per_run": 10,
        "groups": groups,
    }
    output = artifact_root / "20260807_p5_p6_formal_repeats" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "groups": len(groups)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
