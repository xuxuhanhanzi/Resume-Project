"""Apply the preregistered E2 quality gate on aligned frozen validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from forgemm.evaluation.inference import evaluate_inference_rows
from forgemm.evaluation.statistics import paired_bootstrap_delta
from forgemm.evaluation.task_inference import extract_answer
from forgemm.rewards.answer import answer_reward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-format", type=float, default=0.99)
    parser.add_argument("--noninferiority-pp", type=float, default=-0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    result = evaluate_gate(
        _load_rows(args.baseline),
        _load_rows(args.structured),
        min_format=args.min_format,
        noninferiority_pp=args.noninferiority_pp,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.output.exists():
        raise FileExistsError(f"output_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "advance" else 3


def evaluate_gate(
    baseline_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    *,
    min_format: float,
    noninferiority_pp: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a gate result without using the test split or training reward."""

    if not 0 <= min_format <= 1 or not -1 <= noninferiority_pp <= 0:
        raise ValueError("invalid_gate_threshold")
    if len(baseline_rows) == 0 or len(baseline_rows) != len(structured_rows):
        raise ValueError("unaligned_validation_rows")
    baseline = _answer_vector(baseline_rows)
    structured = _answer_vector(structured_rows)
    _verify_alignment(baseline_rows, structured_rows)
    structured_metrics = evaluate_inference_rows(structured_rows)
    interval = paired_bootstrap_delta(
        np.asarray(baseline, dtype=np.float64),
        np.asarray(structured, dtype=np.float64),
        samples=bootstrap_samples,
        seed=seed,
    )
    format_pass = float(structured_metrics["format_compliance"]) >= min_format
    quality_pass = float(interval["ci_low"]) >= noninferiority_pp
    return {
        "schema_version": "forgemm-e2-gate-v1",
        "claim_boundary": "frozen ChartQA validation only; not a test-set result",
        "samples": len(baseline),
        "baseline_task_accuracy": float(np.mean(baseline)),
        "structured_task_accuracy": float(np.mean(structured)),
        "structured_format_compliance": structured_metrics["format_compliance"],
        "task_accuracy_delta": interval,
        "thresholds": {
            "min_format": min_format,
            "noninferiority_pp": noninferiority_pp,
        },
        "format_pass": format_pass,
        "quality_pass": quality_pass,
        "decision": "advance" if format_pass and quality_pass else "stop",
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid_jsonl_row:{path}")
    return rows


def _answer_vector(rows: list[dict[str, Any]]) -> list[float]:
    scores = []
    for row in rows:
        answer, _ = extract_answer(str(row["response"]))
        reference, _ = extract_answer(str(row["labels"]))
        scores.append(answer_reward(answer, reference))
    return scores


def _verify_alignment(
    baseline_rows: list[dict[str, Any]], structured_rows: list[dict[str, Any]]
) -> None:
    for index, (baseline, structured) in enumerate(
        zip(baseline_rows, structured_rows, strict=True)
    ):
        if str(baseline.get("labels")) != str(structured.get("labels")):
            raise ValueError(f"label_identity_mismatch:{index}")
        if _image_paths(baseline.get("images")) != _image_paths(structured.get("images")):
            raise ValueError(f"image_identity_mismatch:{index}")


def _image_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item.get("path", "")) if isinstance(item, dict) else str(item) for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
