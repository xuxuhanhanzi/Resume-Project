"""Paired statistical comparison for aligned structured inference JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from forgemm.evaluation.statistics import paired_bootstrap_delta, paired_mcnemar
from forgemm.reasoning.parser import parse_prediction
from forgemm.rewards.scoring import score_completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    result = compare_rows(
        baseline,
        candidate,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def compare_rows(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired inference rows must be non-empty and aligned")
    baseline_scores = []
    candidate_scores = []
    for index, (base, cand) in enumerate(zip(baseline, candidate, strict=True)):
        for key in ("labels", "images", "messages"):
            if base.get(key) != cand.get(key):
                raise ValueError(f"paired_identity_mismatch:{index}:{key}")
        gold = parse_prediction(str(base["labels"]))
        baseline_scores.append(score_completion(str(base["response"]), gold.answer, gold.evidence))
        candidate_scores.append(score_completion(str(cand["response"]), gold.answer, gold.evidence))

    vectors: dict[str, tuple[np.ndarray[Any, np.dtype[np.float64]], ...]] = {
        "task": _vectors(baseline_scores, candidate_scores, "task"),
        "evidence_f1": _vectors(baseline_scores, candidate_scores, "evidence"),
        "operation_consistency": _vectors(baseline_scores, candidate_scores, "operation"),
        "fcr": (
            _fcr_vector(baseline_scores),
            _fcr_vector(candidate_scores),
        ),
    }
    bootstrap = {
        metric: paired_bootstrap_delta(pair[0], pair[1], samples=bootstrap_samples, seed=seed)
        for metric, pair in vectors.items()
    }
    return {
        "schema_version": "forgemm-paired-comparison-v1",
        "pairs": len(baseline),
        "bootstrap": bootstrap,
        "mcnemar": {
            "task": vars(
                paired_mcnemar(
                    vectors["task"][0].astype(bool).tolist(),
                    vectors["task"][1].astype(bool).tolist(),
                )
            ),
            "fcr": vars(
                paired_mcnemar(
                    vectors["fcr"][0].astype(bool).tolist(),
                    vectors["fcr"][1].astype(bool).tolist(),
                )
            ),
        },
    }


def _vectors(
    baseline: list[Any], candidate: list[Any], field: str
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    return (
        np.asarray([getattr(item, field) for item in baseline], dtype=np.float64),
        np.asarray([getattr(item, field) for item in candidate], dtype=np.float64),
    )


def _fcr_vector(scores: list[Any]) -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray(
        [item.task == 1.0 and item.evidence == 1.0 and item.operation == 1.0 for item in scores],
        dtype=np.float64,
    )


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    raise SystemExit(main())
