"""Enforce the preregistered minimum record counts before costly formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--val-summary", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=1500)
    parser.add_argument("--min-val", type=int, default=200)
    parser.add_argument("--min-test", type=int, default=500)
    args = parser.parse_args()

    result = evaluate_data_gate(
        _load_object(args.train_summary),
        _load_object(args.val_summary),
        _load_object(args.test_summary),
        min_train=args.min_train,
        min_val=args.min_val,
        min_test=args.min_test,
    )
    if args.output.exists():
        raise FileExistsError(f"output_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "advance" else 3


def evaluate_data_gate(
    train_summary: dict[str, Any],
    val_summary: dict[str, Any],
    test_summary: dict[str, Any],
    *,
    min_train: int,
    min_val: int,
    min_test: int,
) -> dict[str, Any]:
    """Return a deterministic data-quality decision before any model run starts."""

    if min(min_train, min_val, min_test) <= 0:
        raise ValueError("minimum_counts_must_be_positive")
    observed = {
        "train_strict_records": _count(train_summary, "unique_strict_records"),
        "val_strict_records": _count(val_summary, "strict_question_records"),
        "test_strict_records": _count(test_summary, "strict_question_records"),
    }
    thresholds = {
        "train_strict_records": min_train,
        "val_strict_records": min_val,
        "test_strict_records": min_test,
    }
    failed = {
        name: observed[name] for name, minimum in thresholds.items() if observed[name] < minimum
    }
    return {
        "schema_version": "forgemm-stage04-data-gate-v1",
        "claim_boundary": (
            "minimum records are required for a preregistered formal comparison; a failed "
            "gate may support data-label analysis only, not a supported method claim"
        ),
        "observed": observed,
        "thresholds": thresholds,
        "failed": failed,
        "decision": "advance" if not failed else "stop",
    }


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_summary:{path}")
    return payload


def _count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid_summary_count:{key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
