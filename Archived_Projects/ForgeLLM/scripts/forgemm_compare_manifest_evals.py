"""Combine four manifest evaluations into one traceable comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct
from forgellm.multimodal.evaluation import (
    load_completed_predictions,
    load_evaluation_manifest,
    paired_sign_test_two_sided_p,
    summarize_predictions,
    transition_counts,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _pairwise_exact(
    left: Sequence[Mapping[str, object]], right: Sequence[Mapping[str, object]]
) -> dict[str, int]:
    left_wins = right_wins = both_correct = both_wrong = 0
    for left_row, right_row in zip(left, right, strict=True):
        left_correct = normalized_exact_correct(
            str(left_row["prediction"]), str(left_row["reference"])
        )
        right_correct = normalized_exact_correct(
            str(right_row["prediction"]), str(right_row["reference"])
        )
        left_wins += int(left_correct and not right_correct)
        right_wins += int(right_correct and not left_correct)
        both_correct += int(left_correct and right_correct)
        both_wrong += int(not left_correct and not right_correct)
    return {
        "left_only_correct": left_wins,
        "right_only_correct": right_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {args.output_dir}")
    model_dirs = {
        "frozen_raw": args.frozen_dir,
        "b1_attention_lr2e4": args.b1_dir,
        "b2_attention_ffn_lr2e4": args.b2_dir,
        "b3_attention_lr5e5": args.b3_dir,
    }
    configs = {
        name: cast(
            dict[str, object],
            json.loads((path / "run_config.json").read_text(encoding="utf-8")),
        )
        for name, path in model_dirs.items()
    }
    manifest_hashes = {str(config["manifest_sha256"]) for config in configs.values()}
    counts = {int(cast(int, config["count"])) for config in configs.values()}
    if len(manifest_hashes) != 1 or len(counts) != 1:
        raise ValueError("Evaluation runs do not use the same manifest and count")
    manifest_path = Path(str(configs["frozen_raw"]["manifest_path"]))
    records = load_evaluation_manifest(manifest_path, limit=next(iter(counts)))
    predictions = {
        name: load_completed_predictions(path / "predictions.jsonl", records)
        for name, path in model_dirs.items()
    }
    if any(len(rows) != len(records) for rows in predictions.values()):
        raise ValueError("At least one evaluation is incomplete")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    comparison_csv = args.output_dir / "model_comparison.csv"
    fields = [
        "index",
        "source_index",
        "image",
        "task_type",
        "answer_type",
        "question",
        "reference",
    ]
    for name in model_dirs:
        fields.extend((f"{name}_prediction", f"{name}_exact", f"{name}_relaxed"))
    with comparison_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            row: dict[str, object] = {
                "index": index,
                "source_index": record["source_index"],
                "image": record["image"],
                "task_type": record["task_type"],
                "answer_type": record["answer_type"],
                "question": record["question"],
                "reference": record["reference"],
            }
            for name, rows in predictions.items():
                prediction = str(rows[index]["prediction"])
                row[f"{name}_prediction"] = prediction
                row[f"{name}_exact"] = normalized_exact_correct(prediction, record["reference"])
                row[f"{name}_relaxed"] = chartqa_relaxed_correct(prediction, record["reference"])
            writer.writerow(row)

    baseline = predictions["frozen_raw"]
    exact_transitions = {
        name: transition_counts(baseline, rows)
        for name, rows in predictions.items()
        if name != "frozen_raw"
    }
    relaxed_transitions = {
        name: transition_counts(baseline, rows, metric="relaxed")
        for name, rows in predictions.items()
        if name != "frozen_raw"
    }
    report = {
        "schema_version": "forgemm-val250-comparison-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "One deterministic stratified 250-example validation subset; category labels are "
            "project heuristics and results are not a final benchmark."
        ),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": next(iter(manifest_hashes)),
        "models": {
            name: {
                "run_dir": str(model_dirs[name].resolve()),
                "config": configs[name],
                "metrics": summarize_predictions(rows),
            }
            for name, rows in predictions.items()
        },
        "transitions_from_frozen_raw": {
            "normalized_exact": exact_transitions,
            "chartqa_relaxed": relaxed_transitions,
        },
        "paired_sign_test_two_sided": {
            metric: {
                name: paired_sign_test_two_sided_p(
                    values["fixed_baseline_errors"],
                    values["regressed_baseline_correct"],
                )
                for name, values in transitions.items()
            }
            for metric, transitions in {
                "normalized_exact": exact_transitions,
                "chartqa_relaxed": relaxed_transitions,
            }.items()
        },
        "pairwise_exact": {
            "b1_vs_b3": _pairwise_exact(
                predictions["b1_attention_lr2e4"], predictions["b3_attention_lr5e5"]
            ),
            "b2_vs_b3": _pairwise_exact(
                predictions["b2_attention_ffn_lr2e4"], predictions["b3_attention_lr5e5"]
            ),
        },
        "comparison_csv": str(comparison_csv.resolve()),
    }
    _write_json(args.output_dir / "comparison_report.json", report)
    print((args.output_dir / "comparison_report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
