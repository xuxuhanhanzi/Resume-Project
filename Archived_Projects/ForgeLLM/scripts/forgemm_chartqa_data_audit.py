"""Audit ChartQA data distributions and ForgeMM B1/B2/B3 evaluation behavior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from forgellm.multimodal.chartqa import chartqa_relaxed_correct, normalized_exact_correct
from forgellm.multimodal.data_audit import (
    classify_answer,
    classify_task,
    compare_distributions,
    deterministic_stratified_indices,
    normalize_question,
    summarize_records,
)

Record = dict[str, str]
REQUIRED_FIELDS = {"imgname", "query", "label"}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_records(path: Path) -> list[Record]:
    payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    records: list[Record] = []
    for index, raw_record in enumerate(payload):
        if not isinstance(raw_record, dict) or not raw_record.keys() >= REQUIRED_FIELDS:
            raise ValueError(f"Invalid ChartQA record {index} in {path}")
        records.append(
            {field: str(cast(dict[str, object], raw_record)[field]) for field in REQUIRED_FIELDS}
        )
    return records


def _source_path(data_root: Path, split: str, source: str) -> Path:
    return data_root / split / f"{split}_{source}.json"


def _integrity_summary(
    records: Sequence[Record], *, image_dir: Path, available_pngs: set[str]
) -> dict[str, object]:
    referenced = {record["imgname"] for record in records}
    missing = sorted(referenced - available_pngs)
    return {
        "referenced_unique_images": len(referenced),
        "missing_image_count": len(missing),
        "missing_image_examples": missing[:20],
        "image_directory": str(image_dir.resolve()),
    }


def _record_keys(records: Iterable[Record]) -> dict[str, set[object]]:
    exact: set[object] = set()
    qa: set[object] = set()
    questions: set[object] = set()
    images: set[object] = set()
    for record in records:
        question = normalize_question(record["query"])
        answer = record["label"].strip().casefold()
        exact.add((record["imgname"], question, answer))
        qa.add((question, answer))
        questions.add(question)
        images.add(record["imgname"])
    return {
        "exact_record": exact,
        "question_answer": qa,
        "question": questions,
        "image_name": images,
    }


def _surface_overlap(left: Sequence[Record], right: Sequence[Record]) -> dict[str, int]:
    left_keys = _record_keys(left)
    right_keys = _record_keys(right)
    return {
        key: len(left_keys[key] & right_keys[key])
        for key in ("exact_record", "question_answer", "question", "image_name")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_hashes(records: Sequence[Record], image_dir: Path) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for filename in sorted({record["imgname"] for record in records}):
        path = image_dir / filename
        if path.is_file():
            hashes[_sha256(path)].append(filename)
    return dict(hashes)


def _content_overlap(
    left: Mapping[str, list[str]], right: Mapping[str, list[str]]
) -> dict[str, object]:
    shared = sorted(set(left) & set(right))
    examples = [
        {"sha256": digest, "left": left[digest], "right": right[digest]} for digest in shared[:20]
    ]
    return {"shared_content_hashes": len(shared), "examples": examples}


def _counts(summary: Mapping[str, object], key: str) -> dict[str, int]:
    return cast(dict[str, int], summary[key])


def _distribution_comparison(
    population: Mapping[str, object], subset: Mapping[str, object]
) -> dict[str, object]:
    return {
        "task": compare_distributions(
            _counts(population, "task_counts"), _counts(subset, "task_counts")
        ),
        "answer": compare_distributions(
            _counts(population, "answer_counts"), _counts(subset, "answer_counts")
        ),
    }


def _load_evaluation(path: Path) -> list[dict[str, object]]:
    payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"Evaluation has no prediction list: {path}")
    return cast(list[dict[str, object]], predictions)


def _metric_summary(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_task: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    by_answer: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    exact_total = 0
    relaxed_total = 0
    for row in predictions:
        question = str(row["question"])
        reference = str(row["reference"])
        prediction = str(row["prediction"])
        exact = normalized_exact_correct(prediction, reference)
        relaxed = chartqa_relaxed_correct(prediction, reference)
        exact_total += int(exact)
        relaxed_total += int(relaxed)
        by_task[classify_task(question, reference)].append((exact, relaxed))
        by_answer[classify_answer(question, reference)].append((exact, relaxed))

    def group_metrics(groups: Mapping[str, Sequence[tuple[bool, bool]]]) -> dict[str, object]:
        return {
            key: {
                "count": len(values),
                "normalized_exact_accuracy": sum(int(value[0]) for value in values) / len(values),
                "chartqa_relaxed_accuracy": sum(int(value[1]) for value in values) / len(values),
            }
            for key, values in sorted(groups.items())
        }

    count = len(predictions)
    return {
        "count": count,
        "normalized_exact_accuracy": exact_total / count,
        "chartqa_relaxed_accuracy": relaxed_total / count,
        "by_task": group_metrics(by_task),
        "by_answer": group_metrics(by_answer),
    }


def _evaluation_audit(
    evaluation_paths: Mapping[str, Path], *, output_csv: Path
) -> dict[str, object]:
    predictions = {name: _load_evaluation(path) for name, path in evaluation_paths.items()}
    lengths = {name: len(rows) for name, rows in predictions.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Evaluation lengths differ: {lengths}")
    count = next(iter(lengths.values()))
    baseline = predictions["frozen_before"]
    for name, rows in predictions.items():
        for index, (expected, actual) in enumerate(zip(baseline, rows, strict=True)):
            expected_key = (expected["image"], expected["question"], expected["reference"])
            actual_key = (actual["image"], actual["question"], actual["reference"])
            if expected_key != actual_key:
                raise ValueError(f"Evaluation order mismatch for {name} at row {index}")

    fieldnames = [
        "index",
        "image",
        "task_type",
        "answer_type",
        "question",
        "reference",
    ]
    for name in evaluation_paths:
        fieldnames.extend((f"{name}_prediction", f"{name}_exact", f"{name}_relaxed"))
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            reference_row = baseline[index]
            question = str(reference_row["question"])
            reference = str(reference_row["reference"])
            row: dict[str, object] = {
                "index": index,
                "image": reference_row["image"],
                "task_type": classify_task(question, reference),
                "answer_type": classify_answer(question, reference),
                "question": question,
                "reference": reference,
            }
            for name, rows in predictions.items():
                prediction = str(rows[index]["prediction"])
                row[f"{name}_prediction"] = prediction
                row[f"{name}_exact"] = normalized_exact_correct(prediction, reference)
                row[f"{name}_relaxed"] = chartqa_relaxed_correct(prediction, reference)
            writer.writerow(row)

    baseline_correct = [
        normalized_exact_correct(str(row["prediction"]), str(row["reference"])) for row in baseline
    ]
    transitions: dict[str, object] = {}
    for name, rows in predictions.items():
        if name == "frozen_before":
            continue
        candidate_correct = [
            normalized_exact_correct(str(row["prediction"]), str(row["reference"])) for row in rows
        ]
        transitions[name] = {
            "fixed_baseline_errors": sum(
                int(not before and after)
                for before, after in zip(baseline_correct, candidate_correct, strict=True)
            ),
            "regressed_baseline_correct": sum(
                int(before and not after)
                for before, after in zip(baseline_correct, candidate_correct, strict=True)
            ),
            "unchanged_correct": sum(
                int(before and after)
                for before, after in zip(baseline_correct, candidate_correct, strict=True)
            ),
            "unchanged_wrong": sum(
                int(not before and not after)
                for before, after in zip(baseline_correct, candidate_correct, strict=True)
            ),
        }
    return {
        "sources": {name: str(path.resolve()) for name, path in evaluation_paths.items()},
        "models": {name: _metric_summary(rows) for name, rows in predictions.items()},
        "transitions_from_frozen_before": transitions,
        "comparison_csv": str(output_csv.resolve()),
    }


def _write_training_manifest(
    records: Sequence[Record], indices: Sequence[int], output_csv: Path
) -> None:
    fields = [
        "sequence",
        "source_index",
        "image",
        "task_type",
        "answer_type",
        "question",
        "reference",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sequence, index in enumerate(indices):
            record = records[index]
            writer.writerow(
                {
                    "sequence": sequence,
                    "source_index": index,
                    "image": record["imgname"],
                    "task_type": classify_task(record["query"], record["label"]),
                    "answer_type": classify_answer(record["query"], record["label"]),
                    "question": record["query"],
                    "reference": record["label"],
                }
            )


def _max_shift(comparison: Mapping[str, object]) -> float:
    return max(
        cast(
            float,
            cast(dict[str, object], comparison[distribution_name])[
                "max_absolute_proportion_difference"
            ],
        )
        for distribution_name in ("task", "answer")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--b1-dir", type=Path, required=True)
    parser.add_argument("--b2-dir", type=Path, required=True)
    parser.add_argument("--b3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    datasets: dict[str, list[Record]] = {}
    available_pngs: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        image_dir = args.data_root / split / "png"
        available_pngs[split] = {path.name for path in image_dir.glob("*.png")}
        for source in ("human", "augmented"):
            path = _source_path(args.data_root, split, source)
            datasets[f"{split}_{source}"] = _load_records(path)

    train_human = datasets["train_human"]
    val_human = datasets["val_human"]
    train_prefix512 = train_human[:512]
    val_prefix50 = val_human[:50]
    b1_report = cast(
        dict[str, Any], json.loads((args.b1_dir / "report.json").read_text(encoding="utf-8"))
    )
    sample_groups = cast(list[list[int]], b1_report["training"]["sample_indices"])
    actual_indices = [index for group in sample_groups for index in group]
    actual_train = [train_human[index] for index in actual_indices]

    train_hashes = _image_hashes(train_human, args.data_root / "train" / "png")
    val_hashes = _image_hashes(val_human, args.data_root / "val" / "png")
    test_hashes = _image_hashes(datasets["test_human"], args.data_root / "test" / "png")
    train_shared_hashes = (set(train_hashes) & set(val_hashes)) | (
        set(train_hashes) & set(test_hashes)
    )
    val_shared_hashes = (set(val_hashes) & set(train_hashes)) | (set(val_hashes) & set(test_hashes))
    excluded_train_images = {
        filename for digest in train_shared_hashes for filename in train_hashes[digest]
    }
    excluded_val_images = {
        filename for digest in val_shared_hashes for filename in val_hashes[digest]
    }
    excluded_train_indices = {
        index
        for index, record in enumerate(train_human)
        if record["imgname"] in excluded_train_images
    }
    excluded_val_indices = {
        index for index, record in enumerate(val_human) if record["imgname"] in excluded_val_images
    }
    proposed_train_indices = deterministic_stratified_indices(
        train_human,
        sample_size=200,
        seed=42,
        excluded_indices=excluded_train_indices,
    )
    proposed_val_indices = deterministic_stratified_indices(
        val_human,
        sample_size=250,
        seed=20260806,
        excluded_indices=excluded_val_indices,
    )
    proposed_train = [train_human[index] for index in proposed_train_indices]
    proposed_val = [val_human[index] for index in proposed_val_indices]

    summaries = {name: summarize_records(records) for name, records in datasets.items()}
    summaries.update(
        {
            "train_human_prefix512": summarize_records(train_prefix512),
            "train_human_actual200": summarize_records(actual_train),
            "train_human_proposed_stratified200": summarize_records(proposed_train),
            "val_human_prefix50": summarize_records(val_prefix50),
            "val_human_proposed_stratified250": summarize_records(proposed_val),
        }
    )

    integrity: dict[str, object] = {}
    for name, records in datasets.items():
        split = name.split("_", maxsplit=1)[0]
        integrity[name] = _integrity_summary(
            records,
            image_dir=args.data_root / split / "png",
            available_pngs=available_pngs[split],
        )
    for split in ("train", "val", "test"):
        all_referenced = {
            record["imgname"]
            for source in ("human", "augmented")
            for record in datasets[f"{split}_{source}"]
        }
        cast(dict[str, object], integrity[f"{split}_human"])["png_files_in_directory"] = len(
            available_pngs[split]
        )
        cast(dict[str, object], integrity[f"{split}_human"])["unreferenced_png_count"] = len(
            available_pngs[split] - all_referenced
        )

    leakage = {
        "surface_overlap": {
            "train_vs_val": _surface_overlap(train_human, val_human),
            "train_vs_test": _surface_overlap(train_human, datasets["test_human"]),
            "val_vs_test": _surface_overlap(val_human, datasets["test_human"]),
        },
        "image_content_overlap": {
            "train_vs_val": _content_overlap(train_hashes, val_hashes),
            "train_vs_test": _content_overlap(train_hashes, test_hashes),
            "val_vs_test": _content_overlap(val_hashes, test_hashes),
        },
        "within_split_duplicate_content_hashes": {
            "train": sum(int(len(files) > 1) for files in train_hashes.values()),
            "val": sum(int(len(files) > 1) for files in val_hashes.values()),
            "test": sum(int(len(files) > 1) for files in test_hashes.values()),
        },
        "stratified_manifest_exclusions": {
            "train_record_indices": sorted(excluded_train_indices),
            "train_images": sorted(excluded_train_images),
            "val_record_indices": sorted(excluded_val_indices),
            "val_images": sorted(excluded_val_images),
        },
    }

    comparisons = {
        "train_prefix512_vs_full_human": _distribution_comparison(
            summaries["train_human"], summaries["train_human_prefix512"]
        ),
        "actual_train200_vs_full_human": _distribution_comparison(
            summaries["train_human"], summaries["train_human_actual200"]
        ),
        "val_prefix50_vs_full_human": _distribution_comparison(
            summaries["val_human"], summaries["val_human_prefix50"]
        ),
        "proposed_train200_vs_full_human": _distribution_comparison(
            summaries["train_human"], summaries["train_human_proposed_stratified200"]
        ),
        "proposed_val250_vs_full_human": _distribution_comparison(
            summaries["val_human"], summaries["val_human_proposed_stratified250"]
        ),
    }

    evaluation_paths = {
        "frozen_before": args.b3_dir / "evaluation_before.json",
        "b1_attention_lr2e4": args.b1_dir / "evaluation_step_000050.json",
        "b2_attention_ffn_lr2e4": args.b2_dir / "evaluation_step_000050.json",
        "b3_attention_lr5e5": args.b3_dir / "evaluation_step_000050.json",
    }
    evaluation_audit = _evaluation_audit(
        evaluation_paths,
        output_csv=args.output_dir / "val50_model_comparison.csv",
    )
    _write_training_manifest(
        train_human,
        actual_indices,
        args.output_dir / "actual_train200_manifest.csv",
    )
    _write_training_manifest(
        train_human,
        proposed_train_indices,
        args.output_dir / "proposed_stratified_train200_manifest.csv",
    )
    _write_training_manifest(
        val_human,
        proposed_val_indices,
        args.output_dir / "proposed_stratified_val250_manifest.csv",
    )

    val_shift = _max_shift(comparisons["val_prefix50_vs_full_human"])
    train_shift = _max_shift(comparisons["actual_train200_vs_full_human"])
    decision = {
        "thresholds": {
            "material_max_absolute_proportion_difference": 0.05,
            "minimum_next_validation_examples": 250,
        },
        "val50_material_distribution_shift": val_shift >= 0.05,
        "actual_train200_material_distribution_shift": train_shift >= 0.05,
        "val50_max_absolute_shift": val_shift,
        "actual_train200_max_absolute_shift": train_shift,
        "next_stage": "build_deterministic_stratified_val250_before_b4_training",
        "conditional_b4": (
            "joint_task_answer_stratified_train200_with_b3_recipe"
            if train_shift >= 0.05
            else "increase_unique_training_exposure_with_b3_recipe"
        ),
        "claim_boundary": (
            "Task labels are deterministic project heuristics, not official ChartQA labels. "
            "A 50-example validation result remains exploratory."
        ),
    }

    report = {
        "schema_version": "forgemm-chartqa-data-audit-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "summaries": summaries,
        "integrity": integrity,
        "leakage": leakage,
        "distribution_comparisons": comparisons,
        "evaluation_audit": evaluation_audit,
        "decision": decision,
        "artifacts": {
            "actual_train200_manifest": str(
                (args.output_dir / "actual_train200_manifest.csv").resolve()
            ),
            "val50_model_comparison": str(
                (args.output_dir / "val50_model_comparison.csv").resolve()
            ),
            "proposed_stratified_train200_manifest": str(
                (args.output_dir / "proposed_stratified_train200_manifest.csv").resolve()
            ),
            "proposed_stratified_val250_manifest": str(
                (args.output_dir / "proposed_stratified_val250_manifest.csv").resolve()
            ),
        },
    }
    _write_json(args.output_dir / "audit_report.json", report)
    print((args.output_dir / "audit_report.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
