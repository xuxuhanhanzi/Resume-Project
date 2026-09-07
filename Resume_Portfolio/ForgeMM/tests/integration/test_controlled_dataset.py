from __future__ import annotations

from pathlib import Path

from forgemm.controlled.dataset import (
    audit_controlled_dataset,
    create_controlled_dataset,
    load_controlled_records,
)
from forgemm.controlled.protocol import render_controlled_completion
from forgemm.evaluation.controlled_inference import (
    evaluate_controlled_rows,
    evaluate_controlled_sft_gate,
)


def test_controlled_dataset_is_auditable_and_has_gold_free_prompt_views(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = project / "datasets" / "ForgeMM-Controlled-v1-Lite"
    summary = create_controlled_dataset(
        output,
        project_root=project,
        counts={"train": 7, "val": 3, "test": 5},
        root_seed=11,
    )

    audit = audit_controlled_dataset(
        output,
        expected_counts={"train": 7, "val": 3, "test": 5},
    )

    assert summary["images"] == 15
    assert audit["decision"] == "pass"
    assert audit["counts"] == {"train": 7, "val": 3, "test": 5}
    assert all(audit["attacks_rejected"].values())
    assert (
        len((output / "views" / "answer_sft_train.jsonl").read_text(encoding="utf-8").splitlines())
        == 7
    )


def test_controlled_evaluation_keeps_oracle_separate_from_inference_prompts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = project / "datasets" / "ForgeMM-Controlled-v1-Lite"
    create_controlled_dataset(
        output,
        project_root=project,
        counts={"train": 3, "val": 3, "test": 3},
        root_seed=17,
    )
    records = load_controlled_records(output / "manifests" / "val_oracle.jsonl")
    structured = [
        {
            "sample_id": record.record_id,
            "images": [record.image_path],
            "response": render_controlled_completion(record),
        }
        for record in records
    ]
    baseline = [
        {
            "sample_id": record.record_id,
            "images": [record.image_path],
            "response": record.reference_answer,
        }
        for record in records
    ]

    metrics = evaluate_controlled_rows(structured, records)
    gate = evaluate_controlled_sft_gate(
        baseline, structured, records, bootstrap_samples=100, seed=3
    )

    assert metrics["full_pass"] == 1.0
    assert gate["decision"] == "advance"
