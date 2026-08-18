from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from forgemm.data.chartqa import ChartQALoader
from forgemm.data.evidence_store import (
    EvidenceStore,
    build_table_evidence,
    derive_supported_label,
    gold_evidence,
)
from forgemm.data.schemas import EvidenceCell, EvidenceRecord


def _record() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="chartqa:train:human:0:chart",
        dataset="ChartQA",
        split="train",
        source="human",
        image_path="train/png/chart.png",
        image_sha256="abc",
        question="Q",
        reference_answer="10",
        cells=(EvidenceCell("t1c1", "2020", "Value", "10"),),
    )


def test_evidence_store_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "store.jsonl")

    assert store.write([_record()]) == 1
    assert list(store.iter_records()) == [_record()]
    with pytest.raises(FileExistsError, match="evidence_store_exists"):
        store.write([_record()])


def test_evidence_store_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "store.jsonl"
    payload = _record().to_dict()
    payload["schema_version"] = "999"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported_schema_version"):
        list(EvidenceStore(path).iter_records())


def test_build_table_evidence_does_not_invent_operation(tmp_path: Path) -> None:
    split = tmp_path / "train"
    for directory in ("png", "annotations", "tables"):
        (split / directory).mkdir(parents=True, exist_ok=True)
    (split / "png" / "chart.png").write_bytes(b"png")
    annotation = {
        "models": [
            {
                "name": "Favorable",
                "x": ["2015"],
                "y": ["38"],
            },
            {
                "name": "Unfavorable",
                "x": ["2015"],
                "y": ["54"],
            },
        ]
    }
    (split / "annotations" / "chart.json").write_text(json.dumps(annotation), encoding="utf-8")
    (split / "tables" / "chart.csv").write_text(
        "Year,Favorable,Unfavorable\n2015,38,54\n", encoding="utf-8"
    )
    qa = [
        {
            "imgname": "chart.png",
            "query": "What was Favorable in 2015?",
            "label": "38",
        }
    ]
    (split / "train_human.json").write_text(json.dumps(qa), encoding="utf-8")
    source = next(ChartQALoader(tmp_path).iter_records("train"))

    evidence = build_table_evidence(source)

    assert evidence.cells[0] == EvidenceCell("t1c1", "2015", "Favorable", "38")
    assert evidence.conflict_status == "exact"
    assert evidence.evidence_mask is False
    assert evidence.gold_operation is None
    assert evidence.operation_mask is False

    labelled = derive_supported_label(evidence)
    assert labelled.evidence_mask is True
    assert labelled.operation_mask is True
    assert labelled.gold_operation is not None
    assert labelled.gold_operation.name == "lookup"
    assert labelled.schema_version == "1.3.0"
    assert gold_evidence(labelled) == (EvidenceCell("t1c1", "2015", "Favorable", "38"),)


def test_derive_supported_label_requires_unique_executable_reasoning() -> None:
    base = _record()
    ambiguous = replace(
        base,
        question="What value is shown in 2020?",
        cells=(
            EvidenceCell("e1", "2020", "First", "10"),
            EvidenceCell("e2", "2020", "Second", "10"),
        ),
        conflict_status="exact",
        exclusion_reason=None,
    )
    difference = replace(
        base,
        question="What is the difference between 2020 and 2021 for Value?",
        reference_answer="6",
        cells=(
            EvidenceCell("e1", "2020", "Value", "10"),
            EvidenceCell("e2", "2021", "Value", "4"),
        ),
        conflict_status="exact",
        exclusion_reason=None,
    )

    assert derive_supported_label(ambiguous).operation_mask is False
    labelled = derive_supported_label(difference)
    assert labelled.operation_mask is True
    assert labelled.gold_operation is not None
    assert labelled.gold_operation.name == "difference"


def test_derive_supported_argmax_and_count_labels() -> None:
    base = _record()
    cells = (
        EvidenceCell("e1", "Lombardy", "ICUs", "861"),
        EvidenceCell("e2", "Lazio", "ICUs", "571"),
        EvidenceCell("e3", "Veneto", "ICUs", "494"),
    )
    maximum = replace(
        base,
        question="Which region has the highest number of ICUs?",
        reference_answer="Lombardy",
        cells=cells,
        conflict_status="exact",
        exclusion_reason=None,
    )
    count = replace(
        base,
        question="How many data points are collected?",
        reference_answer="3",
        cells=cells,
        conflict_status="exact",
        exclusion_reason=None,
    )

    labelled_maximum = derive_supported_label(maximum)
    labelled_count = derive_supported_label(count)

    assert labelled_maximum.gold_operation is not None
    assert labelled_maximum.gold_operation.name == "argmax"
    assert labelled_count.gold_operation is not None
    assert labelled_count.gold_operation.name == "count"


def test_lookup_label_uses_exact_cell_value_not_relaxed_metric() -> None:
    base = _record()
    record = replace(
        base,
        question="What was Spain's rate in 2018?",
        reference_answer="71.4",
        cells=(
            EvidenceCell("e1", "Spain", "2018", "71.4"),
            EvidenceCell("e2", "Spain", "2019", "73.8"),
        ),
        conflict_status="exact",
        exclusion_reason=None,
    )

    labelled = derive_supported_label(record)

    assert labelled.gold_evidence_ids == ("e1",)
    assert labelled.gold_operation is not None
    assert labelled.gold_operation.name == "lookup"


def test_gold_operation_rejects_chartqa_relaxed_but_unfaithful_result() -> None:
    base = _record()
    misleading = replace(
        base,
        question="What is the sum of A and B?",
        reference_answer="161",
        cells=(
            EvidenceCell("e1", "A", "Value", "31"),
            EvidenceCell("e2", "B", "Value", "123"),
        ),
        conflict_status="exact",
        exclusion_reason=None,
    )

    labelled = derive_supported_label(misleading)

    assert labelled.operation_mask is False
    assert labelled.exclusion_reason == "no_unique_supported_label"
