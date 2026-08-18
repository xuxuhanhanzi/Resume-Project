from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forgemm.data.chartqa import ChartQALoader
from forgemm.data.chartqapro import ChartQAProLoader


def _make_chartqa(root: Path) -> None:
    split = root / "train"
    for directory in ("png", "annotations", "tables"):
        (split / directory).mkdir(parents=True, exist_ok=True)
    (split / "png" / "chart.png").write_bytes(b"png")
    (split / "annotations" / "chart.json").write_text("{}", encoding="utf-8")
    (split / "tables" / "chart.csv").write_text("Year,Value\n2020,10\n", encoding="utf-8")
    payload = [{"imgname": "chart.png", "query": "What is the value?", "label": "10"}]
    (split / "train_human.json").write_text(json.dumps(payload), encoding="utf-8")


def test_chartqa_loader_resolves_assets_and_stable_id(tmp_path: Path) -> None:
    _make_chartqa(tmp_path)

    record = next(ChartQALoader(tmp_path).iter_records("train"))

    assert record.record_id == "chartqa:train:human:0:chart"
    assert record.table_path.name == "chart.csv"
    assert record.answer == "10"


def test_chartqa_loader_rejects_missing_asset(tmp_path: Path) -> None:
    _make_chartqa(tmp_path)
    (tmp_path / "train" / "png" / "chart.png").unlink()

    with pytest.raises(FileNotFoundError, match="missing_image"):
        next(ChartQALoader(tmp_path).iter_records("train"))


def test_chartqapro_loader_is_test_only_and_omits_image(tmp_path: Path) -> None:
    path = tmp_path / "pro.parquet"
    table = pa.table(
        {
            "Question": [["Q1", "Q2"]],
            "Answer": [["A1", "A2"]],
            "Question Type": ["Factoid"],
            "image": [b"large-image-placeholder"],
            "Year": [["NO"]],
            "Paragraph": ["context"],
        }
    )
    pq.write_table(table, path)
    loader = ChartQAProLoader(path)

    records = list(loader.iter_records())

    assert [record.record_id for record in records] == [
        "chartqapro:test:0:0",
        "chartqapro:test:0:1",
    ]
    assert not hasattr(records[0], "image")
    with pytest.raises(ValueError, match="chartqapro_test_only"):
        next(loader.iter_records("train"))


def test_chartqapro_loader_retains_invalid_slot_with_reason(tmp_path: Path) -> None:
    path = tmp_path / "pro.parquet"
    pq.write_table(
        pa.table(
            {
                "Question": [["valid", ""]],
                "Answer": [["answer", "orphan"]],
                "Question Type": ["Factoid"],
                "image": [b"image"],
                "Year": [["NO"]],
                "Paragraph": [""],
            }
        ),
        path,
    )

    records = list(ChartQAProLoader(path).iter_records())

    assert records[0].valid is True
    assert records[1].valid is False
    assert records[1].exclusion_reason == "empty_question"
