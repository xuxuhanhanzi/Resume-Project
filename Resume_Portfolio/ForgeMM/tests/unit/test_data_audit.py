from __future__ import annotations

import json
from pathlib import Path

from forgemm.data.audit import compare_table_annotation


def _annotation(path: Path, y_values: list[str]) -> None:
    payload = {
        "models": [
            {
                "name": "Favorable",
                "x": ["2015", "2016"],
                "y": y_values,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_table_annotation_exact_and_value_conflict(tmp_path: Path) -> None:
    table = tmp_path / "table.csv"
    annotation = tmp_path / "annotation.json"
    table.write_text("Year,Favorable\n2015,38\n2016,43\n", encoding="utf-8")
    _annotation(annotation, ["38", "43"])

    assert compare_table_annotation(table, annotation) == "exact"

    _annotation(annotation, ["38", "99"])
    assert compare_table_annotation(table, annotation) == "conflict_value"


def test_compare_table_annotation_unverifiable_layout(tmp_path: Path) -> None:
    table = tmp_path / "table.csv"
    annotation = tmp_path / "annotation.json"
    table.write_text("Unknown,Other\nA,1\n", encoding="utf-8")
    _annotation(annotation, ["38", "43"])

    assert compare_table_annotation(table, annotation) == "unverifiable_layout"
