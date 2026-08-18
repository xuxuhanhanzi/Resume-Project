from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/merge_infer_metadata.py"
    spec = importlib.util.spec_from_file_location("merge_infer_metadata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_rows_restores_group_metadata() -> None:
    module = _load_script()
    dataset = [
        {
            "sample_id": "one",
            "source": "chartqapro",
            "question_type": "Factoid",
            "images": [{"bytes": None, "path": "one.jpg"}],
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
    ]
    inference = [
        {
            "response": "answer",
            "labels": "answer",
            "images": ["one.jpg"],
            "messages": [{"role": "user", "content": "question"}],
        }
    ]
    result = module.merge_rows(dataset, inference)
    assert result[0]["sample_id"] == "one"
    assert result[0]["question_type"] == "Factoid"
