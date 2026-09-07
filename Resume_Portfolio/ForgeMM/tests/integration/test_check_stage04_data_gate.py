from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/check_stage04_data_gate.py"
    spec = importlib.util.spec_from_file_location("check_stage04_data_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_data_gate_advances_only_when_all_preregistered_counts_are_met() -> None:
    module = _load_script()
    result = module.evaluate_data_gate(
        {"unique_strict_records": 1763},
        {"strict_question_records": 200},
        {"strict_question_records": 500},
        min_train=1500,
        min_val=200,
        min_test=500,
    )

    assert result["decision"] == "advance"
    assert result["failed"] == {}


def test_data_gate_stops_without_rewriting_thresholds() -> None:
    module = _load_script()
    result = module.evaluate_data_gate(
        {"unique_strict_records": 1763},
        {"strict_question_records": 95},
        {"strict_question_records": 108},
        min_train=1500,
        min_val=200,
        min_test=500,
    )

    assert result["decision"] == "stop"
    assert result["failed"] == {"val_strict_records": 95, "test_strict_records": 108}
