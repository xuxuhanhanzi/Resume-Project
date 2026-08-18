from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_frozen_val_matrix.py"
    spec = importlib.util.spec_from_file_location("summarize_frozen_val_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_uses_preregistered_tie_breakers(tmp_path: Path) -> None:
    module = _load_script()
    base = {
        "format_compliance": 0.99,
        "task_accuracy": 0.80,
        "evidence_f1": 0.65,
        "evidence_exact": 0.60,
        "operation_consistency": 0.95,
        "fcr": 0.58,
        "inconsistency_rate": 0.22,
        "truncation_suspected_rate": 0.01,
    }
    for seed, task in ((17, 0.80), (42, 0.82), (2026, 0.82)):
        payload = {**base, "task_accuracy": task}
        (tmp_path / f"frozen_val_e5_dynamic_seed{seed}.metrics.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    summary = module.summarize(tmp_path)
    assert summary["aggregates"]["e5_dynamic"]["runs"] == 3
    assert summary["selected_checkpoints"]["e5_dynamic"]["seed"] == 42
