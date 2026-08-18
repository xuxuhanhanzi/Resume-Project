from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_cloud_final.py"
    spec = importlib.util.spec_from_file_location("summarize_cloud_final", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_run(run_dir: Path, *, task_exit: int = 0) -> None:
    for name in (
        "formal_matrix_driver.exit",
        "frozen_val_matrix_driver.exit",
        "frozen_val_summary.exit",
        "frozen_test_strict_driver.exit",
        "frozen_baseline_driver.exit",
    ):
        (run_dir / name).write_text("0\n", encoding="utf-8")
    (run_dir / "frozen_task_driver.exit").write_text(f"{task_exit}\n", encoding="utf-8")
    (run_dir / "frozen_val_e3_task_seed17.metrics.json").write_text(
        json.dumps({"samples": 3, "task_accuracy": 2 / 3}), encoding="utf-8"
    )
    (run_dir / "formal_training_summary.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "frozen_val_summary.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "frozen_test_strict_e3_vs_e5.json").write_text("{}\n", encoding="utf-8")


def test_summary_requires_successful_drivers_and_hashes_metrics(tmp_path: Path) -> None:
    module = _load_script()
    _write_minimal_run(tmp_path)
    result = module.summarize(tmp_path)
    identity = result["evidence_identities"]["frozen_val_e3_task_seed17.metrics.json"]
    assert result["driver_exit_codes"]["frozen_task_driver.exit"] == 0
    assert identity["bytes"] > 0
    assert len(identity["sha256"]) == 64


def test_summary_rejects_failed_driver(tmp_path: Path) -> None:
    module = _load_script()
    _write_minimal_run(tmp_path, task_exit=7)
    with pytest.raises(RuntimeError, match="cloud driver failure"):
        module.summarize(tmp_path)
