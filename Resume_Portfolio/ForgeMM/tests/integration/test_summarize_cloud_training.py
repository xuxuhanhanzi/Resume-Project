from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_cloud_training.py"
    spec = importlib.util.spec_from_file_location("summarize_cloud_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_reward_variance_and_clipping(tmp_path: Path) -> None:
    module = _load_script()
    run = tmp_path / "formal50_e3_task_seed17"
    run.mkdir()
    rows = [
        {
            "loss": 0.1,
            "reward": 0.5,
            "reward_std": 0.7,
            "completions/mean_length": 64,
            "completions/clipped_ratio": 0.5,
            "memory(GiB)": 9.5,
        },
        {"loss": 0.2, "memory(GiB)": 9.0},
    ]
    (run / "logging.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = module.summarize(tmp_path)
    assert result["all_loss_finite"] is True
    assert result["total_nonzero_reward_std_records"] == 1
    assert result["runs"][0]["completion_clipped_nonzero_records"] == 1
