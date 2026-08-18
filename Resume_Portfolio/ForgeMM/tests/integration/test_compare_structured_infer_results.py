from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/compare_structured_infer_results.py"
    spec = importlib.util.spec_from_file_location("compare_structured_infer_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLD = """<evidence>
e1=cell(row="2015", column="Value", value=38)
</evidence>
<operation>
lookup(ref=e1)
</operation>
<answer>
38
</answer>"""


def test_compare_rows_reports_positive_candidate_delta() -> None:
    module = _load_script()
    identity = {"labels": GOLD, "images": ["chart.png"], "messages": []}
    result = module.compare_rows(
        [{**identity, "response": "invalid"}, {**identity, "response": GOLD}],
        [{**identity, "response": GOLD}, {**identity, "response": GOLD}],
        bootstrap_samples=100,
    )
    assert result["pairs"] == 2
    assert result["bootstrap"]["fcr"]["mean_delta"] == 0.5
    assert result["mcnemar"]["fcr"]["candidate_only_correct"] == 1
