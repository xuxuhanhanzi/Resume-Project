from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/check_structured_sft_gate.py"
    spec = importlib.util.spec_from_file_location("check_structured_sft_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GOLD = """<evidence>
e1=cell(row="2016", column="Value", value=43)
</evidence>
<operation>
lookup(ref=e1)
</operation>
<answer>
43
</answer>"""


def test_e2_gate_allows_a_format_valid_noninferior_structured_control() -> None:
    module = _load_script()
    rows = [
        {"response": "43", "labels": GOLD, "images": ["a.png"]},
        {"response": "wrong", "labels": GOLD, "images": ["b.png"]},
    ]
    structured = [
        {"response": GOLD, "labels": GOLD, "images": ["a.png"]},
        {"response": "wrong", "labels": GOLD, "images": ["b.png"]},
    ]

    result = module.evaluate_gate(
        rows,
        structured,
        min_format=0.5,
        noninferiority_pp=-0.01,
        bootstrap_samples=100,
        seed=17,
    )

    assert result["decision"] == "advance"
    assert result["format_pass"]
    assert result["quality_pass"]


def test_e2_gate_stops_when_structured_format_is_below_threshold() -> None:
    module = _load_script()
    rows = [{"response": "43", "labels": GOLD, "images": ["a.png"]}]
    structured = [{"response": "43", "labels": GOLD, "images": ["a.png"]}]

    result = module.evaluate_gate(
        rows,
        structured,
        min_format=0.99,
        noninferiority_pp=-0.01,
        bootstrap_samples=100,
        seed=17,
    )

    assert result["decision"] == "stop"
    assert not result["format_pass"]
