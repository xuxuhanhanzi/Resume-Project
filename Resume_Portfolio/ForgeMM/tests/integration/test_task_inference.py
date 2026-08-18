from __future__ import annotations

import pytest

from forgemm.evaluation.task_inference import evaluate_task_rows, extract_answer

VALID = """<evidence>
e1=cell(row="2015", column="Value", value=38)
</evidence>
<operation>
lookup(ref=e1)
</operation>
<answer>
38
</answer>"""


def test_extract_answer_separates_answer_quality_from_format() -> None:
    assert extract_answer(VALID) == ("38", None)
    assert extract_answer("prefix <answer>38</answer>") == ("38", "invalid_protocol")
    assert extract_answer("38") == ("38", "invalid_protocol")


def test_evaluate_task_rows_reports_type_breakdown() -> None:
    result = evaluate_task_rows(
        [
            {"response": VALID, "labels": "38.0", "question_type": "Factoid"},
            {"response": "no", "labels": "yes", "question_type": "Factoid"},
        ]
    )
    assert result["task_accuracy"] == pytest.approx(0.5)
    assert result["format_compliance"] == pytest.approx(0.5)
    assert result["by_question_type"]["Factoid"]["samples"] == 2
