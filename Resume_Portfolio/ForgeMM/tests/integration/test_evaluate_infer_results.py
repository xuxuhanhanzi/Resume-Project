from __future__ import annotations

from forgemm.evaluation.inference import evaluate_inference_rows

GOLD = """<evidence>
e1=cell(row="2015", column="Value", value=38)
e2=cell(row="2016", column="Value", value=43)
</evidence>
<operation>
difference(ref=e2, ref=e1)
</operation>
<answer>
5
</answer>"""


def test_evaluate_rows_reports_decomposed_metrics() -> None:
    result = evaluate_inference_rows(
        [
            {"response": GOLD, "labels": GOLD},
            {"response": "not structured", "labels": GOLD},
        ]
    )

    assert result["samples"] == 2
    assert result["format_compliance"] == 0.5
    assert result["task_accuracy"] == 0.5
    assert result["evidence_f1"] == 0.5
    assert result["evidence_exact"] == 0.5
    assert result["operation_consistency"] == 0.5
    assert result["fcr"] == 0.5
    assert result["inconsistency_rate"] == 0.0
    assert result["truncation_suspected_rate"] == 0.5
    assert result["parse_errors"] == {"invalid_protocol": 1}
