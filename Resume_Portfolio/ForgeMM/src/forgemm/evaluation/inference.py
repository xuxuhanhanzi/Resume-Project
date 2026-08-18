"""Metrics for ms-swift JSONL inference outputs with embedded gold labels."""

from __future__ import annotations

from collections import Counter
from typing import Any

from forgemm.reasoning.parser import parse_prediction
from forgemm.rewards.scoring import score_completion


def evaluate_inference_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter[str]()
    parse_errors = Counter[str]()
    evidence_f1_sum = 0.0
    for row in rows:
        response = str(row["response"])
        gold = parse_prediction(str(row["labels"]))
        bundle = score_completion(response, gold.answer, gold.evidence)
        totals["samples"] += 1
        totals["format_pass"] += bundle.parse_error is None
        totals["task_pass"] += bundle.task == 1.0
        evidence_f1_sum += bundle.evidence
        totals["evidence_pass"] += bundle.evidence == 1.0
        totals["operation_pass"] += bundle.operation == 1.0
        fcr_pass = bundle.task == 1.0 and bundle.evidence == 1.0 and bundle.operation == 1.0
        totals["fcr_pass"] += fcr_pass
        totals["inconsistent_pass"] += bundle.task == 1.0 and not fcr_pass
        totals["truncation_suspected"] += (
            bundle.parse_error == "invalid_protocol" and "</answer>" not in response
        )
        if bundle.parse_error is not None:
            parse_errors[bundle.parse_error] += 1

    samples = totals["samples"]
    if samples == 0:
        raise ValueError("inference result is empty")
    return {
        "schema_version": "forgemm-infer-eval-v1",
        "samples": samples,
        "format_compliance": totals["format_pass"] / samples,
        "task_accuracy": totals["task_pass"] / samples,
        "evidence_f1": evidence_f1_sum / samples,
        "evidence_exact": totals["evidence_pass"] / samples,
        "operation_consistency": totals["operation_pass"] / samples,
        "fcr": totals["fcr_pass"] / samples,
        "inconsistency_rate": totals["inconsistent_pass"] / samples,
        "truncation_suspected_rate": totals["truncation_suspected"] / samples,
        "counts": dict(totals),
        "parse_errors": dict(parse_errors),
    }
