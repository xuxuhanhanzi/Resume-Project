"""Evaluation and pre-GRPO gate for independent controlled visual oracles."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from forgemm.controlled.protocol import ControlledRecord, verify_controlled_completion
from forgemm.evaluation.statistics import paired_bootstrap_delta
from forgemm.rewards.answer import answer_reward


def evaluate_controlled_rows(
    inference_rows: list[dict[str, Any]], oracle_records: list[ControlledRecord]
) -> dict[str, Any]:
    """Score ordered inference output without embedding oracle labels in prompts."""

    _verify_alignment(inference_rows, oracle_records)
    totals = Counter[str]()
    errors = Counter[str]()
    iou_sum = 0.0
    for row, record in zip(inference_rows, oracle_records, strict=True):
        response = _response(row)
        verdict = verify_controlled_completion(record, response)
        totals["samples"] += 1
        totals["answer_pass"] += verdict.answer_ok
        totals["format_pass"] += verdict.format_ok
        totals["evidence_pass"] += verdict.evidence_ok
        totals["operation_pass"] += verdict.operation_ok
        totals["full_pass"] += verdict.full_pass
        iou_sum += verdict.visual_iou_min
        for error in verdict.errors:
            errors[error] += 1
    samples = totals["samples"]
    return {
        "schema_version": "forgemm-controlled-infer-eval-v1",
        "samples": samples,
        "answer_accuracy": totals["answer_pass"] / samples,
        "format_compliance": totals["format_pass"] / samples,
        "visual_evidence_exact": totals["evidence_pass"] / samples,
        "operation_consistency": totals["operation_pass"] / samples,
        "visual_iou_min_mean": iou_sum / samples,
        "full_pass": totals["full_pass"] / samples,
        "counts": dict(totals),
        "errors": dict(errors),
    }


def evaluate_controlled_sft_gate(
    baseline_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    oracle_records: list[ControlledRecord],
    *,
    min_format: float = 0.99,
    noninferiority_pp: float = -0.01,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    """Use val only to decide whether structured SFT may enter GRPO."""

    _verify_alignment(baseline_rows, oracle_records)
    _verify_alignment(structured_rows, oracle_records)
    baseline = np.asarray(
        [
            answer_reward(_response(row), record.reference_answer)
            for row, record in zip(baseline_rows, oracle_records, strict=True)
        ],
        dtype=np.float64,
    )
    structured = np.asarray(
        [
            float(verify_controlled_completion(record, _response(row)).answer_ok)
            for row, record in zip(structured_rows, oracle_records, strict=True)
        ],
        dtype=np.float64,
    )
    metrics = evaluate_controlled_rows(structured_rows, oracle_records)
    interval = paired_bootstrap_delta(baseline, structured, samples=bootstrap_samples, seed=seed)
    format_pass = float(metrics["format_compliance"]) >= min_format
    quality_pass = float(interval["ci_low"]) >= noninferiority_pp
    return {
        "schema_version": "forgemm-controlled-sft-gate-v1",
        "claim_boundary": "frozen controlled validation only; test remains untouched",
        "samples": len(oracle_records),
        "baseline_answer_accuracy": float(np.mean(baseline)),
        "structured_answer_accuracy": float(np.mean(structured)),
        "structured_format_compliance": metrics["format_compliance"],
        "structured_visual_full_pass": metrics["full_pass"],
        "answer_accuracy_delta": interval,
        "thresholds": {
            "min_format": min_format,
            "noninferiority_pp": noninferiority_pp,
        },
        "format_pass": format_pass,
        "quality_pass": quality_pass,
        "decision": "advance" if format_pass and quality_pass else "stop",
    }


def _verify_alignment(
    inference_rows: list[dict[str, Any]], records: list[ControlledRecord]
) -> None:
    if not inference_rows or len(inference_rows) != len(records):
        raise ValueError("controlled_inference_and_oracle_must_be_nonempty_and_aligned")
    for index, (row, record) in enumerate(zip(inference_rows, records, strict=True)):
        sample_id = row.get("sample_id")
        if sample_id is not None and str(sample_id) != record.record_id:
            raise ValueError(f"controlled_sample_id_mismatch:{index}")
        images = row.get("images")
        if images is not None and _image_paths(images) != [record.image_path]:
            raise ValueError(f"controlled_image_mismatch:{index}")


def _response(row: dict[str, Any]) -> str:
    response = row.get("response")
    if not isinstance(response, str):
        raise ValueError("controlled_inference_missing_response")
    return response


def _image_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item.get("path", "")) if isinstance(item, dict) else str(item) for item in value]
