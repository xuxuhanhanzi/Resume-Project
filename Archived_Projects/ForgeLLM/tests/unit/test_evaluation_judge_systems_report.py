from __future__ import annotations

from pathlib import Path

import pytest

from forgellm.evaluation.behavioral import BehaviorResult
from forgellm.evaluation.judge import (
    build_blind_comparisons,
    position_consistency,
    write_blind_package,
)
from forgellm.evaluation.report import EvidenceClaim, audit_claim_evidence, model_acceptance
from forgellm.evaluation.systems import (
    approximate_decode_tokens_per_second,
    summarize_timings,
    theoretical_kv_cache_bytes,
)


def _behavior(*, passed: bool, tokens: int = 3) -> BehaviorResult:
    return BehaviorResult(passed, passed, False, None, 0.0, 0.0, tokens, False)


def test_blind_package_is_deterministic_and_hides_models(tmp_path: Path) -> None:
    left = {f"c{i}": f"left{i}" for i in range(4)}
    right = {f"c{i}": f"right{i}" for i in range(4)}
    prompts = {f"c{i}": f"prompt{i}" for i in range(4)}
    first = build_blind_comparisons(
        left,
        right,
        prompts,
        left_model="q1",
        right_model="q2",
        count=3,
        seed=7,
    )
    second = build_blind_comparisons(
        left,
        right,
        prompts,
        left_model="q1",
        right_model="q2",
        count=3,
        seed=7,
    )
    assert first == second
    public, private, html = write_blind_package(tmp_path / "blind", first)
    assert "model_a" not in public.read_text(encoding="utf-8")
    assert "prompt" in public.read_text(encoding="utf-8")
    assert "model_a" in private.read_text(encoding="utf-8")
    assert html.is_file()


def test_rule_judge_is_position_consistent() -> None:
    report = position_consistency([(_behavior(passed=True), _behavior(passed=False))])
    assert report["position_consistency_rate"] == 1.0


def test_timing_summary_and_decode_rate() -> None:
    summary = summarize_timings([1, 2, 3, 4, 5])
    assert summary.median_seconds == 3
    assert (
        approximate_decode_tokens_per_second(
            batch_size=2, generated_tokens=6, total_seconds=3.0, first_token_seconds=1.0
        )
        == 5.0
    )
    with pytest.raises(ValueError):
        approximate_decode_tokens_per_second(
            batch_size=1,
            generated_tokens=2,
            total_seconds=1.0,
            first_token_seconds=-1.0,
        )


def test_kv_cache_formula_includes_key_and_value() -> None:
    assert (
        theoretical_kv_cache_bytes(
            layers=2,
            batch_size=3,
            sequence_length=4,
            key_value_heads=5,
            head_dim=6,
            bytes_per_element=2,
        )
        == 2 * 2 * 3 * 4 * 5 * 6 * 2
    )


def test_acceptance_has_separate_gates_and_no_total_score() -> None:
    result = model_acceptance(
        strict_success_rate=0.8,
        minimum_strict_success_rate=0.5,
        retention_loss_ratio=1.2,
        maximum_retention_loss_ratio=1.1,
        repetition_rate=0.0,
        maximum_repetition_rate=0.25,
        system_matrix_completed=True,
    )
    assert result["model_behavior_accepted"] is False
    assert "total_score" not in result
    assert result["gates"] == {
        "correctness": True,
        "retention": False,
        "stability": True,
        "efficiency": True,
    }


def test_claim_audit_reports_missing_evidence(tmp_path: Path) -> None:
    existing = tmp_path / "evidence.json"
    existing.write_text("{}", encoding="utf-8")
    claims = [EvidenceClaim("c", "statement", ("evidence.json", "missing.json"), "boundary")]
    audit = audit_claim_evidence(claims, project_root=tmp_path)
    assert audit["passed"] is False
    assert audit["missing_evidence"] == {"c": ["missing.json"]}
