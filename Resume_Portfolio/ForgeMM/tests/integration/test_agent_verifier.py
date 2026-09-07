from __future__ import annotations

from dataclasses import replace

from forgemm.agent.state import (
    AgentAction,
    AgentBudget,
    EvidenceItem,
    Proposal,
    TrajectoryState,
    apply_action,
)
from forgemm.agent.verifier import TrajectoryVerifier
from forgemm.data.schemas import EvidenceCell, EvidenceRecord, Operation, OperationArgument


def _record(*, evidence_mask: bool = True) -> EvidenceRecord:
    return EvidenceRecord(
        record_id="record-1",
        dataset="ChartQA",
        split="test",
        source="human",
        image_path="image.png",
        image_sha256="sha256:image",
        question="What is the difference?",
        reference_answer="5",
        cells=(
            EvidenceCell("gold-a", "2015", "Value", "38"),
            EvidenceCell("gold-b", "2016", "Value", "43"),
        ),
        gold_evidence_ids=("gold-a", "gold-b"),
        evidence_mask=evidence_mask,
        operation_mask=True,
    )


def _state(*, second_value: str = "43") -> TrajectoryState:
    state = TrajectoryState(
        episode_id="episode-1",
        question_id="record-1",
        image_id="sha256:image",
        processor_version="processor-1",
        budget=AgentBudget(max_steps=6, max_tool_calls=2, max_repairs=1),
    )
    first = EvidenceItem("ev-1", "value", "2015", "Value", "38", "gold-a", 1.0, "a1")
    state = apply_action(state, AgentAction("a1", "EXTRACT", "2015", "gold-a"), evidence=first)
    second = EvidenceItem("ev-2", "value", "2016", "Value", second_value, "gold-b", 1.0, "a2")
    state = apply_action(state, AgentAction("a2", "EXTRACT", "2016", "gold-b"), evidence=second)
    proposal = Proposal(
        ("ev-1", "ev-2"),
        Operation(
            "difference",
            (OperationArgument("ref", "ev-2", True), OperationArgument("ref", "ev-1", True)),
        ),
        "5",
    )
    state = apply_action(state, AgentAction("a3", "PROPOSE"), proposal=proposal)
    return apply_action(state, AgentAction("a4", "FINISH"))


def test_verifier_requires_answer_evidence_operation_format_and_budget() -> None:
    verdict = TrajectoryVerifier().verify(_record(), _state())

    assert verdict.full_pass
    assert verdict.errors == ()
    assert verdict.cost.steps == 4
    assert verdict.cost.tool_calls == 2


def test_verifier_rejects_fabricated_evidence_without_revealing_gold() -> None:
    verdict = TrajectoryVerifier().verify(_record(), _state(second_value="99"))

    assert verdict.answer_ok
    assert verdict.evidence_ok is False
    assert verdict.operation_ok is False
    assert verdict.full_pass is False
    assert "EVIDENCE_MISMATCH" in verdict.errors
    assert all("43" not in error and "38" not in error for error in verdict.errors)


def test_verifier_flags_manual_budget_bypass_and_marks_masked_evidence_inapplicable() -> None:
    state = _state()
    over_budget = replace(state, budget=AgentBudget(max_steps=3, max_tool_calls=1, max_repairs=0))
    verdict = TrajectoryVerifier().verify(_record(evidence_mask=False), over_budget)

    assert verdict.evidence_ok is None
    assert verdict.budget_ok is False
    assert verdict.full_pass is False
    assert {"STEP_BUDGET_EXCEEDED", "TOOL_BUDGET_EXCEEDED"} <= set(verdict.errors)
