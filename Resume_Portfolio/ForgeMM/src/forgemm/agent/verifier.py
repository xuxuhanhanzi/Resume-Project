"""Independent, no-leak verifier for bounded evidence-agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from forgemm.agent.state import AgentAction, EvidenceItem, TrajectoryState
from forgemm.data.schemas import EvidenceCell, EvidenceRecord
from forgemm.reasoning.executor import execute
from forgemm.reasoning.normalizer import (
    answers_match,
    decimal_to_text,
    normalize_number,
    normalize_text,
)


@dataclass(frozen=True)
class VerificationCost:
    """Observed budget consumption, separate from the model-visible state."""

    steps: int
    tool_calls: int
    repairs: int


@dataclass(frozen=True)
class VerificationVerdict:
    """Component-level result used for reward, audits, and failure bucketing."""

    answer_ok: bool
    evidence_ok: bool | None
    operation_ok: bool
    format_ok: bool
    budget_ok: bool
    cost: VerificationCost
    errors: tuple[str, ...]

    @property
    def full_pass(self) -> bool:
        evidence_ok = self.evidence_ok is not False
        return (
            self.answer_ok
            and evidence_ok
            and self.operation_ok
            and self.format_ok
            and self.budget_ok
        )


class TrajectoryVerifier:
    """Evaluate a completed state without exposing gold answers to the trajectory."""

    version = "forgemm-agent-verifier-1.0.0"

    def verify(self, record: EvidenceRecord, state: TrajectoryState) -> VerificationVerdict:
        """Return stable failure categories that do not contain gold values or locations."""

        format_errors = list(_format_errors(record, state))
        errors = list(format_errors)
        cost = VerificationCost(
            steps=len(state.actions), tool_calls=state.tool_calls, repairs=state.repairs
        )
        budget_errors = _budget_errors(state)
        budget_ok = not budget_errors
        if not budget_ok:
            errors.extend(budget_errors)
        format_ok = not format_errors
        if not format_ok or state.proposal is None:
            return VerificationVerdict(
                answer_ok=False,
                evidence_ok=None if not record.evidence_mask else False,
                operation_ok=False,
                format_ok=format_ok,
                budget_ok=budget_ok,
                cost=cost,
                errors=tuple(dict.fromkeys(errors)),
            )

        answer_ok = answers_match(state.proposal.answer, record.reference_answer)
        evidence_ok, evidence_errors = _verify_evidence(
            record, state.ledger, state.proposal.evidence_ids
        )
        operation_ok = _verify_operation(state)
        if not answer_ok:
            errors.append("ANSWER_MISMATCH")
        errors.extend(evidence_errors)
        if not operation_ok:
            errors.append("INVALID_OPERATION")
        return VerificationVerdict(
            answer_ok=answer_ok,
            evidence_ok=evidence_ok,
            operation_ok=operation_ok,
            format_ok=True,
            budget_ok=budget_ok,
            cost=cost,
            errors=tuple(dict.fromkeys(errors)),
        )


def _format_errors(record: EvidenceRecord, state: TrajectoryState) -> tuple[str, ...]:
    errors: list[str] = []
    if state.question_id != record.record_id:
        errors.append("QUESTION_ID_MISMATCH")
    if state.verifier_version != TrajectoryVerifier.version:
        errors.append("VERIFIER_VERSION_MISMATCH")
    if not state.actions or state.actions[-1].kind != "FINISH":
        errors.append("MISSING_FINISH")
    if len({action.action_id for action in state.actions}) != len(state.actions):
        errors.append("DUPLICATE_ACTION_ID")
    if len({item.evidence_id for item in state.ledger}) != len(state.ledger):
        errors.append("DUPLICATE_EVIDENCE_ID")
    action_by_id = {action.action_id: action for action in state.actions}
    for item in state.ledger:
        creator = action_by_id.get(item.created_by_action)
        if creator is None or creator.kind != "EXTRACT":
            errors.append("UNBOUND_EVIDENCE")
        if not _matches_extract_output(item, creator):
            errors.append("TOOL_TRANSCRIPT_MISMATCH")
    if state.proposal is None:
        errors.append("MISSING_PROPOSAL")
    elif not set(state.proposal.evidence_ids) <= {item.evidence_id for item in state.ledger}:
        errors.append("MISSING_EVIDENCE_REFERENCE")
    if any(action.kind == "REPAIR" for action in state.actions) and not state.parent_snapshot_id:
        errors.append("REPAIR_WITHOUT_SNAPSHOT")
    return tuple(errors)


def _budget_errors(state: TrajectoryState) -> tuple[str, ...]:
    errors: list[str] = []
    if len(state.actions) > state.budget.max_steps:
        errors.append("STEP_BUDGET_EXCEEDED")
    if state.tool_calls > state.budget.max_tool_calls:
        errors.append("TOOL_BUDGET_EXCEEDED")
    if state.repairs > state.budget.max_repairs:
        errors.append("REPAIR_BUDGET_EXCEEDED")
    return tuple(errors)


def _matches_extract_output(item: EvidenceItem, action: AgentAction | None) -> bool:
    if action is None or action.tool_output is None:
        return False
    return action.tool_output == item.source_evidence_id


def _verify_evidence(
    record: EvidenceRecord,
    ledger: tuple[EvidenceItem, ...],
    proposal_evidence_ids: tuple[str, ...],
) -> tuple[bool | None, tuple[str, ...]]:
    if not record.evidence_mask:
        return None, ()
    by_ledger_id = {item.evidence_id: item for item in ledger}
    source_cells = {item.evidence_id: item for item in record.cells}
    cited = [by_ledger_id[item] for item in proposal_evidence_ids]
    errors: list[str] = []
    for item in cited:
        source = source_cells.get(item.source_evidence_id)
        if source is None or not _same_cell(item, source):
            errors.append("EVIDENCE_MISMATCH")
    expected_sources = set(record.gold_evidence_ids)
    cited_sources = {item.source_evidence_id for item in cited}
    if expected_sources and cited_sources != expected_sources:
        errors.append("EVIDENCE_SET_MISMATCH")
    return not errors, tuple(errors)


def _same_cell(item: EvidenceItem, source: EvidenceCell) -> bool:
    return (
        normalize_text(item.row) == normalize_text(source.row)
        and normalize_text(item.column) == normalize_text(source.column)
        and _same_value(item.value, source.value)
    )


def _same_value(left: str, right: str) -> bool:
    try:
        return decimal_to_text(normalize_number(left)) == decimal_to_text(normalize_number(right))
    except ValueError:
        return normalize_text(left) == normalize_text(right)


def _verify_operation(state: TrajectoryState) -> bool:
    if state.proposal is None:
        return False
    cells = tuple(
        EvidenceCell(item.evidence_id, item.row, item.column, item.value) for item in state.ledger
    )
    result = execute(state.proposal.operation, cells)
    return result.success and answers_match(result.value, state.proposal.answer)
