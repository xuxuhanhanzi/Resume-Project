"""Immutable, replayable state for the bounded ForgeMM chart agent.

The state deliberately stores evidence and tool metadata instead of model hidden
reasoning.  This makes snapshots portable, auditable, and safe to replay without
an inference server.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from forgemm.data.schemas import Operation, OperationArgument

ActionKind = Literal["OBSERVE", "EXTRACT", "CALCULATE", "PROPOSE", "REPAIR", "FINISH"]
EvidenceKind = Literal["entity", "value", "unit", "relation", "visual_region"]
_ACTION_KINDS = {"OBSERVE", "EXTRACT", "CALCULATE", "PROPOSE", "REPAIR", "FINISH"}
_EVIDENCE_KINDS = {"entity", "value", "unit", "relation", "visual_region"}


@dataclass(frozen=True)
class AgentBudget:
    """Fixed per-episode resource limits."""

    max_steps: int = 6
    max_tool_calls: int = 2
    max_repairs: int = 1

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.max_tool_calls < 0 or self.max_repairs < 0:
            raise ValueError("invalid_budget")


@dataclass(frozen=True)
class AgentAction:
    """One declared action and its deterministic tool transcript metadata."""

    action_id: str
    kind: ActionKind
    tool_input: str | None = None
    tool_output: str | None = None
    parent_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id or self.kind not in _ACTION_KINDS:
            raise ValueError("invalid_action")
        is_tool = self.kind in {"EXTRACT", "CALCULATE"}
        if is_tool != (self.tool_input is not None and self.tool_output is not None):
            raise ValueError("invalid_tool_transcript")
        if self.kind == "REPAIR" and not self.parent_snapshot_id:
            raise ValueError("repair_requires_parent_snapshot")
        if self.kind != "REPAIR" and self.parent_snapshot_id is not None:
            raise ValueError("unexpected_parent_snapshot")

    @property
    def is_tool_call(self) -> bool:
        return self.kind in {"EXTRACT", "CALCULATE"}


@dataclass(frozen=True)
class EvidenceItem:
    """A typed evidence fact bound to a source cell or visual region."""

    evidence_id: str
    kind: EvidenceKind
    row: str
    column: str
    value: str
    source_evidence_id: str
    confidence: float
    created_by_action: str
    region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if (
            not self.evidence_id
            or self.kind not in _EVIDENCE_KINDS
            or not self.value
            or not self.source_evidence_id
            or not self.created_by_action
        ):
            raise ValueError("invalid_evidence")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid_evidence_confidence")
        if self.region is not None and (
            len(self.region) != 4 or any(coordinate < 0 for coordinate in self.region)
        ):
            raise ValueError("invalid_evidence_region")


@dataclass(frozen=True)
class Proposal:
    """The answer candidate, evidence citations, and executable operation."""

    evidence_ids: tuple[str, ...]
    operation: Operation
    answer: str

    def __post_init__(self) -> None:
        if not self.evidence_ids or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("proposal_requires_unique_evidence")
        if not self.answer.strip():
            raise ValueError("proposal_requires_answer")


@dataclass(frozen=True)
class TrajectoryState:
    """Canonical state persisted after every bounded agent transition."""

    episode_id: str
    question_id: str
    image_id: str
    processor_version: str
    budget: AgentBudget
    ledger: tuple[EvidenceItem, ...] = ()
    actions: tuple[AgentAction, ...] = ()
    proposal: Proposal | None = None
    verifier_version: str = "forgemm-agent-verifier-1.0.0"
    parent_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if not all((self.episode_id, self.question_id, self.image_id, self.processor_version)):
            raise ValueError("invalid_trajectory_identity")
        if not self.verifier_version:
            raise ValueError("invalid_verifier_version")

    @property
    def finished(self) -> bool:
        return bool(self.actions) and self.actions[-1].kind == "FINISH"

    @property
    def tool_calls(self) -> int:
        return sum(action.is_tool_call for action in self.actions)

    @property
    def repairs(self) -> int:
        return sum(action.kind == "REPAIR" for action in self.actions)

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical, JSON-compatible state representation."""

        return {
            "episode_id": self.episode_id,
            "question_id": self.question_id,
            "image": {"id": self.image_id, "processor_version": self.processor_version},
            "budget": {
                "max_steps": self.budget.max_steps,
                "max_tool_calls": self.budget.max_tool_calls,
                "max_repairs": self.budget.max_repairs,
            },
            "ledger": [_evidence_to_dict(item) for item in self.ledger],
            "actions": [_action_to_dict(action) for action in self.actions],
            "proposal": None if self.proposal is None else _proposal_to_dict(self.proposal),
            "verifier_version": self.verifier_version,
            "parent_snapshot_id": self.parent_snapshot_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryState:
        """Restore a canonical state without silently accepting unknown structure."""

        image = _require_dict(payload, "image")
        budget_payload = _require_dict(payload, "budget")
        ledger_payload = _require_list(payload, "ledger")
        actions_payload = _require_list(payload, "actions")
        proposal_payload = payload.get("proposal")
        if proposal_payload is not None and not isinstance(proposal_payload, dict):
            raise ValueError("invalid_proposal")
        return cls(
            episode_id=_require_string(payload, "episode_id"),
            question_id=_require_string(payload, "question_id"),
            image_id=_require_string(image, "id"),
            processor_version=_require_string(image, "processor_version"),
            budget=AgentBudget(
                max_steps=_require_int(budget_payload, "max_steps"),
                max_tool_calls=_require_int(budget_payload, "max_tool_calls"),
                max_repairs=_require_int(budget_payload, "max_repairs"),
            ),
            ledger=tuple(_evidence_from_dict(item) for item in ledger_payload),
            actions=tuple(_action_from_dict(item) for item in actions_payload),
            proposal=None if proposal_payload is None else _proposal_from_dict(proposal_payload),
            verifier_version=_require_string(payload, "verifier_version"),
            parent_snapshot_id=_optional_string(payload, "parent_snapshot_id"),
        )

    def sha256(self) -> str:
        """Hash the exact canonical state used by snapshots and replay checks."""

        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable serialized checkpoint owned by :class:`SnapshotStore`."""

    snapshot_id: str
    parent_snapshot_id: str | None
    state_sha256: str
    payload: str


class SnapshotStore:
    """In-memory immutable snapshot registry with deterministic restore and fork."""

    def __init__(self) -> None:
        self._snapshots: dict[str, StateSnapshot] = {}

    def save(self, snapshot_id: str, state: TrajectoryState) -> StateSnapshot:
        """Store an immutable snapshot; identifiers may never be overwritten."""

        if not snapshot_id or snapshot_id in self._snapshots:
            raise ValueError("duplicate_or_empty_snapshot_id")
        payload = _canonical_json(state.to_dict())
        snapshot = StateSnapshot(
            snapshot_id=snapshot_id,
            parent_snapshot_id=state.parent_snapshot_id,
            state_sha256=_canonical_sha256(state.to_dict()),
            payload=payload,
        )
        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def load(self, snapshot_id: str) -> TrajectoryState:
        """Restore a snapshot and verify that its serialized state was not altered."""

        try:
            snapshot = self._snapshots[snapshot_id]
        except KeyError as exc:
            raise ValueError("unknown_snapshot") from exc
        payload = json.loads(snapshot.payload)
        if not isinstance(payload, dict):
            raise ValueError("invalid_snapshot_payload")
        state = TrajectoryState.from_dict(payload)
        if state.sha256() != snapshot.state_sha256:
            raise ValueError("snapshot_hash_mismatch")
        return state

    def fork(self, snapshot_id: str) -> TrajectoryState:
        """Create a repair branch linked to, but unable to overwrite, its parent."""

        return replace(self.load(snapshot_id), parent_snapshot_id=snapshot_id)


def apply_action(
    state: TrajectoryState,
    action: AgentAction,
    *,
    evidence: EvidenceItem | None = None,
    proposal: Proposal | None = None,
) -> TrajectoryState:
    """Apply one valid state transition without mutating its parent state."""

    if state.finished:
        raise ValueError("trajectory_already_finished")
    if len(state.actions) >= state.budget.max_steps:
        raise ValueError("step_budget_exceeded")
    if action.action_id in {item.action_id for item in state.actions}:
        raise ValueError("duplicate_action_id")
    if action.is_tool_call and state.tool_calls >= state.budget.max_tool_calls:
        raise ValueError("tool_budget_exceeded")
    if action.kind == "REPAIR":
        if state.repairs >= state.budget.max_repairs:
            raise ValueError("repair_budget_exceeded")
        if action.parent_snapshot_id != state.parent_snapshot_id:
            raise ValueError("repair_parent_mismatch")

    next_ledger = state.ledger
    next_proposal = state.proposal
    if action.kind == "EXTRACT":
        if evidence is None or evidence.created_by_action != action.action_id:
            raise ValueError("extract_requires_bound_evidence")
        if evidence.evidence_id in {item.evidence_id for item in state.ledger}:
            raise ValueError("duplicate_evidence_id")
        if evidence.source_evidence_id in {item.source_evidence_id for item in state.ledger}:
            raise ValueError("duplicate_evidence_source")
        next_ledger = (*state.ledger, evidence)
    elif evidence is not None:
        raise ValueError("evidence_only_allowed_for_extract")

    if action.kind == "PROPOSE":
        if proposal is None:
            raise ValueError("proposal_action_requires_proposal")
        available = {item.evidence_id for item in next_ledger}
        if not set(proposal.evidence_ids) <= available:
            raise ValueError("proposal_references_missing_evidence")
        next_proposal = proposal
    elif proposal is not None:
        raise ValueError("proposal_only_allowed_for_propose")

    if action.kind == "FINISH" and next_proposal is None:
        raise ValueError("finish_requires_proposal")
    return replace(
        state,
        ledger=next_ledger,
        actions=(*state.actions, action),
        proposal=next_proposal,
    )


def replay_actions(
    initial: TrajectoryState,
    transitions: tuple[tuple[AgentAction, EvidenceItem | None, Proposal | None], ...],
) -> TrajectoryState:
    """Reconstruct a state from deterministic transition records."""

    state = initial
    for action, evidence, proposal in transitions:
        state = apply_action(state, action, evidence=evidence, proposal=proposal)
    return state


def _action_to_dict(action: AgentAction) -> dict[str, str | None]:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "tool_input": action.tool_input,
        "tool_output": action.tool_output,
        "parent_snapshot_id": action.parent_snapshot_id,
    }


def _action_from_dict(payload: Any) -> AgentAction:
    if not isinstance(payload, dict):
        raise ValueError("invalid_action")
    kind = _require_string(payload, "kind")
    return AgentAction(
        action_id=_require_string(payload, "action_id"),
        kind=kind,  # type: ignore[arg-type]
        tool_input=_optional_string(payload, "tool_input"),
        tool_output=_optional_string(payload, "tool_output"),
        parent_snapshot_id=_optional_string(payload, "parent_snapshot_id"),
    )


def _evidence_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "kind": item.kind,
        "row": item.row,
        "column": item.column,
        "value": item.value,
        "source_evidence_id": item.source_evidence_id,
        "confidence": item.confidence,
        "created_by_action": item.created_by_action,
        "region": None if item.region is None else list(item.region),
    }


def _evidence_from_dict(payload: Any) -> EvidenceItem:
    if not isinstance(payload, dict):
        raise ValueError("invalid_evidence")
    region = payload.get("region")
    if region is not None and (
        not isinstance(region, list)
        or len(region) != 4
        or not all(isinstance(item, int) for item in region)
    ):
        raise ValueError("invalid_evidence_region")
    return EvidenceItem(
        evidence_id=_require_string(payload, "evidence_id"),
        kind=_require_string(payload, "kind"),  # type: ignore[arg-type]
        row=_require_string(payload, "row"),
        column=_require_string(payload, "column"),
        value=_require_string(payload, "value"),
        source_evidence_id=_require_string(payload, "source_evidence_id"),
        confidence=_require_float(payload, "confidence"),
        created_by_action=_require_string(payload, "created_by_action"),
        region=None if region is None else tuple(region),
    )


def _proposal_to_dict(proposal: Proposal) -> dict[str, Any]:
    return {
        "evidence_ids": list(proposal.evidence_ids),
        "operation": {
            "name": proposal.operation.name,
            "arguments": [
                {"name": item.name, "value": item.value, "is_reference": item.is_reference}
                for item in proposal.operation.arguments
            ],
        },
        "answer": proposal.answer,
    }


def _proposal_from_dict(payload: dict[str, Any]) -> Proposal:
    evidence_ids = _require_list(payload, "evidence_ids")
    if not all(isinstance(item, str) for item in evidence_ids):
        raise ValueError("invalid_proposal_evidence_ids")
    operation_payload = _require_dict(payload, "operation")
    arguments_payload = _require_list(operation_payload, "arguments")
    arguments: list[OperationArgument] = []
    for argument_payload in arguments_payload:
        if not isinstance(argument_payload, dict) or not isinstance(
            argument_payload.get("is_reference"), bool
        ):
            raise ValueError("invalid_operation_argument")
        arguments.append(
            OperationArgument(
                name=_require_string(argument_payload, "name"),
                value=_require_string(argument_payload, "value"),
                is_reference=argument_payload["is_reference"],
            )
        )
    return Proposal(
        evidence_ids=tuple(evidence_ids),
        operation=Operation(
            name=_require_string(operation_payload, "name"),
            arguments=tuple(arguments),
        ),
        answer=_require_string(payload, "answer"),
    )


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"invalid_{key}")
    return value


def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"invalid_{key}")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid_{key}")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid_{key}")
    return value


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"invalid_{key}")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise ValueError(f"invalid_{key}")
    return float(value)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
