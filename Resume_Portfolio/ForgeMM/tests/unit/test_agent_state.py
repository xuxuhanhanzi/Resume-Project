from __future__ import annotations

import pytest

from forgemm.agent.state import (
    AgentAction,
    AgentBudget,
    EvidenceItem,
    Proposal,
    SnapshotStore,
    TrajectoryState,
    apply_action,
    replay_actions,
)
from forgemm.data.schemas import Operation, OperationArgument


def _initial(*, budget: AgentBudget | None = None) -> TrajectoryState:
    return TrajectoryState(
        episode_id="episode-1",
        question_id="record-1",
        image_id="sha256:image",
        processor_version="processor-1",
        budget=budget or AgentBudget(),
    )


def _evidence(action_id: str = "a1") -> EvidenceItem:
    return EvidenceItem(
        evidence_id="ev-1",
        kind="value",
        row="2016",
        column="Value",
        value="43",
        source_evidence_id="gold-1",
        confidence=0.9,
        created_by_action=action_id,
        region=(0, 0, 4, 4),
    )


def _proposal() -> Proposal:
    return Proposal(
        evidence_ids=("ev-1",),
        operation=Operation("lookup", (OperationArgument("ref", "ev-1", True),)),
        answer="43",
    )


def test_snapshot_fork_preserves_parent_and_replay_is_hash_equivalent() -> None:
    initial = _initial()
    extract = AgentAction("a1", "EXTRACT", tool_input="value", tool_output="gold-1")
    extracted = apply_action(initial, extract, evidence=_evidence())
    store = SnapshotStore()
    snapshot = store.save("s1", extracted)

    branch = store.fork(snapshot.snapshot_id)
    repaired = apply_action(
        branch,
        AgentAction("a2", "REPAIR", parent_snapshot_id="s1"),
    )
    proposed = apply_action(repaired, AgentAction("a3", "PROPOSE"), proposal=_proposal())
    finished = apply_action(proposed, AgentAction("a4", "FINISH"))

    assert store.load("s1") == extracted
    assert finished.parent_snapshot_id == "s1"
    replayed = replay_actions(
        branch,
        (
            (AgentAction("a2", "REPAIR", parent_snapshot_id="s1"), None, None),
            (AgentAction("a3", "PROPOSE"), None, _proposal()),
            (AgentAction("a4", "FINISH"), None, None),
        ),
    )
    assert replayed.sha256() == finished.sha256()


def test_budget_and_duplicate_source_guards_reject_reward_shortcuts() -> None:
    limited = apply_action(
        _initial(budget=AgentBudget(max_steps=3, max_tool_calls=1, max_repairs=0)),
        AgentAction("a1", "EXTRACT", tool_input="value", tool_output="gold-1"),
        evidence=_evidence(),
    )
    with pytest.raises(ValueError, match="tool_budget_exceeded"):
        apply_action(limited, AgentAction("a2", "CALCULATE", tool_input="1+1", tool_output="2"))

    state = apply_action(
        _initial(budget=AgentBudget(max_steps=3, max_tool_calls=2, max_repairs=0)),
        AgentAction("a1", "EXTRACT", tool_input="value", tool_output="gold-1"),
        evidence=_evidence(),
    )
    with pytest.raises(ValueError, match="duplicate_evidence_source"):
        apply_action(
            state,
            AgentAction("a2", "EXTRACT", tool_input="value", tool_output="gold-1"),
            evidence=EvidenceItem(
                evidence_id="ev-2",
                kind="value",
                row="2016",
                column="Value",
                value="43",
                source_evidence_id="gold-1",
                confidence=0.9,
                created_by_action="a2",
            ),
        )


def test_repair_requires_a_forked_snapshot_parent() -> None:
    with pytest.raises(ValueError, match="repair_parent_mismatch"):
        apply_action(_initial(), AgentAction("a1", "REPAIR", parent_snapshot_id="never-saved"))
