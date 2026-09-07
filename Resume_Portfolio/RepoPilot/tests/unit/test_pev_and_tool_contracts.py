from __future__ import annotations

import pytest

from repopilot.core.contracts import Permission, ToolCall, ToolSpec
from repopilot.orchestration.pev import (
    PEVController,
    PEVStage,
    ReviewerStrategy,
    RiskSignals,
    reviewer_decision,
)
from repopilot.tools.contracts import RecoveryCode, StageScopedToolContracts


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name,
        name,
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "max_depth": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
            "x-path-fields": ["path"],
        },
        permission=Permission.READ,
    )


def test_pev_requires_evidence_and_applies_a_bounded_replan() -> None:
    controller = PEVController(max_replans=1)
    controller.submit_plan(("inspect", "patch"))
    controller.record_execution("inspect")
    controller.record_execution("patch")
    controller.record_verification(passed=False, evidence_ids=("test-log-1",))

    assert controller.stage is PEVStage.EXECUTE
    assert controller.replans == 1
    controller.record_execution("inspect")
    controller.record_execution("patch")
    controller.record_verification(passed=True, evidence_ids=("test-log-2",))
    assert controller.can_claim_success
    decision = reviewer_decision(
        ReviewerStrategy.RISK_GATED, RiskSignals(tests_failed=True, dependency_changed=True)
    )
    assert decision.required and decision.reasons == ("tests_failed", "dependency_changed")


def test_tool_contracts_are_stage_scoped_schema_checked_and_recoverable() -> None:
    contracts = StageScopedToolContracts(
        (_spec("read_file"), _spec("apply_patch")),
        allowlist={"explore": frozenset({"read_file"}), "modify": frozenset({"apply_patch"})},
    )

    good = contracts.validate(ToolCall("1", "read_file", {"path": "src/main.py"}), stage="explore")
    blocked = contracts.validate(
        ToolCall("2", "apply_patch", {"path": "src/main.py"}), stage="explore"
    )
    invalid = contracts.validate(ToolCall("3", "read_file", {"path": "../secret"}), stage="explore")
    malformed = contracts.validate(
        ToolCall("4", "read_file", {"path": "src/a.py", "extra": 1}), stage="explore"
    )

    assert good.allowed
    assert blocked.code is RecoveryCode.STAGE_NOT_ALLOWED
    assert invalid.code is RecoveryCode.PATH_OUT_OF_SCOPE
    assert malformed.code is RecoveryCode.SCHEMA_INVALID
    with pytest.raises(ValueError, match="unknown tool stage"):
        contracts.visible_specs("verify")
