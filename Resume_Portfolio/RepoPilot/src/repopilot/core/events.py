"""Provider-neutral runtime events shared by CLI and durable runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from repopilot.core.contracts import JSONValue


class RuntimeEventKind(StrEnum):
    """Ordered lifecycle events emitted by :class:`AgentKernel`."""

    SESSION_INITIALIZING = "session_initializing"
    TURN_STARTED = "turn_started"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_RETRYING = "model_retrying"
    MODEL_OUTPUT_DELTA = "model_output_delta"
    MODEL_CALL_COMPLETED = "model_call_completed"
    TOOL_CALL_PROPOSED = "tool_call_proposed"
    TOOL_REQUIREMENT_ENFORCED = "tool_requirement_enforced"
    PATCH_RECOVERY_ENFORCED = "patch_recovery_enforced"
    POLICY_DECISION = "policy_decision"
    PERMISSION_REQUESTED = "permission_requested"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    WORKFLOW_STAGE_CHANGED = "workflow_stage_changed"
    WORKFLOW_REPORT_CREATED = "workflow_report_created"
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETED = "review_completed"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    HOOK_TRIGGERED = "hook_triggered"
    TURN_COMPLETED = "turn_completed"
    TURN_FAILED = "turn_failed"
    TURN_CANCELLED = "turn_cancelled"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """A structured event; consumers decide how to render or persist it."""

    kind: RuntimeEventKind
    data: dict[str, JSONValue] = field(default_factory=dict)
