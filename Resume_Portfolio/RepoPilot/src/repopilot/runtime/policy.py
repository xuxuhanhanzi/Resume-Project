"""Permission policy and human-approval boundary."""

from __future__ import annotations

from typing import Protocol

from repopilot.core.contracts import (
    Permission,
    PolicyDecision,
    PolicyOutcome,
    ToolCall,
    ToolSpec,
)
from repopilot.task import PublicTaskSpec

_HIGH_RISK_PREFIXES = (
    ".github/",
    "deploy/",
    "deployment/",
    "Dockerfile",
    "requirements",
    "pyproject.toml",
)


class ApprovalHandler(Protocol):
    """Application boundary for an explicit human decision."""

    async def approve(self, call: ToolCall, reason: str) -> bool:
        """Return true only after an explicit approval decision."""
        ...


class StaticApprovalHandler:
    """Deterministic handler used by tests and non-interactive demos."""

    def __init__(self, approved: bool = False) -> None:
        self.approved = approved
        self.requests: list[tuple[ToolCall, str]] = []

    async def approve(self, call: ToolCall, reason: str) -> bool:
        self.requests.append((call, reason))
        return self.approved


class PolicyEngine:
    """Decide structurally; model text can never bypass this layer."""

    def decide(self, call: ToolCall, spec: ToolSpec, task: PublicTaskSpec) -> PolicyDecision:
        if spec.permission is Permission.HIGH_RISK:
            return PolicyDecision(PolicyOutcome.REQUIRE_APPROVAL, "tool is classified high-risk")
        path = call.arguments.get("path")
        if isinstance(path, str):
            portable = path.replace("\\", "/").lstrip("./")
            if any(
                portable == prefix or portable.startswith(prefix) for prefix in _HIGH_RISK_PREFIXES
            ):
                return PolicyDecision(
                    PolicyOutcome.REQUIRE_APPROVAL,
                    f"path {path!r} is a high-risk project file",
                )
        if spec.permission is Permission.EXECUTE and not task.test_command:
            return PolicyDecision(
                PolicyOutcome.DENY, "task does not define an executable test command"
            )
        return PolicyDecision(PolicyOutcome.ALLOW, "tool is allowed by the task policy")
