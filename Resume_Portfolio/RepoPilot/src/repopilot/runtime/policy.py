"""Permission policy and human-approval boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol
from urllib.parse import urlsplit

from repopilot.core.contracts import (
    Permission,
    PolicyDecision,
    PolicyOutcome,
    ToolCall,
    ToolSpec,
)
from repopilot.workspace.contracts import InteractiveTask, WorkspaceTask
from repopilot.workspace.settings import ToolPermissionRules

_HIGH_RISK_PREFIXES = (
    ".github/",
    "deploy/",
    "deployment/",
    ".repopilot",
    "dockerfile",
    "requirements",
    "pyproject.toml",
    "repopilot.md",
    "agents.md",
)
_SECRET_NAMES = (".env", "credentials", "credential", "id_rsa")


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

    def decide(self, call: ToolCall, spec: ToolSpec, task: WorkspaceTask) -> PolicyDecision:
        if spec.permission is Permission.HIGH_RISK:
            return PolicyDecision(PolicyOutcome.REQUIRE_APPROVAL, "tool is classified high-risk")
        for path in _argument_paths(call):
            # Workspace paths are case-insensitive on the supported Windows host.  Apply
            # the same rule here so a casing variant cannot bypass protected controls.
            portable = path.replace("\\", "/")
            while portable.startswith("./"):
                portable = portable[2:]
            portable = portable.casefold()
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


class PermissionMode(StrEnum):
    """Interactive approval modes; benchmark tasks continue to use PolicyEngine."""

    MANUAL = "manual"
    ACCEPT_EDITS = "accept_edits"
    PLAN = "plan"


class PermissionEngine(PolicyEngine):
    """Policy decisions for a real workspace before general shell support exists."""

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.MANUAL,
        rules: ToolPermissionRules | None = None,
    ) -> None:
        self.mode = mode
        self.rules = rules or ToolPermissionRules()

    def decide(self, call: ToolCall, spec: ToolSpec, task: WorkspaceTask) -> PolicyDecision:
        if call.name == "start_subagent":
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL,
                "background analysis sends supplied evidence to the configured model provider",
            )
        if call.name == "web_fetch":
            raw_url = call.arguments.get("url")
            origin = "an invalid URL"
            if isinstance(raw_url, str):
                parsed = urlsplit(raw_url)
                if parsed.scheme and parsed.netloc:
                    origin = f"{parsed.scheme}://{parsed.netloc}"
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL,
                f"web fetch sends a request to {origin}",
            )
        if self.rules.denies(call.name):
            return PolicyDecision(
                PolicyOutcome.DENY, "tool is denied by configured permission rules"
            )
        if self.mode is PermissionMode.PLAN and not spec.read_only:
            return PolicyDecision(PolicyOutcome.DENY, "plan mode permits read-only tools only")
        inherited = super().decide(call, spec, task)
        # Benchmark tasks deliberately permit only their immutable test command.
        # Interactive sessions opt into general argv execution, but every execution
        # still requires an explicit approval below.
        interactive_shell = (
            isinstance(task, InteractiveTask)
            and spec.permission is Permission.EXECUTE
            and not task.test_command
            and inherited.reason == "task does not define an executable test command"
        )
        for path in _argument_paths(call):
            basename = path.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
            if (
                basename == ".env"
                or basename.startswith(".env.")
                or any(name in basename for name in _SECRET_NAMES[1:])
            ):
                return PolicyDecision(PolicyOutcome.DENY, "protected secret path")
        if inherited.outcome is not PolicyOutcome.ALLOW and not interactive_shell:
            return inherited
        if spec.permission is Permission.WRITE and self.mode is PermissionMode.MANUAL:
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL, "manual mode requires edit approval"
            )
        if spec.permission is Permission.EXECUTE:
            if self.rules.allows(call.name):
                return PolicyDecision(
                    PolicyOutcome.ALLOW,
                    "tool is allowed by an explicit user permission rule",
                )
            return PolicyDecision(
                PolicyOutcome.REQUIRE_APPROVAL,
                "interactive execution requires explicit approval",
            )
        if inherited.outcome is PolicyOutcome.REQUIRE_APPROVAL and self.rules.allows(call.name):
            return PolicyDecision(
                PolicyOutcome.ALLOW,
                "tool is allowed by an explicit user permission rule",
            )
        return inherited


def _argument_paths(call: ToolCall) -> tuple[str, ...]:
    """Read path-like tool arguments without trusting a free-form command string."""
    values: list[str] = []
    path = call.arguments.get("path")
    if isinstance(path, str):
        values.append(path)
    paths = call.arguments.get("paths")
    if isinstance(paths, list):
        values.extend(item for item in paths if isinstance(item, str))
    return tuple(values)
