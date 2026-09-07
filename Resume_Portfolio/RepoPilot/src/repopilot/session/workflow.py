"""Evidence-backed workflow summaries for one interactive coding turn."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from repopilot.core.contracts import JSONValue, ToolResult


class WorkflowStage(StrEnum):
    """Advisory, durable milestones of a coding workflow."""

    EXPLORE = "explore"
    PLAN = "plan"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    REVIEW = "review"
    ANSWER = "answer"


@dataclass(frozen=True, slots=True)
class WorkflowCheck:
    """One verification command actually executed during this turn."""

    label: str
    command: tuple[str, ...]
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "label": self.label,
            "command": list(self.command),
            "ok": self.ok,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class TurnWorkflowReport:
    """Only recorded observations are eligible for the final coding summary."""

    changed_files: tuple[str, ...] = ()
    checks: tuple[WorkflowCheck, ...] = ()
    snapshot_error: str | None = None

    @property
    def has_evidence(self) -> bool:
        return bool(self.changed_files or self.checks or self.snapshot_error)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "changed_files": list(self.changed_files),
            "checks": [check.to_dict() for check in self.checks],
            "snapshot_error": self.snapshot_error,
        }

    def render(self) -> str:
        """Return concise user-facing evidence without model-supplied claims."""

        if not self.has_evidence:
            return ""
        lines = ["Workflow evidence:", "Changed:"]
        if self.snapshot_error:
            lines.append(f"- Change inventory unavailable: {self.snapshot_error}")
        elif self.changed_files:
            lines.extend(f"- {path}" for path in self.changed_files)
        else:
            lines.append("- No workspace files changed in this turn.")
        lines.append("Verification:")
        if self.checks:
            for check in self.checks:
                status = "passed" if check.ok else "failed"
                command = " ".join(check.command) or "(unknown command)"
                suffix = f" — {check.error}" if check.error else ""
                lines.append(f"- {status}: {check.label} ({command}){suffix}")
        elif self.changed_files:
            lines.append("- Not run in this turn.")
        else:
            lines.append("- Not applicable to this read-only turn.")
        return "\n".join(lines)


def checks_from_tool_results(results: list[ToolResult]) -> tuple[WorkflowCheck, ...]:
    """Extract only immutable ``run_tests`` evidence from executed tool results."""

    checks: list[WorkflowCheck] = []
    for result in results:
        if result.tool_name != "run_tests":
            continue
        payload = result.data if isinstance(result.data, dict) else {}
        raw_command = payload.get("command", [])
        command = (
            tuple(item for item in raw_command if isinstance(item, str))
            if isinstance(raw_command, list)
            else ()
        )
        checks.append(WorkflowCheck("run_tests", command, result.ok, result.error))
    return tuple(checks)
