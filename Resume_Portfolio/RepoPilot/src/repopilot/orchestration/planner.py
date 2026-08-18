"""A transparent bounded plan used by the single controlling agent."""

from __future__ import annotations

from dataclasses import dataclass

from repopilot.task import PublicTaskSpec


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One observable step, not a separate autonomous agent."""

    step_id: str
    objective: str


@dataclass(frozen=True, slots=True)
class Plan:
    """A small plan that can be projected into model context."""

    steps: tuple[PlanStep, ...]
    revision: int = 1

    def objectives(self) -> list[str]:
        return [step.objective for step in self.steps]


class SimplePlanner:
    """Seed a stable workflow; the model decides concrete searches and edits."""

    def create(self, task: PublicTaskSpec) -> Plan:
        del task
        return Plan(
            (
                PlanStep(
                    "inspect", "Inspect repository structure and reproduce or locate the failure"
                ),
                PlanStep("diagnose", "Form a falsifiable root-cause hypothesis from relevant code"),
                PlanStep("edit", "Apply the narrowest allowed patch"),
                PlanStep("verify", "Run immutable tests and inspect any failure"),
                PlanStep("review", "Review the final diff and propose completion"),
            )
        )

    def replan(self, current: Plan, failure: str) -> Plan:
        return Plan(
            (
                PlanStep("failure", f"Inspect the latest failure: {failure[:300]}"),
                PlanStep(
                    "revise", "Revise the hypothesis without repeating the same failed action"
                ),
                PlanStep("verify", "Apply a bounded correction and rerun immutable tests"),
            ),
            revision=current.revision + 1,
        )
