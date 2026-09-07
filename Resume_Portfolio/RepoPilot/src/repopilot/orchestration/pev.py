"""Deterministic Plan--Execute--Verify control and reviewer-risk policy for R2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PEVStage(StrEnum):
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ReviewerStrategy(StrEnum):
    NONE = "none"
    ALWAYS = "always"
    RISK_GATED = "risk_gated"


@dataclass(frozen=True, slots=True)
class RiskSignals:
    """Only observed, deterministic risk inputs; no reviewer-model guesswork."""

    tests_failed: bool = False
    high_risk_path_changed: bool = False
    diff_bytes: int = 0
    diff_byte_threshold: int = 8_000
    dependency_changed: bool = False
    static_antipattern_found: bool = False
    model_low_confidence: bool = False
    key_evidence_missing: bool = False

    def __post_init__(self) -> None:
        if self.diff_bytes < 0 or self.diff_byte_threshold < 1:
            raise ValueError("diff sizes must be non-negative with a positive threshold")

    def reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.tests_failed:
            reasons.append("tests_failed")
        if self.high_risk_path_changed:
            reasons.append("high_risk_path_changed")
        if self.diff_bytes > self.diff_byte_threshold:
            reasons.append("diff_exceeds_threshold")
        if self.dependency_changed:
            reasons.append("dependency_changed")
        if self.static_antipattern_found:
            reasons.append("static_antipattern_found")
        if self.model_low_confidence:
            reasons.append("model_low_confidence")
        if self.key_evidence_missing:
            reasons.append("key_evidence_missing")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ReviewerDecision:
    required: bool
    reasons: tuple[str, ...]


def reviewer_decision(strategy: ReviewerStrategy, signals: RiskSignals) -> ReviewerDecision:
    """Decide reviewer exposure without invoking a reviewer or changing policy."""

    if strategy is ReviewerStrategy.NONE:
        return ReviewerDecision(False, ())
    if strategy is ReviewerStrategy.ALWAYS:
        return ReviewerDecision(True, ("always",))
    reasons = signals.reasons()
    return ReviewerDecision(bool(reasons), reasons)


@dataclass(slots=True)
class PEVController:
    """A minimal, serializable PEV state machine with an explicit replan cap."""

    max_replans: int = 1
    stage: PEVStage = PEVStage.PLAN
    plan_steps: tuple[str, ...] = ()
    completed_steps: list[str] = field(default_factory=list)
    verification_attempts: int = 0
    replans: int = 0
    acceptance_evidence: tuple[str, ...] = ()
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.max_replans <= 2:
            raise ValueError("max_replans must be in the R2 ablation range 0..2")

    def submit_plan(self, steps: tuple[str, ...]) -> None:
        if self.stage is not PEVStage.PLAN:
            raise ValueError("a plan may only be submitted in the plan stage")
        if not steps or any(not step.strip() for step in steps):
            raise ValueError("PEV requires at least one non-empty plan step")
        self.plan_steps = steps
        self.stage = PEVStage.EXECUTE

    def record_execution(self, step: str) -> None:
        if self.stage is not PEVStage.EXECUTE:
            raise ValueError("execution evidence is only valid in the execute stage")
        if step not in self.plan_steps:
            raise ValueError("execution step is not present in the approved plan")
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        if len(self.completed_steps) == len(self.plan_steps):
            self.stage = PEVStage.VERIFY

    def record_verification(
        self,
        *,
        passed: bool,
        evidence_ids: tuple[str, ...],
        reason: str = "",
    ) -> None:
        if self.stage is not PEVStage.VERIFY:
            raise ValueError("verification can only follow completed planned execution")
        if not evidence_ids or any(not item.strip() for item in evidence_ids):
            raise ValueError("verification requires durable evidence identifiers")
        self.verification_attempts += 1
        self.acceptance_evidence = evidence_ids
        if passed:
            self.stage = PEVStage.COMPLETED
            self.failure_reason = ""
            return
        self.failure_reason = reason.strip() or "verification_failed"
        if self.replans >= self.max_replans:
            self.stage = PEVStage.BLOCKED
            return
        self.replans += 1
        self.completed_steps.clear()
        self.stage = PEVStage.EXECUTE

    @property
    def can_claim_success(self) -> bool:
        return self.stage is PEVStage.COMPLETED and bool(self.acceptance_evidence)
