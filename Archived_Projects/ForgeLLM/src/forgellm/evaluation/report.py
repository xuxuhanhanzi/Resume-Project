"""Claim-evidence audits and multidimensional acceptance decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    """One conservative statement and the local evidence files supporting it."""

    claim_id: str
    statement: str
    evidence_paths: tuple[str, ...]
    boundary: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.statement.strip() or not self.boundary.strip():
            raise ValueError("claim fields must be non-empty")
        if not self.evidence_paths:
            raise ValueError("every claim requires at least one evidence path")

    def as_dict(self) -> dict[str, str | list[str]]:
        """Return serializable claim fields."""
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "evidence_paths": list(self.evidence_paths),
            "boundary": self.boundary,
        }


def audit_claim_evidence(claims: list[EvidenceClaim], *, project_root: Path) -> dict[str, object]:
    """Fail the audit result, not the process, when a claim has missing files."""
    if not claims or len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("claims must be non-empty with unique IDs")
    missing: dict[str, list[str]] = {}
    for claim in claims:
        absent = [path for path in claim.evidence_paths if not (project_root / path).is_file()]
        if absent:
            missing[claim.claim_id] = absent
    return {
        "claims": [claim.as_dict() for claim in claims],
        "missing_evidence": missing,
        "passed": not missing,
    }


def model_acceptance(
    *,
    strict_success_rate: float,
    minimum_strict_success_rate: float,
    retention_loss_ratio: float,
    maximum_retention_loss_ratio: float,
    repetition_rate: float,
    maximum_repetition_rate: float,
    system_matrix_completed: bool,
) -> dict[str, object]:
    """Return separate gates and never collapse them into a total score."""
    rates = (
        strict_success_rate,
        minimum_strict_success_rate,
        retention_loss_ratio,
        maximum_retention_loss_ratio,
        repetition_rate,
        maximum_repetition_rate,
    )
    if any(not math_is_finite(value) or value < 0 for value in rates):
        raise ValueError("acceptance values must be finite and non-negative")
    gates = {
        "correctness": strict_success_rate >= minimum_strict_success_rate,
        "retention": retention_loss_ratio <= maximum_retention_loss_ratio,
        "stability": repetition_rate <= maximum_repetition_rate,
        "efficiency": system_matrix_completed,
    }
    return {
        "gates": gates,
        "model_behavior_accepted": all(gates.values()),
        "status": (
            "model behavior accepted"
            if all(gates.values())
            else "training pipeline accepted, model behavior not accepted"
        ),
    }


def math_is_finite(value: float) -> bool:
    """Avoid importing a numerical stack for scalar report validation."""
    return value == value and value not in (float("inf"), float("-inf"))
