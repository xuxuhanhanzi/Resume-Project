"""Checkpointable Lagrange multipliers for Chart-FGRPO constraints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DualConfig:
    tau_evidence: float = 0.90
    tau_operation: float = 0.95
    dual_lr: float = 0.01
    lambda_max: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.tau_evidence <= 1 or not 0 <= self.tau_operation <= 1:
            raise ValueError("constraint thresholds must be in [0, 1]")
        if self.dual_lr <= 0 or self.lambda_max <= 0:
            raise ValueError("dual_lr and lambda_max must be positive")


@dataclass
class DualState:
    config: DualConfig
    lambda_evidence: float = 0.0
    lambda_operation: float = 0.0
    update_step: int = 0

    def update(
        self,
        evidence_rewards: NDArray[np.float64],
        operation_rewards: NDArray[np.float64],
        evidence_mask: NDArray[np.bool_],
        operation_mask: NDArray[np.bool_],
    ) -> dict[str, float | int | None]:
        evidence_mean = _masked_mean(evidence_rewards, evidence_mask)
        operation_mean = _masked_mean(operation_rewards, operation_mask)
        if evidence_mean is not None:
            self.lambda_evidence = _clip(
                self.lambda_evidence
                + self.config.dual_lr * (self.config.tau_evidence - evidence_mean),
                self.config.lambda_max,
            )
        if operation_mean is not None:
            self.lambda_operation = _clip(
                self.lambda_operation
                + self.config.dual_lr * (self.config.tau_operation - operation_mean),
                self.config.lambda_max,
            )
        self.update_step += 1
        return {
            "update_step": self.update_step,
            "evidence_mean": evidence_mean,
            "operation_mean": operation_mean,
            "lambda_evidence": self.lambda_evidence,
            "lambda_operation": self.lambda_operation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "config": asdict(self.config),
            "lambda_evidence": self.lambda_evidence,
            "lambda_operation": self.lambda_operation,
            "update_step": self.update_step,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DualState:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported dual-state schema")
        return cls(
            config=DualConfig(**payload["config"]),
            lambda_evidence=float(payload["lambda_evidence"]),
            lambda_operation=float(payload["lambda_operation"]),
            update_step=int(payload["update_step"]),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> DualState:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("dual state must be a JSON object")
        return cls.from_dict(payload)


def _masked_mean(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("reward and mask shapes must match")
    if not np.isfinite(values).all():
        raise ValueError("rewards must be finite")
    return float(values[mask].mean()) if mask.any() else None


def _clip(value: float, maximum: float) -> float:
    return float(min(max(value, 0.0), maximum))
