"""ms-swift Chart-FGRPO trainer extension with checkpointable dual weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from swift.rlhf_trainers import GRPOTrainer  # type: ignore[import-not-found]
from swift.trainers.trainer_factory import TrainerFactory  # type: ignore[import-not-found]

from forgemm.trainers.dual_state import DualConfig, DualState
from forgemm.trainers.reward_channels import reward_channel_indices


class ChartFGRPOTrainer(GRPOTrainer):  # type: ignore[misc]
    """Apply current Lagrange weights to GDPO-normalized reward channels."""

    reward_weights: torch.Tensor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = DualConfig(
            tau_evidence=float(os.environ.get("FORGEMM_TAU_EVIDENCE", "0.90")),
            tau_operation=float(os.environ.get("FORGEMM_TAU_OPERATION", "0.95")),
            dual_lr=float(os.environ.get("FORGEMM_DUAL_LR", "0.01")),
            lambda_max=float(os.environ.get("FORGEMM_LAMBDA_MAX", "5.0")),
        )
        self._dual_path = Path(self.args.output_dir) / "chart_fgrpo_dual_state.json"
        self._dual_trace_path = Path(self.args.output_dir) / "chart_fgrpo_dual_trace.jsonl"
        resume_dir = getattr(self.args, "resume_from_checkpoint", None)
        resume_dual_path = Path(resume_dir) / "chart_fgrpo_dual_state.json" if resume_dir else None
        self._dual_state = (
            DualState.load(resume_dual_path)
            if resume_dual_path is not None and resume_dual_path.exists()
            else DualState.load(self._dual_path)
            if self._dual_path.exists()
            else DualState(config)
        )

    def _compute_advantages(
        self,
        inputs: Any,
        rewards_per_func: torch.Tensor,
        batch_encoded_inputs: Any,
    ) -> torch.Tensor:
        indices = self._reward_indices()
        current_weights = torch.zeros_like(self.reward_weights)
        current_weights[indices["task"]] = 1.0
        current_weights[indices["evidence"]] = self._dual_state.lambda_evidence
        current_weights[indices["operation"]] = self._dual_state.lambda_operation
        self.reward_weights = current_weights

        advantages = super()._compute_advantages(inputs, rewards_per_func, batch_encoded_inputs)
        if self.model.training:
            evidence_mask = np.asarray(
                [_as_bool(row.get("evidence_mask", True)) for row in inputs], dtype=np.bool_
            )
            operation_mask = np.asarray(
                [_as_bool(row.get("operation_mask", True)) for row in inputs], dtype=np.bool_
            )
            update = self._dual_state.update(
                rewards_per_func[:, indices["evidence"]].detach().float().cpu().numpy(),
                rewards_per_func[:, indices["operation"]].detach().float().cpu().numpy(),
                evidence_mask,
                operation_mask,
            )
            self._dual_state.save(self._dual_path)
            self._dual_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._dual_trace_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(update, sort_keys=True) + "\n")
        return advantages

    def _save_checkpoint(self, model: Any, trial: Any) -> None:
        super()._save_checkpoint(model, trial)
        checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        self._dual_state.save(checkpoint / "chart_fgrpo_dual_state.json")

    def _reward_indices(self) -> dict[str, int]:
        return reward_channel_indices(self.reward_funcs)


def register_chart_fgrpo_trainer() -> None:
    """Select the extension for GRPO runs in this process."""

    TrainerFactory.TRAINER_MAPPING["grpo"] = "forgemm.swift_trainer.ChartFGRPOTrainer"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


__all__ = ["ChartFGRPOTrainer", "register_chart_fgrpo_trainer"]
