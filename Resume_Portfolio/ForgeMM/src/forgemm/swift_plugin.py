"""ms-swift external reward plugin for ForgeMM.

The module remains importable without ms-swift so its dataset-field contract can be
tested in the CPU development environment. ms-swift injects all non-message columns
as keyword arguments to each ORM reward function.
"""

from __future__ import annotations

import json
import os
from typing import Any

from forgemm.controlled.protocol import ControlledRecord, verify_controlled_completion
from forgemm.data.schemas import EvidenceCell
from forgemm.rewards.scoring import RewardBundle, score_completion

try:
    from swift.rewards import ORM, orms  # type: ignore[import-not-found]
except ImportError:  # CPU contract tests do not install the GPU training stack.

    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, type[ORM]] = {}  # type: ignore[no-redef]


def _install_trl_transformers_451_compat() -> None:
    """Provide the optional Trackio probe expected by TRL 0.26.

    Transformers 4.51.3 is retained for its working Qwen2.5-VL generation path,
    while TRL 0.26 only needs this newer helper to decide whether optional
    Trackio logging is installed.  ForgeMM disables that logger, so returning
    ``False`` is equivalent to the upstream behavior when Trackio is absent.
    """
    if os.environ.get("FORGEMM_TRL_TRANSFORMERS_451") != "1":
        return
    import transformers  # type: ignore[import-not-found]

    if not hasattr(transformers, "is_trackio_available"):
        transformers.is_trackio_available = lambda: False
    trainer_cls = transformers.Trainer
    if not hasattr(trainer_cls, "current_gradient_accumulation_steps"):
        trainer_cls.current_gradient_accumulation_steps = property(
            lambda self: self.args.gradient_accumulation_steps
        )


_install_trl_transformers_451_compat()

if os.environ.get("FORGEMM_ENABLE_CHART_FGRPO") == "1":
    from forgemm.swift_trainer import register_chart_fgrpo_trainer

    register_chart_fgrpo_trainer()


class _ForgeMMReward(ORM):  # type: ignore[misc]
    channel: str

    def __call__(
        self,
        completions: list[str],
        reference_answer: list[str] | str,
        gold_evidence: list[Any] | Any,
        evidence_mask: list[Any] | Any = True,
        operation_mask: list[Any] | Any = True,
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        count = len(completions)
        answers = _broadcast(reference_answer, count)
        evidence = _broadcast(gold_evidence, count)
        evidence_masks = _broadcast(evidence_mask, count)
        operation_masks = _broadcast(operation_mask, count)
        result = []
        for index, completion in enumerate(completions):
            bundle = score_completion(
                completion,
                str(answers[index]),
                _decode_evidence(evidence[index]),
                evidence_mask=_as_bool(evidence_masks[index]),
                operation_mask=_as_bool(operation_masks[index]),
            )
            result.append(float(getattr(bundle, self.channel)))
        return result


class ForgeMMTaskReward(_ForgeMMReward):
    channel = "task"


class ForgeMMEvidenceReward(_ForgeMMReward):
    channel = "evidence"


class ForgeMMOperationReward(_ForgeMMReward):
    channel = "operation"


class _ForgeMMControlledReward(ORM):  # type: ignore[misc]
    """Reward channel for generator-backed visual evidence with pixel boxes."""

    channel: str

    def __call__(
        self,
        completions: list[str],
        controlled_gold: list[Any] | Any,
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        records = _broadcast(controlled_gold, len(completions))
        return [
            _controlled_channel(
                verify_controlled_completion(_decode_controlled(item), completion), self.channel
            )
            for completion, item in zip(completions, records, strict=True)
        ]


class ForgeMMControlledTaskReward(_ForgeMMControlledReward):
    channel = "task"


class ForgeMMControlledEvidenceReward(_ForgeMMControlledReward):
    channel = "evidence"


class ForgeMMControlledOperationReward(_ForgeMMControlledReward):
    channel = "operation"


def score_swift_batch(completions: list[str], **columns: Any) -> list[RewardBundle]:
    count = len(completions)
    answers = _broadcast(columns["reference_answer"], count)
    evidence = _broadcast(columns["gold_evidence"], count)
    evidence_masks = _broadcast(columns.get("evidence_mask", True), count)
    operation_masks = _broadcast(columns.get("operation_mask", True), count)
    return [
        score_completion(
            completion,
            str(answers[index]),
            _decode_evidence(evidence[index]),
            evidence_mask=_as_bool(evidence_masks[index]),
            operation_mask=_as_bool(operation_masks[index]),
        )
        for index, completion in enumerate(completions)
    ]


def score_controlled_swift_batch(
    completions: list[str], **columns: Any
) -> list[tuple[float, float, float]]:
    """CPU-testable scoring mirror for the controlled ms-swift reward channels."""

    records = _broadcast(columns["controlled_gold"], len(completions))
    return [
        (
            float(verdict.answer_ok),
            float(verdict.evidence_ok),
            float(verdict.operation_ok),
        )
        for completion, item in zip(completions, records, strict=True)
        for verdict in (verify_controlled_completion(_decode_controlled(item), completion),)
    ]


def _broadcast(value: Any, count: int) -> list[Any]:
    if isinstance(value, list) and len(value) == count:
        return value
    return [value] * count


def _decode_evidence(value: Any) -> tuple[EvidenceCell, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("cells", [])
    if not isinstance(value, (list, tuple)):
        raise ValueError("gold_evidence must be a list or JSON-encoded list")
    return tuple(EvidenceCell(**item) if isinstance(item, dict) else item for item in value)


def _decode_controlled(value: Any) -> ControlledRecord:
    if isinstance(value, ControlledRecord):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("controlled_gold must be a ControlledRecord or JSON object")
    return ControlledRecord.from_dict(value)


def _controlled_channel(verdict: Any, channel: str) -> float:
    if channel == "task":
        return float(verdict.answer_ok)
    if channel == "evidence":
        return float(verdict.evidence_ok)
    if channel == "operation":
        return float(verdict.operation_ok)
    raise ValueError(f"unknown_controlled_reward_channel:{channel}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


orms["forgemm_task"] = ForgeMMTaskReward
orms["forgemm_evidence"] = ForgeMMEvidenceReward
orms["forgemm_operation"] = ForgeMMOperationReward
orms["forgemm_controlled_task"] = ForgeMMControlledTaskReward
orms["forgemm_controlled_evidence"] = ForgeMMControlledEvidenceReward
orms["forgemm_controlled_operation"] = ForgeMMControlledOperationReward

__all__ = [
    "ForgeMMEvidenceReward",
    "ForgeMMControlledEvidenceReward",
    "ForgeMMControlledOperationReward",
    "ForgeMMControlledTaskReward",
    "ForgeMMOperationReward",
    "ForgeMMTaskReward",
    "orms",
    "score_controlled_swift_batch",
    "score_swift_batch",
]
