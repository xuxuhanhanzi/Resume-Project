"""ms-swift external reward plugin for ForgeMM.

The module remains importable without ms-swift so its dataset-field contract can be
tested in the CPU development environment. ms-swift injects all non-message columns
as keyword arguments to each ORM reward function.
"""

from __future__ import annotations

import json
import os
from typing import Any

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


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


orms["forgemm_task"] = ForgeMMTaskReward
orms["forgemm_evidence"] = ForgeMMEvidenceReward
orms["forgemm_operation"] = ForgeMMOperationReward

__all__ = [
    "ForgeMMEvidenceReward",
    "ForgeMMOperationReward",
    "ForgeMMTaskReward",
    "orms",
    "score_swift_batch",
]
