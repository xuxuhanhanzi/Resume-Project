"""Deterministic verifier-backed preference data for Stage 5."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, cast

from forgellm.alignment.schema import PreferenceRecord
from forgellm.post_training.schema import Message
from forgellm.structured_logging import JsonValue

Split = Literal["train", "validation", "test"]
VERIFIER_VERSION = "forgellm-preference-verifier-v1"
GENERATOR_VERSION = "forgellm-preference-generator-v1"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Task identity frozen before rejected-answer construction."""

    source_index: int
    family: str
    prompt: str
    answer: str
    metadata: dict[str, JsonValue]

    def fingerprint(self) -> str:
        """Hash task semantics only."""
        raw = json.dumps(
            {
                "family": self.family,
                "prompt": self.prompt,
                "answer": self.answer,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _task(index: int) -> TaskSpec:
    family_index = index % 5
    item = f"ITEM-{index:03d}"
    if family_index == 0:
        return TaskSpec(
            index,
            "exact_identifier",
            f"Reply with exactly {item} and nothing else.",
            item,
            {"expected": item},
        )
    if family_index == 1:
        left = 7 + index
        right = 3 + (index % 19)
        answer = str(left + right)
        return TaskSpec(
            index,
            "integer_sum",
            f"Return only the integer result of {left} + {right}.",
            answer,
            {"left": left, "right": right, "expected": answer},
        )
    if family_index == 2:
        answer = json.dumps({"id": item}, ensure_ascii=False, separators=(",", ":"))
        return TaskSpec(
            index,
            "json_object",
            f'Output only compact JSON with key "id" and value "{item}".',
            answer,
            {"expected": answer, "json_required": True},
        )
    if family_index == 3:
        values = [index % 17, (index * 7) % 23, (index * 11) % 29]
        answer = ",".join(str(value) for value in sorted(values))
        return TaskSpec(
            index,
            "integer_sort",
            f"Sort these integers ascending and return comma-separated values only: {values}.",
            answer,
            {"values": cast(JsonValue, values), "expected": answer},
        )
    source = f"code{index:03d}"
    answer = source[::-1]
    return TaskSpec(
        index,
        "reverse_string",
        f"Reverse this ASCII string and output only the result: {source}",
        answer,
        {"source": source, "expected": answer},
    )


def _wrong_answer(spec: TaskSpec) -> str:
    if spec.family == "integer_sum":
        return str(int(spec.answer) + 1)
    if spec.family == "json_object":
        return spec.answer.replace("ITEM", "WRONG")
    if spec.family == "integer_sort":
        return ",".join(reversed(spec.answer.split(",")))
    return f"WRONG-{spec.source_index:03d}"


def _rejected(spec: TaskSpec, reason: str) -> str:
    if reason == "wrong_answer":
        return _wrong_answer(spec)
    if reason == "format_error":
        return f"[{spec.answer}]"
    if reason == "extra_explanation":
        return f"The answer is {spec.answer}."
    if reason == "truncated":
        return spec.answer[:-1] if len(spec.answer) > 1 else "?"
    if reason == "repetition":
        return f"{spec.answer} {spec.answer} {spec.answer}"
    if reason == "length_gaming":
        return f"{spec.answer}\n" + ("additional filler " * 12).strip()
    raise ValueError(f"unknown rejection reason: {reason}")


def build_preference_splits() -> dict[Split, list[PreferenceRecord]]:
    """Build 384/64/64 task-disjoint pairs using hash order before rejection."""
    specs = sorted((_task(index) for index in range(512)), key=TaskSpec.fingerprint)
    split_specs: dict[Split, list[TaskSpec]] = {
        "train": specs[:384],
        "validation": specs[384:448],
        "test": specs[448:],
    }
    reasons = (
        "wrong_answer",
        "format_error",
        "extra_explanation",
        "truncated",
        "repetition",
        "length_gaming",
    )
    result: dict[Split, list[PreferenceRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for split, items in split_specs.items():
        for position, spec in enumerate(items):
            reason = reasons[(spec.source_index + position) % len(reasons)]
            result[split].append(
                PreferenceRecord(
                    record_id=f"preference-{spec.source_index:03d}",
                    prompt=(Message(role="user", content=spec.prompt),),
                    chosen=spec.answer,
                    rejected=_rejected(spec, reason),
                    task_family=spec.family,
                    preference_source="deterministic_verifier",
                    rejection_reason=reason,
                    verifier_version=VERIFIER_VERSION,
                    metadata={
                        **spec.metadata,
                        "generator_version": GENERATOR_VERSION,
                        "split": split,
                    },
                )
            )
    return result


def verify_response(record: PreferenceRecord, response: str) -> dict[str, float]:
    """Return separated, auditable reward components; exactness is deliberately strict."""
    expected = record.metadata.get("expected")
    if not isinstance(expected, str):
        raise ValueError(f"record {record.record_id} has no string expected answer")
    exact = float(response == expected)
    non_empty = float(bool(response.strip()))
    no_extra = float(response.strip() == response and len(response) <= max(len(expected) * 2, 16))
    no_repetition = float(response.count(expected) <= 1)
    structural = 1.0
    if record.metadata.get("json_required") is True:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            structural = 0.0
        else:
            structural = float(
                isinstance(parsed, dict)
                and set(parsed) == {"id"}
                and parsed["id"] == expected[7:-2]
            )
    return {
        "exact": exact,
        "non_empty": non_empty,
        "no_extra": no_extra,
        "no_repetition": no_repetition,
        "structural": structural,
        "total": exact,
    }


def adversarial_responses(record: PreferenceRecord) -> dict[str, str]:
    """Construct attacks that preserve a correct prefix but must receive zero total reward."""
    return {
        "correct_then_garbage": f"{record.chosen} garbage",
        "wrapped": f"Answer: {record.chosen}",
        "repeated": f"{record.chosen} {record.chosen}",
        "leading_space": f" {record.chosen}",
        "trailing_space": f"{record.chosen} ",
    }


def reward_total(components: dict[str, float]) -> float:
    """Extract the frozen training reward without silently combining audit-only components."""
    return components["total"]
