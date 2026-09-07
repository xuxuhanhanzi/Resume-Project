"""Answer-centric metrics for structured or plain ms-swift inference outputs."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from forgemm.reasoning.parser import ParseError, parse_prediction
from forgemm.rewards.answer import answer_reward

_ANSWER_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def evaluate_task_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score relaxed answer accuracy without conflating it with protocol validity."""

    if not rows:
        raise ValueError("inference result is empty")
    counts = Counter[str]()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    parse_errors = Counter[str]()
    for row in rows:
        response = str(row["response"])
        reference, _ = extract_answer(str(row["labels"]))
        answer, parse_error = extract_answer(response)
        correct = answer_reward(answer, reference) == 1.0
        question_type = str(row.get("question_type", "unknown") or "unknown")
        counts["samples"] += 1
        counts["correct"] += correct
        counts["format_pass"] += parse_error is None
        by_type[question_type]["samples"] += 1
        by_type[question_type]["correct"] += correct
        if parse_error is not None:
            parse_errors[parse_error] += 1

    return {
        "schema_version": "forgemm-task-infer-eval-v1",
        "samples": counts["samples"],
        "correct": counts["correct"],
        "task_accuracy": counts["correct"] / counts["samples"],
        "format_compliance": counts["format_pass"] / counts["samples"],
        "parse_errors": dict(sorted(parse_errors.items())),
        "by_question_type": {
            key: {
                "samples": value["samples"],
                "correct": value["correct"],
                "task_accuracy": value["correct"] / value["samples"],
            }
            for key, value in sorted(by_type.items())
        },
    }


def extract_answer(response: str) -> tuple[str, str | None]:
    """Return an answer plus a protocol diagnostic, tolerating plain-answer baselines."""

    try:
        return parse_prediction(response).answer, None
    except ParseError as exc:
        matches = _ANSWER_BLOCK.findall(response)
        if matches:
            return matches[-1].strip(), exc.code
        return response.strip(), exc.code


__all__ = ["evaluate_task_rows", "extract_answer"]
