"""Build and read the frozen Stage 6 task set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from forgellm.alignment.schema import read_preference_jsonl
from forgellm.evaluation.schema import EvaluationCase
from forgellm.post_training.schema import Message, load_instruction_jsonl
from forgellm.structured_logging import JsonValue


def _correctness_constraints(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    allowed = {
        "starts_with",
        "required_substrings",
        "forbidden_substrings",
        "exact_word_count",
        "require_json",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def build_stage6_cases(
    correctness_test: Path,
    preference_test: Path,
) -> list[EvaluationCase]:
    """Create original and frozen semantically equivalent robustness cases."""
    cases: list[EvaluationCase] = []
    correctness = load_instruction_jsonl(correctness_test)
    preferences = read_preference_jsonl(preference_test)
    for correctness_record in correctness:
        expected = correctness_record.metadata.get("expected_response")
        if not isinstance(expected, str):
            raise ValueError(
                f"correctness record {correctness_record.record_id} has no expected response"
            )
        # Keep historical assistant turns. Only the final answer is hidden from
        # evaluation; dropping every assistant turn would create user/user runs.
        prompt = correctness_record.messages[:-1]
        constraints = _correctness_constraints(correctness_record.metadata)
        cases.append(
            EvaluationCase(
                case_id=correctness_record.record_id,
                task_type="correctness",
                prompt=prompt,
                expected_response=expected,
                source=correctness_record.source,
                split="test",
                constraints=constraints,
            )
        )
        restated = tuple(
            Message(
                message.role,
                f"Please follow this equivalent request carefully: {message.content}",
            )
            if message.role == "user"
            else message
            for message in prompt
        )
        cases.append(
            EvaluationCase(
                case_id=f"robust-{correctness_record.record_id}",
                task_type="robustness",
                prompt=restated,
                expected_response=expected,
                source="forgellm-stage6-robustness-v1",
                split="test",
                constraints=constraints,
                parent_case_id=correctness_record.record_id,
            )
        )
    # The frozen Stage 6 budget is 16 correctness + 16 preference originals,
    # each with exactly one robustness partner (64 cases total).
    for preference_record in preferences[:16]:
        metadata_constraints: dict[str, JsonValue] = {}
        if preference_record.metadata.get("json_required") is True:
            metadata_constraints["require_json"] = True
        cases.append(
            EvaluationCase(
                case_id=preference_record.record_id,
                task_type="preference",
                prompt=preference_record.prompt,
                expected_response=preference_record.chosen,
                source="forgellm-stage5-preference-data-v1",
                split="test",
                constraints=metadata_constraints,
            )
        )
        restated_prompt = tuple(
            Message(
                message.role,
                f"Please answer this restated task exactly and add nothing else: {message.content}",
            )
            if message.role == "user"
            else message
            for message in preference_record.prompt
        )
        cases.append(
            EvaluationCase(
                case_id=f"robust-{preference_record.record_id}",
                task_type="robustness",
                prompt=restated_prompt,
                expected_response=preference_record.chosen,
                source="forgellm-stage6-robustness-v1",
                split="test",
                constraints=metadata_constraints,
                parent_case_id=preference_record.record_id,
            )
        )
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Stage 6 case IDs are not unique")
    return sorted(cases, key=lambda case: case.case_id)


def write_cases(path: Path, cases: list[EvaluationCase]) -> None:
    """Create a deterministic case file without overwriting evidence."""
    if not cases or path.exists():
        raise FileExistsError(f"Stage 6 case output is empty or already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(case.as_dict(), ensure_ascii=False, sort_keys=True) + "\n" for case in cases
        ),
        encoding="utf-8",
        newline="\n",
    )


def read_cases(path: Path) -> list[EvaluationCase]:
    """Read and revalidate every frozen case."""
    cases = [
        EvaluationCase.from_dict(cast(object, json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Stage 6 case file is empty or contains duplicate IDs")
    return cases
