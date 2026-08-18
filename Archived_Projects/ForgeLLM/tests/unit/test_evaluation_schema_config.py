from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgellm.evaluation.cases import build_stage6_cases, read_cases, write_cases
from forgellm.evaluation.config import EvaluationConfigError, load_stage6_config
from forgellm.evaluation.schema import (
    EvaluationCase,
    EvaluationManifest,
    EvaluationSchemaError,
    GenerationSettings,
    ModelIdentity,
)
from forgellm.post_training.schema import Message


def _manifest() -> EvaluationManifest:
    model = ModelIdentity("q0", "model", "revision", "a" * 64, None)
    generation = GenerationSettings(32, False, 0.0, 1.0, 7, "b" * 64)
    return EvaluationManifest(
        "forgellm-stage6-evaluation-v1",
        "test",
        (model,),
        "c" * 64,
        {"test": "d" * 64},
        generation,
        "commit+dirty",
    )


def test_manifest_round_trip_and_fingerprint() -> None:
    manifest = _manifest()
    assert EvaluationManifest.from_dict(manifest.as_dict()) == manifest
    assert len(manifest.fingerprint()) == 64


def test_manifest_rejects_tampering() -> None:
    raw = _manifest().as_dict()
    raw["run_name"] = "changed"
    with pytest.raises(EvaluationSchemaError, match="fingerprint"):
        EvaluationManifest.from_dict(raw)


def test_greedy_generation_rejects_nonzero_temperature() -> None:
    with pytest.raises(EvaluationSchemaError, match="greedy"):
        GenerationSettings(1, False, 0.5, 1.0, 0, "a" * 64)


def test_case_round_trip_and_historical_assistant_turn(tmp_path: Path) -> None:
    case = EvaluationCase(
        "case",
        "correctness",
        (Message("user", "question"),),
        "answer",
        "test",
        "test",
        {},
    )
    path = tmp_path / "cases.jsonl"
    write_cases(path, [case])
    assert read_cases(path) == [case]
    multi_turn = EvaluationCase(
        "multi",
        "correctness",
        (
            Message("user", "first"),
            Message("assistant", "history"),
            Message("user", "second"),
        ),
        "answer",
        "test",
        "test",
        {},
    )
    assert multi_turn.prompt[-2].role == "assistant"


def test_frozen_project_case_budget_is_64() -> None:
    cases = build_stage6_cases(
        Path("data/processed/stage4_correctness_v1/test.jsonl"),
        Path("data/processed/stage5_preference_constraints_v1/test.jsonl"),
    )
    assert len(cases) == 64
    assert sum(case.task_type == "correctness" for case in cases) == 16
    assert sum(case.task_type == "preference" for case in cases) == 16
    assert sum(case.task_type == "robustness" for case in cases) == 32
    multi_turn = next(case for case in cases if case.case_id == "correctness-087")
    assert [message.role for message in multi_turn.prompt] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_stage6_config_loads_and_rejects_unknown_section(tmp_path: Path) -> None:
    config = load_stage6_config(Path("configs/evaluation/stage6_final.toml"))
    assert config.systems.repeats == 5
    assert config.fingerprint() == config.fingerprint()
    raw = Path("configs/evaluation/stage6_final.toml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(raw + "\n[unknown]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(EvaluationConfigError, match="five sections"):
        load_stage6_config(invalid)


def test_case_reader_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "case.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "x",
                "task_type": "correctness",
                "prompt": [{"role": "user", "content": "x"}],
                "expected_response": "y",
                "source": "z",
                "split": "test",
                "constraints": {},
                "parent_case_id": None,
                "unknown": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvaluationSchemaError, match="fields differ"):
        read_cases(path)
