from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot.benchmarks.adapters import DABenchAdapter, DABenchEvaluator
from repopilot.benchmarks.contracts import EnvironmentHandle, TaskOutput

_REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"
_QUESTIONS_SHA256 = "430306c5813bf4f2b0e38f1b6ccee9693c36d5920b3322bd7c8ae46f475ce443"
_LABELS_SHA256 = "83b8fb8133c1794fcb6b42435e9cce124d22f68b71e2eedb37c973e40b5a18c2"
_TABLE_SHA256 = "411cf03455d6026823fbd3ab65e2839075a22f9a5c088b85aef0d272d79cca00"


def _data_dir() -> Path:
    project = Path(__file__).resolve().parents[2]
    return project / "evaluation" / "benchmarks" / "data" / "dabench" / _REVISION


def _adapter() -> DABenchAdapter:
    data_dir = _data_dir()
    return DABenchAdapter(
        data_dir / "da-dev-questions.jsonl",
        data_dir / "da-dev-labels.jsonl",
        data_dir,
        dataset_revision=_REVISION,
        questions_sha256=_QUESTIONS_SHA256,
        labels_sha256=_LABELS_SHA256,
        table_sha256={"test_ave.csv": _TABLE_SHA256},
    )


@pytest.mark.integration
def test_dabench_frozen_metadata_and_stratified_public_tasks() -> None:
    adapter = _adapter()
    tasks = list(adapter.load("dev", ["dabench-dev-0000", "dabench-dev-0005", "dabench-dev-0007"]))

    assert adapter.task_count == 257
    assert [task.public_metadata["level"] for task in tasks] == ["easy", "medium", "hard"]
    assert all(task.assets[0].sha256 == _TABLE_SHA256 for task in tasks)
    assert all("common_answers" not in repr(task) for task in tasks)


@pytest.mark.integration
@pytest.mark.parametrize("task_id", ["dabench-dev-0000", "dabench-dev-0005", "dabench-dev-0007"])
def test_three_real_dabench_cases_pass_closed_form_protocol_smoke(task_id: str) -> None:
    adapter = _adapter()
    answer = ", ".join(f"@{key}[{value}]" for key, value in adapter.oracle_answers(task_id))
    output = TaskOutput(final_answer=answer)
    case = adapter.evaluator_case(task_id)

    outcome = asyncio.run(
        DABenchEvaluator(adapter).evaluate(output, case, EnvironmentHandle("smoke", Path.cwd()))
    )

    assert outcome.task_success
    assert outcome.domain_metrics["subquestion_accuracy"] == 1.0
    assert outcome.domain_metrics["output_format_valid"] == 1.0


@pytest.mark.integration
def test_dabench_prepare_copies_only_current_table(tmp_path: Path) -> None:
    adapter = _adapter()
    task = next(iter(adapter.load("dev", ["dabench-dev-0000"])))

    prepared = adapter.prepare(task, tmp_path)

    assert [path.name for path in prepared.work_dir.iterdir()] == ["input.csv"]
    assert prepared.public_payload == {"table_path": "input.csv", "source_file": "test_ave.csv"}


def test_dabench_evaluator_rejects_wrong_value() -> None:
    adapter = _adapter()
    case = adapter.evaluator_case("dabench-dev-0000")
    outcome = asyncio.run(
        DABenchEvaluator(adapter).evaluate(
            TaskOutput(final_answer="@mean_fare[0]"),
            case,
            EnvironmentHandle("smoke", Path.cwd()),
        )
    )

    assert not outcome.task_success
    assert outcome.failure_type == "answer"
