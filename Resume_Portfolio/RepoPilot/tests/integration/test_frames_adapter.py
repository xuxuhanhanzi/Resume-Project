from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot.benchmarks.adapters import FramesAdapter, FramesEvaluator, FramesMode
from repopilot.benchmarks.contracts import EnvironmentHandle, EvidenceRef, TaskOutput

_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"


def _dataset_path() -> Path:
    project = Path(__file__).resolve().parents[2]
    return project / "evaluation" / "benchmarks" / "data" / "frames" / _REVISION / "test.tsv"


def _adapter(mode: FramesMode = FramesMode.ORACLE_DOCUMENT) -> FramesAdapter:
    return FramesAdapter(
        _dataset_path(), dataset_revision=_REVISION, dataset_sha256=_SHA256, mode=mode
    )


@pytest.mark.integration
def test_frozen_frames_dataset_has_expected_shape_and_no_public_answers() -> None:
    adapter = _adapter()
    tasks = list(adapter.load("test", ["frames-test-0000", "frames-test-0001", "frames-test-0002"]))

    assert adapter.task_count == 824
    assert len(tasks) == 3
    assert all(task.assets for task in tasks)
    assert all(adapter.oracle_answer(task.task_id) not in repr(task) for task in tasks)


@pytest.mark.integration
def test_retrieval_mode_does_not_expose_gold_document_urls() -> None:
    adapter = _adapter(FramesMode.RETRIEVAL)
    task = next(iter(adapter.load("test", ["frames-test-0000"])))

    assert task.assets == ()
    assert "wikipedia" not in repr(task).casefold()


@pytest.mark.integration
@pytest.mark.parametrize("task_id", ["frames-test-0000", "frames-test-0001", "frames-test-0002"])
def test_three_real_frames_cases_pass_deterministic_protocol_smoke(task_id: str) -> None:
    adapter = _adapter()
    task = next(iter(adapter.load("test", [task_id])))
    answer = adapter.oracle_answer(task_id)
    evidence = tuple(EvidenceRef(asset.uri, asset.uri) for asset in task.assets)
    output = TaskOutput(final_answer=answer, evidence=evidence)
    case = adapter.evaluator_case(task_id)

    outcome = asyncio.run(
        FramesEvaluator(adapter).evaluate(output, case, EnvironmentHandle("smoke", Path.cwd()))
    )

    assert outcome.task_success
    assert outcome.primary_metric_value == 1.0
    assert outcome.domain_metrics["evidence_recall"] == 1.0


def test_frames_adapter_rejects_unpinned_content() -> None:
    with pytest.raises(ValueError, match="hash"):
        FramesAdapter(_dataset_path(), dataset_revision=_REVISION, dataset_sha256="0" * 64)
