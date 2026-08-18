from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from repopilot.benchmarks.adapters import (
    SWEbenchLiveAdapter,
    SWEbenchLiveEvaluator,
    normalize_official_swebench_result,
)
from repopilot.benchmarks.contracts import EnvironmentHandle, TaskOutput

_REVISION = "a637bd46829f3132e12938c8a0ca93173a977b8e"
_PUBLIC_SHA256 = "1b8b16fba4eb6a892606382b1ad9d4b5bd7e5a3d2fe0d652b910c74d59cc8bb4"
_EVALUATOR_SHA256 = "604d0ba5c721a820a193fcc3f222137e10cdf088a20f8700cef494f512b38979"
_TASK_IDS = (
    "aws-cloudformation__cfn-lint-3767",
    "python-babel__babel-1141",
    "projectmesa__mesa-2394",
)


def _data_dir() -> Path:
    project = Path(__file__).resolve().parents[2]
    return project / "evaluation" / "benchmarks" / "data" / "swebench_live" / _REVISION


def _adapter() -> SWEbenchLiveAdapter:
    data_dir = _data_dir()
    return SWEbenchLiveAdapter(
        data_dir / "smoke_public.jsonl",
        data_dir / "smoke_evaluator.jsonl",
        dataset_revision=_REVISION,
        public_sha256=_PUBLIC_SHA256,
        evaluator_sha256=_EVALUATOR_SHA256,
    )


@pytest.mark.integration
def test_swebench_live_public_tasks_exclude_gold_patches_and_tests() -> None:
    adapter = _adapter()
    tasks = list(adapter.load("lite", _TASK_IDS))

    assert adapter.task_count == 3
    assert len(tasks) == 3
    assert all("test_patch" not in repr(task) for task in tasks)
    assert all("FAIL_TO_PASS" not in repr(task) for task in tasks)
    assert all(task.public_metadata["base_commit"] for task in tasks)


def test_swebench_live_prepare_fails_closed_without_instance_image(tmp_path: Path) -> None:
    adapter = _adapter()
    task = next(iter(adapter.load("lite", [_TASK_IDS[0]])))

    with pytest.raises(RuntimeError, match="not materialized"):
        adapter.prepare(task, tmp_path)


def test_swebench_live_rejects_non_official_score() -> None:
    adapter = _adapter()
    case = adapter.evaluator_case(_TASK_IDS[0])

    with pytest.raises(ValueError, match="official_evaluator"):
        asyncio.run(
            SWEbenchLiveEvaluator().evaluate(
                TaskOutput(final_answer="looks fixed"),
                case,
                EnvironmentHandle("smoke", Path.cwd()),
            )
        )


def test_swebench_live_normalizes_official_evaluator_result() -> None:
    adapter = _adapter()
    case = adapter.evaluator_case(_TASK_IDS[0])
    output = TaskOutput(
        structured_payload={
            "official_evaluator": {
                "resolved": True,
                "environment_built": True,
                "patch_applied": True,
                "fail_to_pass": 1.0,
                "pass_to_pass": 1.0,
            }
        }
    )

    outcome = asyncio.run(
        SWEbenchLiveEvaluator().evaluate(output, case, EnvironmentHandle("smoke", Path.cwd()))
    )

    assert outcome.task_success
    assert outcome.primary_metric_name == "resolved"
    assert outcome.domain_metrics["pass_to_pass_rate"] == 1.0


def test_swebench_live_imports_untouched_upstream_report() -> None:
    normalized = normalize_official_swebench_result(
        {
            _TASK_IDS[1]: {
                "patch_successfully_applied": True,
                "resolved": True,
                "tests_status": {
                    "FAIL_TO_PASS": {"success": ["target"], "failure": []},
                    "PASS_TO_PASS": {"success": ["regression-a", "regression-b"], "failure": []},
                },
            }
        },
        _TASK_IDS[1],
    )

    assert normalized == {
        "resolved": True,
        "environment_built": True,
        "patch_applied": True,
        "fail_to_pass": 1.0,
        "pass_to_pass": 1.0,
    }
