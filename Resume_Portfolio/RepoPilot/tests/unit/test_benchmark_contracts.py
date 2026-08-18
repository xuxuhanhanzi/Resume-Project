from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.benchmarks.adapters import CodeRepairAdapter
from repopilot.benchmarks.contracts import BenchmarkDomain, BenchmarkTask
from repopilot.benchmarks.manifest import BenchmarkManifest, ManifestAsset, ManifestTask
from repopilot.core.budgets import RunBudget
from repopilot.task import EvaluatorTaskSpec, PublicTaskSpec

_SHA256 = "a" * 64


def test_public_benchmark_task_rejects_evaluator_metadata() -> None:
    with pytest.raises(ValueError, match="evaluator-only key"):
        BenchmarkTask(
            "frames",
            "revision",
            "task-1",
            BenchmarkDomain.RESEARCH,
            "Answer with evidence",
            "offline-corpus",
            public_metadata={"nested": {"gold_answer": "secret"}},
        )


def test_manifest_round_trip_keeps_reproducibility_fields(tmp_path: Path) -> None:
    manifest = BenchmarkManifest(
        benchmark_id="frames",
        split="test",
        dataset_revision="dataset-commit",
        evaluator_revision="evaluator-commit",
        tasks=(ManifestTask("task-1", (ManifestAsset("asset://one", _SHA256),)),),
        model_id="local-model",
        model_revision="model-revision",
        prompt_revision="prompt-v1",
        tool_config_revision="tools-v1",
        environment_revision="environment-v1",
        dependency_lock_sha256=_SHA256,
        random_seed=7,
        budget=RunBudget(max_iterations=3),
        model_parameters={"temperature": 0.0},
    )
    path = tmp_path / "manifest.json"

    manifest.save(path)
    restored = BenchmarkManifest.load(path)

    assert restored == manifest
    assert "oracle" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name",
    [
        "frames_protocol_smoke_v1.json",
        "dabench_protocol_smoke_v1.json",
        "swebench_live_protocol_smoke_v1.json",
    ],
)
def test_committed_protocol_manifests_are_valid(name: str) -> None:
    project = Path(__file__).resolve().parents[2]

    manifest = BenchmarkManifest.load(project / "evaluation" / "benchmarks" / "manifests" / name)

    assert len(manifest.tasks) == 3


def test_legacy_code_task_has_public_and_private_boundaries(tmp_path: Path) -> None:
    public = PublicTaskSpec(
        "repair-1",
        tmp_path,
        "Fix the bug",
        trusted_fixture=True,
    )
    adapter = CodeRepairAdapter(
        [EvaluatorTaskSpec(public, hidden_tests=("secret_test.py",), gold_patch_sha256=_SHA256)]
    )

    task = next(iter(adapter.load("local")))
    case = adapter.evaluator_case(task.task_id)

    assert task.domain is BenchmarkDomain.SOFTWARE_ENGINEERING
    assert "secret_test.py" not in repr(task)
    assert case.evaluator_config["hidden_test_count"] == 1
    assert adapter.evaluator_spec(task.task_id).hidden_tests == ("secret_test.py",)
