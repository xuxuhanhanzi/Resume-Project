import pytest

from drivevla_guard.backends import SyntheticBackend
from drivevla_guard.config import GenerationConfig, GuardConfig, RouterConfig
from drivevla_guard.evaluation import EvaluationRunner, compare_results
from drivevla_guard.io import write_jsonl
from drivevla_guard.pipeline import GuardPipeline
from drivevla_guard.synthetic import build_synthetic_manifest


def make_pipeline(rerank: bool, candidates: int) -> GuardPipeline:
    config = GuardConfig(
        rerank_enabled=rerank,
        generation=GenerationConfig(fast_candidates=candidates),
        router=RouterConfig(enabled=False),
    )
    return GuardPipeline(SyntheticBackend(), config)


def test_evaluation_resume_and_paired_compare(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    guarded = tmp_path / "guarded.jsonl"
    write_jsonl(manifest, build_synthetic_manifest(8))
    baseline_summary = EvaluationRunner(make_pipeline(False, 1)).run(manifest, baseline)
    guarded_runner = EvaluationRunner(make_pipeline(True, 4))
    guarded_summary = guarded_runner.run(manifest, guarded)
    assert guarded_summary["collision_failures"] <= baseline_summary["collision_failures"]
    assert guarded_runner.run(manifest, guarded) == guarded_summary
    comparison = compare_results(baseline, guarded)
    assert comparison["paired_scenes"] == 8


def test_resume_rejects_config_drift(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "results.jsonl"
    write_jsonl(manifest, build_synthetic_manifest(2))
    EvaluationRunner(make_pipeline(False, 1)).run(manifest, output)
    with pytest.raises(ValueError, match="config hash"):
        EvaluationRunner(make_pipeline(True, 4)).run(manifest, output)
