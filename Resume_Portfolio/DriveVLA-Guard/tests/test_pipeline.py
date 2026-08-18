from drivevla_guard.backends import SyntheticBackend
from drivevla_guard.config import GenerationConfig, GuardConfig, RouterConfig
from drivevla_guard.pipeline import GuardPipeline
from drivevla_guard.synthetic import build_synthetic_manifest
from drivevla_guard.types import SceneContext


def blocked_context() -> SceneContext:
    return SceneContext.from_dict(build_synthetic_manifest(2)[1]["scene"])


def test_reranker_changes_unsafe_first_choice() -> None:
    config = GuardConfig(
        rerank_enabled=True,
        generation=GenerationConfig(fast_candidates=4),
        router=RouterConfig(enabled=False),
    )
    result = GuardPipeline(SyntheticBackend(), config).plan({}, blocked_context())
    assert result.selected.candidate.candidate_id != "fast-0"
    assert (result.selected.risk.collision or 0.0) < 1.0


def test_router_uses_slow_candidate_for_high_risk_single_path() -> None:
    config = GuardConfig(
        rerank_enabled=False,
        generation=GenerationConfig(fast_candidates=1, slow_candidates=1),
        router=RouterConfig(enabled=True, risk_threshold=1.0),
    )
    result = GuardPipeline(SyntheticBackend(), config).plan({}, blocked_context())
    assert result.routed_to_slow
    assert result.selected.candidate.mode == "slow"
    assert "high_risk" in result.route_reasons


def test_pipeline_is_deterministic() -> None:
    config = GuardConfig(router=RouterConfig(enabled=False))
    pipeline = GuardPipeline(SyntheticBackend(), config)
    first = pipeline.plan({}, blocked_context()).to_dict()
    second = pipeline.plan({}, blocked_context()).to_dict()
    first.pop("total_latency_ms")
    second.pop("total_latency_ms")
    for item in first["fast_candidates"]:
        item["candidate"].pop("latency_ms")
    for item in second["fast_candidates"]:
        item["candidate"].pop("latency_ms")
    assert first == second
