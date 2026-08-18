import pytest

from drivevla_guard.config import GenerationConfig, RiskWeights, RouterConfig, load_config


def test_load_e4_config(project_root) -> None:
    config = load_config(project_root / "configs/e4_guard.yaml")
    assert config.rerank_enabled
    assert config.generation.fast_candidates == 4
    assert config.router.enabled


def test_unknown_config_key_fails(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: GenerationConfig(fast_candidates=0), "candidate counts"),
        (lambda: GenerationConfig(top_p=1.2), "top_p"),
        (lambda: RiskWeights(collision=-1.0), "non-negative"),
        (lambda: RouterConfig(uncertainty_threshold=1.1), "uncertainty_threshold"),
    ],
)
def test_invalid_config_is_rejected(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
