"""Stage 3 experiment configuration tests."""

from pathlib import Path

import pytest

from forgellm.training.config import TrainingConfig, TrainingConfigError, load_experiment_config


def test_bounded_config_resolves_token_budget_and_model_shape() -> None:
    experiment = load_experiment_config(Path("configs/training/stage3_bounded.toml"))

    assert experiment.model.d_model == 256
    assert experiment.training.max_tokens == 1_000_000
    assert experiment.training.max_duration_seconds == 3600.0
    assert experiment.training.effective_tokens_per_step == 8 * 4 * 127
    assert len(experiment.fingerprint()) == 64


def test_mtp_configuration_requires_a_weight_and_room_for_future_tokens() -> None:
    with pytest.raises(TrainingConfigError, match="mtp_loss_weight"):
        TrainingConfig(mtp_future_tokens=2, mtp_loss_weight=0.0)
    with pytest.raises(TrainingConfigError, match="smaller"):
        TrainingConfig(sequence_length=2, mtp_future_tokens=2, mtp_loss_weight=0.1)


def test_training_config_rejects_silent_cuda_or_precision_typos() -> None:
    with pytest.raises(TrainingConfigError, match="device"):
        TrainingConfig(device="gpu")  # type: ignore[arg-type]
    with pytest.raises(TrainingConfigError, match="precision"):
        TrainingConfig(precision="float32")  # type: ignore[arg-type]
