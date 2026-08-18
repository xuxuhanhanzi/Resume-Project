"""Inspectable single-GPU/CPU pretraining loop with exact-resume state."""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler

from forgellm.model.decoder import DecoderLM, next_token_loss
from forgellm.model.frontier_layers import (
    MultiTokenPredictionHead,
    multi_token_prediction_loss,
)
from forgellm.structured_logging import JsonValue, write_jsonl_event
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.training.checkpoint import (
    CheckpointError,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from forgellm.training.config import ExperimentConfig
from forgellm.training.data import (
    DeterministicBatchStream,
    PackedTokenDataset,
    validation_batches,
)
from forgellm.training.metrics import (
    StepMetrics,
    ValidationMetrics,
    bits_per_byte,
    perplexity_from_loss,
)
from forgellm.training.optimization import (
    OptimizerBundle,
    WarmupCosineScheduler,
    build_optimizer,
)

_TRAINER_STATE_SCHEMA = "forgellm-trainer-state-v1"


class TrainingDivergedError(RuntimeError):
    """Raised immediately for a non-finite loss or gradient."""


@dataclass(slots=True)
class TrainerState:
    """Small mutable progress state that is serialized in every checkpoint."""

    step: int = 0
    tokens_seen: int = 0
    elapsed_seconds: float = 0.0

    def state_dict(self) -> dict[str, object]:
        return {"schema_version": _TRAINER_STATE_SCHEMA, **asdict(self)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if set(state) != {"schema_version", "step", "tokens_seen", "elapsed_seconds"}:
            raise CheckpointError("trainer state has unexpected fields")
        if state["schema_version"] != _TRAINER_STATE_SCHEMA:
            raise CheckpointError("unsupported trainer state schema")
        step = state["step"]
        tokens_seen = state["tokens_seen"]
        elapsed_seconds = state["elapsed_seconds"]
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or isinstance(tokens_seen, bool)
            or not isinstance(tokens_seen, int)
            or tokens_seen < 0
            or isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, int | float)
            or elapsed_seconds < 0
        ):
            raise CheckpointError("trainer state values are invalid")
        self.step = step
        self.tokens_seen = tokens_seen
        self.elapsed_seconds = float(elapsed_seconds)


def seed_everything(seed: int) -> None:
    """Seed every RNG used in the Stage 3 baseline."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    """Resolve an explicit device and never silently ignore a CUDA request."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training config requests CUDA, but CUDA is unavailable")
    return torch.device(choice)


class Trainer:
    """Bounded trainer whose transitions are deliberately easy to audit."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        model: DecoderLM,
        tokenizer: ByteBPETokenizer,
        train_stream: DeterministicBatchStream,
        validation_dataset: PackedTokenDataset,
        output_dir: Path,
    ) -> None:
        self.experiment = experiment
        self.config = experiment.training
        self.device = resolve_device(self.config.device)
        if self.config.precision == "fp16" and self.device.type != "cuda":
            raise RuntimeError("FP16 Stage 3 training requires CUDA")
        if (
            self.config.precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("selected CUDA device does not support BF16")
        if model.config != experiment.model:
            raise ValueError("model architecture does not match experiment config")
        if tokenizer.vocab_size != experiment.model.vocab_size:
            raise ValueError("tokenizer and model vocabulary sizes do not match")
        if train_stream.batch_size != self.config.micro_batch_size:
            raise ValueError("training stream batch size does not match config")
        if train_stream.dataset.sequence_length != self.config.sequence_length:
            raise ValueError("training sequence length does not match config")
        if validation_dataset.sequence_length != self.config.sequence_length:
            raise ValueError("validation sequence length does not match config")

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_stream = train_stream
        self.validation_dataset = validation_dataset
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "metrics.jsonl"
        self.checkpoint_path = output_dir / "latest.pt"
        self.state = TrainerState()
        self.history: list[StepMetrics] = []

        self.mtp_head: MultiTokenPredictionHead | None = None
        named_parameters = list(self.model.named_parameters())
        if self.config.mtp_future_tokens:
            self.mtp_head = MultiTokenPredictionHead(
                experiment.model.d_model,
                experiment.model.vocab_size,
                self.config.mtp_future_tokens,
            ).to(self.device)
            named_parameters.extend(
                (f"mtp_head.{name}", parameter)
                for name, parameter in self.mtp_head.named_parameters()
            )
        self.optimizer: OptimizerBundle = build_optimizer(named_parameters, self.config)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.max_steps,
            min_lr_ratio=self.config.min_lr_ratio,
        )
        self.scaler = GradScaler("cuda", enabled=self.config.precision == "fp16")

    @property
    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = list(self.model.parameters())
        if self.mtp_head is not None:
            parameters.extend(self.mtp_head.parameters())
        unique: dict[int, nn.Parameter] = {}
        for parameter in parameters:
            if parameter.requires_grad:
                unique[id(parameter)] = parameter
        return tuple(unique.values())

    def _autocast_dtype(self) -> torch.dtype | None:
        if self.config.precision == "bf16":
            return torch.bfloat16
        if self.config.precision == "fp16":
            return torch.float16
        return None

    def _forward_losses(self, token_ids: Tensor) -> tuple[Tensor, Tensor | None, Tensor]:
        use_mtp = self.mtp_head is not None
        output = self.model(token_ids, return_hidden_states=use_mtp)
        main_loss = next_token_loss(output.logits, token_ids)
        mtp_loss: Tensor | None = None
        total_loss = main_loss
        if self.mtp_head is not None:
            if output.hidden_states is None:
                raise RuntimeError("decoder did not expose hidden states required by MTP")
            mtp_logits = self.mtp_head(output.hidden_states)
            mtp_loss, _ = multi_token_prediction_loss(mtp_logits, token_ids)
            total_loss = main_loss + self.config.mtp_loss_weight * mtp_loss
        return main_loss, mtp_loss, total_loss

    def _check_gradients_finite(self) -> None:
        for name, parameter in list(self.model.named_parameters()) + (
            list(self.mtp_head.named_parameters()) if self.mtp_head is not None else []
        ):
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise TrainingDivergedError(f"non-finite gradient detected in {name}")

    def train_step(self) -> StepMetrics:
        """Perform exactly one optimizer update from one or more micro-batches."""
        if self.state.step >= self.config.max_steps:
            raise RuntimeError("configured max_steps has already been reached")
        self.model.train()
        if self.mtp_head is not None:
            self.mtp_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        main_loss_sum = 0.0
        mtp_loss_sum = 0.0
        total_loss_sum = 0.0
        target_tokens = 0

        autocast_dtype = self._autocast_dtype()
        for _ in range(self.config.gradient_accumulation_steps):
            batch = self.train_stream.next_batch(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                main_loss, mtp_loss, total_loss = self._forward_losses(batch.token_ids)
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
            if not bool(torch.isfinite(total_loss)):
                raise TrainingDivergedError(
                    f"non-finite loss at optimizer step {self.state.step + 1}"
                )
            self.scaler.scale(scaled_loss).backward()  # type: ignore[no-untyped-call]
            main_loss_sum += float(main_loss.detach())
            mtp_loss_sum += float(mtp_loss.detach()) if mtp_loss is not None else 0.0
            total_loss_sum += float(total_loss.detach())
            target_tokens += batch.target_tokens

        for optimizer in self.optimizer:
            self.scaler.unscale_(optimizer)
        self._check_gradients_finite()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters,
            self.config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        gradient_norm = float(gradient_norm_tensor)
        for optimizer in self.optimizer:
            self.scaler.step(optimizer)
        self.scaler.update()
        learning_rates = tuple(self.scheduler.current_lrs())
        self.scheduler.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        step_seconds = time.perf_counter() - started
        self.state.step += 1
        self.state.tokens_seen += target_tokens
        accumulation = self.config.gradient_accumulation_steps
        peak_memory = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else None
        )
        metrics = StepMetrics(
            step=self.state.step,
            loss=total_loss_sum / accumulation,
            main_loss=main_loss_sum / accumulation,
            mtp_loss=mtp_loss_sum / accumulation if self.mtp_head is not None else None,
            learning_rates=learning_rates,
            gradient_norm=gradient_norm,
            target_tokens=target_tokens,
            tokens_seen=self.state.tokens_seen,
            step_seconds=step_seconds,
            tokens_per_second=target_tokens / max(step_seconds, 1e-12),
            peak_memory_mib=peak_memory,
        )
        self.history.append(metrics)
        if self.state.step % self.config.log_interval == 0:
            write_jsonl_event(self.log_path, "train_step", metrics.as_dict())
        return metrics

    def validate(self) -> ValidationMetrics:
        """Evaluate a fixed prefix without changing parameters, RNG, or data cursor."""
        model_was_training = self.model.training
        mtp_was_training = self.mtp_head.training if self.mtp_head is not None else False
        self.model.eval()
        if self.mtp_head is not None:
            self.mtp_head.eval()
        batches = validation_batches(
            self.validation_dataset,
            batch_size=self.config.micro_batch_size,
            batch_count=self.config.validation_batches,
            device=self.device,
        )
        total_nll = 0.0
        target_tokens = 0
        target_bytes = 0
        autocast_dtype = self._autocast_dtype()
        with torch.inference_mode():
            for batch in batches:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    output = self.model(batch.token_ids)
                    loss = next_token_loss(output.logits, batch.token_ids)
                total_nll += float(loss) * batch.target_tokens
                target_tokens += batch.target_tokens
                target_bytes += batch.target_bytes
        if model_was_training:
            self.model.train()
        if self.mtp_head is not None and mtp_was_training:
            self.mtp_head.train()
        mean_loss = total_nll / target_tokens
        metrics = ValidationMetrics(
            loss=mean_loss,
            perplexity=perplexity_from_loss(mean_loss),
            bits_per_byte=bits_per_byte(total_nll_nats=total_nll, target_bytes=target_bytes),
            target_tokens=target_tokens,
            target_bytes=target_bytes,
        )
        write_jsonl_event(
            self.log_path,
            "validation",
            {"step": self.state.step, **metrics.as_dict()},
        )
        return metrics

    def _checkpoint_payload(self) -> dict[str, object]:
        return {
            "config_fingerprint": self.experiment.fingerprint(),
            "tokenizer_fingerprint": self.tokenizer.fingerprint(),
            "model_state": self.model.state_dict(),
            "mtp_head_state": self.mtp_head.state_dict() if self.mtp_head is not None else None,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "trainer_state": self.state.state_dict(),
            "train_stream_state": self.train_stream.state_dict(),
            "rng_state": capture_rng_state(),
        }

    def save_checkpoint(self, path: Path | None = None) -> Path:
        """Persist all state needed to reproduce the next batch and update."""
        destination = path or self.checkpoint_path
        save_checkpoint_atomic(destination, self._checkpoint_payload())
        write_jsonl_event(
            self.log_path,
            "checkpoint_saved",
            {"path": str(destination), "step": self.state.step},
        )
        return destination

    def load_checkpoint(self, path: Path | None = None) -> None:
        """Validate compatibility before restoring the complete training state."""
        source = path or self.checkpoint_path
        payload = load_checkpoint(source, map_location=self.device)
        expected_fields = {
            "config_fingerprint",
            "tokenizer_fingerprint",
            "model_state",
            "mtp_head_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "trainer_state",
            "train_stream_state",
            "rng_state",
        }
        if set(payload) != expected_fields:
            raise CheckpointError("checkpoint payload has unexpected fields")
        if payload["config_fingerprint"] != self.experiment.fingerprint():
            raise CheckpointError("checkpoint experiment config does not match")
        if payload["tokenizer_fingerprint"] != self.tokenizer.fingerprint():
            raise CheckpointError("checkpoint tokenizer does not match")
        mtp_state = payload["mtp_head_state"]
        if (self.mtp_head is None) != (mtp_state is None):
            raise CheckpointError("checkpoint MTP layout does not match")

        model_state = payload["model_state"]
        optimizer_state = payload["optimizer_state"]
        scheduler_state = payload["scheduler_state"]
        scaler_state = payload["scaler_state"]
        trainer_state = payload["trainer_state"]
        stream_state = payload["train_stream_state"]
        rng_state = payload["rng_state"]
        if not all(
            isinstance(value, dict)
            for value in (
                model_state,
                optimizer_state,
                scheduler_state,
                scaler_state,
                trainer_state,
                stream_state,
                rng_state,
            )
        ):
            raise CheckpointError("checkpoint contains an invalid state mapping")
        self.model.load_state_dict(cast(dict[str, Tensor], model_state))
        if self.mtp_head is not None:
            if not isinstance(mtp_state, dict):
                raise CheckpointError("checkpoint MTP state is invalid")
            self.mtp_head.load_state_dict(cast(dict[str, Tensor], mtp_state))
        self.optimizer.load_state_dict(cast(dict[str, object], optimizer_state))
        self.scheduler.load_state_dict(cast(dict[str, object], scheduler_state))
        self.scaler.load_state_dict(cast(dict[str, object], scaler_state))
        self.state.load_state_dict(cast(dict[str, object], trainer_state))
        self.train_stream.load_state_dict(cast(dict[str, object], stream_state))
        restore_rng_state(cast(dict[str, object], rng_state))
        write_jsonl_event(
            self.log_path,
            "checkpoint_loaded",
            {"path": str(source), "step": self.state.step},
        )

    def _stop_reason(self, *, elapsed_seconds: float) -> str | None:
        if self.state.step >= self.config.max_steps:
            return "max_steps"
        if self.config.max_tokens is not None and self.state.tokens_seen >= self.config.max_tokens:
            return "max_tokens"
        if (
            self.config.max_duration_seconds is not None
            and elapsed_seconds >= self.config.max_duration_seconds
        ):
            return "max_duration_seconds"
        return None

    def train(self, *, stop_after_step: int | None = None) -> tuple[list[StepMetrics], str]:
        """Run until the first explicit step, token, or wall-clock stop condition."""
        if stop_after_step is not None and not (
            self.state.step < stop_after_step <= self.config.max_steps
        ):
            raise ValueError("stop_after_step must be after current step and within max_steps")
        run_started = time.perf_counter()
        prior_elapsed = self.state.elapsed_seconds
        stop_reason = self._stop_reason(elapsed_seconds=prior_elapsed)
        while stop_reason is None:
            self.train_step()
            if self.state.step % self.config.validation_interval == 0:
                self.validate()
            if (
                self.config.checkpoint_interval
                and self.state.step % self.config.checkpoint_interval == 0
            ):
                self.save_checkpoint()
            self.state.elapsed_seconds = prior_elapsed + (time.perf_counter() - run_started)
            if stop_after_step is not None and self.state.step >= stop_after_step:
                stop_reason = "manual_step_limit"
            else:
                stop_reason = self._stop_reason(elapsed_seconds=self.state.elapsed_seconds)
        self.save_checkpoint()
        write_jsonl_event(
            self.log_path,
            "training_complete",
            {
                "step": self.state.step,
                "tokens_seen": self.state.tokens_seen,
                "stop_reason": stop_reason,
            },
        )
        return self.history, stop_reason


def summarize_history(history: list[StepMetrics]) -> dict[str, JsonValue]:
    """Create a compact report without hiding the unsmoothed JSONL metrics."""
    if not history:
        return {"steps": 0}
    first_window = history[: min(10, len(history))]
    last_window = history[-min(10, len(history)) :]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "steps": len(history),
        "first_window_mean_loss": mean([item.loss for item in first_window]),
        "last_window_mean_loss": mean([item.loss for item in last_window]),
        "loss_declined": mean([item.loss for item in last_window])
        < mean([item.loss for item in first_window]),
        "mean_tokens_per_second": mean([item.tokens_per_second for item in history]),
        "peak_memory_mib": max((item.peak_memory_mib or 0.0 for item in history), default=0.0),
        "all_losses_finite": all(math.isfinite(item.loss) for item in history),
    }
