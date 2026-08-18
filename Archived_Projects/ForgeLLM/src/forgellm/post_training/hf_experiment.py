"""Auditable bounded Qwen LoRA/QLoRA training and evaluation for Stage 4."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from forgellm.post_training.adapters import (
    HFStack,
    adapter_base_gradients_are_absent,
    lora_is_initial_noop,
    tokenize_qwen_generation_prompt,
    tokenize_qwen_record,
)
from forgellm.post_training.chat_template import IGNORE_INDEX, TokenizedConversation
from forgellm.post_training.collator import collate_tokenized_conversations
from forgellm.post_training.config import Stage4RunConfig
from forgellm.post_training.evaluation import repeated_ngram_fraction, verify_constraints
from forgellm.post_training.schema import InstructionRecord
from forgellm.post_training.sft import assistant_only_causal_loss, scale_gradients_by_token_count
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class LossMetrics:
    """Token-weighted teacher-forced loss metrics."""

    loss: float
    perplexity: float
    token_accuracy: float
    target_tokens: int

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible values."""
        return cast(dict[str, JsonValue], asdict(self))


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Deterministic constrained-generation metrics."""

    examples: int
    constraint_pass_rate: float
    exact_match_rate: float
    mean_response_tokens: float
    mean_repeated_trigram_fraction: float
    outputs: tuple[dict[str, JsonValue], ...]

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible values."""
        return cast(dict[str, JsonValue], asdict(self))


def _autocast_context(device: torch.device) -> AbstractContextManager[object]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("model has no materialized parameter")


def _model_logits(model: nn.Module, **kwargs: Tensor) -> Tensor:
    output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise RuntimeError("causal LM output does not expose Tensor logits")
    return logits


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_records(
    records: list[InstructionRecord],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> list[TokenizedConversation]:
    """Tokenize records and reject any example with no shifted supervision."""
    tokenized: list[TokenizedConversation] = []
    for record in records:
        example = tokenize_qwen_record(record, tokenizer, max_length=max_length)
        if example.supervised_tokens <= 0:
            raise ValueError(f"record {record.record_id} lost all assistant targets")
        tokenized.append(example)
    return tokenized


@torch.no_grad()
def evaluate_assistant_loss(
    model: nn.Module,
    examples: list[TokenizedConversation],
    *,
    tokenizer: PreTrainedTokenizerBase,
    limit: int,
) -> LossMetrics:
    """Compute token-weighted assistant-only held-out loss."""
    model.eval()
    device = _model_device(model)
    nll_sum = 0.0
    target_tokens = 0
    correct_tokens = 0
    for example in examples[:limit]:
        batch = collate_tokenized_conversations(
            [example], pad_token_id=cast(int, tokenizer.pad_token_id)
        )
        with _autocast_context(device):
            result = assistant_only_causal_loss(
                _model_logits(
                    model,
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                ),
                batch.labels.to(device),
            )
        nll_sum += float(result.negative_log_likelihood_sum.item())
        target_tokens += result.target_tokens
        correct_tokens += result.correct_tokens
    if target_tokens <= 0:
        raise RuntimeError("assistant evaluation selected no target tokens")
    loss = nll_sum / target_tokens
    return LossMetrics(
        loss=loss,
        perplexity=math.exp(min(loss, 20.0)),
        token_accuracy=correct_tokens / target_tokens,
        target_tokens=target_tokens,
    )


@torch.no_grad()
def evaluate_retention_loss(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    path: Path,
    *,
    limit: int,
    max_length: int,
) -> LossMetrics:
    """Evaluate plain-text causal loss as a small pretraining-retention sentinel."""
    model.eval()
    device = _model_device(model)
    nll_sum = 0.0
    target_tokens = 0
    correct_tokens = 0
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        raw = json.loads(line)
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise ValueError(f"invalid retention record in {path}")
        token_ids = tokenizer.encode(raw["text"], add_special_tokens=False)[:max_length]
        if len(token_ids) < 2:
            continue
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        labels = input_ids.clone()
        labels[:, 0] = IGNORE_INDEX
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        with _autocast_context(device):
            result = assistant_only_causal_loss(
                _model_logits(model, input_ids=input_ids, attention_mask=attention_mask), labels
            )
        nll_sum += float(result.negative_log_likelihood_sum.item())
        target_tokens += result.target_tokens
        correct_tokens += result.correct_tokens
    if target_tokens <= 0:
        raise RuntimeError("retention evaluation selected no target tokens")
    loss = nll_sum / target_tokens
    return LossMetrics(
        loss=loss,
        perplexity=math.exp(min(loss, 20.0)),
        token_accuracy=correct_tokens / target_tokens,
        target_tokens=target_tokens,
    )


def _metadata_str(metadata: dict[str, JsonValue], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _metadata_str_list(metadata: dict[str, JsonValue], key: str) -> list[str] | None:
    value = metadata.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast(list[str], value)
    return None


@torch.no_grad()
def evaluate_generation(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    records: list[InstructionRecord],
    *,
    limit: int,
    max_length: int,
    max_new_tokens: int,
) -> GenerationMetrics:
    """Greedily generate and run only explicit deterministic task checks."""
    model.eval()
    device = _model_device(model)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    outputs: list[dict[str, JsonValue]] = []
    passes = 0
    exact_matches = 0
    response_tokens: list[int] = []
    repetitions: list[float] = []
    for record in records[:limit]:
        prompt_ids = tokenize_qwen_generation_prompt(record, tokenizer, max_length=max_length)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(  # type: ignore[no-untyped-call]
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=cast(int, im_end),
            pad_token_id=cast(int, tokenizer.pad_token_id),
            use_cache=True,
        )
        generated_ids = generated[0, len(prompt_ids) :].tolist()
        response = cast(str, tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()
        metadata = record.metadata
        word_count = metadata.get("exact_word_count")
        result = verify_constraints(
            response,
            expected_response=_metadata_str(metadata, "expected_response"),
            starts_with=_metadata_str(metadata, "starts_with"),
            required_substrings=_metadata_str_list(metadata, "required_substrings"),
            forbidden_substrings=_metadata_str_list(metadata, "forbidden_substrings"),
            exact_word_count=(word_count if isinstance(word_count, int) else None),
            require_json=metadata.get("require_json") is True,
        )
        passes += int(result.all_constraints_passed)
        exact_matches += int(result.exact_match)
        response_tokens.append(len(generated_ids))
        repetition = repeated_ngram_fraction(response)
        repetitions.append(repetition)
        outputs.append(
            {
                "id": record.record_id,
                "response": response,
                "response_tokens": len(generated_ids),
                "repeated_trigram_fraction": repetition,
                "constraints": result.as_dict(),
            }
        )
    count = len(outputs)
    if count == 0:
        raise RuntimeError("generation evaluation selected no examples")
    return GenerationMetrics(
        examples=count,
        constraint_pass_rate=passes / count,
        exact_match_rate=exact_matches / count,
        mean_response_tokens=sum(response_tokens) / count,
        mean_repeated_trigram_fraction=sum(repetitions) / count,
        outputs=tuple(outputs),
    )


def evaluate_stage4(
    stack: HFStack,
    config: Stage4RunConfig,
    validation_examples: list[TokenizedConversation],
    task_records: list[InstructionRecord],
) -> dict[str, JsonValue]:
    """Run the three frozen Stage 4 evaluation views."""
    assistant = evaluate_assistant_loss(
        stack.model,
        validation_examples,
        tokenizer=stack.tokenizer,
        limit=config.training.eval_examples,
    )
    task_loss = evaluate_assistant_loss(
        stack.model,
        tokenize_records(task_records, stack.tokenizer, max_length=config.training.max_length),
        tokenizer=stack.tokenizer,
        limit=config.training.eval_examples,
    )
    retention = evaluate_retention_loss(
        stack.model,
        stack.tokenizer,
        Path(config.data.retention_path),
        limit=config.training.eval_examples,
        max_length=config.training.max_length,
    )
    generation = evaluate_generation(
        stack.model,
        stack.tokenizer,
        task_records,
        limit=config.training.generation_examples,
        max_length=config.training.max_length,
        max_new_tokens=config.training.max_new_tokens,
    )
    return {
        "assistant_validation": assistant.as_dict(),
        "correctness_teacher_forced": task_loss.as_dict(),
        "pretraining_retention": retention.as_dict(),
        "correctness_generation": generation.as_dict(),
    }


def _training_batches(
    examples: list[TokenizedConversation], *, seed: int, micro_batch_size: int
) -> list[list[TokenizedConversation]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    ordered = [examples[index] for index in indices]
    return [
        ordered[index : index + micro_batch_size]
        for index in range(0, len(ordered), micro_batch_size)
    ]


def warmup_cosine_multiplier(
    completed_tokens: int, *, maximum_tokens: int, warmup_ratio: float
) -> float:
    """Return a token-progress 3%-warmup/cosine multiplier in [0, 1]."""
    if maximum_tokens <= 0 or completed_tokens < 0:
        raise ValueError("token schedule requires non-negative progress and positive maximum")
    if not 0.0 < warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in (0, 1)")
    progress = min(completed_tokens / maximum_tokens, 1.0)
    if progress < warmup_ratio:
        return progress / warmup_ratio
    cosine_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
    return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))


def train_bounded(
    stack: HFStack,
    config: Stage4RunConfig,
    train_examples: list[TokenizedConversation],
) -> dict[str, JsonValue]:
    """Train adapters until the first frozen step/token/time boundary is met."""
    model = stack.model
    tokenizer = stack.tokenizer
    device = _model_device(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer_steps = 0
    assistant_tokens = 0
    loss_numerator = 0.0
    accumulated_tokens = 0
    micro_steps = 0
    epoch = 0
    stop_reason = "max_steps"
    final_learning_rate = 0.0
    model.train()
    while optimizer_steps < config.training.max_steps:
        batches = _training_batches(
            train_examples,
            seed=config.training.seed + epoch,
            micro_batch_size=config.training.micro_batch_size,
        )
        for examples in batches:
            batch = collate_tokenized_conversations(
                examples, pad_token_id=cast(int, tokenizer.pad_token_id), pad_to_multiple_of=8
            )
            with _autocast_context(device):
                result = assistant_only_causal_loss(
                    _model_logits(
                        model,
                        input_ids=batch.input_ids.to(device),
                        attention_mask=batch.attention_mask.to(device),
                    ),
                    batch.labels.to(device),
                )
            torch.autograd.backward(result.negative_log_likelihood_sum)
            loss_numerator += float(result.negative_log_likelihood_sum.detach().item())
            accumulated_tokens += result.target_tokens
            micro_steps += 1
            if micro_steps % config.training.gradient_accumulation_steps != 0:
                continue
            scale_gradients_by_token_count(cast(list[Tensor], trainable), accumulated_tokens)
            if not adapter_base_gradients_are_absent(model):
                raise RuntimeError("a frozen Base parameter acquired a gradient")
            gradient_norm = nn.utils.clip_grad_norm_(trainable, config.training.gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("adapter gradient norm is not finite")
            scheduled_tokens = assistant_tokens + accumulated_tokens
            multiplier = warmup_cosine_multiplier(
                scheduled_tokens,
                maximum_tokens=config.training.max_assistant_tokens,
                warmup_ratio=config.training.warmup_ratio,
            )
            final_learning_rate = config.training.learning_rate * multiplier
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = final_learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            assistant_tokens += accumulated_tokens
            accumulated_tokens = 0
            elapsed = time.perf_counter() - started
            if optimizer_steps == 1 or optimizer_steps % 10 == 0:
                print(
                    "training_progress "
                    f"steps={optimizer_steps} assistant_tokens={assistant_tokens} "
                    f"elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
            if assistant_tokens >= config.training.max_assistant_tokens:
                stop_reason = "max_assistant_tokens"
                break
            if elapsed >= config.training.max_duration_seconds:
                stop_reason = "max_duration_seconds"
                break
            if optimizer_steps >= config.training.max_steps:
                stop_reason = "max_steps"
                break
        else:
            epoch += 1
            continue
        break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return {
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "epochs_started": epoch + 1,
        "assistant_tokens": assistant_tokens,
        "mean_train_nll": loss_numerator / max(assistant_tokens, 1),
        "elapsed_seconds": elapsed,
        "assistant_tokens_per_second": assistant_tokens / elapsed,
        "peak_allocated_bytes": peak_memory,
        "stop_reason": stop_reason,
        "scheduler": "assistant-token 3%-warmup then cosine",
        "final_learning_rate": final_learning_rate,
        "base_gradients_absent": True,
    }


def package_versions() -> dict[str, JsonValue]:
    """Capture exact framework versions used by the run."""
    names = ("accelerate", "bitsandbytes", "datasets", "peft", "torch", "transformers", "trl")
    return {name: importlib.metadata.version(name) for name in names}


def build_run_provenance(config: Stage4RunConfig) -> dict[str, JsonValue]:
    """Hash every frozen input file and the resolved config."""
    paths = {
        "train": Path(config.data.train_path),
        "validation": Path(config.data.validation_path),
        "task_test": Path(config.data.task_test_path),
        "retention": Path(config.data.retention_path),
    }
    return {
        "config": config.as_dict(),
        "config_fingerprint": config.fingerprint(),
        "input_sha256": {name: _file_sha256(path) for name, path in paths.items()},
        "packages": package_versions(),
    }


def assert_stack_ready(stack: HFStack) -> None:
    """Fail before evaluation/training when the adapter boundary is not trustworthy."""
    if not lora_is_initial_noop(stack.model):
        raise RuntimeError("PEFT LoRA B matrices are not exact-zero at initialization")
    if not adapter_base_gradients_are_absent(stack.model):
        raise RuntimeError("Base model unexpectedly has gradients before training")
