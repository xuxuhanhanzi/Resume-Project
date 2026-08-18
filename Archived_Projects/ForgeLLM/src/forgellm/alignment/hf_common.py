"""Shared Qwen/PEFT boundaries for Stage 5 framework experiments."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel
from torch import Tensor, nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from forgellm.alignment.config import Stage5RunConfig
from forgellm.alignment.schema import PreferenceRecord
from forgellm.post_training.adapters import tokenize_qwen_generation_prompt, tokenize_qwen_record
from forgellm.post_training.chat_template import TokenizedConversation
from forgellm.post_training.schema import InstructionRecord, Message


@dataclass(frozen=True, slots=True)
class AlignmentHFStack:
    """Loaded Stage 4 policy prepared for Stage 5 Adapter-only updates."""

    tokenizer: PreTrainedTokenizerBase
    model: PeftModel
    total_parameters: int
    trainable_parameters: int
    initial_adapter_sha256: str


def file_sha256(path: Path) -> str:
    """Hash one file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    """Seed Python and Torch generators used by framework runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model: nn.Module) -> torch.device:
    """Find the first materialized parameter device."""
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("model has no materialized parameter")


def model_logits(model: nn.Module, **kwargs: Tensor) -> Tensor:
    """Extract causal-LM logits with a strict Tensor contract."""
    output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise RuntimeError("causal LM did not return Tensor logits")
    return logits


def load_stage5_policy(config: Stage5RunConfig, *, trainable: bool) -> AlignmentHFStack:
    """Load exact BF16 Qwen Base and the frozen Stage 4 SFT Adapter."""
    adapter_dir = Path(config.model.initial_adapter_path)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"initial Stage 4 Adapter is missing: {adapter_weights}")
    tokenizer_raw = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        use_fast=True,
    )
    if tokenizer_raw is None:
        raise RuntimeError("AutoTokenizer returned no tokenizer")
    tokenizer = cast(PreTrainedTokenizerBase, tokenizer_raw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_raw = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base = cast(PreTrainedModel, base_raw)
    base.to(torch.device("cuda"))  # type: ignore[arg-type]
    base.config.use_cache = False
    if trainable:
        base.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    policy = PeftModel.from_pretrained(
        base,
        adapter_dir,
        is_trainable=trainable,
    )
    for module in policy.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    total = sum(parameter.numel() for parameter in policy.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    if trainable and trainable_parameters <= 0:
        raise RuntimeError("Stage 5 policy has no trainable Adapter parameters")
    if trainable_parameters >= total:
        raise RuntimeError("Stage 5 unexpectedly made the Base trainable")
    return AlignmentHFStack(
        tokenizer,
        policy,
        total,
        trainable_parameters,
        file_sha256(adapter_weights),
    )


def preference_instruction(
    record: PreferenceRecord, response: str, *, suffix: str
) -> InstructionRecord:
    """Turn one side of a preference pair into Stage 4's segmented ChatML input."""
    messages = (*record.prompt, Message(role="assistant", content=response))
    return InstructionRecord(
        record_id=f"{record.record_id}-{suffix}",
        messages=messages,
        source="stage5-preference",
        license="project-original",
        metadata={"preference_id": record.record_id, "side": suffix},
    )


def tokenize_preference_pair(
    record: PreferenceRecord,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> tuple[TokenizedConversation, TokenizedConversation]:
    """Tokenize chosen and rejected under the exact same prompt/template."""
    chosen = tokenize_qwen_record(
        preference_instruction(record, record.chosen, suffix="chosen"),
        tokenizer,
        max_length=max_length,
    )
    rejected = tokenize_qwen_record(
        preference_instruction(record, record.rejected, suffix="rejected"),
        tokenizer,
        max_length=max_length,
    )
    if chosen.supervised_tokens <= 0 or rejected.supervised_tokens <= 0:
        raise ValueError(f"pair {record.record_id} lost response supervision")
    return chosen, rejected


def generation_prompt_ids(
    record: PreferenceRecord,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> list[int]:
    """Build the Qwen assistant-prefix input for preference test generation."""
    placeholder = preference_instruction(record, record.chosen, suffix="generation")
    return tokenize_qwen_generation_prompt(placeholder, tokenizer, max_length=max_length)


def base_gradients_are_absent(model: nn.Module) -> bool:
    """Verify only LoRA parameters ever receive gradients."""
    return all(
        "lora_" in name or parameter.grad is None for name, parameter in model.named_parameters()
    )


def dropout_is_disabled(model: nn.Module) -> bool:
    """Verify rollout, policy and reference distributions share deterministic dropout semantics."""
    return all(not isinstance(module, nn.Dropout) or module.p == 0.0 for module in model.modules())
