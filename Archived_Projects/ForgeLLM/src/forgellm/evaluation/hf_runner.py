"""Frozen Qwen Base/Adapter quality evaluation for Stage 6."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from forgellm.alignment.hf_common import model_device, model_logits, tokenize_preference_pair
from forgellm.alignment.schema import PreferenceRecord
from forgellm.evaluation.behavioral import BehaviorResult, evaluate_behavior
from forgellm.evaluation.identity import directory_sha256
from forgellm.evaluation.language_modeling import LanguageModelingTotals
from forgellm.evaluation.schema import EvaluationCase, GenerationRecord, StopReason
from forgellm.post_training.adapters import tokenize_qwen_generation_prompt, tokenize_qwen_record
from forgellm.post_training.collator import collate_tokenized_conversations
from forgellm.post_training.schema import InstructionRecord, Message
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class EvaluationHFStack:
    """One immutable Qwen policy and its shared tokenizer."""

    tokenizer: PreTrainedTokenizerBase
    model: nn.Module
    adapter_sha256: str | None


def load_evaluation_stack(
    *,
    model_id: str,
    revision: str,
    tokenizer_path: Path,
    adapter_path: Path | None,
) -> EvaluationHFStack:
    """Load exact BF16 Base and optional non-trainable Adapter for inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen Stage 6 Qwen evaluation requires CUDA")
    tokenizer_raw = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer_raw is None:
        raise RuntimeError("AutoTokenizer returned no tokenizer")
    tokenizer = cast(PreTrainedTokenizerBase, tokenizer_raw)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_raw = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base = cast(PreTrainedModel, base_raw)
    base.to(torch.device("cuda"))  # type: ignore[arg-type]
    base.config.use_cache = True
    model: nn.Module = base
    adapter_sha256 = None
    if adapter_path is not None:
        model = cast(
            nn.Module,
            PeftModel.from_pretrained(base, adapter_path, is_trainable=False),
        )
        adapter_sha256 = directory_sha256(adapter_path)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return EvaluationHFStack(tokenizer, model, adapter_sha256)


def _case_instruction(case: EvaluationCase) -> InstructionRecord:
    return InstructionRecord(
        record_id=case.case_id,
        messages=(*case.prompt, Message("assistant", case.expected_response)),
        source=case.source,
        license="evaluation-only",
        metadata={"split": "test"},
    )


@torch.no_grad()
def generate_case(
    stack: EvaluationHFStack,
    case: EvaluationCase,
    *,
    run_fingerprint: str,
    model_key: str,
    max_length: int,
    max_new_tokens: int,
) -> tuple[GenerationRecord, BehaviorResult]:
    """Greedily generate one raw response and score it independently."""
    tokenizer = stack.tokenizer
    placeholder = _case_instruction(case)
    prompt_ids = tokenize_qwen_generation_prompt(placeholder, tokenizer, max_length=max_length)
    device = model_device(stack.model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    im_end_raw = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_raw, int) or im_end_raw < 0:
        raise RuntimeError("Qwen tokenizer has no <|im_end|> token")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    generate = getattr(stack.model, "generate", None)
    if not callable(generate):
        raise RuntimeError("Qwen policy has no generate method")
    generated_raw = generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=im_end_raw,
        pad_token_id=cast(int, tokenizer.pad_token_id),
        use_cache=True,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated = cast(Tensor, generated_raw)
    response_ids = [int(value) for value in generated[0, len(prompt_ids) :].tolist()]
    response = cast(str, tokenizer.decode(response_ids, skip_special_tokens=True)).strip()
    reached_budget = len(response_ids) >= max_new_tokens and im_end_raw not in response_ids
    stop_reason: StopReason = "max_new_tokens" if reached_budget else "eos"
    record = GenerationRecord(
        run_fingerprint=run_fingerprint,
        model_key=model_key,
        case_id=case.case_id,
        response=response,
        prompt_tokens=len(prompt_ids),
        response_tokens=len(response_ids),
        stop_reason=stop_reason,
        elapsed_seconds=elapsed,
    )
    behavior = evaluate_behavior(
        case,
        response,
        response_ids,
        reached_token_budget=reached_budget,
    )
    return record, behavior


@torch.no_grad()
def evaluate_expected_nll(
    stack: EvaluationHFStack,
    cases: list[EvaluationCase],
    *,
    tokenizer_sha256: str,
    max_length: int,
) -> LanguageModelingTotals:
    """Score frozen expected responses with assistant-only teacher forcing."""
    device = model_device(stack.model)
    nll = 0.0
    target_tokens = 0
    target_bytes = 0
    im_end_raw = stack.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(im_end_raw, int) or im_end_raw < 0:
        raise RuntimeError("Qwen tokenizer has no <|im_end|> token")
    for case in cases:
        tokenized = tokenize_qwen_record(
            _case_instruction(case), stack.tokenizer, max_length=max_length
        )
        batch = collate_tokenized_conversations(
            [tokenized], pad_token_id=cast(int, stack.tokenizer.pad_token_id)
        )
        labels = batch.labels.to(device)
        # BPB describes response text bytes. The training objective also
        # supervises ChatML im_end, but that protocol token has no source-text
        # byte denominator, so exclude it from this evaluation total.
        labels = labels.masked_fill(labels == im_end_raw, -100)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_logits(
                stack.model,
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            loss_sum = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        supervised = int((labels[:, 1:] != -100).sum().item())
        if supervised <= 0:
            raise RuntimeError(f"case lost all targets: {case.case_id}")
        nll += float(loss_sum.item())
        target_tokens += supervised
        target_bytes += len(case.expected_response.encode("utf-8"))
    return LanguageModelingTotals(nll, target_tokens, target_bytes, tokenizer_sha256)


@torch.no_grad()
def evaluate_retention_nll(
    stack: EvaluationHFStack,
    path: Path,
    *,
    tokenizer_sha256: str,
    limit: int,
    max_length: int,
) -> LanguageModelingTotals:
    """Score a fixed plain-text prefix and bind token/byte denominators."""
    device = model_device(stack.model)
    nll = 0.0
    target_tokens = 0
    target_bytes = 0
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        raw = cast(object, json.loads(line))
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise ValueError(f"invalid retention record in {path}")
        text = cast(str, raw["text"])
        token_ids = [
            int(value)
            for value in stack.tokenizer.encode(text, add_special_tokens=False)[:max_length]
        ]
        if len(token_ids) < 2:
            continue
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_logits(stack.model, input_ids=input_ids)
            loss_sum = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction="sum",
            )
        nll += float(loss_sum.item())
        target_tokens += len(token_ids) - 1
        decoded_targets = cast(
            str, stack.tokenizer.decode(token_ids[1:], skip_special_tokens=False)
        )
        target_bytes += len(decoded_targets.encode("utf-8"))
    return LanguageModelingTotals(nll, target_tokens, target_bytes, tokenizer_sha256)


@torch.no_grad()
def evaluate_preference_pairs(
    stack: EvaluationHFStack,
    records: list[PreferenceRecord],
    *,
    max_length: int,
) -> dict[str, JsonValue]:
    """Measure raw chosen-minus-rejected response log-probability margins."""
    device = model_device(stack.model)
    margins: list[float] = []
    length_differences: list[int] = []
    per_pair: list[JsonValue] = []
    for record in records:
        chosen, rejected = tokenize_preference_pair(record, stack.tokenizer, max_length=max_length)
        batch = collate_tokenized_conversations(
            [chosen, rejected],
            pad_token_id=cast(int, stack.tokenizer.pad_token_id),
            pad_to_multiple_of=8,
        )
        labels = batch.labels.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_logits(
                stack.model,
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = labels[:, 1:]
            mask = targets != -100
            safe_targets = targets.masked_fill(~mask, 0)
            selected = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
            sums = (selected * mask).sum(dim=-1)
            counts = mask.sum(dim=-1)
        margin = float((sums[0] - sums[1]).item())
        length_difference = int((counts[0] - counts[1]).item())
        margins.append(margin)
        length_differences.append(length_difference)
        per_pair.append(
            {
                "record_id": record.record_id,
                "chosen_minus_rejected_log_prob": margin,
                "chosen_minus_rejected_tokens": length_difference,
                "correct": margin > 0,
            }
        )
    count = len(margins)
    if count == 0:
        raise ValueError("preference evaluation requires records")
    mean_margin = sum(margins) / count
    return {
        "pairs": count,
        "preference_accuracy": sum(value > 0 for value in margins) / count,
        "mean_chosen_minus_rejected_log_prob": mean_margin,
        "mean_chosen_minus_rejected_tokens": sum(length_differences) / count,
        "finite": math.isfinite(mean_margin),
        "per_pair": per_pair,
    }
