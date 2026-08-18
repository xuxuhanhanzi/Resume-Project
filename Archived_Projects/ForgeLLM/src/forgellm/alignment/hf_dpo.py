"""Bounded Qwen DPO experiment with a precomputed immutable SFT reference."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import cast

import torch
from peft import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from forgellm.alignment.config import Stage5RunConfig
from forgellm.alignment.dpo import dpo_loss, response_sequence_log_probs
from forgellm.alignment.hf_common import (
    AlignmentHFStack,
    base_gradients_are_absent,
    generation_prompt_ids,
    model_device,
    model_logits,
    tokenize_preference_pair,
)
from forgellm.alignment.preference_data import verify_response
from forgellm.alignment.reward import pearson_correlation
from forgellm.alignment.schema import PreferenceRecord
from forgellm.post_training.collator import collate_tokenized_conversations
from forgellm.post_training.evaluation import repeated_character_ngram_fraction
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class ReferencePair:
    """Immutable initial-SFT response log-probs for one pair."""

    chosen_log_prob: float
    rejected_log_prob: float
    chosen_tokens: int
    rejected_tokens: int


def _pair_log_probs(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    record: PreferenceRecord,
    *,
    max_length: int,
) -> tuple[Tensor, Tensor, int, int]:
    chosen, rejected = tokenize_preference_pair(record, tokenizer, max_length=max_length)
    batch = collate_tokenized_conversations(
        [chosen, rejected],
        pad_token_id=cast(int, tokenizer.pad_token_id),
        pad_to_multiple_of=8,
    )
    device = model_device(model)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sequence = response_sequence_log_probs(
            model_logits(
                model,
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            ),
            batch.labels.to(device),
        )
    return (
        sequence.sums[0],
        sequence.sums[1],
        int(sequence.token_counts[0].item()),
        int(sequence.token_counts[1].item()),
    )


@torch.no_grad()
def precompute_reference(
    stack: AlignmentHFStack,
    records: list[PreferenceRecord],
    *,
    max_length: int,
) -> dict[str, ReferencePair]:
    """Cache the initial SFT policy once so it cannot drift during DPO."""
    stack.model.eval()
    cache: dict[str, ReferencePair] = {}
    for record in records:
        chosen, rejected, chosen_tokens, rejected_tokens = _pair_log_probs(
            stack.model, stack.tokenizer, record, max_length=max_length
        )
        cache[record.record_id] = ReferencePair(
            float(chosen.item()),
            float(rejected.item()),
            chosen_tokens,
            rejected_tokens,
        )
    return cache


@torch.no_grad()
def evaluate_preference_log_probs(
    stack: AlignmentHFStack,
    records: list[PreferenceRecord],
    reference: dict[str, ReferencePair],
    *,
    beta: float,
    max_length: int,
    limit: int,
) -> dict[str, JsonValue]:
    """Measure held-out DPO margins, accuracy and length correlation."""
    stack.model.eval()
    margins: list[float] = []
    losses: list[float] = []
    accuracies: list[float] = []
    response_lengths: list[float] = []
    for record in records[:limit]:
        chosen, rejected, chosen_tokens, rejected_tokens = _pair_log_probs(
            stack.model, stack.tokenizer, record, max_length=max_length
        )
        cached = reference[record.record_id]
        result = dpo_loss(
            chosen.unsqueeze(0),
            rejected.unsqueeze(0),
            torch.tensor([cached.chosen_log_prob], device=chosen.device),
            torch.tensor([cached.rejected_log_prob], device=chosen.device),
            beta=beta,
        )
        margins.append(float(result.logits.item()))
        losses.append(float(result.loss.item()))
        accuracies.append(float(result.preference_accuracy.item()))
        response_lengths.append(float(chosen_tokens - rejected_tokens))
    margin_tensor = torch.tensor(margins)
    length_tensor = torch.tensor(response_lengths)
    return {
        "pairs": len(margins),
        "mean_loss": sum(losses) / len(losses),
        "implicit_preference_accuracy": sum(accuracies) / len(accuracies),
        "mean_logit_margin": sum(margins) / len(margins),
        "margin_length_difference_correlation": pearson_correlation(margin_tensor, length_tensor),
    }


@torch.no_grad()
def evaluate_preference_generation(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    records: list[PreferenceRecord],
    *,
    max_length: int,
    max_new_tokens: int,
    limit: int,
) -> dict[str, JsonValue]:
    """Greedily test strict held-out behavior independently of training log-probs."""
    model.eval()
    device = model_device(model)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    outputs: list[dict[str, JsonValue]] = []
    passed = 0
    repetitions: list[float] = []
    lengths: list[int] = []
    for record in records[:limit]:
        prompt_ids = generation_prompt_ids(record, tokenizer, max_length=max_length)
        input_ids = torch.tensor([prompt_ids], device=device)
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
        components = verify_response(record, response)
        repetition = repeated_character_ngram_fraction(response)
        passed += int(components["total"] == 1.0)
        repetitions.append(repetition)
        lengths.append(len(generated_ids))
        outputs.append(
            {
                "id": record.record_id,
                "response": response,
                "response_tokens": len(generated_ids),
                "character_8gram_repetition": repetition,
                "reward_components": cast(dict[str, JsonValue], components),
            }
        )
    count = len(outputs)
    return {
        "examples": count,
        "strict_pass_rate": passed / count,
        "mean_response_tokens": sum(lengths) / count,
        "mean_character_8gram_repetition": sum(repetitions) / count,
        "outputs": cast(list[JsonValue], outputs),
    }


def train_dpo_bounded(
    stack: AlignmentHFStack,
    config: Stage5RunConfig,
    records: list[PreferenceRecord],
    reference: dict[str, ReferencePair],
) -> dict[str, JsonValue]:
    """Train only the Adapter until the first pair-token/step/time boundary."""
    model = stack.model
    dpo_config = config.dpo
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=dpo_config.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    device = model_device(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer_steps = 0
    micro_steps = 0
    pair_tokens = 0
    accumulated_pair_tokens = 0
    accumulated_pairs = 0
    loss_sum = 0.0
    epoch = 0
    stop_reason = "max_steps"
    model.train()
    while optimizer_steps < dpo_config.max_steps:
        order = list(records)
        random.Random(dpo_config.seed + epoch).shuffle(order)
        for record in order:
            policy_chosen, policy_rejected, chosen_tokens, rejected_tokens = _pair_log_probs(
                model, stack.tokenizer, record, max_length=dpo_config.max_length
            )
            cached = reference[record.record_id]
            result = dpo_loss(
                policy_chosen.unsqueeze(0),
                policy_rejected.unsqueeze(0),
                torch.tensor([cached.chosen_log_prob], device=device),
                torch.tensor([cached.rejected_log_prob], device=device),
                beta=dpo_config.beta,
            )
            (result.loss / dpo_config.gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
            loss_sum += float(result.loss.detach().item())
            accumulated_pairs += 1
            accumulated_pair_tokens += chosen_tokens + rejected_tokens
            micro_steps += 1
            if micro_steps % dpo_config.gradient_accumulation_steps != 0:
                continue
            if not base_gradients_are_absent(model):
                raise RuntimeError("a frozen Base parameter acquired a DPO gradient")
            gradient_norm = nn.utils.clip_grad_norm_(trainable, dpo_config.gradient_clip_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("DPO gradient norm is non-finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pair_tokens += accumulated_pair_tokens
            accumulated_pair_tokens = 0
            elapsed = time.perf_counter() - started
            if optimizer_steps == 1 or optimizer_steps % 10 == 0:
                print(
                    f"dpo_progress steps={optimizer_steps} pair_tokens={pair_tokens} "
                    f"elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
            if pair_tokens >= dpo_config.max_pair_tokens:
                stop_reason = "max_pair_tokens"
                break
            if elapsed >= dpo_config.max_duration_seconds:
                stop_reason = "max_duration_seconds"
                break
            if optimizer_steps >= dpo_config.max_steps:
                stop_reason = "max_steps"
                break
        else:
            epoch += 1
            continue
        break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "pairs_consumed": accumulated_pairs,
        "pair_response_tokens": pair_tokens,
        "mean_pair_loss": loss_sum / max(accumulated_pairs, 1),
        "elapsed_seconds": elapsed,
        "pair_tokens_per_second": pair_tokens / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "stop_reason": stop_reason,
        "base_gradients_absent": True,
        "reference_mode": "precomputed initial-SFT log-probs",
    }


def reference_cache_as_json(cache: dict[str, ReferencePair]) -> dict[str, JsonValue]:
    """Serialize the reference cache for independent auditing."""
    return {
        record_id: {
            "chosen_log_prob": pair.chosen_log_prob,
            "rejected_log_prob": pair.rejected_log_prob,
            "chosen_tokens": pair.chosen_tokens,
            "rejected_tokens": pair.rejected_tokens,
        }
        for record_id, pair in cache.items()
    }
