"""One-step real Qwen GRPO rollout -> reward -> log-prob -> update chain."""

from __future__ import annotations

import difflib
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from forgellm.alignment.config import Stage5RunConfig
from forgellm.alignment.grpo import (
    assert_initial_on_policy,
    flatten_group_advantages,
    group_relative_advantages,
    grpo_policy_loss,
)
from forgellm.alignment.hf_common import (
    AlignmentHFStack,
    base_gradients_are_absent,
    generation_prompt_ids,
    model_device,
    model_logits,
)
from forgellm.alignment.preference_data import verify_response
from forgellm.alignment.schema import PreferenceRecord, RolloutRecord
from forgellm.post_training.evaluation import repeated_character_ngram_fraction
from forgellm.structured_logging import JsonValue


def _response_log_probs(
    model: nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    with_grad: bool,
) -> Tensor:
    if not response_ids:
        raise ValueError("rollout response must contain at least one token")
    device = model_device(model)
    full = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    context = (
        torch.enable_grad()  # type: ignore[no-untyped-call]
        if with_grad
        else torch.no_grad()
    )
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model_logits(
            model,
            input_ids=full,
            attention_mask=torch.ones_like(full),
        )
        start = len(prompt_ids) - 1
        response_logits = logits[:, start : start + len(response_ids)].float()
        targets = torch.tensor([response_ids], dtype=torch.long, device=device)
        return (
            torch.log_softmax(response_logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(0)
            .squeeze(-1)
        )


def online_training_reward(record: PreferenceRecord, response: str) -> dict[str, float]:
    """Deterministic partial-credit reward kept separate from strict held-out success."""
    strict = verify_response(record, response)
    expected = record.chosen
    similarity = difflib.SequenceMatcher(a=expected, b=response).ratio()
    repetition = repeated_character_ngram_fraction(response)
    excess = max(0, len(response) - max(2 * len(expected), 16))
    length_penalty = min(excess / 64.0, 1.0)
    format_score = statistics.mean(
        [strict["non_empty"], strict["no_extra"], strict["no_repetition"], strict["structural"]]
    )
    total = strict["exact"] + 0.25 * similarity + 0.05 * format_score
    total -= 0.1 * repetition + 0.1 * length_penalty
    return {
        "strict_exact": strict["exact"],
        "similarity": similarity,
        "format_score": format_score,
        "character_8gram_repetition": repetition,
        "length_penalty": length_penalty,
        "total": total,
    }


def _sample_group(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    record: PreferenceRecord,
    config: Stage5RunConfig,
) -> tuple[list[int], list[list[int]]]:
    device = model_device(model)
    prompt_ids = generation_prompt_ids(record, tokenizer, max_length=config.grpo.max_length)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    generated = model.generate(  # type: ignore[no-untyped-call]
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=True,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        num_return_sequences=config.grpo.group_size,
        max_new_tokens=config.grpo.max_new_tokens,
        min_new_tokens=1,
        eos_token_id=cast(int, im_end),
        pad_token_id=cast(int, tokenizer.pad_token_id),
        use_cache=True,
    )
    responses: list[list[int]] = []
    for row in generated:
        response = [
            int(token)
            for token in row[len(prompt_ids) :].tolist()
            if int(token) != cast(int, tokenizer.pad_token_id)
        ]
        if not response:
            raise RuntimeError(f"rollout {record.record_id} produced no response tokens")
        responses.append(response)
    return prompt_ids, responses


def run_one_step_grpo(
    stack: AlignmentHFStack,
    config: Stage5RunConfig,
    records: list[PreferenceRecord],
    *,
    output_dir: Path,
) -> dict[str, JsonValue]:
    """Generate exactly the frozen budget and apply exactly one optimizer step."""
    selected = records[: config.grpo.prompts]
    if len(selected) != config.grpo.prompts:
        raise ValueError("not enough prompts for the frozen GRPO smoke")
    model = stack.model
    tokenizer = stack.tokenizer
    device = model_device(model)
    started = time.perf_counter()
    model.eval()
    rollout_rows: list[RolloutRecord] = []
    prompt_ids_by_row: list[list[int]] = []
    response_ids_by_row: list[list[int]] = []
    reward_rows: list[list[float]] = []
    policy_revision = f"sft-adapter:{stack.initial_adapter_sha256}"
    for record in selected:
        prompt_ids, responses = _sample_group(model, tokenizer, record, config)
        group_rewards: list[float] = []
        for response_ids in responses:
            old_log_probs = _response_log_probs(model, prompt_ids, response_ids, with_grad=False)
            response = cast(str, tokenizer.decode(response_ids, skip_special_tokens=True)).strip()
            components = online_training_reward(record, response)
            group_rewards.append(components["total"])
            rollout_rows.append(
                RolloutRecord(
                    prompt_id=record.record_id,
                    policy_revision=policy_revision,
                    adapter_sha256=stack.initial_adapter_sha256,
                    generation_config={
                        "temperature": config.grpo.temperature,
                        "top_p": config.grpo.top_p,
                        "max_new_tokens": config.grpo.max_new_tokens,
                    },
                    seed=config.grpo.seed,
                    token_ids=tuple(response_ids),
                    old_log_probs=tuple(float(value) for value in old_log_probs.cpu().tolist()),
                    reward_components=components,
                    verifier_version="forgellm-grpo-shaped-verifier-v1",
                    response=response,
                    timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                )
            )
            prompt_ids_by_row.append(prompt_ids)
            response_ids_by_row.append(response_ids)
        reward_rows.append(group_rewards)
    if time.perf_counter() - started > config.grpo.max_duration_seconds:
        raise TimeoutError("GRPO rollout exceeded the frozen wall-clock budget")
    rewards = torch.tensor(reward_rows, dtype=torch.float32, device=device)
    advantage_result = group_relative_advantages(rewards)
    if not torch.any(advantage_result.usable_groups):
        raise RuntimeError("every rollout group had zero reward variance; update is undefined")
    old_rows = [
        torch.tensor(row.old_log_probs, dtype=torch.float32, device=device) for row in rollout_rows
    ]
    old_padded = pad_sequence(old_rows, batch_first=True, padding_value=0.0)
    response_mask = pad_sequence(
        [torch.ones(len(row), dtype=torch.bool, device=device) for row in response_ids_by_row],
        batch_first=True,
        padding_value=False,
    )
    # Gradients do not require train() mode. Staying in eval keeps the rollout and
    # update distributions identical by construction (all dropout is also p=0).
    model.eval()
    new_rows = [
        _response_log_probs(model, prompt_ids, response_ids, with_grad=True)
        for prompt_ids, response_ids in zip(prompt_ids_by_row, response_ids_by_row, strict=True)
    ]
    new_padded = pad_sequence(new_rows, batch_first=True, padding_value=0.0)
    assert_initial_on_policy(new_padded, old_padded, response_mask)
    result = grpo_policy_loss(
        new_padded,
        old_padded,
        flatten_group_advantages(advantage_result),
        response_mask,
        clip_epsilon=config.grpo.clip_epsilon,
        reference_log_probs=old_padded,
        kl_beta=config.grpo.kl_beta,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.grpo.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    result.loss.backward()  # type: ignore[no-untyped-call]
    if not base_gradients_are_absent(model):
        raise RuntimeError("a frozen Base parameter acquired a GRPO gradient")
    gradient_norm = nn.utils.clip_grad_norm_(trainable, config.grpo.gradient_clip_norm)
    if not torch.isfinite(gradient_norm) or float(gradient_norm.item()) <= 0:
        raise FloatingPointError("GRPO gradient norm is invalid")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if elapsed > config.grpo.max_duration_seconds:
        raise TimeoutError("GRPO full chain exceeded the frozen wall-clock budget")
    rollout_path = output_dir / "rollouts.jsonl"
    rollout_path.write_text(
        "".join(
            __import__("json").dumps(row.as_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rollout_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "prompts": len(selected),
        "group_size": config.grpo.group_size,
        "rollouts": len(rollout_rows),
        "response_tokens": sum(row.response_length for row in rollout_rows),
        "reward_mean": float(rewards.mean().item()),
        "reward_standard_deviation": float(rewards.std(unbiased=False).item()),
        "usable_groups": int(advantage_result.usable_groups.sum().item()),
        "zero_variance_groups": int((~advantage_result.usable_groups).sum().item()),
        "loss_before_step": float(result.loss.detach().item()),
        "approximate_kl_before_step": float(result.approximate_kl.detach().item()),
        "clip_fraction_before_step": float(result.clip_fraction.detach().item()),
        "gradient_norm": float(gradient_norm.item()),
        "base_gradients_absent": True,
        "optimizer_steps": 1,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "rollout_policy_revision": policy_revision,
        "rollout_artifact": str(rollout_path),
    }
