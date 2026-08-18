"""Answer-only supervision helpers for multimodal SFT."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable

from torch import Tensor

IGNORE_INDEX = -100

LANGUAGE_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LANGUAGE_FFN_TARGETS = ("gate_proj", "up_proj", "down_proj")


def deterministic_sample_index(dataset_size: int, micro_step: int, *, seed: int) -> int:
    """Return one reproducibly shuffled sample index across repeated epochs."""
    if dataset_size <= 0 or micro_step < 0:
        raise ValueError("dataset_size must be positive and micro_step non-negative")
    epoch, offset = divmod(micro_step, dataset_size)
    order = list(range(dataset_size))
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def mask_prompt_labels(input_ids: Tensor, *, prompt_length: int) -> Tensor:
    """Copy token IDs into labels and mask every prompt position."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("B1 smoke training expects input_ids with shape [1, sequence]")
    if prompt_length <= 0 or prompt_length >= input_ids.size(1):
        raise ValueError("prompt_length must leave at least one supervised answer token")
    labels = input_ids.clone()
    labels[:, :prompt_length] = IGNORE_INDEX
    return labels


def assert_language_attention_lora_only(parameter_names: Iterable[str]) -> tuple[str, ...]:
    """Fail if a trainable parameter escapes B1's language-attention LoRA boundary."""
    return assert_language_lora_only(
        parameter_names,
        target_modules=LANGUAGE_ATTENTION_TARGETS,
    )


def language_lora_target_pattern(target_modules: Iterable[str]) -> str:
    """Build a PEFT full-match pattern that cannot select visual namesakes."""
    targets = tuple(target_modules)
    attention = [target for target in targets if target in LANGUAGE_ATTENTION_TARGETS]
    ffn = [target for target in targets if target in LANGUAGE_FFN_TARGETS]
    unsupported = set(targets) - set(LANGUAGE_ATTENTION_TARGETS) - set(LANGUAGE_FFN_TARGETS)
    if unsupported or not targets:
        raise ValueError(f"Unsupported language LoRA targets: {sorted(unsupported)}")
    branches: list[str] = []
    if attention:
        branches.append(r"self_attn\." + "(?:" + "|".join(map(re.escape, attention)) + ")")
    if ffn:
        branches.append(r"mlp\." + "(?:" + "|".join(map(re.escape, ffn)) + ")")
    return r".*language_model\.layers\.\d+\.(?:" + "|".join(branches) + ")"


def assert_language_lora_only(
    parameter_names: Iterable[str], *, target_modules: Iterable[str]
) -> tuple[str, ...]:
    """Validate that LoRA is limited to selected language attention/FFN projections."""
    names = tuple(parameter_names)
    if not names:
        raise ValueError("Language QLoRA has no trainable parameters")
    targets = tuple(target_modules)
    if not targets:
        raise ValueError("At least one language LoRA target is required")
    attention_targets = set(LANGUAGE_ATTENTION_TARGETS)
    ffn_targets = set(LANGUAGE_FFN_TARGETS)
    unsupported = set(targets) - attention_targets - ffn_targets
    if unsupported:
        raise ValueError(f"Unsupported language LoRA targets: {sorted(unsupported)}")
    for name in names:
        if "language_model.layers" not in name or "lora_" not in name:
            raise ValueError(f"Trainable parameter is outside language boundary: {name}")
        matching_targets = [target for target in targets if f".{target}.lora_" in name]
        if len(matching_targets) != 1:
            raise ValueError(f"Trainable parameter is outside selected targets: {name}")
        target = matching_targets[0]
        expected_scope = ".self_attn." if target in attention_targets else ".mlp."
        if expected_scope not in name:
            raise ValueError(f"Trainable parameter has the wrong language scope: {name}")
    for target in targets:
        if not any(f".{target}.lora_" in name for name in names):
            raise ValueError(f"Language QLoRA did not inject LoRA into {target}")
    return names
