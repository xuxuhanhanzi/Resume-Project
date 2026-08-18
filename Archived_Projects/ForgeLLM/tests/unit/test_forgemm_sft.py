import re

import pytest
import torch

from forgellm.multimodal.sft import (
    IGNORE_INDEX,
    assert_language_attention_lora_only,
    assert_language_lora_only,
    deterministic_sample_index,
    language_lora_target_pattern,
    mask_prompt_labels,
)


def test_mask_prompt_labels_supervises_only_answer_suffix() -> None:
    input_ids = torch.tensor([[11, 12, 13, 21, 22]])

    labels = mask_prompt_labels(input_ids, prompt_length=3)

    assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 21, 22]]
    assert input_ids.tolist() == [[11, 12, 13, 21, 22]]


def test_mask_prompt_labels_rejects_empty_answer() -> None:
    with pytest.raises(ValueError, match="answer token"):
        mask_prompt_labels(torch.tensor([[1, 2]]), prompt_length=2)


def test_deterministic_sample_order_is_shuffled_and_epoch_aware() -> None:
    first = [deterministic_sample_index(8, step, seed=17) for step in range(8)]
    repeated = [deterministic_sample_index(8, step, seed=17) for step in range(8)]
    second_epoch = [deterministic_sample_index(8, step, seed=17) for step in range(8, 16)]

    assert first == repeated
    assert sorted(first) == list(range(8))
    assert sorted(second_epoch) == list(range(8))
    assert first != second_epoch


def test_b1_boundary_accepts_language_attention_lora() -> None:
    names = [
        f"base_model.model.model.language_model.layers.0.self_attn.{target}.lora_A.weight"
        for target in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    assert assert_language_attention_lora_only(names) == tuple(names)


def test_b1_boundary_rejects_visual_lora() -> None:
    with pytest.raises(ValueError, match="outside language boundary"):
        assert_language_attention_lora_only(
            ["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"]
        )


def test_b2_boundary_accepts_language_attention_and_ffn_lora() -> None:
    targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = [
        "base_model.model.model.language_model.layers.0."
        f"{'self_attn' if target in {'q_proj', 'k_proj', 'v_proj', 'o_proj'} else 'mlp'}."
        f"{target}.lora_A.weight"
        for target in targets
    ]

    assert assert_language_lora_only(names, target_modules=targets) == tuple(names)


def test_b2_boundary_rejects_ffn_target_in_attention_scope() -> None:
    with pytest.raises(ValueError, match="wrong language scope"):
        assert_language_lora_only(
            ["base_model.model.model.language_model.layers.0.self_attn.gate_proj.lora_A.weight"],
            target_modules=("gate_proj",),
        )


def test_b2_peft_pattern_selects_language_namesakes_only() -> None:
    pattern = language_lora_target_pattern(
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    )

    assert re.fullmatch(pattern, "base_model.model.model.language_model.layers.0.mlp.gate_proj")
    assert re.fullmatch(pattern, "base_model.model.model.language_model.layers.35.self_attn.q_proj")
    assert not re.fullmatch(pattern, "base_model.model.model.visual.blocks.0.mlp.gate_proj")
