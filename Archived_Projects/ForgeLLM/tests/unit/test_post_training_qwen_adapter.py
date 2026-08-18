"""Qwen segmented-label and PEFT initialization tests without Hub downloads."""

from typing import Any, cast

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import PreTrainedTokenizerBase, Qwen3Config, Qwen3ForCausalLM

from forgellm.post_training.adapters import lora_is_initial_noop, tokenize_qwen_record
from forgellm.post_training.lora import LoRALinear
from forgellm.post_training.schema import InstructionRecord, Message


class _FakeQwenTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": 1, "<|im_end|>": 2}.get(token, -1)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [100 + ord(character) for character in text]


def test_qwen_tokenization_supervises_only_assistant_content_and_end() -> None:
    record = InstructionRecord(
        "one",
        (Message("user", "Q"), Message("assistant", "A")),
        "fixture",
        "project-original",
    )
    tokenizer = cast(PreTrainedTokenizerBase, _FakeQwenTokenizer())

    example = tokenize_qwen_record(record, tokenizer, max_length=128)

    supervised_ids = [
        token_id
        for token_id, selected in zip(example.input_ids, example.assistant_mask, strict=True)
        if selected
    ]
    assert supervised_ids == [100 + ord("A"), 2]
    assert all(label == -100 for label in example.labels[:3])


def test_peft_lora_b_matrices_are_exact_zero_and_base_is_frozen() -> None:
    base = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    model = cast(
        PeftModel,
        get_peft_model(
            base,
            LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"),
        ),
    )

    assert lora_is_initial_noop(model)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    assert any(parameter.requires_grad for parameter in model.parameters())
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )


def test_handwritten_lora_matches_peft_forward_gradients_and_merge() -> None:
    base = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    peft_layer = cast(
        Any,
        dict(peft_model.named_modules())["base_model.model.model.layers.0.self_attn.q_proj"],
    )
    manual = LoRALinear(nn.Linear(32, 32, bias=False), rank=4, alpha=8.0, dropout=0.0)
    with torch.no_grad():
        manual.base_layer.weight.copy_(peft_layer.base_layer.weight)
        manual.lora_a.copy_(peft_layer.lora_A["default"].weight)
        torch.manual_seed(19)
        update = torch.randn_like(manual.lora_b)
        manual.lora_b.copy_(update)
        peft_layer.lora_B["default"].weight.copy_(update)
    inputs = torch.randn(2, 3, 32)

    manual_output = manual(inputs)
    peft_output = peft_layer(inputs)
    assert torch.allclose(manual_output, peft_output, atol=1e-6, rtol=1e-6)

    manual_output.square().sum().backward()
    peft_output.square().sum().backward()
    manual_a_gradient = manual.lora_a.grad
    manual_b_gradient = manual.lora_b.grad
    assert manual_a_gradient is not None and manual_b_gradient is not None
    assert torch.allclose(
        manual_a_gradient, peft_layer.lora_A["default"].weight.grad, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        manual_b_gradient, peft_layer.lora_B["default"].weight.grad, atol=1e-5, rtol=1e-5
    )

    peft_layer.merge()
    assert torch.allclose(peft_layer(inputs), manual_output, atol=1e-5, rtol=1e-5)
    peft_layer.unmerge()
    assert torch.allclose(peft_layer(inputs), manual_output, atol=1e-5, rtol=1e-5)
