"""Hugging Face tokenization and PEFT/QLoRA construction boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from forgellm.post_training.chat_template import TokenizedConversation
from forgellm.post_training.config import Stage4RunConfig
from forgellm.post_training.schema import InstructionRecord, Message


@dataclass(frozen=True, slots=True)
class HFStack:
    """Loaded tokenizer, PEFT model and measured parameter counts."""

    tokenizer: PreTrainedTokenizerBase
    model: PeftModel
    total_parameters: int
    trainable_parameters: int
    quantized_parameters: int


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(item) for item in encoded]


def _token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0 or token_id == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer does not define required special token {token!r}")
    return token_id


def render_qwen_messages(messages: tuple[Message, ...]) -> str:
    """Render the Qwen ChatML form used by the segmented tokenizer."""
    return "".join(
        f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n" for message in messages
    )


def tokenize_qwen_record(
    record: InstructionRecord,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> TokenizedConversation:
    """Tokenize Qwen ChatML while making assistant content and im_end explicit targets."""
    im_start = _token_id(tokenizer, "<|im_start|>")
    im_end = _token_id(tokenizer, "<|im_end|>")
    input_ids: list[int] = []
    mask: list[bool] = []
    for message in record.messages:
        prefix = [im_start, *_encode(tokenizer, f"{message.role}\n")]
        content = _encode(tokenizer, message.content)
        suffix = [im_end]
        input_ids.extend(prefix)
        mask.extend([False] * len(prefix))
        input_ids.extend(content)
        mask.extend([message.role == "assistant"] * len(content))
        input_ids.extend(suffix)
        mask.extend([message.role == "assistant"])
        newline = _encode(tokenizer, "\n")
        input_ids.extend(newline)
        mask.extend([False] * len(newline))
    input_ids = input_ids[:max_length]
    mask = mask[:max_length]
    labels = [
        token_id if selected else -100 for token_id, selected in zip(input_ids, mask, strict=True)
    ]
    return TokenizedConversation(
        record_id=record.record_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        assistant_mask=tuple(mask),
        serialized_text=render_qwen_messages(record.messages),
    )


def tokenize_qwen_generation_prompt(
    record: InstructionRecord,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> list[int]:
    """Serialize all messages before the final assistant and append its role prefix."""
    im_start = _token_id(tokenizer, "<|im_start|>")
    im_end = _token_id(tokenizer, "<|im_end|>")
    input_ids: list[int] = []
    for message in record.messages[:-1]:
        input_ids.extend([im_start, *_encode(tokenizer, f"{message.role}\n")])
        input_ids.extend(_encode(tokenizer, message.content))
        input_ids.extend([im_end, *_encode(tokenizer, "\n")])
    input_ids.extend([im_start, *_encode(tokenizer, "assistant\n")])
    if len(input_ids) > max_length:
        raise ValueError(f"generation prompt {record.record_id} exceeds max_length")
    return input_ids


def load_stage4_hf_stack(config: Stage4RunConfig) -> HFStack:
    """Load the exact Qwen revision, optionally quantize Base, then inject LoRA."""
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
    load_kwargs: dict[str, object] = {
        "revision": config.model.revision,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if config.model.quantization == "nf4":
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    base_raw = AutoModelForCausalLM.from_pretrained(config.model.model_id, **load_kwargs)
    base = cast(PreTrainedModel, base_raw)
    if config.model.quantization == "nf4":
        base = cast(
            PreTrainedModel,
            prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
                base,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            ),
        )
    else:
        base.to(torch.device("cuda"))  # type: ignore[arg-type]
        base.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    base.config.use_cache = False
    peft_config = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = cast(PeftModel, get_peft_model(base, peft_config))
    total = sum(parameter.numel() for parameter in peft_model.parameters())
    trainable = sum(
        parameter.numel() for parameter in peft_model.parameters() if parameter.requires_grad
    )
    quantized = sum(
        parameter.numel()
        for parameter in peft_model.parameters()
        if parameter.__class__.__name__ == "Params4bit"
    )
    if trainable <= 0 or trainable >= total:
        raise RuntimeError("PEFT trainable parameter boundary is invalid")
    return HFStack(tokenizer, peft_model, total, trainable, quantized)


def adapter_base_gradients_are_absent(model: nn.Module) -> bool:
    """Check that every non-LoRA parameter remains gradient-free."""
    for name, parameter in model.named_parameters():
        if "lora_" not in name and parameter.grad is not None:
            return False
    return True


def lora_is_initial_noop(model: nn.Module) -> bool:
    """Verify every PEFT LoRA B matrix starts at exact zero."""
    found = False
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            found = True
            if torch.count_nonzero(parameter.detach()).item() != 0:
                return False
    return found
