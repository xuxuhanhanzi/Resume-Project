"""Supervised post-training primitives for ForgeLLM Stage 4."""

from forgellm.post_training.collator import SFTBatch, collate_tokenized_conversations
from forgellm.post_training.lora import LoRALinear, inject_lora
from forgellm.post_training.schema import InstructionRecord, Message
from forgellm.post_training.sft import SFTLoss, assistant_only_causal_loss

__all__ = [
    "InstructionRecord",
    "LoRALinear",
    "Message",
    "SFTBatch",
    "SFTLoss",
    "assistant_only_causal_loss",
    "collate_tokenized_conversations",
    "inject_lora",
]
