"""Right-padding SFT collator that preserves assistant-only labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forgellm.post_training.chat_template import IGNORE_INDEX, TokenizedConversation


@dataclass(frozen=True, slots=True)
class SFTBatch:
    """One padded batch plus exact supervised-token accounting."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    supervised_tokens: int


def collate_tokenized_conversations(
    examples: list[TokenizedConversation],
    *,
    pad_token_id: int,
    pad_to_multiple_of: int | None = None,
) -> SFTBatch:
    """Right-pad conversations and keep padding outside both attention and loss."""
    if not examples:
        raise ValueError("examples must be non-empty")
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be positive")
    max_length = max(len(example.input_ids) for example in examples)
    if pad_to_multiple_of is not None:
        max_length = (
            (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    attention_rows: list[list[bool]] = []
    supervised_tokens = 0
    for example in examples:
        padding = max_length - len(example.input_ids)
        input_rows.append([*example.input_ids, *([pad_token_id] * padding)])
        label_rows.append([*example.labels, *([IGNORE_INDEX] * padding)])
        attention_rows.append([*([True] * len(example.input_ids)), *([False] * padding)])
        supervised_tokens += example.supervised_tokens
    if supervised_tokens <= 0:
        raise ValueError("batch has no shifted assistant target tokens")
    return SFTBatch(
        input_ids=torch.tensor(input_rows, dtype=torch.long),
        labels=torch.tensor(label_rows, dtype=torch.long),
        attention_mask=torch.tensor(attention_rows, dtype=torch.bool),
        supervised_tokens=supervised_tokens,
    )
