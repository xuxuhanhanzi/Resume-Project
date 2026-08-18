"""Inspectable chat serialization with exact assistant-token supervision spans."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from forgellm.post_training.schema import InstructionRecord, Message
from forgellm.tokenization.bpe import BOS_ID, EOS_ID, ByteBPETokenizer

IGNORE_INDEX = -100


class ChatTemplateError(ValueError):
    """Raised when serialization would create ambiguous or empty supervision."""


@dataclass(frozen=True, slots=True)
class TokenizedConversation:
    """Token IDs, labels and assistant mask for one conversation."""

    record_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_mask: tuple[bool, ...]
    serialized_text: str

    def __post_init__(self) -> None:
        lengths = {len(self.input_ids), len(self.labels), len(self.assistant_mask)}
        if len(lengths) != 1 or not self.input_ids:
            raise ChatTemplateError("input_ids, labels and assistant_mask must be non-empty/equal")
        for token_id, label, selected in zip(
            self.input_ids, self.labels, self.assistant_mask, strict=True
        ):
            if selected != (label != IGNORE_INDEX):
                raise ChatTemplateError("labels and assistant_mask disagree")
            if selected and label != token_id:
                raise ChatTemplateError("supervised labels must copy the target token ID")
        if not any(self.assistant_mask[1:]):
            raise ChatTemplateError("conversation has no shifted assistant target token")

    @property
    def supervised_tokens(self) -> int:
        """Number of assistant targets after the causal one-token shift."""
        return sum(self.assistant_mask[1:])

    def fingerprint(self) -> str:
        """Hash all model-facing arrays."""
        payload = repr((self.input_ids, self.labels, self.assistant_mask)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def render_conversation(messages: Sequence[Message]) -> str:
    """Render the human-readable dependency-free Stage 4 template."""
    chunks: list[str] = []
    for message in messages:
        chunks.append(f"<{message.role}>\n{message.content}\n</{message.role}>\n")
    return "".join(chunks)


def _append_segment(
    *,
    text: str,
    selected: bool,
    encode: Callable[[str], list[int]],
    input_ids: list[int],
    assistant_mask: list[bool],
) -> None:
    encoded = encode(text)
    input_ids.extend(encoded)
    assistant_mask.extend([selected] * len(encoded))


def tokenize_with_segments(
    record: InstructionRecord,
    *,
    encode: Callable[[str], list[int]],
    bos_token_id: int | None,
    eos_token_id: int,
    max_length: int | None = None,
    supervise_eos: bool = True,
) -> TokenizedConversation:
    """Tokenize template segments separately so supervision spans are exact.

    Segment boundaries are part of this educational template contract. They avoid
    BPE merges crossing a role/content boundary and make every assistant span
    auditable without guessing character-to-token offsets.
    """
    if max_length is not None and max_length < 2:
        raise ChatTemplateError("max_length must be at least two")
    input_ids: list[int] = []
    assistant_mask: list[bool] = []
    if bos_token_id is not None:
        input_ids.append(bos_token_id)
        assistant_mask.append(False)
    for message in record.messages:
        _append_segment(
            text=f"<{message.role}>\n",
            selected=False,
            encode=encode,
            input_ids=input_ids,
            assistant_mask=assistant_mask,
        )
        _append_segment(
            text=message.content,
            selected=message.role == "assistant",
            encode=encode,
            input_ids=input_ids,
            assistant_mask=assistant_mask,
        )
        if message.role == "assistant":
            input_ids.append(eos_token_id)
            assistant_mask.append(supervise_eos)
        _append_segment(
            text=f"\n</{message.role}>\n",
            selected=False,
            encode=encode,
            input_ids=input_ids,
            assistant_mask=assistant_mask,
        )
    if max_length is not None:
        input_ids = input_ids[:max_length]
        assistant_mask = assistant_mask[:max_length]
    labels = [
        token_id if selected else IGNORE_INDEX
        for token_id, selected in zip(input_ids, assistant_mask, strict=True)
    ]
    return TokenizedConversation(
        record_id=record.record_id,
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        assistant_mask=tuple(assistant_mask),
        serialized_text=render_conversation(record.messages),
    )


def tokenize_byte_bpe_record(
    record: InstructionRecord,
    tokenizer: ByteBPETokenizer,
    *,
    max_length: int | None = None,
    supervise_eos: bool = True,
) -> TokenizedConversation:
    """Apply the Stage 1 tokenizer under the explicit Stage 4 template."""
    return tokenize_with_segments(
        record,
        encode=tokenizer.encode,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        max_length=max_length,
        supervise_eos=supervise_eos,
    )
