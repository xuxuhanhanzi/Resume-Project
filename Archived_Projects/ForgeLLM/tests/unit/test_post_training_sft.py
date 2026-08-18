"""Chat Template, label mask, collator and token-normalized SFT tests."""

import pytest
import torch
from torch.nn import functional as F

from forgellm.post_training.chat_template import (
    IGNORE_INDEX,
    ChatTemplateError,
    tokenize_byte_bpe_record,
)
from forgellm.post_training.collator import collate_tokenized_conversations
from forgellm.post_training.hf_experiment import warmup_cosine_multiplier
from forgellm.post_training.schema import InstructionRecord, Message
from forgellm.post_training.sft import assistant_only_causal_loss, scale_gradients_by_token_count
from forgellm.tokenization.bpe import EOS_ID, PAD_ID, ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig


def _tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        TokenizerConfig(vocab_size=280, min_pair_frequency=2),
        ["<user> answer </user> <assistant> yes </assistant> " * 8],
    )


def _record(record_id: str, answer: str) -> InstructionRecord:
    return InstructionRecord(
        record_id,
        (Message("user", f"Answer {record_id}."), Message("assistant", answer)),
        "fixture",
        "project-original",
    )


def test_template_masks_role_and_user_tokens_but_supervises_assistant_and_eos() -> None:
    example = tokenize_byte_bpe_record(_record("one", "yes"), _tokenizer())
    selected_ids = [
        token_id
        for token_id, selected in zip(example.input_ids, example.assistant_mask, strict=True)
        if selected
    ]

    assert selected_ids[-1] == EOS_ID
    assert _tokenizer().decode(selected_ids[:-1]) == "yes"
    assert all(
        label == (token_id if selected else IGNORE_INDEX)
        for token_id, label, selected in zip(
            example.input_ids, example.labels, example.assistant_mask, strict=True
        )
    )
    assert example.supervised_tokens == len(selected_ids)


def test_truncation_rejects_zero_shifted_supervision() -> None:
    with pytest.raises(ChatTemplateError, match="no shifted assistant"):
        tokenize_byte_bpe_record(_record("one", "yes"), _tokenizer(), max_length=2)


def test_collator_right_padding_stays_outside_attention_and_loss() -> None:
    examples = [
        tokenize_byte_bpe_record(_record("short", "ok"), _tokenizer()),
        tokenize_byte_bpe_record(_record("longer", "a longer answer"), _tokenizer()),
    ]

    batch = collate_tokenized_conversations(examples, pad_token_id=PAD_ID, pad_to_multiple_of=8)

    assert batch.input_ids.shape == batch.labels.shape == batch.attention_mask.shape
    assert batch.input_ids.size(1) % 8 == 0
    padding = ~batch.attention_mask
    assert torch.all(batch.input_ids[padding] == PAD_ID)
    assert torch.all(batch.labels[padding] == IGNORE_INDEX)
    assert batch.supervised_tokens == sum(example.supervised_tokens for example in examples)


def test_assistant_only_loss_matches_manual_shifted_cross_entropy() -> None:
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 1]], dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )

    result = assistant_only_causal_loss(logits, labels)
    expected = F.cross_entropy(torch.stack((logits[0, 1], logits[0, 2])), torch.tensor([2, 1]))

    assert torch.allclose(result.loss, expected)
    assert result.target_tokens == 2
    assert result.correct_tokens == 2
    assert result.token_accuracy == 1.0


def test_zero_supervision_fails_fast() -> None:
    logits = torch.zeros(1, 3, 4)
    labels = torch.full((1, 3), IGNORE_INDEX)

    with pytest.raises(ValueError, match="no supervised"):
        assistant_only_causal_loss(logits, labels)


def test_accumulated_gradient_is_normalized_by_true_token_count() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    torch.autograd.backward(parameter * 6.0)

    scale_gradients_by_token_count([parameter], 3)

    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(2.0)


def test_token_warmup_cosine_schedule_boundaries() -> None:
    assert warmup_cosine_multiplier(0, maximum_tokens=1000, warmup_ratio=0.03) == 0.0
    assert warmup_cosine_multiplier(30, maximum_tokens=1000, warmup_ratio=0.03) == 1.0
    assert warmup_cosine_multiplier(1000, maximum_tokens=1000, warmup_ratio=0.03) == 0.0
    assert 0.0 < warmup_cosine_multiplier(500, maximum_tokens=1000, warmup_ratio=0.03) < 1.0
