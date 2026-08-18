"""Tiny Stage 5 RM/DPO/PG/PPO/GRPO integration tests without Hub downloads."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer

from forgellm.alignment.dpo import dpo_loss, response_sequence_log_probs
from forgellm.alignment.method_lab import (
    reward_model_overfit,
    toy_grpo_training,
    toy_ppo_training,
)
from forgellm.alignment.preference_data import build_preference_splits
from forgellm.post_training.chat_template import IGNORE_INDEX


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<unk>": 3,
                "Q": 4,
                "A": 5,
                "B": 6,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def test_tiny_reward_ppo_and_grpo_learning_gates() -> None:
    records = build_preference_splits()["train"]
    reward = reward_model_overfit(records, seed=41, steps=100)
    ppo = toy_ppo_training(seed=41, iterations=15)
    grpo = toy_grpo_training(seed=41, iterations=20)
    assert reward.pair_accuracy == 1.0
    assert reward.final_loss < reward.initial_loss * 0.2
    assert ppo.final_expected_reward > ppo.initial_expected_reward
    assert grpo.final_expected_reward > grpo.initial_expected_reward


def test_handwritten_dpo_matches_trl_1_8_fixed_batch(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
        Qwen3Config(
            vocab_size=7,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=8,
        )
    )
    reference = copy.deepcopy(model)
    with torch.no_grad():
        model.lm_head.weight[5].add_(0.05)
        model.lm_head.weight[6].sub_(0.03)
    dataset = Dataset.from_dict({"prompt": ["Q"], "chosen": ["A"], "rejected": ["B"]})
    trainer = DPOTrainer(
        model=model,
        ref_model=reference,
        args=DPOConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            gradient_checkpointing=False,
            max_length=16,
            per_device_train_batch_size=1,
            beta=0.1,
        ),
        train_dataset=dataset,
        processing_class=_tiny_tokenizer(),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    trl_loss = trainer.compute_loss(model, batch)  # type: ignore[no-untyped-call]
    input_ids = batch["input_ids"]
    completion_mask = batch["completion_mask"].bool()
    labels = input_ids.masked_fill(~completion_mask, IGNORE_INDEX)
    policy_sequence = response_sequence_log_probs(
        model(input_ids=input_ids, attention_mask=batch["attention_mask"]).logits,
        labels,
    )
    with torch.no_grad():
        reference_sequence = response_sequence_log_probs(
            reference(input_ids=input_ids, attention_mask=batch["attention_mask"]).logits,
            labels,
        )
    policy_chosen, policy_rejected = policy_sequence.sums.chunk(2)
    reference_chosen, reference_rejected = reference_sequence.sums.chunk(2)
    manual = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    assert isinstance(trl_loss, torch.Tensor)
    torch.testing.assert_close(manual.loss, trl_loss, atol=1e-6, rtol=1e-6)
