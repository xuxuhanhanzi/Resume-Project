"""Small end-to-end SFT/LoRA and TRL loss tests without Hub downloads."""

from pathlib import Path

import torch
from datasets import Dataset  # type: ignore[import-untyped]
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from forgellm.model.config import ModelConfig
from forgellm.post_training.chat_template import tokenize_byte_bpe_record
from forgellm.post_training.correctness_data import build_correctness_records
from forgellm.post_training.method_lab import create_frozen_tiny_state, run_tiny_variant
from forgellm.post_training.sft import assistant_only_causal_loss
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.tokenization.config import TokenizerConfig


def test_tiny_assistant_sft_and_lora_reduce_training_loss() -> None:
    tokenizer = ByteBPETokenizer.train(
        TokenizerConfig(300, 2),
        ["follow output format item result assistant user " * 20],
    )
    examples = [
        tokenize_byte_bpe_record(record, tokenizer, max_length=128)
        for record in build_correctness_records()[:2]
    ]
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=128,
        attention_backend="sdpa",
    )
    state = create_frozen_tiny_state(config, seed=8)

    assistant = run_tiny_variant(
        base_state=state,
        model_config=config,
        examples=examples,
        variant="assistant_only",
        seed=8,
        steps=20,
        learning_rate=1e-2,
        device=torch.device("cpu"),
    )
    lora = run_tiny_variant(
        base_state=state,
        model_config=config,
        examples=examples,
        variant="lora",
        seed=8,
        steps=30,
        learning_rate=2e-2,
        device=torch.device("cpu"),
    )

    assert assistant.losses_finite and lora.losses_finite
    assert assistant.final_train_loss < assistant.first_loss
    assert lora.final_train_loss < lora.first_loss
    assert lora.trainable_parameters < assistant.trainable_parameters


def test_pretokenized_assistant_loss_matches_trl_compute_loss(tmp_path: Path) -> None:
    model = Qwen3ForCausalLM(  # type: ignore[no-untyped-call]
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
    input_ids = torch.tensor([[1, 7, 8, 9, 10, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, -100, 9, 10, 2]], dtype=torch.long)
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=Tokenizer(
            WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}, unk_token="<unk>")
        ),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        }
    )
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            gradient_checkpointing=False,
            max_length=16,
            loss_type="nll",
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    trl_loss = trainer.compute_loss(model, inputs)  # type: ignore[no-untyped-call]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    manual = assistant_only_causal_loss(logits, labels)

    assert isinstance(trl_loss, torch.Tensor)
    assert torch.allclose(trl_loss, manual.loss, atol=1e-6, rtol=1e-6)
