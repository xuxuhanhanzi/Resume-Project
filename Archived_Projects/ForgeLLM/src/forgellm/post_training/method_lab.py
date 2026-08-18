"""Low-cost Stage 4 SFT/LoRA experiments on the inspectable DecoderLM."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import Tensor, nn

from forgellm.model.config import ModelConfig
from forgellm.model.decoder import DecoderLM
from forgellm.post_training.chat_template import IGNORE_INDEX, TokenizedConversation
from forgellm.post_training.lora import inject_lora
from forgellm.post_training.sft import assistant_only_causal_loss
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class TinyVariantResult:
    """One bounded tiny-model result."""

    variant: str
    seed: int
    steps: int
    first_loss: float
    final_train_loss: float
    final_train_token_accuracy: float
    trainable_parameters: int
    elapsed_seconds: float
    losses_finite: bool

    def as_dict(self) -> dict[str, JsonValue]:
        """Return JSON-compatible values."""
        return cast(dict[str, JsonValue], asdict(self))


def _labels_for_variant(example: TokenizedConversation, *, assistant_only: bool) -> Tensor:
    labels = list(example.labels if assistant_only else example.input_ids)
    labels[0] = IGNORE_INDEX
    return torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def evaluate_examples(
    model: DecoderLM,
    examples: list[TokenizedConversation],
    *,
    device: torch.device,
    assistant_only: bool,
) -> tuple[float, float]:
    """Return token-weighted NLL and accuracy."""
    model.eval()
    nll_sum = 0.0
    target_tokens = 0
    correct_tokens = 0
    for example in examples:
        input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        labels = _labels_for_variant(example, assistant_only=assistant_only).to(device)
        result = assistant_only_causal_loss(model(input_ids).logits, labels)
        nll_sum += float(result.negative_log_likelihood_sum.item())
        target_tokens += result.target_tokens
        correct_tokens += result.correct_tokens
    return nll_sum / target_tokens, correct_tokens / target_tokens


def run_tiny_variant(
    *,
    base_state: dict[str, Tensor],
    model_config: ModelConfig,
    examples: list[TokenizedConversation],
    variant: str,
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> TinyVariantResult:
    """Run full-sequence, assistant-only or LoRA SFT with fixed cyclic samples."""
    if variant not in ("full_sequence", "assistant_only", "lora"):
        raise ValueError("unsupported tiny SFT variant")
    if steps <= 0 or not examples:
        raise ValueError("steps and examples must be positive/non-empty")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DecoderLM(model_config)
    model.load_state_dict(copy.deepcopy(base_state))
    if variant == "lora":
        inject_lora(model, rank=4, alpha=8.0, dropout=0.0)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    assistant_only = variant != "full_sequence"
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        example = examples[step % len(examples)]
        input_ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
        labels = _labels_for_variant(example, assistant_only=assistant_only).to(device)
        optimizer.zero_grad(set_to_none=True)
        result = assistant_only_causal_loss(model(input_ids).logits, labels)
        torch.autograd.backward(result.loss)
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(result.loss.detach().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_loss, final_accuracy = evaluate_examples(
        model, examples, device=device, assistant_only=assistant_only
    )
    return TinyVariantResult(
        variant=variant,
        seed=seed,
        steps=steps,
        first_loss=losses[0],
        final_train_loss=final_loss,
        final_train_token_accuracy=final_accuracy,
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        elapsed_seconds=elapsed,
        losses_finite=all(math.isfinite(value) for value in losses),
    )


def create_frozen_tiny_state(config: ModelConfig, *, seed: int) -> dict[str, Tensor]:
    """Create one CPU state reused by every controlled variant for a seed."""
    torch.manual_seed(seed)
    model = DecoderLM(config)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
