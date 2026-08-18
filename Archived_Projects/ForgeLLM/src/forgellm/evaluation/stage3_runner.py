"""Frozen language-modeling evaluation for the Stage 3 checkpoint."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from forgellm.evaluation.identity import file_sha256
from forgellm.evaluation.language_modeling import LanguageModelingTotals
from forgellm.model import DecoderLM
from forgellm.structured_logging import JsonValue
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.training.checkpoint import load_checkpoint
from forgellm.training.config import load_experiment_config
from forgellm.training.data import load_or_create_packed_jsonl, validation_batches


@torch.no_grad()
def evaluate_stage3_checkpoint(
    *,
    config_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    validation_path: Path,
) -> dict[str, JsonValue]:
    """Evaluate the frozen Stage 3 model on its fixed validation prefix."""
    config = load_experiment_config(config_path)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise RuntimeError("Stage 3 tokenizer/model vocabulary mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoderLM(config.model).to(device)
    payload = load_checkpoint(checkpoint_path, map_location=device)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("Stage 3 checkpoint has no model_state")
    model.load_state_dict(cast(dict[str, Tensor], state), strict=True)
    model.eval()
    dataset = load_or_create_packed_jsonl(
        validation_path,
        tokenizer,
        sequence_length=config.training.sequence_length,
    )
    batches = validation_batches(
        dataset,
        batch_size=config.training.micro_batch_size,
        batch_count=config.training.validation_batches,
        device=device,
    )
    nll = 0.0
    target_tokens = 0
    target_bytes = 0
    for batch in batches:
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            logits = model(batch.token_ids).logits
            loss_sum = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                batch.token_ids[:, 1:].reshape(-1),
                reduction="sum",
            )
        nll += float(loss_sum.item())
        target_tokens += batch.target_tokens
        target_bytes += batch.target_bytes
    totals = LanguageModelingTotals(
        negative_log_likelihood=nll,
        target_tokens=target_tokens,
        target_bytes=target_bytes,
        tokenizer_sha256=file_sha256(tokenizer_path),
    )
    return {
        "model_key": "m3",
        "suite": "stage3_language_modeling_validation",
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "validation_path": validation_path.as_posix(),
        "validation_sha256": file_sha256(validation_path),
        "batches": len(batches),
        "metrics": cast(JsonValue, totals.as_dict()),
        "comparison_boundary": (
            "M3 uses a different tokenizer and task suite; do not rank its PPL against Q0-Q2."
        ),
    }
