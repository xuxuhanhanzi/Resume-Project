"""Small construction helper shared by Stage 3 command-line laboratories."""

from __future__ import annotations

from pathlib import Path

from forgellm.model.decoder import DecoderLM
from forgellm.tokenization.bpe import ByteBPETokenizer
from forgellm.training.config import ExperimentConfig
from forgellm.training.data import DeterministicBatchStream, load_or_create_packed_jsonl
from forgellm.training.trainer import Trainer, seed_everything


def create_trainer(
    experiment: ExperimentConfig,
    *,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
) -> Trainer:
    """Construct all state in a deterministic order suitable for restart tests."""
    seed_everything(experiment.training.seed)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    train_dataset = load_or_create_packed_jsonl(
        train_path,
        tokenizer,
        sequence_length=experiment.training.sequence_length,
    )
    validation_dataset = load_or_create_packed_jsonl(
        validation_path,
        tokenizer,
        sequence_length=experiment.training.sequence_length,
    )
    train_stream = DeterministicBatchStream(
        train_dataset,
        batch_size=experiment.training.micro_batch_size,
        seed=experiment.training.seed,
    )
    model = DecoderLM(experiment.model)
    return Trainer(
        experiment,
        model,
        tokenizer,
        train_stream,
        validation_dataset,
        output_dir,
    )
