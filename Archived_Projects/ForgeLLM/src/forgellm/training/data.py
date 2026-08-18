"""Document-safe token packing and a checkpointable deterministic batch stream."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from forgellm.tokenization.bpe import ByteBPETokenizer

_STREAM_SCHEMA = "forgellm-deterministic-batch-stream-v1"
_CACHE_SCHEMA = "forgellm-packed-token-cache-v1"


class TrainingDataError(ValueError):
    """Raised when pretraining data cannot satisfy the deterministic contract."""


@dataclass(frozen=True, slots=True)
class TrainingDocument:
    """One already-split document supplied to the pretraining packer."""

    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """One tensor batch plus auditable sample and byte counts."""

    token_ids: Tensor
    sample_indices: Tensor
    target_tokens: int
    target_bytes: int


class PackedTokenDataset:
    """Non-overlapping target windows over a packed document token stream.

    Consecutive samples overlap by one context token. Therefore every retained
    target transition appears exactly once even though the boundary context token
    appears in two adjacent samples.
    """

    def __init__(
        self,
        token_ids: list[int] | Tensor,
        token_byte_lengths: list[int] | Tensor,
        *,
        sequence_length: int,
        document_ids: tuple[str, ...],
        source_path: Path,
    ) -> None:
        if sequence_length < 2:
            raise TrainingDataError("sequence_length must be at least 2")
        resolved_token_ids = torch.as_tensor(token_ids, dtype=torch.int32).contiguous()
        resolved_byte_lengths = torch.as_tensor(token_byte_lengths, dtype=torch.int16).contiguous()
        if resolved_token_ids.ndim != 1 or resolved_byte_lengths.ndim != 1:
            raise TrainingDataError("packed token arrays must be one-dimensional")
        if len(resolved_token_ids) != len(resolved_byte_lengths):
            raise TrainingDataError("token IDs and byte lengths must align")
        if len(resolved_token_ids) < sequence_length:
            raise TrainingDataError("token stream is shorter than one sequence")
        if bool((resolved_token_ids < 0).any()) or bool((resolved_byte_lengths < 0).any()):
            raise TrainingDataError("packed token values must be non-negative")
        if not document_ids:
            raise TrainingDataError("at least one document is required")
        self._token_ids = resolved_token_ids.clone()
        self._token_byte_lengths = resolved_byte_lengths.clone()
        self.sequence_length = sequence_length
        self.document_ids = document_ids
        self.source_path = source_path
        self._stride = sequence_length - 1
        self._sample_count = (len(self._token_ids) - 1) // self._stride
        if self._sample_count == 0:
            raise TrainingDataError("token stream does not contain one prediction window")
        self._fingerprint = self._compute_fingerprint()

    @classmethod
    def from_documents(
        cls,
        documents: list[TrainingDocument],
        tokenizer: ByteBPETokenizer,
        *,
        sequence_length: int,
        source_path: Path,
    ) -> PackedTokenDataset:
        """Encode documents independently and insert EOS without crossing merges."""
        if not documents:
            raise TrainingDataError("document collection must be non-empty")
        token_ids: list[int] = []
        byte_lengths: list[int] = []
        seen_ids: set[str] = set()
        for document in documents:
            if not document.document_id or document.document_id in seen_ids:
                raise TrainingDataError("document IDs must be non-empty and unique")
            seen_ids.add(document.document_id)
            encoded = tokenizer.encode(document.text, add_eos=True)
            token_ids.extend(encoded)
            for token_id in encoded:
                payload = tokenizer.token_bytes(token_id)
                byte_lengths.append(len(payload) if payload is not None else 0)
        return cls(
            token_ids,
            byte_lengths,
            sequence_length=sequence_length,
            document_ids=tuple(document.document_id for document in documents),
            source_path=source_path,
        )

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("sample index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self._stride
        end = start + self.sequence_length
        tokens = self._token_ids[start:end].to(dtype=torch.long)
        target_bytes = int(self._token_byte_lengths[start + 1 : end].sum())
        return tokens, target_bytes

    @property
    def total_tokens(self) -> int:
        return len(self._token_ids)

    @property
    def retained_target_tokens(self) -> int:
        return len(self) * self._stride

    @property
    def dropped_tail_tokens(self) -> int:
        return (len(self._token_ids) - 1) - self.retained_target_tokens

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"sequence_length={self.sequence_length}\n".encode())
        digest.update(self._token_ids.numpy().tobytes())
        return digest.hexdigest()

    def cache_payload(self, *, source_sha256: str, tokenizer_sha256: str) -> dict[str, object]:
        """Return the safe tensor/primitives payload used by the local token cache."""
        return {
            "schema_version": _CACHE_SCHEMA,
            "source_sha256": source_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "sequence_length": self.sequence_length,
            "document_ids": list(self.document_ids),
            "token_ids": self._token_ids,
            "token_byte_lengths": self._token_byte_lengths,
            "dataset_fingerprint": self.fingerprint,
        }


def read_training_documents(path: Path) -> list[TrainingDocument]:
    """Read processed JSONL while rejecting malformed or duplicate records."""
    documents: list[TrainingDocument] = []
    seen_ids: set[str] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as error:
        raise TrainingDataError(f"could not open training data {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise TrainingDataError(f"blank JSONL line at {path}:{line_number}")
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingDataError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise TrainingDataError(f"record at {path}:{line_number} must be an object")
            record = cast(dict[str, object], raw)
            if "id" not in record or "text" not in record:
                raise TrainingDataError(f"record at {path}:{line_number} requires id and text")
            document_id = record["id"]
            text = record["text"]
            if not isinstance(document_id, str) or not document_id:
                raise TrainingDataError(f"invalid document id at {path}:{line_number}")
            if document_id in seen_ids:
                raise TrainingDataError(f"duplicate document id {document_id!r}")
            if not isinstance(text, str) or not text:
                raise TrainingDataError(f"invalid document text at {path}:{line_number}")
            seen_ids.add(document_id)
            documents.append(TrainingDocument(document_id, text))
    if not documents:
        raise TrainingDataError(f"no documents found in {path}")
    return documents


def load_packed_jsonl(
    path: Path,
    tokenizer: ByteBPETokenizer,
    *,
    sequence_length: int,
) -> PackedTokenDataset:
    """Load one pre-split JSONL file and freeze its packed token windows."""
    return PackedTokenDataset.from_documents(
        read_training_documents(path),
        tokenizer,
        sequence_length=sequence_length,
        source_path=path,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_cache_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def load_or_create_packed_jsonl(
    path: Path,
    tokenizer: ByteBPETokenizer,
    *,
    sequence_length: int,
    cache_path: Path | None = None,
) -> PackedTokenDataset:
    """Reuse a fingerprinted token cache or create it once from document JSONL."""
    resolved_cache = cache_path or path.with_suffix(f".seq{sequence_length}.tokens.pt")
    source_sha256 = _sha256_file(path)
    tokenizer_sha256 = tokenizer.fingerprint()
    if not resolved_cache.exists():
        dataset = load_packed_jsonl(path, tokenizer, sequence_length=sequence_length)
        _save_cache_atomic(
            resolved_cache,
            dataset.cache_payload(
                source_sha256=source_sha256,
                tokenizer_sha256=tokenizer_sha256,
            ),
        )
        return dataset
    try:
        raw: object = torch.load(resolved_cache, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingDataError(f"could not load token cache {resolved_cache}: {error}") from error
    if not isinstance(raw, dict):
        raise TrainingDataError("token cache must be a mapping")
    cache = cast(dict[str, object], raw)
    expected_fields = {
        "schema_version",
        "source_sha256",
        "tokenizer_sha256",
        "sequence_length",
        "document_ids",
        "token_ids",
        "token_byte_lengths",
        "dataset_fingerprint",
    }
    if set(cache) != expected_fields or cache["schema_version"] != _CACHE_SCHEMA:
        raise TrainingDataError("token cache schema does not match")
    if cache["source_sha256"] != source_sha256:
        raise TrainingDataError("token cache source fingerprint does not match")
    if cache["tokenizer_sha256"] != tokenizer_sha256:
        raise TrainingDataError("token cache tokenizer fingerprint does not match")
    if cache["sequence_length"] != sequence_length:
        raise TrainingDataError("token cache sequence length does not match")
    document_ids = cache["document_ids"]
    token_ids = cache["token_ids"]
    byte_lengths = cache["token_byte_lengths"]
    if (
        not isinstance(document_ids, list)
        or not all(isinstance(item, str) for item in document_ids)
        or not isinstance(token_ids, Tensor)
        or not isinstance(byte_lengths, Tensor)
    ):
        raise TrainingDataError("token cache values are invalid")
    dataset = PackedTokenDataset(
        token_ids,
        byte_lengths,
        sequence_length=sequence_length,
        document_ids=tuple(cast(list[str], document_ids)),
        source_path=path,
    )
    if dataset.fingerprint != cache["dataset_fingerprint"]:
        raise TrainingDataError("token cache dataset fingerprint does not match")
    return dataset


class DeterministicBatchStream:
    """Seeded epoch permutations represented only by epoch and cursor state."""

    def __init__(self, dataset: PackedTokenDataset, *, batch_size: int, seed: int) -> None:
        if batch_size <= 0 or seed < 0:
            raise TrainingDataError("batch_size must be positive and seed non-negative")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.position = 0
        self._order = self._order_for_epoch(0)

    def _order_for_epoch(self, epoch: int) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + epoch)
        return torch.randperm(len(self.dataset), generator=generator)

    def _next_indices(self) -> Tensor:
        pieces: list[Tensor] = []
        remaining = self.batch_size
        while remaining:
            available = len(self.dataset) - self.position
            take = min(available, remaining)
            if take:
                pieces.append(self._order[self.position : self.position + take])
                self.position += take
                remaining -= take
            if self.position == len(self.dataset):
                self.epoch += 1
                self.position = 0
                self._order = self._order_for_epoch(self.epoch)
        return torch.cat(pieces)

    def next_batch(self, device: torch.device | str = "cpu") -> TrainingBatch:
        """Return one full batch and advance the checkpointable cursor."""
        indices = self._next_indices()
        samples = [self.dataset[int(index)] for index in indices]
        token_ids = torch.stack([sample[0] for sample in samples]).to(device)
        return TrainingBatch(
            token_ids=token_ids,
            sample_indices=indices,
            target_tokens=token_ids.size(0) * (token_ids.size(1) - 1),
            target_bytes=sum(sample[1] for sample in samples),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": _STREAM_SCHEMA,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected_keys = {
            "schema_version",
            "dataset_fingerprint",
            "batch_size",
            "seed",
            "epoch",
            "position",
        }
        if set(state) != expected_keys or state["schema_version"] != _STREAM_SCHEMA:
            raise TrainingDataError("unsupported deterministic stream checkpoint")
        if state["dataset_fingerprint"] != self.dataset.fingerprint:
            raise TrainingDataError("checkpoint dataset fingerprint does not match")
        if state["batch_size"] != self.batch_size or state["seed"] != self.seed:
            raise TrainingDataError("checkpoint batch stream settings do not match")
        epoch = state["epoch"]
        position = state["position"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < len(self.dataset)
        ):
            raise TrainingDataError("checkpoint batch cursor is invalid")
        self.epoch = epoch
        self.position = position
        self._order = self._order_for_epoch(epoch)


def validation_batches(
    dataset: PackedTokenDataset,
    *,
    batch_size: int,
    batch_count: int,
    device: torch.device | str,
) -> list[TrainingBatch]:
    """Build a fixed, non-mutating validation prefix."""
    if batch_size <= 0 or batch_count <= 0:
        raise TrainingDataError("validation batch size and count must be positive")
    batches: list[TrainingBatch] = []
    for batch_index in range(batch_count):
        indices = torch.tensor(
            [((batch_index * batch_size) + offset) % len(dataset) for offset in range(batch_size)]
        )
        samples = [dataset[int(index)] for index in indices]
        token_ids = torch.stack([sample[0] for sample in samples]).to(device)
        batches.append(
            TrainingBatch(
                token_ids=token_ids,
                sample_indices=indices,
                target_tokens=token_ids.size(0) * (token_ids.size(1) - 1),
                target_bytes=sum(sample[1] for sample in samples),
            )
        )
    return batches
