"""Deterministic, educational byte-level BPE implementation."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from forgellm.structured_logging import JsonValue
from forgellm.tokenization.config import BASE_VOCAB_SIZE, TokenizerConfig

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
SCHEMA_VERSION = "forgellm-byte-bpe-v1"
SPECIAL_TOKENS: dict[str, int] = {
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
    "<pad>": PAD_ID,
    "<unk>": UNK_ID,
}
_SPECIAL_BY_ID = {token_id: token for token, token_id in SPECIAL_TOKENS.items()}
_MODEL_FIELDS = frozenset(
    {
        "config",
        "merges",
        "model_sha256",
        "schema_version",
        "special_tokens",
        "vocabulary_hex",
    }
)


class TokenizerError(ValueError):
    """Raised when tokenizer training, encoding, or model validation fails."""


@dataclass(frozen=True, slots=True)
class Merge:
    """One ranked BPE merge."""

    left_id: int
    right_id: int
    new_id: int

    def as_dict(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible mapping."""
        return {
            "left_id": self.left_id,
            "new_id": self.new_id,
            "right_id": self.right_id,
        }


def _canonical_json(payload: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _replace_pair(sequence: Sequence[int], pair: tuple[int, int], new_id: int) -> list[int]:
    output: list[int] = []
    index = 0
    left_id, right_id = pair
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left_id
            and sequence[index + 1] == right_id
        ):
            output.append(new_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


class ByteBPETokenizer:
    """A deterministic byte-level BPE tokenizer with fixed special-token IDs."""

    def __init__(
        self,
        config: TokenizerConfig,
        vocabulary: Sequence[bytes | None],
        merges: Sequence[Merge],
    ) -> None:
        self.config = config
        self._vocabulary = tuple(vocabulary)
        self._merges = tuple(merges)
        self._validate_structure()

    @classmethod
    def train(cls, config: TokenizerConfig, texts: Iterable[str]) -> ByteBPETokenizer:
        """Train byte-level BPE without crossing document boundaries."""
        sequences = [
            [byte + BYTE_OFFSET for byte in text.encode("utf-8", errors="strict")] for text in texts
        ]
        vocabulary: list[bytes | None] = [None] * BYTE_OFFSET
        vocabulary.extend(bytes([byte]) for byte in range(256))
        merges: list[Merge] = []

        while len(vocabulary) < config.vocab_size:
            counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:], strict=False))
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < config.min_pair_frequency:
                break

            new_id = len(vocabulary)
            left_payload = vocabulary[pair[0]]
            right_payload = vocabulary[pair[1]]
            if left_payload is None or right_payload is None:
                raise TokenizerError("training pair unexpectedly contains a special token")
            vocabulary.append(left_payload + right_payload)
            merges.append(Merge(pair[0], pair[1], new_id))
            sequences = [_replace_pair(sequence, pair, new_id) for sequence in sequences]

        return cls(config, vocabulary, merges)

    @property
    def vocab_size(self) -> int:
        """Return the actual vocabulary size after early stopping."""
        return len(self._vocabulary)

    @property
    def merge_count(self) -> int:
        """Return the number of learned merges."""
        return len(self._merges)

    @property
    def unknown_token_id(self) -> int:
        """Return the fixed unknown-token ID."""
        return UNK_ID

    @property
    def merges(self) -> tuple[Merge, ...]:
        """Expose the immutable ranked merge sequence."""
        return self._merges

    def token_bytes(self, token_id: int) -> bytes | None:
        """Return the byte payload for one valid token ID."""
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TokenizerError("token ID must be an integer")
        if not 0 <= token_id < self.vocab_size:
            raise TokenizerError(f"token ID out of range: {token_id}")
        return self._vocabulary[token_id]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode one string with a heap that preserves the frozen merge order."""
        if not isinstance(text, str):
            raise TokenizerError("text must be a string")
        sequence = [byte + BYTE_OFFSET for byte in text.encode("utf-8", errors="strict")]
        sequence = self._apply_ranked_merges(sequence)
        if add_bos:
            sequence.insert(0, BOS_ID)
        if add_eos:
            sequence.append(EOS_ID)
        return sequence

    def _apply_ranked_merges(self, sequence: list[int]) -> list[int]:
        """Apply standard ranked BPE merges in near O(n log n) time.

        Heap entries are validated when popped, so stale neighboring pairs are
        harmless. Ordering by ``(rank, left_position)`` exactly matches the old
        implementation's rank-by-rank, left-to-right non-overlapping replay.
        """
        if len(sequence) < 2 or not self._merges:
            return sequence
        pair_rules = {
            (merge.left_id, merge.right_id): (rank, merge.new_id)
            for rank, merge in enumerate(self._merges)
        }
        tokens = sequence.copy()
        previous = [index - 1 for index in range(len(tokens))]
        following = [index + 1 for index in range(len(tokens))]
        following[-1] = -1
        alive = [True] * len(tokens)
        versions = [0] * len(tokens)
        heap: list[tuple[int, int, int, int, int, int]] = []

        def push_pair(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            rule = pair_rules.get((tokens[left], tokens[right]))
            if rule is not None:
                rank, new_id = rule
                heapq.heappush(
                    heap,
                    (rank, left, right, versions[left], versions[right], new_id),
                )

        for index in range(len(tokens) - 1):
            push_pair(index)
        while heap:
            _, left, right, left_version, right_version, new_id = heapq.heappop(heap)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or versions[left] != left_version
                or versions[right] != right_version
            ):
                continue
            tokens[left] = new_id
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            next_index = following[right]
            following[left] = next_index
            if next_index >= 0:
                previous[next_index] = left
            push_pair(previous[left])
            push_pair(left)

        output: list[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(tokens[index])
            index = following[index]
        return output

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode token IDs by concatenating their exact byte payloads."""
        payload = bytearray()
        for token_id in token_ids:
            token_payload = self.token_bytes(token_id)
            if token_id in _SPECIAL_BY_ID:
                if not skip_special_tokens:
                    payload.extend(_SPECIAL_BY_ID[token_id].encode("utf-8"))
                continue
            if token_payload is None:
                raise TokenizerError(f"ordinary token {token_id} has no byte payload")
            payload.extend(token_payload)
        try:
            return bytes(payload).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise TokenizerError(f"token sequence is not valid UTF-8: {error}") from error

    def _payload(self) -> dict[str, JsonValue]:
        return {
            "config": self.config.as_dict(),
            "merges": [merge.as_dict() for merge in self._merges],
            "schema_version": SCHEMA_VERSION,
            "special_tokens": dict(sorted(SPECIAL_TOKENS.items())),
            "vocabulary_hex": [
                token_payload.hex() if token_payload is not None else None
                for token_payload in self._vocabulary
            ],
        }

    def fingerprint(self) -> str:
        """Hash the canonical model payload, excluding the hash field itself."""
        return hashlib.sha256(_canonical_json(self._payload())).hexdigest()

    def as_dict(self) -> dict[str, JsonValue]:
        """Return the complete serialized model including its fingerprint."""
        payload = self._payload()
        payload["model_sha256"] = self.fingerprint()
        return payload

    def save(self, path: Path) -> None:
        """Save a validated JSON model while refusing to overwrite an existing file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"tokenizer model already exists: {path}")
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: Path) -> ByteBPETokenizer:
        """Load and fully validate a saved model."""
        try:
            parsed: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TokenizerError(f"Could not read tokenizer model {path}: {error}") from error
        if not isinstance(parsed, dict) or set(parsed) != _MODEL_FIELDS:
            raise TokenizerError("tokenizer model has invalid top-level fields")
        model = cast(dict[str, object], parsed)
        if model["schema_version"] != SCHEMA_VERSION:
            raise TokenizerError(f"unsupported tokenizer schema: {model['schema_version']!r}")
        if model["special_tokens"] != SPECIAL_TOKENS:
            raise TokenizerError("tokenizer special-token mapping does not match v1 contract")

        config_values = model["config"]
        if not isinstance(config_values, Mapping):
            raise TokenizerError("tokenizer config must be an object")
        try:
            config = TokenizerConfig.from_mapping(cast(Mapping[str, object], config_values))
        except ValueError as error:
            raise TokenizerError(f"invalid tokenizer config: {error}") from error

        raw_vocabulary = model["vocabulary_hex"]
        if not isinstance(raw_vocabulary, list):
            raise TokenizerError("vocabulary_hex must be a list")
        vocabulary: list[bytes | None] = []
        for index, value in enumerate(raw_vocabulary):
            if value is None:
                vocabulary.append(None)
            elif isinstance(value, str):
                try:
                    vocabulary.append(bytes.fromhex(value))
                except ValueError as error:
                    raise TokenizerError(f"invalid vocabulary hex at ID {index}") from error
            else:
                raise TokenizerError(f"invalid vocabulary entry at ID {index}")

        raw_merges = model["merges"]
        if not isinstance(raw_merges, list):
            raise TokenizerError("merges must be a list")
        merges: list[Merge] = []
        for rank, value in enumerate(raw_merges):
            if not isinstance(value, dict) or set(value) != {"left_id", "new_id", "right_id"}:
                raise TokenizerError(f"invalid merge at rank {rank}")
            merge_values = cast(dict[str, object], value)
            ids = (
                merge_values["left_id"],
                merge_values["right_id"],
                merge_values["new_id"],
            )
            if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
                raise TokenizerError(f"merge IDs must be integers at rank {rank}")
            merges.append(Merge(cast(int, ids[0]), cast(int, ids[1]), cast(int, ids[2])))

        instance = cls(config, vocabulary, merges)
        expected_hash = model["model_sha256"]
        if not isinstance(expected_hash, str) or expected_hash != instance.fingerprint():
            raise TokenizerError("tokenizer model SHA-256 mismatch")
        return instance

    def _validate_structure(self) -> None:
        if not BASE_VOCAB_SIZE <= len(self._vocabulary) <= self.config.vocab_size:
            raise TokenizerError("actual vocabulary size violates configured bounds")
        if len(self._merges) != len(self._vocabulary) - BASE_VOCAB_SIZE:
            raise TokenizerError("merge count must equal learned vocabulary size")
        if self._vocabulary[:BYTE_OFFSET] != (None,) * BYTE_OFFSET:
            raise TokenizerError("special-token vocabulary entries must have no byte payload")
        for byte in range(256):
            if self._vocabulary[byte + BYTE_OFFSET] != bytes([byte]):
                raise TokenizerError("base byte vocabulary does not match the v1 ID contract")

        seen_pairs: set[tuple[int, int]] = set()
        for rank, merge in enumerate(self._merges):
            expected_new_id = BASE_VOCAB_SIZE + rank
            if merge.new_id != expected_new_id:
                raise TokenizerError(f"merge rank {rank} has non-contiguous new_id")
            if not 0 <= merge.left_id < merge.new_id or not 0 <= merge.right_id < merge.new_id:
                raise TokenizerError(f"merge rank {rank} references an unavailable token")
            pair = (merge.left_id, merge.right_id)
            if pair in seen_pairs:
                raise TokenizerError(f"merge rank {rank} repeats an earlier pair")
            seen_pairs.add(pair)
            left_payload = self._vocabulary[merge.left_id]
            right_payload = self._vocabulary[merge.right_id]
            actual_payload = self._vocabulary[merge.new_id]
            if left_payload is None or right_payload is None:
                raise TokenizerError(f"merge rank {rank} references a special token")
            if actual_payload != left_payload + right_payload:
                raise TokenizerError(f"merge rank {rank} payload does not match its parents")
