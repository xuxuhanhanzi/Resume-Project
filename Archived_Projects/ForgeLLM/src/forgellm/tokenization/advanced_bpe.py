"""Educational BPE variants for the Stage 1 method laboratory.

The implementations favor visible state transitions over production performance.
They are suitable for learning and deterministic tests, not large-corpus training.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from forgellm.tokenization.pretokenization import Pretokenization, split_text


class BPEEventKind(StrEnum):
    """An event replayed by the encoder in training order."""

    MERGE = "merge"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class BPEEvent:
    """One merge or vocabulary-removal event."""

    kind: BPEEventKind
    token_id: int
    left_id: int
    right_id: int
    frequency: int
    phase: int = 1


def _replace_pair(
    sequence: Sequence[int],
    pair: tuple[int, int],
    token_id: int,
    *,
    dropout: float = 0.0,
    generator: random.Random | None = None,
) -> list[int]:
    output: list[int] = []
    index = 0
    while index < len(sequence):
        matches = (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        )
        dropped = (
            matches and dropout > 0.0 and generator is not None and generator.random() < dropout
        )
        if matches and not dropped:
            output.append(token_id)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return output


def _expand_token(sequence: Sequence[int], token_id: int, left_id: int, right_id: int) -> list[int]:
    output: list[int] = []
    for item in sequence:
        if item == token_id:
            output.extend((left_id, right_id))
        else:
            output.append(item)
    return output


def _pair_counts(sequences: Iterable[Sequence[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:], strict=False))
    return counts


def _best_pair(
    sequences: Iterable[Sequence[int]], min_pair_frequency: int
) -> tuple[tuple[int, int], int] | None:
    counts = _pair_counts(sequences)
    if not counts:
        return None
    pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
    if frequency < min_pair_frequency:
        return None
    return pair, frequency


def _pieces_for_documents(texts: Sequence[str], mode: Pretokenization) -> list[list[list[int]]]:
    return [[[byte for byte in piece] for piece in split_text(text, mode)] for text in texts]


def _all_pieces(documents: Sequence[Sequence[Sequence[int]]]) -> list[list[int]]:
    return [list(piece) for document in documents for piece in document]


def _replace_in_documents(
    documents: list[list[list[int]]], pair: tuple[int, int], token_id: int
) -> None:
    for document_index, document in enumerate(documents):
        documents[document_index] = [_replace_pair(piece, pair, token_id) for piece in document]


class EducationalBPE:
    """Replayable byte BPE with optional boundaries, dropout, and two phases."""

    def __init__(
        self,
        *,
        token_payloads: dict[int, bytes],
        active_token_ids: set[int],
        events: Sequence[BPEEvent],
        pretokenization: Pretokenization,
    ) -> None:
        self._token_payloads = dict(token_payloads)
        self._active_internal_ids = frozenset(active_token_ids)
        self._events = tuple(events)
        self.pretokenization = pretokenization
        ordered_internal_ids = sorted(active_token_ids)
        self._internal_to_external = {
            internal_id: external_id for external_id, internal_id in enumerate(ordered_internal_ids)
        }
        self._external_payloads = tuple(
            self._token_payloads[internal_id] for internal_id in ordered_internal_ids
        )
        self._validate()

    @property
    def vocab_size(self) -> int:
        return len(self._external_payloads)

    @property
    def unknown_token_id(self) -> None:
        return None

    @property
    def events(self) -> tuple[BPEEvent, ...]:
        return self._events

    @property
    def merge_count(self) -> int:
        return sum(event.kind is BPEEventKind.MERGE for event in self._events)

    @property
    def removal_count(self) -> int:
        return sum(event.kind is BPEEventKind.REMOVE for event in self._events)

    @property
    def vocabulary(self) -> tuple[bytes, ...]:
        return self._external_payloads

    def _validate(self) -> None:
        if not set(range(256)).issubset(self._active_internal_ids):
            raise ValueError("all byte tokens must remain active")
        if not self._active_internal_ids <= self._token_payloads.keys():
            raise ValueError("active token has no byte payload")
        for event in self._events:
            expected = self._token_payloads[event.left_id] + self._token_payloads[event.right_id]
            if self._token_payloads[event.token_id] != expected:
                raise ValueError("event payload does not equal its parent bytes")

    def encode(
        self,
        text: str,
        *,
        dropout: float = 0.0,
        seed: int = 0,
    ) -> list[int]:
        """Encode text; non-zero dropout randomly skips individual merge occurrences."""
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        pieces = [[byte for byte in piece] for piece in split_text(text, self.pretokenization)]
        generator = random.Random(seed)
        phase_one = [event for event in self._events if event.phase == 1]
        phase_two = [event for event in self._events if event.phase == 2]
        for event in phase_one:
            if event.kind is BPEEventKind.MERGE:
                pieces = [
                    _replace_pair(
                        piece,
                        (event.left_id, event.right_id),
                        event.token_id,
                        dropout=dropout,
                        generator=generator,
                    )
                    for piece in pieces
                ]
            else:
                pieces = [
                    _expand_token(piece, event.token_id, event.left_id, event.right_id)
                    for piece in pieces
                ]

        sequence = [token_id for piece in pieces for token_id in piece]
        for event in phase_two:
            if event.kind is BPEEventKind.MERGE:
                sequence = _replace_pair(
                    sequence,
                    (event.left_id, event.right_id),
                    event.token_id,
                    dropout=dropout,
                    generator=generator,
                )
            else:
                sequence = _expand_token(sequence, event.token_id, event.left_id, event.right_id)
        try:
            return [self._internal_to_external[token_id] for token_id in sequence]
        except KeyError as error:
            raise AssertionError("encoding produced a removed token") from error

    def decode(self, token_ids: Iterable[int]) -> str:
        payload = bytearray()
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            payload.extend(self._external_payloads[token_id])
        return bytes(payload).decode("utf-8", errors="strict")


def train_classic_bpe(
    texts: Sequence[str],
    *,
    vocab_size: int,
    min_pair_frequency: int = 2,
    pretokenization: Pretokenization = Pretokenization.NONE,
) -> EducationalBPE:
    """Train classical greedy BPE while respecting fixed pre-token boundaries."""
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")
    documents = _pieces_for_documents(texts, pretokenization)
    payloads = {byte: bytes([byte]) for byte in range(256)}
    active = set(range(256))
    events: list[BPEEvent] = []
    next_id = 256
    while len(active) < vocab_size:
        selected = _best_pair(_all_pieces(documents), min_pair_frequency)
        if selected is None:
            break
        pair, frequency = selected
        payloads[next_id] = payloads[pair[0]] + payloads[pair[1]]
        events.append(BPEEvent(BPEEventKind.MERGE, next_id, pair[0], pair[1], frequency))
        active.add(next_id)
        _replace_in_documents(documents, pair, next_id)
        next_id += 1
    return EducationalBPE(
        token_payloads=payloads,
        active_token_ids=active,
        events=events,
        pretokenization=pretokenization,
    )


def train_super_bpe(
    texts: Sequence[str],
    *,
    vocab_size: int,
    subword_vocab_size: int,
    min_pair_frequency: int = 2,
) -> EducationalBPE:
    """Train a two-phase SuperBPE-style curriculum.

    Phase 1 learns within Unicode-class pieces. Phase 2 removes those boundaries and
    permits merges across whitespace. This is a small educational reproduction of the
    curriculum idea, not the paper's large-scale training system.
    """
    if not 256 <= subword_vocab_size <= vocab_size:
        raise ValueError("subword_vocab_size must be in [256, vocab_size]")
    documents = _pieces_for_documents(texts, Pretokenization.UNICODE_CLASS)
    payloads = {byte: bytes([byte]) for byte in range(256)}
    active = set(range(256))
    events: list[BPEEvent] = []
    next_id = 256
    while len(active) < subword_vocab_size:
        selected = _best_pair(_all_pieces(documents), min_pair_frequency)
        if selected is None:
            break
        pair, frequency = selected
        payloads[next_id] = payloads[pair[0]] + payloads[pair[1]]
        events.append(BPEEvent(BPEEventKind.MERGE, next_id, pair[0], pair[1], frequency, phase=1))
        active.add(next_id)
        _replace_in_documents(documents, pair, next_id)
        next_id += 1

    whole_documents = [
        [token_id for piece in document for token_id in piece] for document in documents
    ]
    while len(active) < vocab_size:
        selected = _best_pair(whole_documents, min_pair_frequency)
        if selected is None:
            break
        pair, frequency = selected
        payloads[next_id] = payloads[pair[0]] + payloads[pair[1]]
        events.append(BPEEvent(BPEEventKind.MERGE, next_id, pair[0], pair[1], frequency, phase=2))
        active.add(next_id)
        whole_documents = [_replace_pair(document, pair, next_id) for document in whole_documents]
        next_id += 1
    return EducationalBPE(
        token_payloads=payloads,
        active_token_ids=active,
        events=events,
        pretokenization=Pretokenization.UNICODE_CLASS,
    )


def _token_usage(sequences: Iterable[Sequence[int]]) -> Counter[int]:
    usage: Counter[int] = Counter()
    for sequence in sequences:
        usage.update(sequence)
    return usage


def train_picky_bpe(
    texts: Sequence[str],
    *,
    vocab_size: int,
    min_pair_frequency: int = 2,
    removal_threshold: float = 0.5,
    refinement_steps: int = 16,
) -> EducationalBPE:
    """Train a deterministic Picky-BPE-inspired vocabulary refinement model.

    First reach the requested active vocabulary size. Each refinement step then adds
    one useful merge and removes one older learned token whose current use divided by
    its creation frequency is at or below ``removal_threshold``. The encoder replays
    both merge and removal events, making the algorithm inspectable.

    The paper uses a more complete likelihood-aware removal procedure. This compact
    version reproduces the central add/remove event mechanism for study and tests.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")
    if not 0.0 <= removal_threshold <= 1.0:
        raise ValueError("removal_threshold must be in [0, 1]")
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")

    documents = _pieces_for_documents(texts, Pretokenization.UNICODE_CLASS)
    payloads = {byte: bytes([byte]) for byte in range(256)}
    parents: dict[int, tuple[int, int]] = {}
    creation_frequency: dict[int, int] = {}
    active = set(range(256))
    events: list[BPEEvent] = []
    next_id = 256

    def add_best_merge() -> bool:
        nonlocal next_id
        selected = _best_pair(_all_pieces(documents), min_pair_frequency)
        if selected is None:
            return False
        pair, frequency = selected
        payloads[next_id] = payloads[pair[0]] + payloads[pair[1]]
        parents[next_id] = pair
        creation_frequency[next_id] = frequency
        active.add(next_id)
        events.append(BPEEvent(BPEEventKind.MERGE, next_id, pair[0], pair[1], frequency))
        _replace_in_documents(documents, pair, next_id)
        next_id += 1
        return True

    while len(active) < vocab_size and add_best_merge():
        pass

    for _ in range(refinement_steps):
        if len(active) < vocab_size or not add_best_merge():
            break
        usage = _token_usage(_all_pieces(documents))
        new_token_id = next_id - 1
        candidates = []
        for token_id in active:
            if token_id < 256 or token_id == new_token_id:
                continue
            ratio = usage[token_id] / creation_frequency[token_id]
            if ratio <= removal_threshold:
                candidates.append((ratio, usage[token_id], token_id))
        if not candidates:
            active.remove(new_token_id)
            pair = parents[new_token_id]
            for document_index, document in enumerate(documents):
                documents[document_index] = [
                    _expand_token(piece, new_token_id, pair[0], pair[1]) for piece in document
                ]
            events.append(
                BPEEvent(
                    BPEEventKind.REMOVE,
                    new_token_id,
                    pair[0],
                    pair[1],
                    usage[new_token_id],
                )
            )
            break
        _, token_frequency, removed_id = min(candidates)
        left_id, right_id = parents[removed_id]
        active.remove(removed_id)
        for document_index, document in enumerate(documents):
            documents[document_index] = [
                _expand_token(piece, removed_id, left_id, right_id) for piece in document
            ]
        events.append(
            BPEEvent(
                BPEEventKind.REMOVE,
                removed_id,
                left_id,
                right_id,
                token_frequency,
            )
        )

    return EducationalBPE(
        token_payloads=payloads,
        active_token_ids=active,
        events=events,
        pretokenization=Pretokenization.UNICODE_CLASS,
    )
