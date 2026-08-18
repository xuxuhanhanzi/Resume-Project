"""Small byte-level Unigram tokenizer with EM and stochastic segmentation."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from forgellm.tokenization.pretokenization import Pretokenization, split_text

_NEGATIVE_INFINITY = float("-inf")


def _logsumexp(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return _NEGATIVE_INFINITY
    maximum = max(items)
    if maximum == _NEGATIVE_INFINITY:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in items))


def _candidate_counts(texts: Sequence[str], max_piece_bytes: int) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        for piece in split_text(text, Pretokenization.UNICODE_CLASS):
            for start in range(len(piece)):
                upper = min(len(piece), start + max_piece_bytes)
                for end in range(start + 2, upper + 1):
                    counts[piece[start:end]] += 1
    return counts


class EducationalUnigram:
    """A fixed byte vocabulary scored by a simple Unigram language model."""

    def __init__(self, vocabulary: Sequence[bytes], log_probabilities: Sequence[float]) -> None:
        if len(vocabulary) != len(log_probabilities):
            raise ValueError("vocabulary and score lengths differ")
        if tuple(vocabulary[:256]) != tuple(bytes([byte]) for byte in range(256)):
            raise ValueError("the first 256 Unigram pieces must cover all bytes")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Unigram vocabulary contains duplicate pieces")
        self._vocabulary = tuple(vocabulary)
        self._scores = tuple(log_probabilities)
        by_first_byte: defaultdict[int, list[int]] = defaultdict(list)
        for token_id, piece in enumerate(self._vocabulary):
            by_first_byte[piece[0]].append(token_id)
        self._by_first_byte = {
            byte: tuple(
                sorted(token_ids, key=lambda token_id: (-len(self._vocabulary[token_id]), token_id))
            )
            for byte, token_ids in by_first_byte.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self._vocabulary)

    @property
    def unknown_token_id(self) -> None:
        return None

    @property
    def vocabulary(self) -> tuple[bytes, ...]:
        return self._vocabulary

    @property
    def log_probabilities(self) -> tuple[float, ...]:
        return self._scores

    def _edges(self, data: bytes, position: int) -> list[tuple[int, int]]:
        return [
            (position + len(self._vocabulary[token_id]), token_id)
            for token_id in self._by_first_byte[data[position]]
            if data.startswith(self._vocabulary[token_id], position)
        ]

    def _backward_scores(self, data: bytes, *, score_scale: float = 1.0) -> list[float]:
        backward = [_NEGATIVE_INFINITY] * (len(data) + 1)
        backward[len(data)] = 0.0
        for position in range(len(data) - 1, -1, -1):
            backward[position] = _logsumexp(
                self._scores[token_id] * score_scale + backward[end]
                for end, token_id in self._edges(data, position)
            )
        return backward

    def encode(self, text: str) -> list[int]:
        """Return the maximum-score segmentation using Viterbi dynamic programming."""
        data = text.encode("utf-8", errors="strict")
        best = [_NEGATIVE_INFINITY] * (len(data) + 1)
        backpointer: list[tuple[int, int] | None] = [None] * (len(data) + 1)
        best[0] = 0.0
        for position in range(len(data)):
            if best[position] == _NEGATIVE_INFINITY:
                continue
            for end, token_id in self._edges(data, position):
                score = best[position] + self._scores[token_id]
                current = backpointer[end]
                current_length = len(self._vocabulary[current[1]]) if current is not None else -1
                if score > best[end] or (
                    math.isclose(score, best[end])
                    and (len(self._vocabulary[token_id]), -token_id)
                    > (current_length, -(current[1] if current is not None else token_id))
                ):
                    best[end] = score
                    backpointer[end] = (position, token_id)
        if best[-1] == _NEGATIVE_INFINITY:
            raise AssertionError("byte fallback should make every input segmentable")
        output: list[int] = []
        position = len(data)
        while position:
            step = backpointer[position]
            if step is None:
                raise AssertionError("Viterbi backpointer is missing")
            position, token_id = step
            output.append(token_id)
        output.reverse()
        return output

    def sample_encode(self, text: str, *, seed: int, temperature: float = 1.0) -> list[int]:
        """Sample one valid segmentation from the Unigram posterior."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        data = text.encode("utf-8", errors="strict")
        score_scale = 1.0 / temperature
        backward = self._backward_scores(data, score_scale=score_scale)
        generator = random.Random(seed)
        output: list[int] = []
        position = 0
        while position < len(data):
            edges = self._edges(data, position)
            logits = [
                self._scores[token_id] * score_scale + backward[end] for end, token_id in edges
            ]
            maximum = max(logits)
            weights = [math.exp(logit - maximum) for logit in logits]
            selected = generator.choices(range(len(edges)), weights=weights, k=1)[0]
            end, token_id = edges[selected]
            output.append(token_id)
            position = end
        return output

    def decode(self, token_ids: Iterable[int]) -> str:
        payload = bytearray()
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("token IDs must be integers")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID out of range: {token_id}")
            payload.extend(self._vocabulary[token_id])
        return bytes(payload).decode("utf-8", errors="strict")


def _expected_counts(model: EducationalUnigram, byte_sequences: Sequence[bytes]) -> list[float]:
    expected = [0.0] * model.vocab_size
    for data in byte_sequences:
        if not data:
            continue
        forward = [_NEGATIVE_INFINITY] * (len(data) + 1)
        forward[0] = 0.0
        for position in range(len(data)):
            if forward[position] == _NEGATIVE_INFINITY:
                continue
            for end, token_id in model._edges(data, position):
                forward[end] = _logsumexp(
                    (forward[end], forward[position] + model.log_probabilities[token_id])
                )
        backward = model._backward_scores(data)
        normalizer = forward[-1]
        for position in range(len(data)):
            for end, token_id in model._edges(data, position):
                log_count = (
                    forward[position]
                    + model.log_probabilities[token_id]
                    + backward[end]
                    - normalizer
                )
                expected[token_id] += math.exp(log_count)
    return expected


def _scores_from_counts(counts: Sequence[float]) -> list[float]:
    smoothed = [count + 1e-8 for count in counts]
    total = sum(smoothed)
    return [math.log(count / total) for count in smoothed]


def train_unigram(
    texts: Sequence[str],
    *,
    vocab_size: int,
    max_piece_bytes: int = 8,
    em_iterations: int = 4,
    seed_multiplier: int = 4,
) -> EducationalUnigram:
    """Train a compact byte Unigram model with forward-backward EM.

    Candidate pruning uses expected token counts rather than SentencePiece's exact
    loss-change pruning rule. The forward-backward E-step and Viterbi/sampling paths
    are implemented directly so the core Unigram method remains visible.
    """
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256")
    if max_piece_bytes < 2:
        raise ValueError("max_piece_bytes must be at least 2")
    if em_iterations < 1:
        raise ValueError("em_iterations must be at least 1")
    if seed_multiplier < 1:
        raise ValueError("seed_multiplier must be at least 1")
    if not texts:
        raise ValueError("Unigram training requires at least one text")

    candidate_counts = _candidate_counts(texts, max_piece_bytes)
    learned_budget = max(0, vocab_size - 256)
    seed_budget = learned_budget * seed_multiplier
    learned_candidates = [
        piece
        for piece, _ in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0]))[
            :seed_budget
        ]
    ]
    vocabulary = [bytes([byte]) for byte in range(256)] + learned_candidates
    raw_byte_counts: Counter[int] = Counter(
        byte for text in texts for byte in text.encode("utf-8", errors="strict")
    )
    initial_counts = [float(raw_byte_counts[token_id] + 1) for token_id in range(256)]
    initial_counts.extend(float(candidate_counts[piece]) for piece in learned_candidates)
    model = EducationalUnigram(vocabulary, _scores_from_counts(initial_counts))
    byte_sequences = [text.encode("utf-8", errors="strict") for text in texts]

    expected = initial_counts
    for _ in range(em_iterations):
        expected = _expected_counts(model, byte_sequences)
        model = EducationalUnigram(vocabulary, _scores_from_counts(expected))

    if len(vocabulary) > vocab_size:
        retained_learned_ids = sorted(
            range(256, len(vocabulary)),
            key=lambda token_id: (-expected[token_id], vocabulary[token_id]),
        )[:learned_budget]
        retained_ids = [*range(256), *retained_learned_ids]
        vocabulary = [vocabulary[token_id] for token_id in retained_ids]
        retained_counts = [expected[token_id] for token_id in retained_ids]
        model = EducationalUnigram(vocabulary, _scores_from_counts(retained_counts))
        final_expected = _expected_counts(model, byte_sequences)
        model = EducationalUnigram(vocabulary, _scores_from_counts(final_expected))
    return model
