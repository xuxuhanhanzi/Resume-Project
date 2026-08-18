"""Toy entropy patching for understanding tokenizer-free byte models."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BytePatch:
    """One exact byte span and the surprisal that opened it."""

    start: int
    end: int
    opening_surprisal_bits: float
    payload: bytes


class BigramEntropyPatcher:
    """Group bytes dynamically using a smoothed byte-bigram predictor.

    BLT uses a learned entropy model. This toy version makes the boundary rule visible:
    a surprising next byte starts a new patch, while predictable bytes share a patch.
    """

    def __init__(self, transition_counts: dict[int, Counter[int]], *, smoothing: float) -> None:
        if smoothing <= 0.0:
            raise ValueError("smoothing must be positive")
        self._transition_counts = {
            previous: Counter(counts) for previous, counts in transition_counts.items()
        }
        self.smoothing = smoothing

    @classmethod
    def train(cls, texts: Sequence[str], *, smoothing: float = 0.1) -> BigramEntropyPatcher:
        counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        for text in texts:
            previous = 256  # explicit beginning-of-document context
            for byte in text.encode("utf-8", errors="strict"):
                counts[previous][byte] += 1
                previous = byte
        return cls(dict(counts), smoothing=smoothing)

    def surprisal_bits(self, previous: int, byte: int) -> float:
        counts = self._transition_counts.get(previous, Counter())
        denominator = sum(counts.values()) + self.smoothing * 256
        probability = (counts[byte] + self.smoothing) / denominator
        return -math.log2(probability)

    def patch(
        self,
        text: str,
        *,
        threshold_bits: float = 4.0,
        max_patch_bytes: int = 8,
    ) -> list[BytePatch]:
        if threshold_bits < 0.0:
            raise ValueError("threshold_bits must be non-negative")
        if max_patch_bytes < 1:
            raise ValueError("max_patch_bytes must be at least 1")
        data = text.encode("utf-8", errors="strict")
        if not data:
            return []
        patches: list[BytePatch] = []
        start = 0
        opening_surprisal = self.surprisal_bits(256, data[0])
        previous = data[0]
        for position in range(1, len(data)):
            surprisal = self.surprisal_bits(previous, data[position])
            should_split = surprisal >= threshold_bits or position - start >= max_patch_bytes
            if should_split:
                patches.append(BytePatch(start, position, opening_surprisal, data[start:position]))
                start = position
                opening_surprisal = surprisal
            previous = data[position]
        patches.append(BytePatch(start, len(data), opening_surprisal, data[start:]))
        return patches

    @staticmethod
    def decode(patches: Iterable[BytePatch]) -> str:
        return b"".join(patch.payload for patch in patches).decode("utf-8", errors="strict")
