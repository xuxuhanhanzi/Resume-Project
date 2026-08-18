"""Algorithm tests for classical, stochastic, Picky, and Super BPE variants."""

import pytest

from forgellm.tokenization.advanced_bpe import (
    BPEEventKind,
    train_classic_bpe,
    train_picky_bpe,
    train_super_bpe,
)
from forgellm.tokenization.pretokenization import Pretokenization


@pytest.mark.parametrize("text", ["", "new york", "中文🙂", "a  b\n"])
def test_classic_pretokenized_bpe_round_trips(text: str) -> None:
    model = train_classic_bpe(
        ["new york new york", "中文 中文", text],
        vocab_size=280,
        min_pair_frequency=1,
        pretokenization=Pretokenization.UNICODE_CLASS,
    )

    assert model.decode(model.encode(text)) == text


def test_pretokenization_prevents_cross_boundary_tokens() -> None:
    texts = ["new york new york new york"] * 4
    model = train_classic_bpe(
        texts,
        vocab_size=280,
        min_pair_frequency=2,
        pretokenization=Pretokenization.UNICODE_CLASS,
    )

    assert all(b"new york" not in piece for piece in model.vocabulary)


def test_super_bpe_curriculum_learns_cross_space_expression() -> None:
    texts = ["new york new york new york"] * 8
    model = train_super_bpe(
        texts,
        vocab_size=276,
        subword_vocab_size=268,
        min_pair_frequency=2,
    )

    assert any(event.phase == 2 for event in model.events)
    assert any(b"new york" in piece for piece in model.vocabulary)
    assert model.decode(model.encode("new york")) == "new york"


def test_picky_bpe_replaces_underused_tokens_and_preserves_round_trip() -> None:
    texts = [
        "lowest lower newest widest",
        "low low lower newest",
        "中文 中文 tokenizer tokenizer",
    ] * 5
    model = train_picky_bpe(
        texts,
        vocab_size=276,
        min_pair_frequency=2,
        removal_threshold=1.0,
        refinement_steps=6,
    )

    assert model.vocab_size == 276
    assert model.removal_count > 0
    assert any(event.kind is BPEEventKind.REMOVE for event in model.events)
    for text in [*texts[:3], "unseen bytes 🙂"]:
        assert model.decode(model.encode(text)) == text


def test_bpe_dropout_changes_segmentation_without_changing_bytes() -> None:
    text = "banana banana banana"
    model = train_classic_bpe(
        [text] * 8,
        vocab_size=280,
        min_pair_frequency=2,
        pretokenization=Pretokenization.NONE,
    )
    segmentations = {tuple(model.encode(text, dropout=0.35, seed=seed)) for seed in range(12)}

    assert len(segmentations) > 1
    assert all(model.decode(segmentation) == text for segmentation in segmentations)
